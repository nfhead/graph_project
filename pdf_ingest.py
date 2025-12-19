import argparse
import json
from pathlib import Path
from typing import List

import fitz  # pymupdf
import cv2
import pytesseract

from .config_loader import Config
from .graph_metadata import GraphMetadata
from .utils import split_into_paragraphs, chunk_text, extract_figure_captions


def ocr_graph_image(img_path: Path) -> str:
    img = cv2.imread(str(img_path))
    if img is None:
        return ""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    h, w = gray.shape[:2]
    if max(h, w) < 800:
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # tesseract config: you can tune
    text = pytesseract.image_to_string(gray, config="--psm 6")
    text = text.replace("\r", "\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


def _extract_bottom_table_like_text(page: fitz.Page) -> str:
    """
    Heuristic extraction of table-like text near bottom half of the page using PDF text (NOT OCR).
    Works when table is real selectable text.
    """
    d = page.get_text("dict")
    blocks = d.get("blocks", [])
    page_h = float(page.rect.height)

    lines_out = []
    for b in blocks:
        if b.get("type") != 0:
            continue  # not text
        x0, y0, x1, y1 = b.get("bbox", [0, 0, 0, 0])
        # focus bottom region (where tables often are)
        if y0 < 0.55 * page_h:
            continue

        # collect block text
        block_text_parts = []
        for ln in b.get("lines", []):
            span_text = "".join(sp.get("text", "") for sp in ln.get("spans", []))
            span_text = span_text.strip()
            if span_text:
                block_text_parts.append(span_text)
        block_text = "\n".join(block_text_parts).strip()
        if not block_text:
            continue

        # heuristic: lots of numbers / columns
        num_count = sum(ch.isdigit() for ch in block_text)
        if num_count >= 8 and (("  " in block_text) or ("\t" in block_text) or ("\n" in block_text)):
            lines_out.append(block_text)

    # merge
    merged = "\n\n".join(lines_out).strip()
    return merged


def ingest_pdf(pdf_path: Path, cfg: Config):
    pdf_name = pdf_path.name
    pages_dir = Path(cfg.paths["pages_dir"])
    graphs_dir = Path(cfg.paths["graphs_dir"])
    meta_dir = Path(cfg.paths["metadata_dir"])

    pages_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    max_chars = cfg.ingest.get("max_chars_per_chunk", 2500)
    overlap = cfg.ingest.get("chunk_overlap", 200)
    min_caption_len = cfg.ingest.get("min_graph_caption_len", 20)

    doc = fitz.open(pdf_path)
    docs_jsonl: List[dict] = []
    graphs_meta: List[GraphMetadata] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_number = page_idx + 1

        # ---------- TEXT ----------
        page_text = page.get_text("text")
        page_text = page_text.replace("\r", "\n")
        lines = [ln for ln in page_text.splitlines() if len(ln.strip()) > 3]
        page_text = "\n".join(lines)

        paragraphs = split_into_paragraphs(page_text)
        chunks = chunk_text(paragraphs, max_chars, overlap)
        for ci, ch in enumerate(chunks):
            doc_id = f"{pdf_name}_page_{page_number:04d}_chunk_{ci:03d}"
            docs_jsonl.append({
                "doc_id": doc_id,
                "type": "text_chunk",
                "pdf_name": pdf_name,
                "page_number": page_number,
                "text": ch,
            })

        # ---------- TABLE-LIKE TEXT FROM PDF (NEW) ----------
        table_text = _extract_bottom_table_like_text(page)
        if table_text:
            table_doc_id = f"{pdf_name}_page_{page_number:04d}_table_000"
            docs_jsonl.append({
                "doc_id": table_doc_id,
                "type": "table",
                "pdf_name": pdf_name,
                "page_number": page_number,
                "text": (
                    f"TABLE on page {page_number} of PDF {pdf_name} (extracted from PDF text, not OCR):\n"
                    f"{table_text}"
                ),
            })

        # ---------- CAPTIONS ----------
        captions = extract_figure_captions(page_text)

        # ---------- IMAGES (GRAPHS) ----------
        images = page.get_images(full=True)

        for img_idx, img_info in enumerate(images):
            xref = img_info[0]
            try:
                pix = fitz.Pixmap(doc, xref)
            except Exception:
                continue

            img_name = f"{pdf_name}_page_{page_number:04d}_img_{img_idx:03d}.png"
            img_path = graphs_dir / img_name

            try:
                if pix.colorspace is None or pix.colorspace.n != 3:
                    pix_converted = fitz.Pixmap(fitz.csRGB, pix)
                else:
                    pix_converted = pix
                pix_converted.save(img_path)
            except Exception as e:
                print(f"[WARN] Skipping image {img_name} due to error: {e}")
            finally:
                pix = None
                pix_converted = None

            # Caption/nearby
            caption_text = ""
            nearby_text = ""
            if captions:
                cap_index = min(img_idx, len(captions) - 1)
                cap_line, line_idx = captions[cap_index]
                caption_text = cap_line

                lines_page = page_text.splitlines()
                start = max(0, line_idx - 3)
                end = min(len(lines_page), line_idx + 4)
                nearby_text = "\n".join(lines_page[start:end])

            if len(caption_text) < min_caption_len:
                pass

            # OCR inside graph image (keep)
            ocr_text = ocr_graph_image(img_path)

            graph_id = f"{pdf_name}_page_{page_number:04d}_graph_{img_idx:03d}"

            gm = GraphMetadata(
                graph_id=graph_id,
                pdf_name=pdf_name,
                page_number=page_number,
                image_path=str(img_path),
                caption=caption_text,
                nearby_text=nearby_text,
                axes_info={},
                curves=[],
                extra_tags=[],
            )
            graphs_meta.append(gm)

            graph_doc_id = f"graph_{graph_id}"
            graph_text_repr = (
                f"GRAPH {graph_id} in PDF {pdf_name} on page {page_number}.\n"
                f"Caption: {caption_text}\n"
                f"Nearby text on page:\n{nearby_text}\n"
                f"Text inside the graph image (OCR):\n{ocr_text}"
            )

            docs_jsonl.append({
                "doc_id": graph_doc_id,
                "type": "graph",
                "graph_id": graph_id,
                "pdf_name": pdf_name,
                "page_number": page_number,
                "text": graph_text_repr,
            })

    docs_path = meta_dir / f"{pdf_name}_docs.jsonl"
    with open(docs_path, "w", encoding="utf-8") as f:
        for d in docs_jsonl:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    graphs_path = meta_dir / f"{pdf_name}_graphs.jsonl"
    with open(graphs_path, "w", encoding="utf-8") as f:
        for gm in graphs_meta:
            f.write(json.dumps(gm.to_dict(), ensure_ascii=False) + "\n")

    print(f"[OK] Saved docs to:   {docs_path}")
    print(f"[OK] Saved graphs to: {graphs_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str, required=True, help="Path to input PDF (e.g. data/pdfs/manual.pdf)")
    args = parser.parse_args()
    cfg = Config()
    ingest_pdf(Path(args.pdf), cfg)


if __name__ == "__main__":
    main()
