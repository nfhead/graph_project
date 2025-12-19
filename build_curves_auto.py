import json
import signal
from pathlib import Path
from typing import Dict, Any

from .config_loader import Config
from .curve_extractor import detect_curves


# ===============================
# TIMEOUT SUPPORT
# ===============================
class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutException()


signal.signal(signal.SIGALRM, _timeout_handler)


# ===============================
# MAIN FUNCTION
# ===============================
def build_curves_for_pdf(
    pdf_name: str,
    max_curves_per_graph: int = 5,
    timeout_seconds: int = 30,
):
    """
    For a given PDF (by name):
    - Read <pdf>_graphs.jsonl
    - Run curve detection on each graph image
    - Write curve-level docs to manual_curves.jsonl

    SAFETY FEATURES:
    - Per-graph timeout
    - Skip bad / slow images
    - Continue on all errors
    """

    cfg = Config()
    meta_dir = Path(cfg.paths["metadata_dir"])

    graphs_file = meta_dir / f"{pdf_name}_graphs.jsonl"
    curves_file = meta_dir / "manual_curves.jsonl"

    if not graphs_file.exists():
        raise FileNotFoundError(f"Graphs metadata not found: {graphs_file}")

    print(f"[INFO] Reading graphs metadata: {graphs_file}")
    print(f"[INFO] Writing curves to: {curves_file}")
    print(f"[INFO] Max curves per graph: {max_curves_per_graph}")
    print(f"[INFO] Timeout per graph: {timeout_seconds}s")

    # Overwrite old file
    curves_file.unlink(missing_ok=True)

    total_graphs = 0
    total_curves = 0
    skipped_graphs = 0

    with open(graphs_file, "r", encoding="utf-8") as gf, \
         open(curves_file, "w", encoding="utf-8") as cf:

        for line_no, line in enumerate(gf, start=1):
            line = line.strip()
            if not line:
                continue

            g = json.loads(line)
            graph_id = g["graph_id"]
            pdf_name_in_meta = g.get("pdf_name", pdf_name)
            page_number = g["page_number"]
            image_path = g["image_path"]
            caption = g.get("caption", "")

            total_graphs += 1
            print(
                f"[INFO] ({total_graphs}) Processing graph {graph_id} "
                f"(page {page_number})",
                flush=True,
            )

            try:
                # -------- HARD TIMEOUT --------
                signal.alarm(timeout_seconds)
                detected = detect_curves(
                    image_path,
                    max_curves=max_curves_per_graph,
                )

            except TimeoutException:
                skipped_graphs += 1
                print(
                    f"[WARN] Timeout ({timeout_seconds}s) for {graph_id}, skipping.",
                    flush=True,
                )
                continue

            except Exception as e:
                skipped_graphs += 1
                print(
                    f"[WARN] Curve detection failed for {graph_id}: {e}",
                    flush=True,
                )
                continue

            finally:
                signal.alarm(0)  # disable alarm

            if not detected:
                print(f"[INFO] No curves detected for {graph_id}.")
                continue

            # -------- WRITE CURVES --------
            for dc in detected:
                curve_id = f"{graph_id}_{dc.local_id}"

                numeric_profile_px: Dict[str, Any] = {
                    "x_min_px": dc.bbox["x_min_px"],
                    "x_max_px": dc.bbox["x_max_px"],
                    "y_min_px": dc.bbox["y_min_px"],
                    "y_max_px": dc.bbox["y_max_px"],
                }

                doc = {
                    "doc_id": f"curve_{curve_id}",
                    "type": "curve",
                    "curve_id": curve_id,
                    "graph_id": graph_id,
                    "pdf_name": pdf_name_in_meta,
                    "page_number": page_number,
                    "label": dc.local_id,          # curve_A / curve_B
                    "color_name": dc.color_name,
                    "color_rgb": dc.color_rgb,
                    "numeric_profile": numeric_profile_px,
                    "text": (
                        f"Curve {dc.local_id} in graph {graph_id} on page {page_number} "
                        f"of PDF {pdf_name_in_meta}. "
                        f"Detected as a {dc.color_name} curve. "
                        f"Caption: {caption}. "
                        f"Approximate pixel extent: {numeric_profile_px}."
                    ),
                }

                cf.write(json.dumps(doc, ensure_ascii=False) + "\n")
                total_curves += 1

    print("\n================ SUMMARY ================")
    print(f"[OK] Total graphs processed : {total_graphs}")
    print(f"[OK] Total curves detected  : {total_curves}")
    print(f"[OK] Graphs skipped         : {skipped_graphs}")
    print(f"[OK] Output file            : {curves_file}")
    print("========================================")


# ===============================
# CLI ENTRY
# ===============================
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf-name",
        type=str,
        required=True,
        help="PDF base name used in metadata (e.g. MIL-STD-810H.pdf)",
    )
    parser.add_argument(
        "--max-curves-per-graph",
        type=int,
        default=5,
        help="Maximum number of curves to detect per graph.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=30,
        help="Maximum time allowed per graph (seconds).",
    )

    args = parser.parse_args()

    build_curves_for_pdf(
        pdf_name=args.pdf_name,
        max_curves_per_graph=args.max_curves_per_graph,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    main()
