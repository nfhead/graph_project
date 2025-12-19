import json
import pickle
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from .config_loader import Config


def _tokenize(text: str) -> List[str]:
    return [t for t in "".join(ch if ch.isalnum() else " " for ch in text.lower()).split() if t]


def load_docs(meta_dir: Path) -> List[Dict]:
    docs: List[Dict] = []

    for jsonl_file in meta_dir.glob("*_docs.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                docs.append(json.loads(line))

    manual_curves = meta_dir / "manual_curves.jsonl"
    if manual_curves.exists():
        print(f"[INFO] Loading manual curves from: {manual_curves}")
        with open(manual_curves, "r", encoding="utf-8") as f:
            for line in f:
                docs.append(json.loads(line))
    else:
        print("[INFO] No manual_curves.jsonl found; index will not contain curve-level docs.")

    return docs


def build_index():
    cfg = Config()
    meta_dir = Path(cfg.paths["metadata_dir"])
    index_dir = Path(cfg.paths["index_dir"])
    index_dir.mkdir(parents=True, exist_ok=True)

    docs = load_docs(meta_dir)
    print(f"[INFO] Loaded {len(docs)} docs for indexing")

    if not docs:
        raise RuntimeError("No docs found to index. Did you run pdf_ingest and create manual_curves.jsonl?")

    model_name = cfg.embedding["model_name"]
    batch_size = cfg.embedding.get("batch_size", 16)
    model = SentenceTransformer(model_name)

    texts = [d.get("text", "") for d in docs]

    # --- FAISS embeddings ---
    embeddings_chunks = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = model.encode(
            batch,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        embeddings_chunks.append(emb.astype("float32"))

    embeddings = np.vstack(embeddings_chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(index_dir / "faiss_index.bin"))

    # --- BM25 ---
    tokenized_corpus = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    with open(index_dir / "bm25_corpus_tokens.pkl", "wb") as f:
        pickle.dump(tokenized_corpus, f)

    # Save metadata in same order as embeddings/corpus
    meta_path = index_dir / "docs_metadata.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"[OK] Index built (FAISS + BM25) and saved to: {index_dir}")


if __name__ == "__main__":
    build_index()
