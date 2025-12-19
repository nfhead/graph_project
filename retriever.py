import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from .config_loader import Config


def _tokenize(text: str) -> List[str]:
    return [t for t in "".join(ch if ch.isalnum() else " " for ch in text.lower()).split() if t]


def _rrf_merge(
    faiss_hits: List[Tuple[int, float]],
    bm25_hits: List[Tuple[int, float]],
    k: int,
    rrf_k: int = 60
) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion:
    score(doc) = sum(1 / (rrf_k + rank))
    """
    scores: Dict[int, float] = {}

    for rank, (idx, _s) in enumerate(faiss_hits):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

    for rank, (idx, _s) in enumerate(bm25_hits):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return merged[:k]


class Retriever:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        index_dir = Path(cfg.paths["index_dir"])

        self.index = faiss.read_index(str(index_dir / "faiss_index.bin"))

        self.metadata: List[Dict] = []
        with open(index_dir / "docs_metadata.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))

        model_name = cfg.embedding["model_name"]
        self.embed_model = SentenceTransformer(model_name)

        # BM25
        bm25_path = index_dir / "bm25.pkl"
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                self.bm25: BM25Okapi = pickle.load(f)
            self.has_bm25 = True
        else:
            self.bm25 = None
            self.has_bm25 = False

    def embed_query(self, query: str):
        emb = self.embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype("float32")

    def _faiss_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        q_emb = self.embed_query(query)
        scores, idxs = self.index.search(q_emb, k)
        idxs = idxs[0]
        scores = scores[0]
        hits = []
        for i, s in zip(idxs, scores):
            if i < 0:
                continue
            hits.append((int(i), float(s)))
        return hits

    def _bm25_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        if not self.has_bm25:
            return []
        q_tok = _tokenize(query)
        scores = self.bm25.get_scores(q_tok)
        top_idx = np.argsort(scores)[::-1][:k]
        hits = [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]
        return hits

    def retrieve(self, query: str, k: int = 10) -> List[Dict]:
        # pull more from each channel then merge
        k_each = max(20, k * 3)

        faiss_hits = self._faiss_search(query, k_each)
        bm25_hits = self._bm25_search(query, k_each)

        if bm25_hits:
            merged = _rrf_merge(faiss_hits, bm25_hits, k=k)
            results = []
            for idx, mscore in merged:
                doc = self.metadata[idx].copy()
                doc["score"] = float(mscore)
                results.append(doc)
            return results

        # fallback: only FAISS
        results = []
        for idx, s in faiss_hits[:k]:
            doc = self.metadata[idx].copy()
            doc["score"] = float(s)
            results.append(doc)
        return results
