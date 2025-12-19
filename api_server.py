from pathlib import Path
import json
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config_loader import Config
from .retriever import Retriever
from .ranker import rank_graphs
from .llm_client import LLMClient

cfg = Config()
INDEX_DIR = Path(cfg.paths["index_dir"])

_ALL_DOCS_CACHE: List[Dict] = []

def get_all_docs() -> List[Dict]:
    global _ALL_DOCS_CACHE
    if _ALL_DOCS_CACHE:
        return _ALL_DOCS_CACHE
    meta_path = INDEX_DIR / "docs_metadata.jsonl"
    docs: List[Dict] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    _ALL_DOCS_CACHE = docs
    return docs
retriever = Retriever(cfg)
llm_client = LLMClient(cfg)

app = FastAPI(title="Graph RAG Assistant")


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 30


class QueryResponse(BaseModel):
    answer: str
    chosen_graph_id: Optional[str]
    chosen_curve_id: Optional[str]
    page_number: Optional[int]
    alternative_graph_ids: List[str]
    alternative_curve_ids: List[str]
    docs: List[Dict]


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    q = req.query.lower().strip()

    # 1) total graphs in PDF
    if "how many" in q and "graph" in q and "this graph" not in q:
        docs = get_all_docs()
        graph_ids = {d.get("graph_id") for d in docs if d.get("type") == "graph"}
        graph_ids = {g for g in graph_ids if g}
        answer = f"There are {len(graph_ids)} graphs in the PDF (from metadata)."
        return QueryResponse(
            answer=answer,
            chosen_graph_id=None,
            chosen_curve_id=None,
            page_number=None,
            alternative_graph_ids=sorted(graph_ids),
            alternative_curve_ids=[],
            docs=[],
        )

    # 2) total curves in PDF
    if "how many" in q and "curve" in q and "this graph" not in q:
        docs = get_all_docs()
        curve_ids = {d.get("curve_id") for d in docs if d.get("type") == "curve"}
        curve_ids = {c for c in curve_ids if c}
        answer = f"There are {len(curve_ids)} curves in the PDF (from metadata)."
        return QueryResponse(
            answer=answer,
            chosen_graph_id=None,
            chosen_curve_id=None,
            page_number=None,
            alternative_graph_ids=[],
            alternative_curve_ids=sorted(curve_ids),
            docs=[],
        )
    docs = retriever.retrieve(req.query, k=req.top_k)
    ranked_docs = rank_graphs(req.query, docs)

    # --- 1) pick the best graph / page first ---
    best_graph = None
    for d in ranked_docs:
        if d.get("type") == "graph":
            best_graph = d
            break

    # If no graph doc found, fall back to old behaviour
    if best_graph is None:
        top_docs = ranked_docs[:6]
    else:
        best_graph_id = best_graph.get("graph_id")
        best_page = best_graph.get("page_number")
        best_pdf = best_graph.get("pdf_name")

        # --- 2) restrict docs to this graph / page ---
        # Keep:
        #  - curve docs of the same graph_id
        #  - any docs (text chunks) on the same page of the same PDF
        #  - the graph doc itself
        local_docs = []
        for d in ranked_docs:
            if d.get("type") == "graph" and d.get("graph_id") == best_graph_id:
                local_docs.append(d)
            elif d.get("type") == "curve" and d.get("graph_id") == best_graph_id:
                local_docs.append(d)
            else:
                # allow text chunks on same pdf + page
                if (
                    d.get("pdf_name") == best_pdf
                    and d.get("page_number") == best_page
                    and d.get("type") == "text_chunk"
                ):
                    local_docs.append(d)

        # if somehow we filtered too much, fall back to top 6
        if local_docs:
            top_docs = local_docs[:6]
        else:
            top_docs = ranked_docs[:6]

    # --- 3) call LLM only with localised docs ---
    llm_out = llm_client.generate(req.query, top_docs)

    return QueryResponse(
        answer=llm_out.get("answer", ""),
        chosen_graph_id=llm_out.get("chosen_graph_id"),
        chosen_curve_id=llm_out.get("chosen_curve_id"),
        page_number=llm_out.get("page_number"),
        alternative_graph_ids=llm_out.get("alternative_graph_ids", []),
        alternative_curve_ids=llm_out.get("alternative_curve_ids", []),
        docs=top_docs,
    )



@app.get("/graphs/{graph_id}")
def get_graph(graph_id: str):
    """
    We saved graph images with names including 'page_XXXX_img_YYY'.
    Our graph_id uses pattern: "<pdf>_page_XXXX_graph_YYY".
    We'll map 'graph_YYY' -> 'img_YYY' and look for png.
    """
    graphs_dir = Path(cfg.paths["graphs_dir"])

    # Extract "page_XXXX_graph_YYY" part
    if "page_" in graph_id and "graph_" in graph_id:
        tail = graph_id.split("page_")[-1]  # e.g. "0005_graph_000"
        page_part, graph_part = tail.split("_graph_")
        img_part = f"page_{page_part}_img_{graph_part}"
        # Match any file containing that pattern
        candidates = list(graphs_dir.glob(f"*{img_part}*.png"))
    else:
        candidates = list(graphs_dir.glob(f"*{graph_id}*.png"))

    if not candidates:
        raise HTTPException(status_code=404, detail="Graph image not found")
    return FileResponse(candidates[0])
