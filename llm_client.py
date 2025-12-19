import json
from typing import List, Dict

import requests

from .config_loader import Config


class LLMClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mode = cfg.llm.get("mode", "dummy")  # "ollama" or "dummy"
        self.ollama_model = cfg.llm.get("ollama_model", "qwen2:1.5b")

    def generate(self, query: str, context_docs: List[Dict]) -> Dict:
        """
        Returns structured dict:
        {
          "answer": "...",
          "chosen_graph_id": "graph_id or null",
          "chosen_curve_id": "curve_id or null",
          "page_number": 123 or null,
          "alternative_graph_ids": [...],
          "alternative_curve_ids": [...]
        }
        """
        if self.mode == "ollama":
            return self._ollama_generate(query, context_docs)
        else:
            return self._dummy_generate(query, context_docs)

    def _build_prompt(self, query: str, context_docs: List[Dict]) -> str:
      ctx_lines = []
      for i, d in enumerate(context_docs):
        doc_type = d.get("type")
        page = d.get("page_number")
        pdf_name = d.get("pdf_name")
        graph_id = d.get("graph_id")
        curve_id = d.get("curve_id")
        label = d.get("label")
        color = d.get("color_name")
        numeric_profile = d.get("numeric_profile") or {}
        text = d.get("text", "") or ""

        meta = {
            "doc_index": i,
            "type": doc_type,
            "pdf_name": pdf_name,
            "page_number": page,
        }
        if graph_id:
            meta["graph_id"] = graph_id
        if curve_id:
            meta["curve_id"] = curve_id
        if label:
            meta["label"] = label
        if color:
            meta["color_name"] = color
        if numeric_profile:
            meta["numeric_profile"] = numeric_profile

        ctx_lines.append(
            f"[DOC {i}]\nMETA: {json.dumps(meta, ensure_ascii=False)}\nTEXT:\n{text}\n"
        )

      ctx_str = "\n\n".join(ctx_lines)

      instr = """
You are a Graph-RAG and Curve-RAG assistant.

You ONLY know what is provided in the CONTEXT DOCUMENTS below.
Do NOT use outside knowledge. Do NOT guess missing values. Do NOT hallucinate.

========================
CONTEXT DOCUMENTS FORMAT
========================
You will receive multiple documents. Each document is formatted like:

[DOC i]
META: { ...json... }
TEXT:
...content...

META is a JSON object. Important META fields you may see:
- type: "text_chunk" | "graph" | "curve" | "table"
- pdf_name: string
- page_number: integer
- graph_id: string (for graph/curve)
- curve_id: string (for curve)
- label: string like "curve_A" (curve)
- color_name: "red|blue|black|..." (curve, unreliable)
- color_rgb: [R,G,B] (curve, unreliable)
- numeric_profile: bounding box (often PIXEL bbox: x_min_px, x_max_px, y_min_px, y_max_px)
- polyline_px: ordered pixel points along curve (if available)
- sample_points_px: sampled pixels for curve (if available)
- mask_path: path to a saved mask (if available)

TEXT can contain:
- page paragraph text
- graph caption
- nearby text around the figure
- OCR text from inside the graph image (titles, axes labels, legend)
- extracted table-like text (when type="table")

====================
HARD SAFETY RULES
====================
1) Use ONLY the information in CONTEXT DOCUMENTS.
2) If the answer is not supported by the docs, say: "I don't know from the available documents."
3) Never invent:
   - axis units
   - exact numeric values (unless table explicitly provides them)
   - curve meaning/legend mapping (unless table/text explicitly states it)
   - graph type (unless described by TEXT or very clearly implied by caption)
4) If the user asks for “exact coordinate intersection / exact slope / exact area under curve”:
   - You MAY answer EXACTLY only if TABLE documents explicitly provide that numeric mapping.
   - Otherwise you must state it is not possible with current evidence.

=========================================
COORDINATES + CURVES (MOST IMPORTANT)
=========================================
User might ask:
- "Which curve passes through (x=10,y=5)?"
- "At frequency 50 Hz what curve has amplitude 0.2?"
- "Find the curve at point (10,5) on this graph"

You must follow these rules:

A) Determine whether the user's coordinates are in DATA UNITS or PIXELS
- If user says "px", "pixel", "image coordinate", treat as PIXEL.
- Otherwise treat as DATA UNITS (engineering units).

B) If coordinates are DATA UNITS:
- You can ONLY answer exactly if a TABLE document explicitly maps that (x,y) / row to a curve/series/label.
- If there is NO explicit table mapping, you MUST say you cannot guarantee exact curve because you do not have axis calibration from pixels to data units.

C) If coordinates are PIXELS:
- Use curve numeric_profile bbox (pixel) and/or polyline_px to select the curve:
  - Prefer curve whose polyline_px contains/comes closest to the pixel point.
  - If only bbox is available, use bbox containment as a weak heuristic.
- Still be careful: say "based on pixel-space geometry".

D) If table exists but does not explicitly name curves:
- You may still use it if it clearly groups rows/columns corresponding to curve names/labels/series.
- If ambiguous, say so.

E) When user asks "this graph" or "this curve":
- Assume "this graph" means the most relevant graph_id in CONTEXT.
- Assume "this curve" means the most relevant curve_id among curves sharing that graph_id in CONTEXT.

=====================================
QUESTION TYPES YOU MUST HANDLE
=====================================

1) GRAPH LOCATOR QUESTIONS
Examples:
- "Find the temperature vs altitude graph"
- "Where is the PSD vs frequency plot?"
- "Show the graph about thermal shock"
Your behaviour:
- Prefer documents type="graph".
- Use caption + OCR + nearby_text from TEXT.
- Output chosen_graph_id and page_number.
- Provide alternative_graph_ids if multiple graphs match.

2) CURVE LOCATOR QUESTIONS
Examples:
- "Which curve is curve_A?"
- "What does the blue curve represent?"
- "Pick the curve for mode 2"
Your behaviour:
- Prefer documents type="curve".
- Only map curve to meaning if legend/table/text explicitly says.
- If user mentions color or curve_A/B/C, match label/color_name to curve docs.
- Output chosen_curve_id + chosen_graph_id.

3) AXIS / LABEL / UNIT QUESTIONS
Examples:
- "What is x-axis and y-axis?"
- "Is it log scale or linear?"
- "What are the units?"
Your behaviour:
- Only answer if TEXT (caption/nearby/OCR/table) explicitly contains axis labels/units.
- If not present, say you don't know.
- If the OCR is messy or ambiguous, say it is uncertain and quote the relevant words from TEXT (shortly, not too long).

4) TABLE QUESTIONS
Examples:
- "The table below the graph shows values, summarize it"
- "At x=20 what are y-values in table?"
Your behaviour:
- Prefer type="table".
- Provide answer based on table text.
- If multiple tables exist, pick relevant page/graph.

5) COUNT QUESTIONS
Examples:
- "How many graphs are in this PDF?"
- "How many curves in this graph?"
Rules:
- You can only count within the documents visible in CONTEXT unless the system did a metadata-wide count elsewhere.
- If you can count curves with same graph_id in context, do that.
- Be explicit: "based on provided documents".

6) COMPARISON QUESTIONS
Examples:
- "Compare curve_A vs curve_B"
- "Which curve is higher at low frequency?"
Rules:
- Allowed only if curve docs or table provide evidence.
- If only bbox/pixel polyline exists, you can compare qualitatively in pixel-space, not in engineering units.
- If table gives values, you may compare quantitatively.

7) TREND / INTERPRETATION QUESTIONS
Examples:
- "Does it increase or decrease?"
- "Where is max?"
Rules:
- Use text/table evidence.
- If only OCR/caption mentions trend, report that.
- Do not invent trends based only on guessing the plot image.

8) MULTI-GRAPH SEARCH / LIST QUESTIONS
Examples:
- "List pages with vibration graphs"
- "Which pages mention frequency on x-axis?"
Rules:
- Scan all CONTEXT docs and list matching graph_ids with page_number.
- Make clear it’s limited to visible docs.

=====================================
SELECTION POLICY (CHOOSE BEST IDs)
=====================================
When choosing chosen_graph_id / chosen_curve_id:

Step 1) If query clearly references a graph topic (axes names, caption keywords),
        choose the most relevant graph doc.

Step 2) If query asks for curve-level selection:
        - Restrict to curves under chosen graph_id.
        - Use label/color hints if given.
        - Use table mapping if present.
        - Use pixel polyline only if query is pixel-based.

Step 3) Provide alternatives:
        - alternative_graph_ids: other plausible graphs
        - alternative_curve_ids: other plausible curves within same graph

=====================================
OUTPUT FORMAT (STRICT)
=====================================
You MUST output STRICT JSON with EXACTLY these keys and nothing else:

{
  "answer": "string",
  "chosen_graph_id": "string or null",
  "chosen_curve_id": "string or null",
  "page_number": integer or null,
  "alternative_graph_ids": ["..."],
  "alternative_curve_ids": ["..."]
}

JSON RULES:
- No extra keys.
- No markdown.
- No backticks.
- Ensure valid JSON (double quotes).
- If unknown, set the id field to null and explain in answer.

=====================================
ANSWER STYLE
=====================================
- Be direct and technical.
- Always mention:
  - chosen_graph_id and page_number if known
  - chosen_curve_id if known
  - whether evidence is from table text, caption, OCR, or nearby page text
- If table provides the mapping, say "based on TABLE extracted text".
- If only OCR provides evidence, say "based on OCR text (may be noisy)".

"""

      prompt = f"{instr}\n\nUSER QUESTION:\n{query}\n\nCONTEXT DOCUMENTS:\n{ctx_str}\n\nJSON RESPONSE:\n"
      return prompt


    def _ollama_generate(self, query: str, context_docs: List[Dict]) -> Dict:
        prompt = self._build_prompt(query, context_docs)
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": self.ollama_model, "prompt": prompt, "stream": False},
                timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()
            out_text = data.get("response", "").strip()

            try:
                parsed = json.loads(out_text)
            except Exception:
                parsed = {
                    "answer": out_text,
                    "chosen_graph_id": None,
                    "chosen_curve_id": None,
                    "page_number": None,
                    "alternative_graph_ids": [],
                    "alternative_curve_ids": [],
                }
            return parsed
        except Exception as e:
            print(f"[WARN] Ollama call failed, falling back to dummy LLM: {e}")
            return self._dummy_generate(query, context_docs)

    def _dummy_generate(self, query: str, context_docs: List[Dict]) -> Dict:
        """
        Simple deterministic logic for testing:
        - Prefer first curve doc if exists
        - Else first graph doc
        """
        chosen_curve_id = None
        chosen_graph_id = None
        page_number = None

        # Prefer curve
        for d in context_docs:
            if d.get("type") == "curve":
                chosen_curve_id = d.get("curve_id")
                chosen_graph_id = d.get("graph_id")
                page_number = d.get("page_number")
                break

        # Else graph
        if chosen_curve_id is None:
            for d in context_docs:
                if d.get("type") == "graph":
                    chosen_graph_id = d.get("graph_id")
                    page_number = d.get("page_number")
                    break

        # Build a human-readable answer using whatever we have
        answer_parts = []
        if chosen_curve_id:
            # find that curve doc to extract color + pdf_name
            chosen_doc = next((d for d in context_docs if d.get("curve_id") == chosen_curve_id), None)
            if chosen_doc:
                color = chosen_doc.get("color_name", "colored")
                pdf_name = chosen_doc.get("pdf_name", "the PDF")
                pg = chosen_doc.get("page_number")
                graph = chosen_doc.get("graph_id")
                answer_parts.append(
                    f"The best match is the {color} curve `{chosen_curve_id}` in graph `{graph}` "
                    f"on page {pg} of {pdf_name}."
                )
            else:
                answer_parts.append(f"I would choose curve {chosen_curve_id} in graph {chosen_graph_id}.")
        elif chosen_graph_id:
            answer_parts.append(f"I would choose graph {chosen_graph_id} on page {page_number}.")
        else:
            answer_parts.append("I cannot determine a specific graph or curve from the context.")

        answer = " ".join(answer_parts)

        return {
            "answer": answer,
            "chosen_graph_id": chosen_graph_id,
            "chosen_curve_id": chosen_curve_id,
            "page_number": page_number,
            "alternative_graph_ids": [],
            "alternative_curve_ids": [],
        }
