import re
from typing import List

def split_into_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]

def chunk_text(paragraphs: List[str], max_chars: int, overlap: int = 0) -> List[str]:
    chunks = []
    buffer = ""
    for p in paragraphs:
        if len(buffer) + len(p) + 2 <= max_chars:
            buffer = buffer + ("\n\n" if buffer else "") + p
        else:
            if buffer:
                chunks.append(buffer)
            buffer = p
    if buffer:
        chunks.append(buffer)

    # simple overlap: join last part of previous
    if overlap > 0 and len(chunks) > 1:
        new_chunks = []
        for i, ch in enumerate(chunks):
            if i == 0:
                new_chunks.append(ch)
            else:
                prev_tail = chunks[i-1][-overlap:]
                new_chunks.append(prev_tail + "\n" + ch)
        chunks = new_chunks
    return chunks

def extract_figure_captions(text: str):
    """
    Very simple heuristic: lines starting with 'Fig', 'Figure'
    Returns list of (caption_text, index_in_text).
    """
    lines = text.splitlines()
    captions = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^(Fig\.?|Figure)\s", stripped, re.IGNORECASE):
            captions.append((stripped, i))
    return captions
