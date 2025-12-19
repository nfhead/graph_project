import re
from typing import List, Dict, Tuple, Optional


def extract_numbers_and_units(text: str):
    nums = [float(x) for x in re.findall(r"[-+]?\d+\.?\d*", text)]
    units = re.findall(r"(Mach|deg|MPa|g|Hz|cycles|epoch|epochs|rpm|nm|kN|m/s|mm|cm|ms|s)", text, flags=re.IGNORECASE)
    return nums, [u.lower() for u in units]


def parse_xy(text: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse patterns like:
    - x=10, y=5
    - (10, 5)
    Returns (x, y) in USER UNITS (data space), not pixels.
    """
    m = re.search(r"x\s*=\s*([-\d\.]+).+?y\s*=\s*([-\d\.]+)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except ValueError:
            pass

    m = re.search(r"\(\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)", text)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except ValueError:
            pass

    return None, None


def rank_graphs(query: str, docs: List[Dict]) -> List[Dict]:
    """
    Re-rank docs based on:
    - Prefer curve docs when query is curve-specific.
    - Boost graph docs over text.
    - Boost TABLE docs heavily when query contains numbers / (x,y).
    - IMPORTANT: Do NOT compare user (x,y) in data units to pixel bbox.
      Only do bbox containment if user explicitly mentions pixel coords.
    """
    q_lower = query.lower()
    q_nums, q_units = extract_numbers_and_units(query)
    qx, qy = parse_xy(query)

    # Only treat (x,y) as pixel coords if user explicitly says px/pixel
    q_is_pixels = ("px" in q_lower) or ("pixel" in q_lower) or ("pixels" in q_lower)

    wants_curve = any(word in q_lower for word in ["curve", "line", "mode", "legend", "series"])
    wants_numeric = (qx is not None and qy is not None) or (len(q_nums) >= 1)

    for d in docs:
        score_boost = 0.0
        text = d.get("text", "") or ""
        text_lower = text.lower()
        dtype = d.get("type", "text_chunk")

        curve_label = (d.get("label") or "").lower()
        color_name = (d.get("color_name") or "").lower()

        # ---------- Base type boosts ----------
        if dtype == "graph":
            score_boost += 0.25
        elif dtype == "curve":
            score_boost += 0.35
            if wants_curve:
                score_boost += 0.15
        elif dtype == "table":
            # tables are extremely useful for coordinate queries
            score_boost += 0.45
            if wants_numeric:
                score_boost += 0.35

        # ---------- Color hints ----------
        basic_colors = ["red", "blue", "green", "yellow", "magenta", "cyan", "black"]
        for cword in basic_colors:
            if cword in q_lower and color_name == cword:
                score_boost += 0.35

        # ---------- Label hints ----------
        for letter in "abcdefghijklmnopqrstuvwxyz":
            pattern = f"curve {letter}"
            if pattern in q_lower and curve_label == f"curve_{letter}":
                score_boost += 0.30

        # ---------- Units match ----------
        for u in q_units:
            if u in text_lower:
                score_boost += 0.08

        # ---------- Numeric match in text (helps table rows) ----------
        doc_nums, _doc_units = extract_numbers_and_units(text)
        for qn in q_nums[:4]:
            # stronger match if numbers are close
            for dn in doc_nums[:30]:
                if abs(qn - dn) <= max(1e-6, 0.02 * max(1.0, abs(dn))):
                    score_boost += 0.10
                    break

        # ---------- numeric_profile bbox hints (PIXEL ONLY) ----------
        numeric_profile = d.get("numeric_profile")
        if isinstance(numeric_profile, dict) and q_is_pixels:
            x_min = numeric_profile.get("x_min_px", numeric_profile.get("x_min"))
            x_max = numeric_profile.get("x_max_px", numeric_profile.get("x_max"))
            y_min = numeric_profile.get("y_min_px", numeric_profile.get("y_min"))
            y_max = numeric_profile.get("y_max_px", numeric_profile.get("y_max"))

            if qx is not None and qy is not None and None not in (x_min, x_max, y_min, y_max):
                if (x_min <= qx <= x_max) and (y_min <= qy <= y_max):
                    score_boost += 0.20

        # ---------- Domain boosts ----------
        vib_keywords = [
            "asd", "psd", "power spectral density",
            "acceleration spectral density", "g^2/hz",
            "vibration", "random vibration", "srs",
            "shock response spectrum", "shock", "pulse"
        ]
        if any(kw in q_lower for kw in vib_keywords) and any(kw in text_lower for kw in vib_keywords):
            score_boost += 0.15

        mil_keywords = ["mil-std-810h", "810h", "thermal shock", "steady-state acceleration"]
        if any(kw in q_lower for kw in mil_keywords) and any(kw in text_lower for kw in mil_keywords):
            score_boost += 0.10

        d["score"] = float(d.get("score", 0.0)) + score_boost

    return sorted(docs, key=lambda x: x.get("score", 0.0), reverse=True)
