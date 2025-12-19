from dataclasses import dataclass, asdict
from typing import List, Dict, Optional


@dataclass
class Curve:
    """
    Metadata for a single curve inside a graph.
    These are primarily for manual / external annotation, not automatically extracted.
    """
    curve_id: str               # unique id, e.g. "<pdf>_page_0005_graph_000_curve_A"
    graph_id: str               # parent graph id
    label: Optional[str] = None
    color: Optional[str] = None
    style: Optional[str] = None
    description: Optional[str] = None
    # e.g. {"x_min": 0.0, "x_max": 50.0, "y_min": 0.1, "y_max": 2.0}
    numeric_profile: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class GraphMetadata:
    """
    Metadata for a graph / figure extracted from the PDF.
    Curves are currently expected to be filled manually or from an external tool.
    """
    graph_id: str
    pdf_name: str
    page_number: int
    image_path: str
    caption: str
    nearby_text: str
    axes_info: Dict[str, str]
    curves: List[Curve]
    extra_tags: List[str]

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["curves"] = [c.to_dict() for c in self.curves]
        return d
