import yaml
from pathlib import Path

class Config:
    def __init__(self, path: str = "configs/config.yaml"):
        self._raw = self._load(path)
        self.paths = self._raw.get("paths", {})
        self.ingest = self._raw.get("ingest", {})
        self.embedding = self._raw.get("embedding", {})
        self.llm = self._raw.get("llm", {})
        self.api = self._raw.get("api", {})

    def _load(self, path: str):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with open(p, "r") as f:
            return yaml.safe_load(f)
