"""KG retrieval tool (paper Sec. III-A). Use `make_kg(cfg.kg)` everywhere so training and
evaluation always talk to the same tool."""
from .base import KGBackend, TwoStageKG
from .client import HttpKG, make_kg

__all__ = ["KGBackend", "TwoStageKG", "HttpKG", "make_kg"]
