"""Build the KG tool from config. Every caller (trainers, evaluator, OpenRLHF rollout, KG server)
goes through make_kg(), so they all use the same retrieval.

kg.backend:
  claude_sim       Freebase simulated by Claude (needs ANTHROPIC_API_KEY)
  static           the cache file of claude_sim only, offline
  freebase_sparql  a real Freebase SPARQL endpoint (kg.sparql_url)
  http             a running `python -m kghopper.kg.server` (kg.url)
"""
import requests

from ..config import resolve_path
from .base import TwoStageKG
from .claude_sim import JsonCache
from .selector import make_selector


class HttpKG:
    """Client for kghopper.kg.server; same search()/search_many() interface as TwoStageKG."""

    def __init__(self, url, timeout=600):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def search_many(self, requests_):
        if not requests_:
            return []
        queries = [{"entity": r[0], "question": r[1], "mids": (r[2] if len(r) > 2 else None) or {}} for r in requests_]
        resp = requests.post(self.url + "/search", json={"queries": queries}, timeout=self.timeout)
        resp.raise_for_status()
        return [[tuple(t) for t in triples] for triples in resp.json()["results"]]

    def search(self, entity, question, hints=None):
        return self.search_many([(entity, question, hints)])[0]


def make_backend(kg_cfg, cache):
    backend = kg_cfg.get("backend", "claude_sim")
    if backend == "claude_sim":
        from .claude_sim import ClaudeSimulatedFreebase

        return ClaudeSimulatedFreebase(cache, model=kg_cfg.get("claude_model", "claude-opus-5-5"),
                                       effort=kg_cfg.get("claude_effort", "low"))
    if backend == "static":
        from .claude_sim import StaticKG

        return StaticKG(cache)
    if backend == "freebase_sparql":
        from .freebase_sparql import FreebaseSparql

        return FreebaseSparql(
            kg_cfg["sparql_url"],
            timeout=kg_cfg.get("sparql_timeout", 60),
            method=kg_cfg.get("sparql_method", "post"),
            language=kg_cfg.get("language", "en"),
            exclude_prefixes=kg_cfg.get("exclude_relation_prefixes"),
            expand_cvt=kg_cfg.get("expand_cvt", True),
            entity_lookup=kg_cfg.get("entity_lookup", "degree"),
            max_relations=kg_cfg.get("max_relations", 1000),
        )
    raise ValueError(f"unknown kg.backend {backend!r} (claude_sim | static | freebase_sparql | http)")


def make_kg(kg_cfg):
    """kg_cfg: the `kg` section of the config."""
    if kg_cfg.get("backend") == "http":
        return HttpKG(kg_cfg["url"])
    cache = JsonCache(resolve_path(kg_cfg.get("cache_path")) if kg_cfg.get("cache_path") else None)
    backend = make_backend(kg_cfg, cache)
    selector = make_selector(kg_cfg, cache, offline=kg_cfg.get("backend") == "static")
    return TwoStageKG(
        backend,
        selector,
        width=kg_cfg.get("width", 3),
        max_objects_per_relation=kg_cfg.get("max_objects_per_relation", 10),
        max_triples=kg_cfg.get("max_triples", 30),
        max_workers=kg_cfg.get("max_workers", 8),
    )
