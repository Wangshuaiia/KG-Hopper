"""HTTP KG tool for multi-process training (OpenRLHF + Ray).

    python -m kghopper.kg.server --config configs/local_cpu.yaml --port 5005

POST /search {"queries": [{"entity": "...", "question": "..."}]} -> {"results": [[[h, r, t], ...], ...]}
"""
import argparse

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from ..config import add_config_args, load_config
from .client import make_kg


class Query(BaseModel):
    entity: str
    question: str = ""
    mids: dict[str, str] = {}  # entity name -> KG id hints, e.g. topic-entity MIDs


class SearchRequest(BaseModel):
    queries: list[Query]


def build_app(kg_cfg):
    kg_cfg = dict(kg_cfg)
    if kg_cfg.get("backend") == "http":
        raise ValueError("the server needs a real backend (claude_sim, static or freebase_sparql), not http")
    kg = make_kg(kg_cfg)
    app = FastAPI()

    @app.post("/search")
    def search(req: SearchRequest):
        results = kg.search_many([(q.entity, q.question, q.mids) for q in req.queries])
        return {"results": [[list(t) for t in triples] for triples in results]}

    @app.get("/health")
    def health():
        return {"ok": True, "calls": kg.calls}

    return app


def main():
    parser = add_config_args(argparse.ArgumentParser())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5005)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    kg_cfg = dict(cfg.get("kg", {}))
    if kg_cfg.get("backend") == "http":  # configs used by clients point at this server; serve the simulation
        kg_cfg["backend"] = kg_cfg.get("server_backend", "claude_sim")
    uvicorn.run(build_app(kg_cfg), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
