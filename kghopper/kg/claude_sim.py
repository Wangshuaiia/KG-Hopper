"""Freebase simulated by Claude, for running the pipeline without a Freebase/Virtuoso server.

Stage-1 neighbourhoods and stage-2 selections are cached in a JSON file shared by training,
evaluation and the KG server, so the simulated graph answers the same way every time.
The question is never shown to stage 1, so the graph is not tailored to the question.
Needs ANTHROPIC_API_KEY. The facts come from Claude's world knowledge, not from Freebase,
so scores obtained on it are not comparable with the paper's.
"""
import fcntl
import json
import logging
import os
import threading
from concurrent.futures import Future

from pydantic import BaseModel

from .base import NeighbourhoodBackend, normalize_entity

log = logging.getLogger(__name__)

NEIGHBOURHOOD_SYSTEM = """You are a simulated Freebase SPARQL endpoint used to test a KBQA pipeline.
Given one entity, return its outgoing edges as they would appear in Freebase.
- Predicates use Freebase schema names (domain.type.property), e.g. people.person.place_of_birth.
- Objects are human-readable entity names or literal values (dates, numbers), not MIDs.
- Return 20-40 predicates covering the facts Freebase would hold for this entity, at most 10 objects each.
- Skip administrative predicates (type.object.*, common.*, freebase.*, *.sameAs).
- Collapse compound-value (CVT) nodes into a direct edge to the meaningful object.
- Use real-world knowledge as of Freebase's shutdown. If the entity is obscure or the string is not an entity,
  return few or no edges; do not invent facts."""

class _Edge(BaseModel):
    relation: str
    objects: list[str]


class _Neighbourhood(BaseModel):
    edges: list[_Edge]


class JsonCache:
    """{"neighbourhood": {entity: {relation: [objects]}}, "selection": {key: [relations]}} on disk.

    Writes merge with what is on disk under an exclusive file lock, so the trainer, the
    evaluator and the KG server can share one file.
    """

    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self.data = {"neighbourhood": {}, "selection": {}}
        if path and os.path.exists(path):
            self.data = self._read()

    def _read(self):
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("neighbourhood", {})
        data.setdefault("selection", {})
        return data

    def get(self, section, key):
        with self._lock:
            return self.data[section].get(key)

    def put(self, section, key, value):
        with self._lock:
            self.data[section][key] = value
            if not self.path:
                return
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path + ".lock", "w") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                if os.path.exists(self.path):
                    on_disk = self._read()
                    for sec in ("neighbourhood", "selection"):
                        on_disk[sec].update(self.data[sec])
                    self.data = on_disk
                tmp = self.path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self.data, f, ensure_ascii=False, indent=1)
                os.replace(tmp, self.path)


class _SingleFlight:
    """Run one computation per key at a time; concurrent callers share its result."""

    def __init__(self):
        self._lock = threading.Lock()
        self._inflight = {}

    def run(self, key, fn):
        with self._lock:
            fut = self._inflight.get(key)
            owner = fut is None
            if owner:
                fut = self._inflight[key] = Future()
        if not owner:
            return fut.result()
        try:
            result = fn()
            fut.set_result(result)
            return result
        except Exception as e:
            fut.set_exception(e)
            raise
        finally:
            with self._lock:
                self._inflight.pop(key, None)


class ClaudeSimulatedFreebase(NeighbourhoodBackend):
    def __init__(self, cache, model="claude-opus-5-5", effort="low"):
        import anthropic  # imported here so offline backends do not need the SDK

        self.client = anthropic.Anthropic(max_retries=5)
        self.model = model
        self.effort = effort
        self.cache = cache
        self._flight = _SingleFlight()
        self.api_calls = 0

    def _fetch(self, entity):
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=8000,
            system=NEIGHBOURHOOD_SYSTEM,
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": f"Entity: {entity}"}],
            output_format=_Neighbourhood,
        )
        self.api_calls += 1
        if response.stop_reason == "refusal" or response.parsed_output is None:
            log.warning("KG simulation got no structured output (%s) for %s", response.stop_reason, entity)
            return {}
        return {e.relation: e.objects for e in response.parsed_output.edges if e.objects}

    def neighbourhood(self, entity, limit=10, hints=None):
        key = normalize_entity(entity)
        if not key:
            return {}
        cached = self.cache.get("neighbourhood", key)
        if cached is not None:
            return cached

        def fetch():
            cached_now = self.cache.get("neighbourhood", key)
            if cached_now is not None:
                return cached_now
            edges = self._fetch(entity)
            self.cache.put("neighbourhood", key, edges)
            return edges

        return self._flight.run(key, fetch)


class StaticKG(NeighbourhoodBackend):
    """Read-only graph from a cache file written by ClaudeSimulatedFreebase (offline runs, tests)."""

    def __init__(self, cache=None, edges=None):
        if edges is not None:
            self.data = {normalize_entity(k): v for k, v in edges.items()}
        else:
            self.data = cache.data["neighbourhood"]

    def neighbourhood(self, entity, limit=10, hints=None):
        return self.data.get(normalize_entity(entity), {})
