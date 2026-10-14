"""Two-stage retrieval shared by every KG backend (paper Sec. III-A).

Stage 1: all outgoing predicates of the entity (backend.relations).
Stage 2: the `width` predicates most relevant to the question (selector), then their objects
(backend.objects). Backends only answer graph queries; relation selection is a separate,
configurable component (kghopper/kg/selector.py).
"""
import re
import threading
from concurrent.futures import ThreadPoolExecutor

_STOPWORDS = set("a an the of in on at to for from by with is are was were be what which who whom whose when where how did does do".split())


def normalize_entity(name):
    return " ".join(str(name).strip().strip('"').split()).lower()


def relation_words(relation):
    """'people.person.place_of_birth' -> {'people', 'person', 'place', 'birth'}"""
    return {w for w in re.split(r"[._\s>-]+", relation.lower()) if w and w not in _STOPWORDS}


def lexical_select(question, relations, width):
    """Rank predicates by word overlap with the question (no model needed)."""
    q = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in _STOPWORDS}
    q |= {w[:-1] for w in q if w.endswith("s")}
    scored = []
    for i, rel in enumerate(relations):
        words = relation_words(rel)
        last = set(re.split(r"[._]", rel.split(".")[-1]))
        scored.append((-(len(q & words) + 0.5 * len(q & last)), i, rel))
    return [rel for _, _, rel in sorted(scored)[:width]]


class KGBackend:
    """Graph access. `hints` maps entity names to KG ids the caller already knows
    (e.g. a dataset's topic-entity MIDs); backends that work on names ignore it."""

    def relations(self, entity, hints=None):
        """Stage 1: outgoing predicates of `entity`."""
        raise NotImplementedError

    def objects(self, entity, relation, limit, hints=None):
        """Stage 2: object names / literal values of (entity, relation)."""
        raise NotImplementedError

    def neighbourhood(self, entity, limit=10, hints=None):
        """{relation: [objects]}: every predicate with its objects (used to build synthetic data)."""
        return {r: objs for r in self.relations(entity, hints) if (objs := self.objects(entity, r, limit, hints))}


class NeighbourhoodBackend(KGBackend):
    """For backends that fetch all edges of an entity at once (simulated / cached graphs)."""

    def neighbourhood(self, entity, limit=10, hints=None):
        raise NotImplementedError

    def relations(self, entity, hints=None):
        return sorted(self.neighbourhood(entity))

    def objects(self, entity, relation, limit, hints=None):
        return self.neighbourhood(entity).get(relation, [])[:limit]


class TwoStageKG:
    """The KG tool the policy calls. search() is thread-safe; search_many() runs requests in parallel."""

    def __init__(self, backend: KGBackend, selector, width=3, max_objects_per_relation=10, max_triples=30, max_workers=8):
        self.backend = backend
        self.selector = selector
        self.width = width
        self.max_objects = max_objects_per_relation
        self.max_triples = max_triples
        self.max_workers = max_workers
        self._lock = threading.Lock()
        self.calls = 0

    def search(self, entity, question, hints=None):
        """entity name + question -> list of (head, relation, tail)."""
        with self._lock:
            self.calls += 1
        relations = self.backend.relations(entity, hints)
        if not relations:
            return []
        if len(relations) > self.width:
            selected = [r for r in self.selector.select(question, entity, relations, self.width) if r in relations]
            selected = selected[: self.width] or relations[: self.width]
        else:
            selected = list(relations)
        triples = [(entity, rel, obj) for rel in selected for obj in self.backend.objects(entity, rel, self.max_objects, hints)]
        return triples[: self.max_triples]

    def search_many(self, requests):
        """[(entity, question) or (entity, question, hints), ...] -> [[triples], ...], in order."""
        if not requests:
            return []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(requests))) as pool:
            return list(pool.map(lambda r: self.search(*r), requests))
