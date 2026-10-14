"""Freebase over a SPARQL endpoint, e.g. Virtuoso loaded with the Freebase dump.

    kg:
      backend: freebase_sparql
      sparql_url: http://localhost:8890/sparql    # your endpoint

Stage 1 lists the entity's outgoing predicates, stage 2 fetches objects of the selected ones,
so hub entities with millions of edges are never pulled in full. Compound value type (CVT)
nodes, which have no name (e.g. a marriage linking two people), are expanded into two-step
predicates such as "people.person.spouse_s -> people.marriage.spouse".

Entity names written by the policy are mapped to MIDs in this order: ids passed as hints
(a dataset's topic-entity MIDs), names already seen in earlier results, then a name lookup
in the graph (the candidate with the most edges wins when names collide).
Only standard SPARQL 1.1 is used.
"""
import logging
import re
import threading
import time

import requests

from .base import KGBackend, normalize_entity

log = logging.getLogger(__name__)

NS = "http://rdf.freebase.com/ns/"
PREFIX = f"PREFIX ns: <{NS}>\n"
CVT_SEP = " -> "
DEFAULT_EXCLUDE = ["type.", "common.", "freebase.", "kg.", "dataworld.", "user."]

_MID = re.compile(r"^[mg]\.[0-9a-z_]+$")
_REL = re.compile(r"^[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)+$")


def _literal(text, lang):
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    return f'"{escaped}"@{lang}'


class FreebaseSparql(KGBackend):
    def __init__(self, sparql_url, timeout=60, method="post", language="en", exclude_prefixes=None,
                 expand_cvt=True, entity_lookup="degree", max_relations=1000, retries=2):
        self.url = sparql_url
        self.timeout = timeout
        self.method = method
        self.lang = language
        self.exclude = tuple(exclude_prefixes if exclude_prefixes is not None else DEFAULT_EXCLUDE)
        self.expand_cvt = expand_cvt
        self.entity_lookup = entity_lookup
        self.max_relations = max_relations
        self.retries = retries
        self._lock = threading.Lock()
        self._learned = {}  # normalized name -> mid, from earlier results
        self._lookup_cache, self._rel_cache, self._obj_cache = {}, {}, {}
        self.queries = 0

    # ------------------------------------------------------------------ transport

    def query(self, sparql):
        """Run a SELECT query; returns the SPARQL-JSON bindings list."""
        with self._lock:
            self.queries += 1
        headers = {"Accept": "application/sparql-results+json"}
        last = None
        for attempt in range(self.retries + 1):
            try:
                if self.method == "get":
                    r = requests.get(self.url, params={"query": sparql}, headers=headers, timeout=self.timeout)
                else:
                    r = requests.post(self.url, data={"query": sparql}, headers=headers, timeout=self.timeout)
                r.raise_for_status()
                return r.json()["results"]["bindings"]
            except (requests.RequestException, ValueError, KeyError) as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        log.warning("SPARQL query failed after %d tries: %s", self.retries + 1, last)
        return []

    # ------------------------------------------------------------------ entity resolution

    def _keep(self, relation):
        return bool(_REL.match(relation)) and not relation.startswith(self.exclude)

    def _learn(self, name, iri):
        if iri.startswith(NS):
            mid = iri[len(NS):]
            if _MID.match(mid):
                with self._lock:
                    self._learned.setdefault(normalize_entity(name), mid)

    def resolve(self, entity, hints=None):
        """Entity name (or MID) -> MID, or None if the graph has no such entity."""
        entity = entity.strip()
        if _MID.match(entity):
            return entity
        key = normalize_entity(entity)
        for name, mid in (hints or {}).items():
            if normalize_entity(name) == key and _MID.match(str(mid)):
                return mid
        with self._lock:
            if key in self._learned:
                return self._learned[key]
            if key in self._lookup_cache:
                return self._lookup_cache[key]
        mid = self._lookup(entity)
        with self._lock:
            self._lookup_cache[key] = mid
        return mid

    def _lookup(self, name):
        for prop in ("ns:type.object.name", "ns:common.topic.alias"):
            lit = _literal(name, self.lang)
            if self.entity_lookup == "degree":
                q = PREFIX + f"SELECT ?e (COUNT(?p) AS ?deg) WHERE {{ ?e {prop} {lit} . ?e ?p ?o . }} GROUP BY ?e ORDER BY DESC(?deg) LIMIT 1"
            else:
                q = PREFIX + f"SELECT ?e WHERE {{ ?e {prop} {lit} . }} LIMIT 1"
            # an aggregate over no matches can still return one row, with ?e unbound
            rows = [r for r in self.query(q) if "e" in r]
            if rows and rows[0]["e"]["value"].startswith(NS):
                mid = rows[0]["e"]["value"][len(NS):]
                if _MID.match(mid):
                    return mid
        return None

    def name_of(self, mid):
        """English name of a MID, or None (used when preparing data that only has MIDs)."""
        if not _MID.match(str(mid)):
            return None
        rows = self.query(PREFIX + f'SELECT ?n WHERE {{ ns:{mid} ns:type.object.name ?n . FILTER(langMatches(lang(?n), "{self.lang}")) }} LIMIT 1')
        if not rows:
            return None
        name = rows[0]["n"]["value"]
        self._learn(name, NS + mid)
        return name

    # ------------------------------------------------------------------ stage 1

    def relations(self, entity, hints=None):
        mid = self.resolve(entity, hints)
        if mid is None:
            return []
        with self._lock:
            if mid in self._rel_cache:
                return self._rel_cache[mid]
        direct = self.query(PREFIX + f"""SELECT DISTINCT ?r WHERE {{
  ns:{mid} ?r ?o .
  FILTER(STRSTARTS(STR(?r), "{NS}"))
  FILTER(isLiteral(?o) || EXISTS {{ ?o ns:type.object.name ?n }})
}} LIMIT {self.max_relations}""")
        rels = [b["r"]["value"][len(NS):] for b in direct]
        if self.expand_cvt:
            cvt = self.query(PREFIX + f"""SELECT DISTINCT ?r1 ?r2 WHERE {{
  ns:{mid} ?r1 ?c .
  FILTER(isIRI(?c) && STRSTARTS(STR(?r1), "{NS}"))
  FILTER NOT EXISTS {{ ?c ns:type.object.name ?cn }}
  ?c ?r2 ?x .
  FILTER(STRSTARTS(STR(?r2), "{NS}") && ?x != ns:{mid})
  FILTER(isLiteral(?x) || EXISTS {{ ?x ns:type.object.name ?n }})
}} LIMIT {self.max_relations}""")
            rels += [b["r1"]["value"][len(NS):] + CVT_SEP + b["r2"]["value"][len(NS):] for b in cvt]
        out = sorted({r for r in rels if all(self._keep(part) for part in r.split(CVT_SEP))})
        with self._lock:
            self._rel_cache[mid] = out
        return out

    # ------------------------------------------------------------------ stage 2

    def objects(self, entity, relation, limit, hints=None):
        mid = self.resolve(entity, hints)
        parts = relation.split(CVT_SEP)
        if mid is None or not all(_REL.match(p) for p in parts) or len(parts) > 2:
            return []
        key = (mid, relation, limit)
        with self._lock:
            if key in self._obj_cache:
                return self._obj_cache[key]
        if len(parts) == 1:
            pattern = f"ns:{mid} ns:{parts[0]} ?x ."
        else:
            pattern = (f"ns:{mid} ns:{parts[0]} ?c . FILTER NOT EXISTS {{ ?c ns:type.object.name ?cn }} "
                       f"?c ns:{parts[1]} ?x . FILTER(?x != ns:{mid})")
        rows = self.query(PREFIX + f"""SELECT DISTINCT ?x ?name WHERE {{
  {pattern}
  OPTIONAL {{ ?x ns:type.object.name ?name . FILTER(langMatches(lang(?name), "{self.lang}")) }}
}} LIMIT {limit * 3}""")
        values = []
        for b in rows:
            x = b["x"]
            if "name" in b:
                name = b["name"]["value"]
                self._learn(name, x["value"])
            elif x["type"] in ("literal", "typed-literal") and x.get("xml:lang", self.lang) == self.lang:
                name = x["value"]
            else:
                continue  # unnamed node or a literal in another language
            if name not in values:
                values.append(name)
            if len(values) >= limit:
                break
        with self._lock:
            self._obj_cache[key] = values
        return values
