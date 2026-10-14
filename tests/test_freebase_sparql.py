"""FreebaseSparql against a real SPARQL 1.1 HTTP endpoint (rdflib) serving a tiny Freebase-shaped graph."""
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import rdflib

from kghopper.kg.base import TwoStageKG
from kghopper.kg.freebase_sparql import FreebaseSparql
from kghopper.kg.selector import LexicalSelector

GRAPH = """
@prefix ns: <http://rdf.freebase.com/ns/> .
ns:m.03_dwn ns:type.object.name "Lou Seal"@en ; ns:type.object.type ns:sports.mascot ;
    ns:sports.mascot.team ns:m.0713r ; ns:common.topic.alias "Luigi Francisco Seal"@en .
ns:m.0lou2 ns:type.object.name "Lou Seal"@en .
ns:m.0713r ns:type.object.name "San Francisco Giants"@en ;
    ns:sports.sports_team.championships ns:m.0ws2014 , ns:m.0ws2012 ;
    ns:sports.sports_team.founded "1883" ;
    ns:sports.sports_team.team_mascot ns:m.03_dwn ;
    ns:sports.sports_team.roster ns:m.cvt1 ;
    ns:common.topic.description "A baseball team"@en .
ns:m.0ws2014 ns:type.object.name "2014 World Series"@en , "Serie mondiale 2014"@fr .
ns:m.0ws2012 ns:type.object.name "2012 World Series"@en .
ns:m.cvt1 ns:sports.sports_team_roster.player ns:m.posey ;
    ns:sports.sports_team_roster.team ns:m.0713r ;
    ns:sports.sports_team_roster.from "2009" .
ns:m.posey ns:type.object.name "Buster Posey"@en .
"""


@pytest.fixture(scope="module")
def endpoint():
    graph = rdflib.Graph().parse(data=GRAPH, format="turtle")
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"])).decode()
            query = urllib.parse.parse_qs(body)["query"][0]
            with lock:  # rdflib's SPARQL parser is not thread-safe
                payload = graph.query(query).serialize(format="json")
            self.send_response(200)
            self.send_header("Content-Type", "application/sparql-results+json")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/sparql"
    server.shutdown()


def test_name_lookup_prefers_the_entity_with_more_edges(endpoint):
    fb = FreebaseSparql(endpoint)
    assert fb.resolve("Lou Seal") == "m.03_dwn"  # two entities share the name
    assert fb.resolve("Luigi Francisco Seal") == "m.03_dwn"  # alias fallback
    assert fb.resolve("Lou Seal", hints={"Lou Seal": "m.0lou2"}) == "m.0lou2"  # dataset MID wins
    assert fb.resolve("No Such Entity") is None
    assert fb.resolve('Bad "quoted\\ name') is None  # escaped, no SPARQL error


def test_relations_filter_and_expand_cvt(endpoint):
    rels = FreebaseSparql(endpoint).relations("San Francisco Giants")
    assert "sports.sports_team.championships" in rels and "sports.sports_team.founded" in rels
    assert "sports.sports_team.roster -> sports.sports_team_roster.player" in rels
    assert "sports.sports_team.roster -> sports.sports_team_roster.from" in rels
    assert "sports.sports_team.roster -> sports.sports_team_roster.team" not in rels  # leads back to the head
    assert "sports.sports_team.roster" not in rels  # only reaches the unnamed CVT node
    assert not any(r.startswith(("common.", "type.")) for r in rels)


def test_objects_names_literals_and_cvt(endpoint):
    fb = FreebaseSparql(endpoint)
    assert sorted(fb.objects("San Francisco Giants", "sports.sports_team.championships", 10)) == ["2012 World Series", "2014 World Series"]
    assert fb.objects("San Francisco Giants", "sports.sports_team.founded", 10) == ["1883"]
    assert fb.objects("San Francisco Giants", "sports.sports_team.roster -> sports.sports_team_roster.player", 10) == ["Buster Posey"]
    assert fb.objects("San Francisco Giants", "bad relation; DROP", 10) == []


def test_names_from_results_resolve_without_lookup(endpoint):
    fb = FreebaseSparql(endpoint)
    assert fb.objects("Lou Seal", "sports.mascot.team", 10) == ["San Francisco Giants"]
    before = fb.queries
    assert fb.resolve("San Francisco Giants") == "m.0713r"
    assert fb.queries == before  # learned from the previous result, no name lookup


def test_two_stage_search_end_to_end(endpoint):
    kg = TwoStageKG(FreebaseSparql(endpoint), LexicalSelector(), width=2)
    triples = kg.search("San Francisco Giants", "Which World Series championships did the Giants win?")
    assert ("San Francisco Giants", "sports.sports_team.championships", "2014 World Series") in triples
    assert kg.search("Lou Seal", "q", hints={"Lou Seal": "m.0lou2"}) == []  # the hinted MID has no edges
