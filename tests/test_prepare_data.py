"""Training-data preparation: converters, RL sampling, cold-start trajectory filtering."""
import pytest
from transformers import AutoTokenizer

from kghopper import protocol
from kghopper.data import load_test_set
from kghopper.kg.base import TwoStageKG
from kghopper.kg.claude_sim import StaticKG
from kghopper.kg.selector import LexicalSelector
from kghopper.prepare_data import (DEMOS, TeacherGenerator, build_rl, convert, generate_sft,
                                   insert_demos, parse_completion, _quotas)

# One record per dataset in the layout of its official release.
OFFICIAL = {
    "cwq": {"ID": "WebQTrn-1_x", "question": "Lou Seal is the mascot for the team that last won the World Series when?",
            "answers": [{"answer": "2014 World Series", "aliases": ["2014 MLB World Series"], "answer_id": "m.0117q3yz"}],
            "compositionality_type": "composition",
            "sparql": "PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?x WHERE { ?c ns:sports.sports_team.team_mascot ns:m.03_dwn . ?c ns:sports.sports_team.championships ?x . }"},
    "webqsp": {"QuestionId": "WebQTrn-0", "RawQuestion": "what is the name of justin bieber brother?",
               "Parses": [{"TopicEntityName": "Justin Bieber", "TopicEntityMid": "m.06w2sn5",
                           "InferentialChain": ["people.person.sibling_s", "people.sibling_relationship.sibling"],
                           "Answers": [{"AnswerType": "Entity", "AnswerArgument": "m.0gxnnwc", "EntityName": "Jaxon Bieber"}]}]},
    "grailqa": {"qid": 1, "question": "which play is produced by the illusion?", "num_edge": 1,
                "answer": [{"answer_type": "Entity", "answer_argument": "m.0yrltsn", "entity_name": "The Illusion"}],
                "graph_query": {"nodes": [{"nid": 0, "node_type": "class", "id": "theater.play", "friendly_name": "Play"},
                                          {"nid": 1, "node_type": "entity", "id": "m.0yrlqjm", "friendly_name": "The Illusion"}]}},
    "webquestions": {"url": "http://www.freebase.com/view/en/justin_bieber", "utterance": "what is the name of justin bieber brother?",
                     "targetValue": '(list (description "Jazmyn Bieber") (description "Jaxon Bieber"))'},
    "trex": {"id": "t1", "input": "Cirith Ungol [SEP] genre", "output": [{"answer": "heavy metal", "provenance": []}]},
    "zeroshotre": {"id": "z1", "input": "Jennings Randolph Bridge [SEP] crosses", "output": [{"answer": "Ohio River"}]},
    "creak": {"ex_id": "train_0", "sentence": "Eating soup means eating only solids.", "label": "false", "entity": "Soup"},
}


def test_converters_on_official_formats():
    out = {name: convert(name, [rec])[0] for name, rec in OFFICIAL.items()}
    assert out["cwq"]["golds"] == ["2014 World Series", "2014 MLB World Series"] and out["cwq"]["hops"] == 2
    assert out["cwq"]["entity_ids"] == {"m.03_dwn": "m.03_dwn"}  # no endpoint given: the MID stands in for the name
    assert out["webqsp"]["entity_ids"] == {"Justin Bieber": "m.06w2sn5"} and out["webqsp"]["hops"] == 2
    assert out["grailqa"]["topic_entities"] == ["The Illusion"] and out["grailqa"]["hops"] == 1
    assert out["webquestions"]["golds"] == ["Jazmyn Bieber", "Jaxon Bieber"]
    assert out["webquestions"]["topic_entities"] == ["Justin Bieber"]
    assert out["trex"]["task"] == "slot_filling" and out["trex"]["topic_entities"] == ["Cirith Ungol"] and out["trex"]["hops"] == 1
    assert out["creak"] == {**out["creak"], "task": "fact_checking", "golds": ["false"], "topic_entities": ["Soup"]}
    for s in out.values():
        assert {"idx", "question", "golds", "task", "topic_entities", "entity_ids", "source", "hops"} <= set(s)


@pytest.mark.parametrize("name", ["cwq", "webqsp", "grailqa", "webquestions", "trex", "zeroshotre", "creak", "qald"])
def test_converters_accept_the_test_set_layout(name):
    import json

    from kghopper.data import TEST_DIR, TEST_SETS

    raw = json.load(open(f"{TEST_DIR}/{TEST_SETS[name][0]}", encoding="utf-8"))[:3]
    converted, reference = convert(name, raw), load_test_set(name, 3)
    for c, r in zip(converted, reference):
        assert (c["question"], c["golds"], c["topic_entities"]) == (r["question"], r["golds"], r["topic_entities"])


def test_quotas_pass_shortfall_on():
    assert _quotas({"a": 10, "b": 1000, "c": 1000}, 300, {}) == {"a": 10, "b": 145, "c": 145}
    assert sum(_quotas({"a": 5, "b": 7}, 100, {}).values()) == 12  # not enough data: take everything


def test_build_rl_dedupes_drops_test_overlap_and_splits_pool():
    samples = [{"idx": f"x{i}", "question": f"question {i}?", "golds": ["a"], "source": "cwq" if i % 2 else "webqsp"} for i in range(40)]
    samples += [{"idx": "dup", "question": "Question 1", "golds": ["a"], "source": "cwq"},
                {"idx": "nogold", "question": "no gold?", "golds": [], "source": "cwq"},
                {"idx": "leak", "question": "In the test set?", "golds": ["a"], "source": "cwq"}]
    rl, pool, dropped = build_rl(samples, 10, exclude_questions=["in the test set"])
    assert len(rl) == 10 and len(pool) == 30
    assert {s["source"] for s in rl} == {"cwq", "webqsp"} and sum(s["source"] == "cwq" for s in rl) == 5
    assert not {s["question"] for s in rl} & {s["question"] for s in pool}
    assert dropped == {"duplicate question": 1, "no gold answer": 1, "also in a test set": 1}
    assert [s["idx"] for s in rl] == [f"rl-{i}" for i in range(10)]


class ScriptedTeacher:
    """complete_batch(): the teacher's reply depends on the question it sees."""

    def __init__(self, replies):
        self.replies, self.seen = replies, []

    def __call__(self, texts, max_new, temperature, top_p, stop):
        outs = []
        for t in texts:
            self.seen.append(t)
            q = t.split("\nUser: ")[-1].split("\n")[0]
            turn = t.split(protocol.RESPONSE_MARKER)[-1].count(protocol.TRIPLES_OPEN)
            text = self.replies[q][min(turn, len(self.replies[q]) - 1)]
            outs.append((text, next((s for s in stop if text.endswith(s)), None), "stop"))
        return outs


def test_generate_sft_keeps_only_grounded_correct_trajectories():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    kg = TwoStageKG(StaticKG(edges={"Lou Seal": {"sports.mascot.team": ["San Francisco Giants"]},
                                    "San Francisco Giants": {"sports.sports_team.championships": ["2010 World Series", "2014 World Series"]}}),
                    LexicalSelector())
    good_q, wrong_q, ungrounded_q = "Lou Seal's team last won the World Series when?", "Which team has Lou Seal?", "Who founded Lou Seal?"
    teacher = ScriptedTeacher({
        good_q: [" <search> Lou Seal </search>", "Team found. <search> San Francisco Giants </search>",
                 "Latest is 2014. </think>\n\n<answer> 2014 World Series </answer>"],
        wrong_q: [" <search> Lou Seal </search>", "So it is </think>\n\n<answer> Boston Red Sox </answer>"],
        ungrounded_q: [" <search> Lou Seal </search>", "I just know it. </think>\n\n<answer> Bob Lurie </answer>"],
    })
    pool = [{"idx": "g", "question": good_q, "golds": ["2014 World Series"], "task": "qa", "topic_entities": ["Lou Seal"]},
            {"idx": "w", "question": wrong_q, "golds": ["San Francisco Giants"], "task": "qa", "topic_entities": ["Lou Seal"]},
            {"idx": "u", "question": ungrounded_q, "golds": ["Bob Lurie"], "task": "qa", "topic_entities": ["Lou Seal"]}]
    kept, report = generate_sft(TeacherGenerator(teacher, tok, DEMOS), kg, pool, n=5,
                                rollout_cfg={"max_turns": 4, "max_new_tokens_per_turn": 64, "max_total_tokens": 4096}, attempts=2)
    assert [k["idx"] for k in kept] == ["g"]
    assert report["rollouts"] == 5 and report["failed_checks"] == {"correct": 2, "grounded": 2}
    segs = kept[0]["segments"]
    assert [s["from_model"] for s in segs] == [True, False, True, False, True]
    assert "Titanic" not in "".join(s["text"] for s in segs)  # demos are shown to the teacher only
    assert all("Who is the spouse of the director of Titanic?" in t for t in teacher.seen)


def test_insert_demos_places_them_before_the_question():
    prompt = protocol.build_prompt("Q?", ["E"])
    shown = insert_demos(prompt, DEMOS)
    assert shown.startswith(protocol.TASK_INSTRUCTIONS["qa"]) and shown.endswith(protocol.RESPONSE_MARKER)
    assert shown.index("Titanic") < shown.index("User: Q?")
    assert protocol.parse_prompt(prompt)[0] == "Q?"


def test_parse_completion_restores_dropped_stop_strings():
    stop = protocol.STOP_STRINGS
    assert parse_completion({"text": " x <search> A </search>", "finish_reason": "stop"}, stop) == (" x <search> A </search>", "</search>", "stop")
    assert parse_completion({"text": " x <search> A ", "finish_reason": "stop", "stop_reason": "</search>"}, stop)[1] == "</search>"
    assert parse_completion({"text": " x <search> A ", "finish_reason": "stop"}, stop)[0].endswith("</search>")  # plain OpenAI server
    assert parse_completion({"text": " ok </think>\n\n<answer> B ", "finish_reason": "stop"}, stop)[1] == "</answer>"
    assert parse_completion({"text": " rambling", "finish_reason": "length"}, stop) == (" rambling", None, "length")
