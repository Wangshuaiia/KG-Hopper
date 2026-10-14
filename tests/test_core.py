"""Unit tests for the shared core. Run: .venv/bin/python -m pytest tests -q
(needs the Qwen2.5 tokenizer from the Hugging Face hub or its local cache)."""
import pytest
from transformers import AutoTokenizer

from kghopper import protocol
from kghopper.generation import GenOutput, HFGenerator
from kghopper.judges import AnswerJudge, ReasoningJudge, exact_match, hit
from kghopper.kg.base import TwoStageKG
from kghopper.kg.claude_sim import StaticKG
from kghopper.kg.selector import LexicalSelector
from kghopper.reward import RewardFn
from kghopper.rollout import rollout

TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"

KG_EDGES = {
    "Lou Seal": {"sports.mascot.team": ["San Francisco Giants"], "people.person.gender": ["Male"]},
    "San Francisco Giants": {
        "sports.sports_team.championships": ["2010 World Series", "2012 World Series", "2014 World Series"],
        "sports.sports_team.arena_stadium": ["Oracle Park"],
        "sports.sports_team.team_mascot": ["Lou Seal"],
        "sports.sports_team.founded": ["1883"],
    },
}
SAMPLE = {"idx": "t0", "question": "Lou Seal is the mascot for the team that last won the World Series when?",
          "golds": ["2014 World Series"], "task": "qa", "topic_entities": ["Lou Seal"]}


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(TOKENIZER)


class ScriptedGenerator:
    """Plays back fixed model turns; behaves like HFGenerator (ids, text, stop)."""

    def __init__(self, tokenizer, turns):
        self.tokenizer = tokenizer
        self.turns = list(turns)
        self.inputs = []

    def generate(self, batch_ids, max_new_tokens, temperature=0.0, top_p=1.0, stop=()):
        self.inputs.append([list(x) for x in batch_ids])
        text = self.turns.pop(0)
        hit_stop = next((s for s in stop if text.endswith(s)), None)
        return [GenOutput(self.tokenizer(text, add_special_tokens=False).input_ids, text, hit_stop,
                          "stop" if hit_stop else "eos") for _ in batch_ids]


def kg():
    return TwoStageKG(StaticKG(edges=KG_EDGES), LexicalSelector(), width=3)


TURNS = [
    " I need the team of Lou Seal. <search> Lou Seal </search>",
    "The team is the San Francisco Giants. <search> San Francisco Giants </search>",
    "The latest title is 2014. </think>\n\n<answer> 2014 World Series </answer>",
]


def run_rollout(tok, turns=TURNS):
    gen = ScriptedGenerator(tok, turns)
    traj = rollout(gen, kg(), [SAMPLE], {"max_turns": 6, "max_new_tokens_per_turn": 64, "max_total_tokens": 4096})[0]
    return traj, gen


def test_rollout_mask_covers_exactly_the_tool_output(tok):
    traj, gen = run_rollout(tok)
    assert traj.status == "answered"
    assert [s["entity"] for s in traj.searches] == ["Lou Seal", "San Francisco Giants"]
    # each later turn is fed the full history: prompt + all previous segments, no re-tokenisation
    assert gen.inputs[1][0] == traj.prompt_ids + traj.segments[0].ids + traj.segments[1].ids
    assert len(traj.loss_mask) == len(traj.response_ids)
    for seg in traj.segments:
        assert (protocol.TRIPLES_OPEN in seg.text) == (not seg.from_model)
    masked = tok.decode([i for i, m in zip(traj.response_ids, traj.loss_mask) if m == 0])
    assert masked.count(protocol.TRIPLES_OPEN) == 2 and "Lou Seal </search>" not in masked
    trained = tok.decode([i for i, m in zip(traj.response_ids, traj.loss_mask) if m == 1])
    assert "<search> Lou Seal </search>" in trained and protocol.TRIPLES_OPEN not in trained
    assert "(sports.sports_team.championships" not in trained


def test_tool_block_is_numbered_and_selected(tok):
    traj, _ = run_rollout(tok)
    tool = traj.segments[3].text
    assert tool.startswith("\n\n<searched_triples>\n(1) (San Francisco Giants, ")
    assert "2014 World Series" in tool


def test_format_check():
    traj_ok = [(" a <search> X </search>", True), (protocol.format_triples([("X", "r", "Y")]), False),
               ("so Y </think>\n\n<answer> Y </answer>", True)]
    assert protocol.check_format(traj_ok)[0]
    fake_tool = [(" a <searched_triples>\n(1) (X, r, Y)\n</searched_triples> </think>\n\n<answer> Y </answer>", True)]
    assert not protocol.check_format(fake_tool)[0]
    assert not protocol.check_format([(" no answer here", True)])[0]
    assert not protocol.check_format([(" x </think>\n\n<answer> Y </answer> extra", True)])[0]


def test_text_and_trajectory_scoring_agree(tok):
    traj, _ = run_rollout(tok)
    fn = RewardFn({"mode": "gated"})
    a, b = fn.score_trajectory(traj), fn.score_text(SAMPLE, traj.response_text)
    assert a["reward"] == b["reward"] and a["n_search"] == b["n_search"] == 2


def _segments(answer, searches):
    segs = []
    for e in ["Lou Seal", "San Francisco Giants"][:searches]:
        segs += [(f" look up <search> {e} </search>", True), (protocol.format_triples([(e, "r", "o")]), False)]
    segs.append((f"done </think>\n\n<answer> {answer} </answer>", True))
    return segs


def test_gated_reward_orders_correct_above_wrong_and_search_above_none():
    fn = RewardFn({"mode": "gated"})
    score = lambda ans, n: fn.score(SAMPLE, _segments(ans, n), n)["reward"]  # noqa: E731
    correct_2, correct_0 = score("2014 World Series", 2), score("2014 World Series", 0)
    wrong_2, wrong_0 = score("2010 World Series", 2), score("Boston Red Sox", 0)
    no_answer = fn.score(SAMPLE, [(" I give up", True)], 0)["reward"]
    assert correct_2 > correct_0 > wrong_2 >= wrong_0 > no_answer
    assert min(correct_2, correct_0) > max(wrong_2, wrong_0)


def test_paper_reward_has_the_documented_flaw():
    fn = RewardFn({"mode": "paper"})
    wrong_2 = fn.score(SAMPLE, _segments("2010 World Series", 2), 2)["reward"]
    correct_0 = fn.score(SAMPLE, _segments("2014 World Series", 0), 0)["reward"]
    assert wrong_2 > correct_0  # why mode=gated is the default


class FakeBackend:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def complete(self, prompt, max_new_tokens):
        self.prompts.append(prompt)
        return self.reply


def test_answer_judge_parses_only_the_reply():
    judge = AnswerJudge("rule")
    judge.backend, judge.kind = FakeBackend("0"), "hf"
    assert judge("capital of Japan?", "Paris", ["Tokyo"]) == 0.0
    assert "1 if correct" in judge.backend.prompts[0]  # the prompt contains a '1' that must not be parsed
    judge.backend = FakeBackend("1")
    assert judge("capital of Japan?", "Tokyo", ["Tokyo"]) == 1.0
    judge.backend = FakeBackend("I am not sure")
    assert judge("capital of Japan?", "Paris", ["Tokyo"]) == 0.0  # falls back to rule matching
    judge.backend = FakeBackend("0")  # a judge that wrongly rejects an exact alias match
    assert judge("who influenced coleridge?", "Baruch Spinoza", ["Giambattista Vico", "Baruch Spinoza"]) == 1.0
    assert judge.backend.prompts == []  # rule_first: not asked
    judge.rule_first = False
    assert judge("who influenced coleridge?", "Baruch Spinoza", ["Giambattista Vico", "Baruch Spinoza"]) == 0.0


def test_reasoning_judge_parsing():
    judge = ReasoningJudge("none")
    for reply, expected in [("Good.\nScore: 0.8", 0.8), ("The score for this reasoning process is **1**.", 1.0),
                            ("**Final Score**: 0.5", 0.5), ("no number", 0.0)]:
        judge.backend = FakeBackend(reply)
        assert judge("q", "r") == pytest.approx(expected)


def test_answer_matching():
    assert exact_match("2014 World Series", ["2014 World Series"])
    assert exact_match("Jamaican English; Jamaican Creole", ["Jamaican English"])
    assert hit("They won the 2014 World Series", ["2014 World Series"])
    assert not exact_match("2010 World Series", ["2014 World Series"])
    assert hit("False", ["false"], "fact_checking") and not hit("true", ["false"], "fact_checking")


def test_hf_generator_cut_ends_at_stop_string(tok):
    gen = HFGenerator.__new__(HFGenerator)
    gen.tokenizer, gen.eos, gen.pad_id = tok, {tok.eos_token_id, 151643}, tok.pad_token_id or 151643
    ids = tok(" think <search> Lou Seal </search> and more text", add_special_tokens=False).input_ids + [gen.pad_id] * 3
    out = gen._cut(ids, protocol.STOP_STRINGS)
    assert out.stop == protocol.SEARCH_CLOSE and out.finish == "stop"
    assert out.text.rstrip().endswith("</search>") and "more" not in out.text
    eos_ids = tok(" just text", add_special_tokens=False).input_ids + [tok.eos_token_id, gen.pad_id]
    out = gen._cut(eos_ids, protocol.STOP_STRINGS)
    assert out.finish == "eos" and out.token_ids[-1] == tok.eos_token_id


@pytest.mark.parametrize("task,topics", [("qa", ["Lou Seal"]), ("slot_filling", []), ("fact_checking", ["Soup", "Water"])])
def test_parse_prompt_roundtrip(task, topics):
    q = "Cirith Ungol [SEP] genre" if task == "slot_filling" else "Is soup\nliquid?"
    assert protocol.parse_prompt(protocol.build_prompt(q, topics, task)) == (q, topics, task)


def test_hf_generator_neutralises_checkpoint_sampling_defaults(tok):
    from transformers import GenerationConfig

    model = type("M", (), {"generation_config": GenerationConfig.from_pretrained(TOKENIZER)})()
    HFGenerator(model, tok)
    cfg = model.generation_config
    assert (cfg.temperature, cfg.top_p, cfg.repetition_penalty) == (1.0, 1.0, 1.0)
    cfg.validate(strict=True)  # save_pretrained() validates strictly; an invalid config breaks checkpointing
