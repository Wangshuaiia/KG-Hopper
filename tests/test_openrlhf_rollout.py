"""Runs OpenRLHF's RemoteExperienceMaker._generate_vllm_with_retrieve without Ray/vLLM/CUDA.

The method's source is executed as written; Ray actors are replaced by a fake vLLM engine that
returns scripted turns in vLLM's output format, the KG by a static graph, and CUDA tensors by
CPU tensors. Checks the packed Samples layout and that retrieve_mask hides exactly the KG output.
"""
import ast
import os
import sys
import textwrap
import types
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoTokenizer

from kghopper import protocol
from kghopper.generation import VLLMGenerator
from kghopper.kg.base import TwoStageKG
from kghopper.kg.claude_sim import StaticKG
from kghopper.kg.selector import LexicalSelector
from kghopper.rollout import rollout

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIENCE_MAKER = os.path.join(REPO, "OpenRLHF-RAG", "openrlhf", "trainer", "ppo_utils", "experience_maker.py")


@dataclass
class Samples:  # same fields as experience_maker.Samples
    sequences: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    action_mask: Optional[torch.Tensor]
    retrieve_mask: Optional[torch.Tensor]
    num_actions: object
    packed_seq_lens: object
    response_length: torch.Tensor
    total_length: torch.Tensor
    retrieve_num: torch.Tensor
    pure_response_length: torch.Tensor


class FakeRequestOutput:
    def __init__(self, ids, stop):
        self.outputs = [types.SimpleNamespace(token_ids=ids, stop_reason=stop, finish_reason="stop")]


class FakeEngine:
    """Stands in for a Ray LLMRayActor handle: engine.generate.remote(prompts, params, use_tqdm=...)."""

    def __init__(self, tokenizer, turns):
        self.tok, self.turns, self.calls = tokenizer, list(turns), []
        self.generate = types.SimpleNamespace(remote=self._generate)

    def _generate(self, prompts, params, use_tqdm=False):
        self.calls.append(prompts)
        text = self.turns.pop(0)
        stop = next((s for s in params.stop if text.endswith(s)), None)
        return [FakeRequestOutput(self.tok(text, add_special_tokens=False).input_ids, stop) for _ in prompts]


def load_method():
    tree = ast.parse(open(EXPERIENCE_MAKER, encoding="utf-8").read())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RemoteExperienceMaker")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_generate_vllm_with_retrieve")
    return ast.get_source_segment(open(EXPERIENCE_MAKER, encoding="utf-8").read(), fn)


def test_generate_vllm_with_retrieve_builds_masked_packed_samples(monkeypatch):
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    turns = [" Find the team. <search> Lou Seal </search>", "So 2014. </think>\n\n<answer> 2014 World Series </answer>"]
    engine = FakeEngine(tok, turns)

    # minimal ray / vllm stand-ins: ray.get returns the (already computed) results, SamplingParams keeps its args
    monkeypatch.setitem(sys.modules, "ray", types.SimpleNamespace(get=lambda refs: refs))
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=lambda **kw: types.SimpleNamespace(**kw)))
    fake_torch = types.SimpleNamespace(
        tensor=lambda data, device=None, dtype=None: torch.tensor(data, dtype=dtype),
        float=torch.float,
        distributed=types.SimpleNamespace(get_rank=lambda: 0, get_world_size=lambda: 1),
    )
    kg = TwoStageKG(StaticKG(edges={"Lou Seal": {"sports.mascot.team": ["San Francisco Giants"]}}), LexicalSelector())
    ns = {"torch": fake_torch, "List": list, "Samples": Samples, "protocol": protocol, "VLLMGenerator": VLLMGenerator,
          "HttpKG": lambda url: kg, "rollout": rollout}
    src = "\n".join(line[4:] for line in load_method().split("\n"))  # dedent the method
    exec(compile("def _f" + src[src.index("("):], EXPERIENCE_MAKER, "exec"), ns)

    args = types.SimpleNamespace(n_samples_per_prompt=2, max_turns=4, max_new_tokens_per_turn=64, prompt_max_len=1024,
                                 generate_max_len=1024, temperature=1.0, micro_rollout_batch_size=1, kg_url="unused")
    self = types.SimpleNamespace(vllm_engines=[engine], strategy=types.SimpleNamespace(args=args), packing_samples=True,
                                 tokenizer=tok, _entity_ids_by_question=lambda: {})
    prompt = protocol.build_prompt("Lou Seal is the mascot for the team that last won the World Series when?", ["Lou Seal"])
    samples = ns["_f"](self, ["7<|idx_prompt_split|>" + prompt])

    assert len(samples) == 2  # n_samples_per_prompt rollouts, micro batch 1
    s = samples[0]
    seq = s.sequences[0].tolist()
    n_resp = s.num_actions[0]
    assert s.packed_seq_lens[0] == len(seq) and s.retrieve_mask.shape[1] == n_resp
    prompt_ids = tok(prompt).input_ids
    assert seq[:len(prompt_ids)] == prompt_ids
    response = seq[len(prompt_ids):]
    mask = s.retrieve_mask[0].tolist()
    masked_text = tok.decode([t for t, m in zip(response, mask) if m == 0])
    trained_text = tok.decode([t for t, m in zip(response, mask) if m == 1])
    assert protocol.TRIPLES_OPEN in masked_text and "San Francisco Giants" in masked_text
    assert protocol.TRIPLES_OPEN not in trained_text and "<answer> 2014 World Series </answer>" in trained_text
    assert s.retrieve_num.tolist() == [1.0] and s.pure_response_length.tolist() == [float(sum(mask))]
    # second turn was fed prompt + first turn + tool output as token ids
    assert engine.calls[1][0]["prompt_token_ids"][:len(prompt_ids)] == prompt_ids


def test_history_resampling_replaces_one_hop_prompts(tmp_path):
    trainer_py = os.path.join(REPO, "OpenRLHF-RAG", "openrlhf", "trainer", "ppo_trainer.py")
    src = open(trainer_py, encoding="utf-8").read()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PPOTrainer")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_history_resampling")
    code = textwrap.dedent("    " + ast.get_source_segment(src, fn))  # the segment's first line has no indent
    ns = {"random": __import__("random")}
    exec(compile(code, trainer_py, "exec"), ns)

    data = tmp_path / "rl.jsonl"
    rows = [{"idx": "rl-0", "question": "q0", "golds": ["a"], "hops": 1},
            {"idx": "rl-1", "question": "q1", "golds": ["a"], "hops": 2},
            {"idx": "rl-2", "question": "q2", "golds": ["a"]}]
    data.write_text("\n".join(__import__("json").dumps(r) for r in rows))
    prompts = [f"{r['idx']}<|idx_prompt_split|>prompt {r['question']}" for r in rows]
    trainer = types.SimpleNamespace(
        prompts_dataloader=types.SimpleNamespace(dataset=types.SimpleNamespace(prompts=prompts)),
        strategy=types.SimpleNamespace(print=lambda *a: None, get_rank=lambda: 0))
    args = types.SimpleNamespace(history_resampling_episode=1, prompt_data=str(data), seed=42)

    assert ns["_history_resampling"](trainer, args, 0, 1, prompts) == prompts  # before the switch: unchanged
    out = ns["_history_resampling"](trainer, args, 1, 7, prompts)
    assert len(out) == len(prompts) and out[1:] == prompts[1:]  # same batch size, multi-hop / unknown kept
    assert out[0] in prompts[1:]  # the one-hop prompt was replaced by a multi-hop (or unlabelled) one
    args.history_resampling_episode = None
    assert ns["_history_resampling"](trainer, args, 5, 7, prompts) == prompts  # disabled
