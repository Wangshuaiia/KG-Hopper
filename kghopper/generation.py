"""Text generators behind one interface, so rollout code is identical on CPU (HF) and GPU (vLLM).

generate(batch_ids, max_new_tokens, temperature, top_p, stop) -> [GenOutput]
  batch_ids: full token ids so far (prompt + response) for each sequence.
  GenOutput.token_ids are the new tokens exactly as sampled, cut right after a stop string
  or the EOS token; .stop is the stop string that ended the turn (None otherwise).
"""
from dataclasses import dataclass

import torch


@dataclass
class GenOutput:
    token_ids: list
    text: str
    stop: str = None
    finish: str = "length"  # "stop" (stop string) | "eos" | "length"


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def eos_ids(tokenizer, model=None):
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(gen_eos, int):
        ids.add(gen_eos)
    elif gen_eos:
        ids.update(gen_eos)
    return ids


class HFGenerator:
    """Batched generation with transformers (works on CPU and Apple MPS)."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.eos = eos_ids(tokenizer, model)
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else min(self.eos)
        # transformers replaces generate() arguments that equal the library defaults (temperature=1.0,
        # top_p=1.0, repetition_penalty=1.0) with the checkpoint's generation_config (Qwen-Instruct:
        # 0.7 / 0.8 / 1.1). Neutralise it so sampling follows exactly the configured distribution.
        gen_cfg = getattr(model, "generation_config", None)
        if gen_cfg is not None:
            gen_cfg.temperature, gen_cfg.top_p, gen_cfg.top_k, gen_cfg.repetition_penalty = 1.0, 1.0, None, 1.0
            gen_cfg.do_sample = False

    def _cut(self, ids, stop):
        """Trim padding, then end at the first stop string or EOS."""
        end = next((i for i, t in enumerate(ids) if t in self.eos or t == self.pad_id), len(ids))
        hit_eos = end < len(ids) and ids[end] in self.eos
        body = ids[:end]
        text = self.tokenizer.decode(body, skip_special_tokens=True)
        found = [(text.find(s), s) for s in stop if s in text]
        if found:
            _, s = min(found)
            lo, hi = 1, len(body)  # smallest prefix whose text contains the stop string
            while lo < hi:
                mid = (lo + hi) // 2
                if s in self.tokenizer.decode(body[:mid], skip_special_tokens=True):
                    hi = mid
                else:
                    lo = mid + 1
            body = body[:lo]
            return GenOutput(body, self.tokenizer.decode(body, skip_special_tokens=True), s, "stop")
        if hit_eos:
            return GenOutput(body + [ids[end]], text, None, "eos")
        return GenOutput(body, text, None, "length")

    @torch.no_grad()
    def generate(self, batch_ids, max_new_tokens, temperature=0.0, top_p=1.0, stop=()):
        from transformers import GenerationConfig

        device = next(self.model.parameters()).device
        width = max(len(x) for x in batch_ids)
        input_ids = torch.tensor([[self.pad_id] * (width - len(x)) + list(x) for x in batch_ids], device=device)
        attention = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in batch_ids], device=device)
        sampling = temperature > 0
        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=sampling,
            temperature=temperature if sampling else 1.0,
            top_p=top_p if sampling else 1.0,
            top_k=0,
            repetition_penalty=1.0,
            pad_token_id=self.pad_id,
            eos_token_id=sorted(self.eos),
            stop_strings=list(stop) or None,
        )
        was_training = self.model.training
        self.model.eval()
        out = self.model.generate(input_ids=input_ids, attention_mask=attention, generation_config=gen_cfg, tokenizer=self.tokenizer)
        if was_training:
            self.model.train()
        new = out[:, width:].tolist()
        return [self._cut(ids, stop) for ids in new]


class VLLMGenerator:
    """vLLM engine(s) for GPU runs. `engines` is a vllm.LLM, or a list of Ray actor handles
    exposing generate() (OpenRLHF's LLMRayActor)."""

    def __init__(self, engines, tokenizer, remote=False):
        self.engines = engines if isinstance(engines, (list, tuple)) else [engines]
        self.tokenizer = tokenizer
        self.remote = remote

    def generate(self, batch_ids, max_new_tokens, temperature=0.0, top_p=1.0, stop=()):
        from vllm import SamplingParams

        params = SamplingParams(temperature=temperature, top_p=top_p if temperature > 0 else 1.0, max_tokens=max_new_tokens,
                                stop=list(stop), include_stop_str_in_output=True, skip_special_tokens=True)
        prompts = [{"prompt_token_ids": list(ids)} for ids in batch_ids]
        n = len(self.engines)
        chunk = (len(prompts) + n - 1) // n
        parts = [prompts[i * chunk:(i + 1) * chunk] for i in range(n)]
        if self.remote:
            import ray

            refs = [eng.generate.remote(p, params, use_tqdm=False) for eng, p in zip(self.engines, parts) if p]
            outputs = [o for res in ray.get(refs) for o in res]
        else:
            outputs = [o for eng, p in zip(self.engines, parts) if p for o in eng.generate(p, params, use_tqdm=False)]
        results = []
        for o in outputs:
            c = o.outputs[0]
            ids = list(c.token_ids)
            stop_str = c.stop_reason if isinstance(c.stop_reason, str) else None
            finish = "stop" if stop_str else ("eos" if c.finish_reason == "stop" else "length")
            results.append(GenOutput(ids, self.tokenizer.decode(ids, skip_special_tokens=True), stop_str, finish))
        return results
