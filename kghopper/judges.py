"""Answer and reasoning judges (paper: f_a and f_r).

Fixes over the original reward server:
  * only the judge's newly generated tokens are parsed (the old code decoded the prompt too,
    and the prompt's "return '1'" made every answer count as correct);
  * greedy decoding, so the same trajectory always gets the same score;
  * long reasoning is shortened from the middle, keeping the conclusion;
  * rule-based matching as default / fallback.
"""
import re
import string

import requests

# ---------------------------------------------------------------- rule-based matching


def normalize_answer(s):
    s = str(s).lower().replace("_", " ")
    s = "".join(" " if ch in set(string.punctuation) | {"‘", "’", "´", "`"} else ch for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


_TRUE = {"true", "yes", "correct", "supported"}
_FALSE = {"false", "no", "incorrect", "refuted"}


def _verdict(s):
    words = normalize_answer(s).split()
    if not words:
        return None
    if words[0] in _TRUE:
        return "true"
    if words[0] in _FALSE:
        return "false"
    return None


def hit(pred, golds, task="qa"):
    """Hits@1 as in ToG-style KBQA evaluation: a gold alias appears in the prediction."""
    if not pred:
        return False
    if task == "fact_checking":
        return _verdict(pred) is not None and any(_verdict(pred) == _verdict(g) for g in golds)
    p = f" {normalize_answer(pred)} "
    return any(normalize_answer(g) and f" {normalize_answer(g)} " in p for g in golds)


def exact_match(pred, golds, task="qa"):
    """The prediction, or one item of a listed prediction, equals a gold alias."""
    if not pred:
        return False
    if task == "fact_checking":
        return hit(pred, golds, task)
    parts = {normalize_answer(x) for x in re.split(r"[;,/|\n]|\band\b", pred)} | {normalize_answer(pred)}
    return any(normalize_answer(g) in parts for g in golds if normalize_answer(g))


def rule_correct(pred, golds, task="qa", max_words=15):
    """Reward-side matching: exact match, or a hit inside a short answer (so listing every
    candidate entity does not pay off)."""
    if exact_match(pred, golds, task):
        return True
    return bool(pred) and len(normalize_answer(pred).split()) <= max_words and hit(pred, golds, task)


# ---------------------------------------------------------------- LLM backends


class HFTextBackend:
    """Local judge model; returns only the newly generated text, greedy."""

    def __init__(self, model_name, device=None, dtype="float32"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .generation import pick_device

        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=getattr(torch, dtype)).to(self.device).eval()

    def complete(self, prompt, max_new_tokens):
        import torch

        if getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                                      top_k=None, pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


class OpenAICompatBackend:
    """A judge served behind an OpenAI-compatible endpoint, e.g. `vllm serve meta-llama/Llama-3.3-70B-Instruct`."""

    def __init__(self, url, model, api_key_env="JUDGE_API_KEY", timeout=120):
        import os

        self.url = url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.headers = {"Authorization": f"Bearer {os.environ[api_key_env]}"} if os.environ.get(api_key_env) else {}
        self.timeout = timeout

    def complete(self, prompt, max_new_tokens):
        resp = requests.post(self.url, headers=self.headers, timeout=self.timeout, json={
            "model": self.model, "temperature": 0, "max_tokens": max_new_tokens,
            "messages": [{"role": "user", "content": prompt}],
        })
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"] or ""


def make_backend(kind, model, url=None):
    if kind == "hf":
        return HFTextBackend(model)
    if kind == "openai":
        return OpenAICompatBackend(url, model)
    raise ValueError(f"unknown judge backend {kind!r}")


# ---------------------------------------------------------------- judges

ANSWER_PROMPT = """You are judging whether a predicted answer to a knowledge question is correct.

Question: {question}
Gold answer(s): {golds}
Predicted answer: {pred}

Is the predicted answer correct, i.e. does it refer to the same entity or value as a gold answer?
Reply with a single digit: 1 if correct, 0 if not."""

REASON_PROMPT = """You are evaluating an agent that answers questions by searching a knowledge graph step by step.

Question: {question}

Agent's reasoning process (its searches, the retrieved triples, and its inferences):
{reasoning}

Rate how sound and coherent the reasoning is: does it decompose the question sensibly, search the right
entities, use the retrieved triples correctly, and reach its conclusion logically?
Give a short evaluation, then end with a final line of the form "Score: X" where X is between 0 and 1."""


class AnswerJudge:
    """kind: rule | hf | openai.

    rule_first: an answer that matches a gold alias (rule_correct) scores 1 without asking the LLM;
    the LLM only decides the rest (paraphrases, other surface forms). Small judges were seen to
    reject exact matches. LLM verdicts that cannot be parsed fall back to rule matching.
    """

    def __init__(self, kind="rule", model=None, url=None, max_words=15, rule_first=True):
        self.kind = kind
        self.max_words = max_words
        self.rule_first = rule_first
        self.backend = None if kind == "rule" else make_backend(kind, model, url)
        self.last_raw = None

    def __call__(self, question, pred, golds, task="qa"):
        self.last_raw = None
        if not pred:
            return 0.0
        by_rule = rule_correct(pred, golds, task, self.max_words)
        if self.backend is None or task == "fact_checking" or (self.rule_first and by_rule):
            return float(by_rule)
        raw = self.backend.complete(ANSWER_PROMPT.format(question=question, golds="; ".join(map(str, golds)), pred=pred), 8)
        self.last_raw = raw
        m = re.search(r"[01]", raw)
        return float(m.group(0)) if m else float(by_rule)


def shorten_middle(text, max_chars):
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n[...]\n" + text[-half:]


class ReasoningJudge:
    """kind: none | hf | openai. Returns a score in [0, 1]."""

    def __init__(self, kind="none", model=None, url=None, max_chars=6000):
        self.kind = kind
        self.max_chars = max_chars
        self.backend = None if kind == "none" else make_backend(kind, model, url)
        self.last_raw = None

    def __call__(self, question, reasoning):
        if self.backend is None:
            return 0.0
        raw = self.backend.complete(REASON_PROMPT.format(question=question, reasoning=shorten_middle(reasoning, self.max_chars)), 256)
        self.last_raw = raw
        scores = re.findall(r"score[^0-9\n]{0,40}?((?<![\d.])[01](?:\.\d+)?|\.\d+)", raw, flags=re.IGNORECASE)
        return max(0.0, min(1.0, float(scores[-1]))) if scores else 0.0
