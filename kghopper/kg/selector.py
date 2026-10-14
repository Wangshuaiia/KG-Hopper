"""Stage-2 relation selection: which of an entity's predicates to expand for a question.

  lexical  word overlap with the question; no model, no API
  claude   Claude picks the predicates (needs ANTHROPIC_API_KEY); results cached on disk
  openai   any OpenAI-compatible chat endpoint, e.g. a self-hosted `vllm serve <model>`
Selections are cached in the KG cache file under "selection", so reruns are reproducible.
"""
import logging
import os
import re

import requests
from pydantic import BaseModel

from .base import lexical_select, normalize_entity

log = logging.getLogger(__name__)

SELECT_SYSTEM = """You are the relation-pruning step of a knowledge-graph retrieval tool.
Given a question, a topic entity and the entity's predicates, pick the predicates whose objects are most
likely to help answer the question (directly or as the next hop). Only pick from the given list."""


def _selection_prompt(question, entity, relations, width):
    return f"Question: {question}\nTopic entity: {entity}\nPick up to {width} predicates from:\n" + "\n".join(relations)


class LexicalSelector:
    def select(self, question, entity, relations, width):
        return lexical_select(question, relations, width)


class _CachedSelector:
    def __init__(self, cache=None):
        self.cache = cache

    def _key(self, question, entity, width):
        return f"{normalize_entity(entity)}||{width}||{question.strip()}"

    def select(self, question, entity, relations, width):
        key = self._key(question, entity, width)
        cached = self.cache.get("selection", key) if self.cache else None
        if cached is not None and set(cached) <= set(relations):
            return cached
        try:
            picked = [r for r in self._pick(question, entity, relations, width) if r in relations][:width]
        except Exception as e:  # never let the selector break a rollout
            log.warning("relation selection failed (%s); using lexical", e)
            picked = []
        picked = picked or lexical_select(question, relations, width)
        if self.cache:
            self.cache.put("selection", key, picked)
        return picked

    def _pick(self, question, entity, relations, width):
        raise NotImplementedError


class _Selection(BaseModel):
    relations: list[str]


class ClaudeSelector(_CachedSelector):
    """offline=True: only replay cached selections (lexical for anything new)."""

    def __init__(self, cache=None, model="claude-opus-5-5", effort="low", offline=False):
        super().__init__(cache)
        self.model, self.effort = model, effort
        self.client = None
        if not offline:
            import anthropic

            self.client = anthropic.Anthropic(max_retries=5)

    def _pick(self, question, entity, relations, width):
        if self.client is None:
            return []
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=2000,
            system=SELECT_SYSTEM,
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": _selection_prompt(question, entity, relations, width)}],
            output_format=_Selection,
        )
        return response.parsed_output.relations if response.parsed_output else []


class OpenAICompatSelector(_CachedSelector):
    def __init__(self, url, model, cache=None, api_key_env="SELECTOR_API_KEY", timeout=60):
        super().__init__(cache)
        self.url = url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.headers = {"Authorization": f"Bearer {os.environ[api_key_env]}"} if os.environ.get(api_key_env) else {}
        self.timeout = timeout

    def _pick(self, question, entity, relations, width):
        resp = requests.post(self.url, headers=self.headers, timeout=self.timeout, json={
            "model": self.model, "temperature": 0, "max_tokens": 300,
            "messages": [{"role": "system", "content": SELECT_SYSTEM + " Answer with the chosen predicates, one per line."},
                         {"role": "user", "content": _selection_prompt(question, entity, relations, width)}],
        })
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"] or ""
        # keep predicates in the order the model mentions them
        found = [(m.start(), r) for r in relations for m in [re.search(re.escape(r) + r"(?![\w.])", reply)] if m]
        return [r for _, r in sorted(found)]


def make_selector(kg_cfg, cache=None, offline=False):
    kind = kg_cfg.get("relation_selector", "claude")
    if kind == "lexical":
        return LexicalSelector()
    if kind in ("claude", "llm"):
        return ClaudeSelector(cache, kg_cfg.get("claude_model", "claude-opus-5-5"), kg_cfg.get("claude_effort", "low"), offline=offline)
    if kind == "openai":
        return OpenAICompatSelector(kg_cfg["selector_url"], kg_cfg["selector_model"], cache)
    raise ValueError(f"unknown kg.relation_selector {kind!r} (lexical | claude | openai)")
