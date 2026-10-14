"""Multi-turn rollout with KG tool calls, shared by SFT data checks, GRPO and evaluation.

A trajectory is kept as token segments: what the model generated (trained on) and what the
KG tool returned (masked out of the loss, paper Sec. III-D). The training sequence is
prompt_ids + the exact sampled ids, so no hardcoded tag token ids and no re-tokenisation.
"""
from dataclasses import dataclass, field

from . import protocol


@dataclass
class Segment:
    ids: list
    text: str
    from_model: bool


@dataclass
class Trajectory:
    sample: dict
    prompt: str
    prompt_ids: list
    segments: list = field(default_factory=list)
    searches: list = field(default_factory=list)
    status: str = "running"  # answered | eos | length | bad_search | max_turns | too_long

    @property
    def response_ids(self):
        return [i for s in self.segments for i in s.ids]

    @property
    def loss_mask(self):
        """1 for tokens the model generated, 0 for KG tool output (one entry per response token)."""
        return [int(s.from_model) for s in self.segments for _ in s.ids]

    @property
    def response_text(self):
        return "".join(s.text for s in self.segments)

    @property
    def text_segments(self):
        return [(s.text, s.from_model) for s in self.segments]

    @property
    def n_search(self):
        return len(self.searches)

    @property
    def answer(self):
        return protocol.extract_answer("".join(s.text for s in self.segments if s.from_model))

    def to_record(self):
        return {
            "idx": self.sample.get("idx"),
            "question": self.sample["question"],
            "golds": self.sample.get("golds", []),
            "task": self.sample.get("task", "qa"),
            "pred": self.answer,
            "status": self.status,
            "searches": self.searches,
            "response": self.response_text,
        }


def make_prompt(sample):
    return protocol.build_prompt(sample["question"], sample.get("topic_entities"), sample.get("task", "qa"))


def tool_segment(tokenizer, triples, previous_text):
    text = protocol.format_triples(triples)
    if previous_text.endswith("\n\n"):
        text = text.lstrip("\n")
    return Segment(tokenizer(text, add_special_tokens=False).input_ids, text, False)


def rollout(generator, kg, samples, rollout_cfg, temperature=0.0, top_p=1.0):
    """Run every sample to an answer (or a limit). Returns one Trajectory per sample, in order."""
    tok = generator.tokenizer
    max_turns = rollout_cfg.get("max_turns", 6)
    per_turn = rollout_cfg.get("max_new_tokens_per_turn", 256)
    max_total = rollout_cfg.get("max_total_tokens", 4096)

    trajs = []
    for s in samples:
        prompt = make_prompt(s)
        trajs.append(Trajectory(sample=s, prompt=prompt, prompt_ids=tok(prompt).input_ids))

    active = list(trajs)
    for _ in range(max_turns):
        if not active:
            break
        budgets = [max_total - len(t.prompt_ids) - len(t.response_ids) for t in active]
        for t, b in zip(active, budgets):
            if b <= 0:
                t.status = "too_long"
        active = [t for t, b in zip(active, budgets) if b > 0]
        if not active:
            break
        max_new = min(per_turn, max(max_total - len(t.prompt_ids) - len(t.response_ids) for t in active))
        outs = generator.generate([t.prompt_ids + t.response_ids for t in active], max_new, temperature, top_p,
                                  stop=protocol.STOP_STRINGS)

        calls = []
        for t, o in zip(active, outs):
            t.segments.append(Segment(list(o.token_ids), o.text, True))
            if o.stop == protocol.SEARCH_CLOSE:
                entity = protocol.extract_search_query(o.text)
                if entity:
                    calls.append((t, entity))
                else:
                    t.status = "bad_search"
            elif o.stop == protocol.ANSWER_CLOSE:
                t.status = "answered"
            else:
                t.status = "eos" if o.finish == "eos" else "length"

        results = kg.search_many([(entity, t.sample["question"], t.sample.get("entity_ids") or {}) for t, entity in calls])
        active = []
        for (t, entity), triples in zip(calls, results):
            t.segments.append(tool_segment(tok, triples, t.segments[-1].text))
            t.searches.append({"entity": entity, "num_triples": len(triples)})
            if len(t.prompt_ids) + len(t.response_ids) >= max_total:
                t.status = "too_long"
            else:
                active.append(t)
    for t in active:
        t.status = "max_turns"
    return trajs
