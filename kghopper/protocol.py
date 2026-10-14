"""The interaction protocol from the paper (Sec. III-B), used by SFT, RL and evaluation alike.

The model thinks inside <think>, calls the KG tool with <search> entity </search>, receives
<searched_triples> ... </searched_triples> from the system, and ends with <answer> ... </answer>.
"""
import re

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
SEARCH_OPEN, SEARCH_CLOSE = "<search>", "</search>"
TRIPLES_OPEN, TRIPLES_CLOSE = "<searched_triples>", "</searched_triples>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"

# Generation stops after either of these; the rollout then calls the KG tool or finishes.
STOP_STRINGS = [SEARCH_CLOSE, ANSWER_CLOSE]

# Everything up to and including this marker is the prompt; the rest is the response.
RESPONSE_MARKER = "Assistant: " + THINK_OPEN

_TOOL_HOWTO = (
    'During thinking, the Assistant can look up one entity in a knowledge graph by writing "<search> entity name </search>". '
    'The system then returns triples about that entity as "<searched_triples>\\n(1) (head, relation, tail)\\n...</searched_triples>". '
    "The Assistant can search several times, follow the triples to new entities, and go back to an earlier entity if a path fails."
)

TASK_INSTRUCTIONS = {
    "qa": (
        "The User asks a question, and the Assistant answers it by reasoning over a knowledge graph.\n"
        "The Assistant first thinks about the reasoning process and then gives the final answer. "
        "The reasoning process and the final answer are enclosed within <think> </think> and <answer> </answer> tags, "
        'i.e., "<think> reasoning process here </think>\\n\\n<answer> final answer here </answer>". '
        "The final answer is a short entity name or value.\n" + _TOOL_HOWTO
    ),
    "slot_filling": (
        'The User gives a subject and a relation separated by "[SEP]", and the Assistant fills in the missing object '
        "by reasoning over a knowledge graph.\n"
        "The Assistant first thinks about the reasoning process and then gives the object. "
        "The reasoning process and the object are enclosed within <think> </think> and <answer> </answer> tags, "
        'i.e., "<think> reasoning process here </think>\\n\\n<answer> object here </answer>".\n' + _TOOL_HOWTO
    ),
    "fact_checking": (
        "The User states a claim, and the Assistant verifies it by reasoning over a knowledge graph.\n"
        "The Assistant first thinks about the reasoning process and then answers true or false. "
        "The reasoning process and the verdict are enclosed within <think> </think> and <answer> </answer> tags, "
        'i.e., "<think> reasoning process here </think>\\n\\n<answer> true or false </answer>".\n' + _TOOL_HOWTO
    ),
}


def build_prompt(question, topic_entities=None, task="qa"):
    """Raw-text prompt (no chat template) shared by every stage."""
    lines = [TASK_INSTRUCTIONS[task], "", f"User: {question}"]
    if topic_entities:
        lines.append("Topic entity: " + "; ".join(topic_entities))
    lines.append(RESPONSE_MARKER)
    return "\n".join(lines)


def parse_prompt(prompt):
    """Inverse of build_prompt: (question, topic_entities, task). Used where only the prompt
    string is available (OpenRLHF passes prompts as text)."""
    task = next((name for name, instr in TASK_INSTRUCTIONS.items() if prompt.startswith(instr)), "qa")
    body = prompt[len(TASK_INSTRUCTIONS[task]):]
    m = re.search(r"\nUser: (.*?)\n(?:Topic entity: (.*?)\n)?" + re.escape(RESPONSE_MARKER) + r"$", body, flags=re.DOTALL)
    if not m:
        raise ValueError("prompt was not built by kghopper.protocol.build_prompt")
    topics = [t for t in (m.group(2) or "").split("; ") if t]
    return m.group(1), topics, task


def split_prompt_response(text):
    """Split a full decoded sequence into (prompt, response) at the first response marker."""
    idx = text.find(RESPONSE_MARKER)
    if idx < 0:
        return None, text
    end = idx + len(RESPONSE_MARKER)
    return text[:end], text[end:]


def format_triples(triples):
    """Tool output block. The leading newlines separate it from the model's </search>."""
    if triples:
        body = "\n".join(f"({i}) ({h}, {r}, {t})" for i, (h, r, t) in enumerate(triples, 1))
    else:
        body = "(no triples found)"
    return f"\n\n{TRIPLES_OPEN}\n{body}\n{TRIPLES_CLOSE}\n\n"


def extract_search_query(model_text):
    """Entity inside the last <search> ... </search> of a model turn, or None."""
    m = re.findall(re.escape(SEARCH_OPEN) + r"(.*?)" + re.escape(SEARCH_CLOSE), model_text, flags=re.DOTALL)
    if not m:
        return None
    query = " ".join(m[-1].replace('"', " ").split())
    return query or None


def extract_answer(text):
    m = re.findall(re.escape(ANSWER_OPEN) + r"(.*?)" + re.escape(ANSWER_CLOSE), text, flags=re.DOTALL)
    return m[-1].strip() if m else None


_TOOL_BLOCK = re.compile(r"\n*" + re.escape(TRIPLES_OPEN) + r".*?" + re.escape(TRIPLES_CLOSE) + r"\n*", flags=re.DOTALL)


def segments_from_text(response):
    """Rebuild (text, from_model) segments from a decoded response.

    Used where only text is available (the OpenRLHF reward server). The local trainer and
    the evaluator keep exact segments from the rollout instead.
    """
    segments, pos = [], 0
    for m in _TOOL_BLOCK.finditer(response):
        if m.start() > pos:
            segments.append((response[pos:m.start()], True))
        segments.append((m.group(0), False))
        pos = m.end()
    if pos < len(response):
        segments.append((response[pos:], True))
    return segments


def check_format(segments):
    """Paper's format condition: <think>/<search>/<answer> used correctly in position and order.

    `segments` is a list of (text, from_model). Returns (ok, reason).
    """
    model_text = "".join(t for t, from_model in segments if from_model)
    full = "".join(t for t, _ in segments)
    if TRIPLES_OPEN in model_text or TRIPLES_CLOSE in model_text:
        return False, "model wrote a <searched_triples> block itself"
    if THINK_OPEN in model_text:
        return False, "extra <think>"
    if full.count(THINK_CLOSE) != 1:
        return False, "need exactly one </think>"
    if model_text.count(ANSWER_OPEN) != 1 or model_text.count(ANSWER_CLOSE) != 1:
        return False, "need exactly one <answer> ... </answer>"
    if model_text.count(SEARCH_OPEN) != model_text.count(SEARCH_CLOSE):
        return False, "unbalanced <search> tags"
    think_end, answer_start = full.find(THINK_CLOSE), full.find(ANSWER_OPEN)
    if not think_end < answer_start:
        return False, "<answer> must come after </think>"
    if full.rfind(SEARCH_CLOSE) > think_end:
        return False, "<search> after </think>"
    tail = full[full.find(ANSWER_CLOSE) + len(ANSWER_CLOSE):]
    if tail.strip():
        return False, "text after </answer>"
    if not (extract_answer(model_text) or "").strip():
        return False, "empty answer"
    return True, "ok"
