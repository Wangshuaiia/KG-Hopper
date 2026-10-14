"""One sample format for every dataset and stage:

{"idx": str, "question": str, "golds": [str], "task": "qa"|"slot_filling"|"fact_checking",
 "topic_entities": [str], "entity_ids": {name: Freebase MID}, "source": str, "hops": int | None}
SFT samples additionally carry "segments": [{"text": str, "from_model": bool}]. See docs/DATA.md.
"""
import json
import os

from .config import REPO_ROOT

TEST_DIR = os.path.join(REPO_ROOT, "data", "test_set")

TEST_SETS = {
    # name: (file, question key, task)
    "cwq": ("cwq.json", "question", "qa"),
    "webqsp": ("WebQSP.json", "RawQuestion", "qa"),
    "webquestions": ("WebQuestions.json", "question", "qa"),
    "grailqa": ("graliqa.json", "question", "qa"),
    "qald": ("qald_10-en.json", "question", "qa"),
    "trex": ("T-REX.json", "input", "slot_filling"),
    "zeroshotre": ("Zero_Shot_RE.json", "input", "slot_filling"),
    "creak": ("creak.json", "sentence", "fact_checking"),
}


def _golds(name, d):
    if name == "webqsp":
        golds = [a.get("EntityName") or a.get("AnswerArgument") for p in d["Parses"] for a in p["Answers"]]
    elif name == "grailqa":
        golds = [a.get("entity_name") or a.get("answer_argument") for a in d["answer"]]
    elif name == "qald":
        golds = list(d["answer"].values()) if isinstance(d["answer"], dict) else [d["answer"]]
    elif name == "creak":
        golds = [d["label"]]
    else:
        golds = list(d.get("alias") or [])
        for key in ("answer", "answers"):
            v = d.get(key)
            golds += v if isinstance(v, list) else ([v] if v else [])
    seen, out = set(), []
    for g in golds:
        if g and str(g) not in seen:
            seen.add(str(g))
            out.append(str(g))
    return out


def _entity_ids(d):
    """Freebase MIDs of the topic entities ({name: mid}); used by the SPARQL backend as hints."""
    ids = d.get("entity_ids")
    if not ids:
        topic = d.get("topic_entity")  # test sets store {mid: name}
        ids = {name: mid for mid, name in topic.items()} if isinstance(topic, dict) else {}
    return {name: mid for name, mid in ids.items() if str(mid).startswith(("m.", "g."))}


def _topics(d):
    topic = d.get("topic_entity") or d.get("qid_topic_entity") or {}
    return list(topic.values()) if isinstance(topic, dict) else list(topic)


def load_test_set(name, n=None, start=0):
    fname, qkey, task = TEST_SETS[name]
    with open(os.path.join(TEST_DIR, fname), encoding="utf-8") as f:
        rows = json.load(f)
    rows = rows[start:] if n is None else rows[start:start + n]
    return [{"idx": f"{name}-{start + i}", "question": d[qkey], "golds": _golds(name, d), "task": task,
             "topic_entities": _topics(d), "entity_ids": _entity_ids(d), "source": name} for i, d in enumerate(rows)]


def normalize_record(r, i, source):
    golds = r.get("golds")
    if golds is None:
        a = r.get("answer", r.get("answers", []))
        golds = a if isinstance(a, list) else [a]
    out = {"idx": str(r.get("idx", f"{source}-{i}")), "question": r["question"], "golds": [str(g) for g in golds],
           "task": r.get("task", "qa"), "topic_entities": r.get("topic_entities") or _topics(r), "entity_ids": _entity_ids(r),
           "source": r.get("source", source)}
    if r.get("hops") is not None:
        out["hops"] = r["hops"]
    if "segments" in r:
        out["segments"] = r["segments"]
    return out


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_samples(spec, n=None):
    """spec: a test-set name (see TEST_SETS) or a .jsonl/.json file in the format above."""
    if spec in TEST_SETS:
        return load_test_set(spec, n)
    path = spec if os.path.isabs(spec) else os.path.join(REPO_ROOT, spec)
    rows = read_jsonl(path)
    rows = rows if n is None else rows[:n]
    source = os.path.splitext(os.path.basename(path))[0]
    return [normalize_record(r, i, source) for i, r in enumerate(rows)]
