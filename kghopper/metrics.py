"""Evaluation metrics (paper Sec. IV-A): Hits@1 for KBQA and slot filling, accuracy for Creak.

Hits@1: the model gives one final answer; it counts as a hit when a gold answer or alias appears in
it as whole words after normalisation (judges.hit). Creak: the verdict (true/false, yes/no, ...)
equals the label. Exact match (EM) is reported as an extra, stricter column. Every test sample
counts, including ones the model never answered.
"""
from collections import Counter

from .judges import exact_match, hit

# (name in data/test_set, label used in the paper), in the paper's table order
PAPER_ORDER = [("cwq", "CWQ"), ("webqsp", "WebQSP"), ("grailqa", "GrailQA"), ("qald", "QALD10-en"),
               ("webquestions", "WebQuestions"), ("trex", "T-REx"), ("zeroshotre", "Zero-Shot RE"), ("creak", "Creak")]
TEST_SIZES = {"cwq": 3531, "webqsp": 1639, "grailqa": 1000, "qald": 333, "webquestions": 2032,
              "trex": 5000, "zeroshotre": 3724, "creak": 1371}


def score_record(r):
    """Adds "hit" and "em" to a record with "pred", "golds" and "task"."""
    task = r.get("task", "qa")
    return {**r, "hit": hit(r.get("pred"), r.get("golds", []), task), "em": exact_match(r.get("pred"), r.get("golds", []), task)}


def summarize(name, records, exclude_no_gold=False):
    records = [score_record(r) for r in records]
    no_gold = sum(not r.get("golds") for r in records)
    if exclude_no_gold:
        records = [r for r in records if r.get("golds")]
    n = len(records)
    task = records[0].get("task", "qa") if records else "qa"
    statuses = Counter(r.get("status", "answered" if r.get("pred") else "no_answer") for r in records)
    out = {
        "dataset": name,
        "metric": "Accuracy" if task == "fact_checking" else "Hits@1",
        "score": sum(r["hit"] for r in records) / n if n else 0.0,
        "em": sum(r["em"] for r in records) / n if n else 0.0,
        "n": n,
        "test_size": TEST_SIZES.get(name),
        "no_gold": no_gold,
        "answered": sum(bool(r.get("pred")) for r in records) / n if n else 0.0,
        "searches_per_q": sum(len(r.get("searches", [])) for r in records) / n if n else 0.0,
        "status": dict(statuses),
    }
    if records and "reward" in records[0]:
        out["format_ok"] = sum(r["reward"].get("r_format", 0) > 0 for r in records) / n
    return out


def order(summaries):
    rank = {name: i for i, (name, _) in enumerate(PAPER_ORDER)}
    return sorted(summaries, key=lambda m: (rank.get(m["dataset"], len(rank)), m["dataset"]))


def render_table(summaries):
    """Markdown table in the paper's dataset order, with the macro average over the benchmarks present."""
    labels = dict(PAPER_ORDER)
    rows = order(summaries)
    lines = ["| Dataset | Metric | Score | EM | n | Answered | Searches/q |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for m in rows:
        size = m.get("test_size")
        n = f"{m['n']}" + (f" / {size} (subset)" if size and m["n"] < size else "")
        lines.append(f"| {labels.get(m['dataset'], m['dataset'])} | {m['metric']} | {100 * m['score']:.1f} | {100 * m['em']:.1f} "
                     f"| {n} | {100 * m['answered']:.0f}% | {m['searches_per_q']:.2f} |")
    bench = [m for m in rows if m["dataset"] in labels]
    if len(bench) > 1:
        avg = sum(m["score"] for m in bench) / len(bench)
        lines.append(f"| **Average** ({len(bench)} benchmarks) | | **{100 * avg:.1f}** | | | | |")
    return "\n".join(lines)
