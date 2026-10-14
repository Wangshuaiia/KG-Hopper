"""Aggregate metrics over evaluation outputs (paper Sec. IV-A): Hits@1 for KBQA and slot filling,
accuracy for Creak, plus EM, answer rate and searches per question, in the paper's dataset order.

Scores are recomputed from the saved predictions, so results can be rescored without rerunning
the model, and predictions from any other system can be scored against the same gold answers.

    python evaluation/metrics.py outputs/eval                     # every <dataset>.jsonl from evaluate.py
    python evaluation/metrics.py outputs/eval/cwq.jsonl outputs/eval/creak.jsonl
    python evaluation/metrics.py --predictions my_preds.jsonl --dataset cwq
        # my_preds.jsonl: one {"idx": "cwq-17", "pred": "..."} (or {"question": ..., "pred": ...}) per line
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kghopper.data import TEST_SETS, load_test_set, read_jsonl  # noqa: E402
from kghopper.judges import normalize_answer  # noqa: E402
from kghopper.metrics import order, render_table, summarize  # noqa: E402


def load_outputs(paths):
    """{dataset: records} from evaluate.py outputs (files or directories)."""
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "*.jsonl"))) if os.path.isdir(p) else [p]
    return {os.path.splitext(os.path.basename(f))[0]: read_jsonl(f) for f in files}


def join_predictions(pred_path, dataset):
    """External predictions -> records with the test set's gold answers. Unmatched test samples count as unanswered."""
    gold = load_test_set(dataset)
    by_idx = {s["idx"]: s for s in gold}
    by_q = {normalize_answer(s["question"]): s for s in gold}
    preds = {}
    for p in read_jsonl(pred_path):
        s = by_idx.get(str(p.get("idx"))) or by_q.get(normalize_answer(p.get("question", "")))
        if s is None:
            raise SystemExit(f"prediction matches no {dataset} test sample: {str(p)[:120]}")
        preds[s["idx"]] = p.get("pred")
    return [{**s, "pred": preds.get(s["idx"]), "status": "answered" if preds.get(s["idx"]) else "no_answer"} for s in gold]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="*", help="evaluate.py output directories or <dataset>.jsonl files")
    ap.add_argument("--predictions", help="jsonl of external predictions: {idx or question, pred}")
    ap.add_argument("--dataset", choices=sorted(TEST_SETS), help="test set the --predictions belong to")
    ap.add_argument("--exclude-no-gold", action="store_true", help="drop test samples without any gold answer (WebQSP: 11, QALD-10: 1)")
    ap.add_argument("--out", default=None, help="write metrics.json and metrics.md here (default: the first directory given)")
    args = ap.parse_args()

    if args.predictions:
        if not args.dataset:
            ap.error("--predictions needs --dataset")
        outputs = {args.dataset: join_predictions(args.predictions, args.dataset)}
    elif args.paths:
        outputs = load_outputs(args.paths)
    else:
        ap.error("give evaluation outputs or --predictions")

    summaries = order([summarize(name, recs, args.exclude_no_gold) for name, recs in outputs.items() if recs])
    table = render_table(summaries)
    print(table)
    for m in summaries:
        if m["no_gold"]:
            note = "excluded" if args.exclude_no_gold else "counted as misses"
            print(f"note: {m['dataset']} has {m['no_gold']} samples without a gold answer ({note})")

    out = args.out or next((p for p in args.paths if os.path.isdir(p)), None)
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"metrics": summaries}, f, ensure_ascii=False, indent=1)
        with open(os.path.join(out, "metrics.md"), "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"\nwritten: {out}/metrics.json, {out}/metrics.md")


if __name__ == "__main__":
    main()
