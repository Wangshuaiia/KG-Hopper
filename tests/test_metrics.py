"""Evaluation metrics: Hits@1 / accuracy, aggregation table, scoring external predictions."""
import importlib.util
import json
import os

import pytest

from kghopper.metrics import render_table, summarize

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rec(pred, golds, task="qa", status="answered"):
    return {"pred": pred, "golds": golds, "task": task, "status": status, "searches": [{"entity": "x", "num_triples": 1}]}


def test_hits_at_1_em_and_unanswered():
    records = [rec("2014 World Series", ["2014 World Series"]),       # hit + EM
               rec("They won the 2014 World Series", ["2014 World Series"]),  # hit, not EM
               rec("2010 World Series", ["2014 World Series"]),       # miss
               rec(None, ["2014 World Series"], status="length")]     # never answered: still counted
    m = summarize("cwq", records)
    assert (m["metric"], m["score"], m["em"], m["n"], m["answered"]) == ("Hits@1", 0.5, 0.25, 4, 0.75)


def test_creak_is_accuracy():
    m = summarize("creak", [rec("False.", ["false"], "fact_checking"), rec("true", ["false"], "fact_checking")])
    assert (m["metric"], m["score"]) == ("Accuracy", 0.5)


def test_samples_without_gold():
    records = [rec("Tokyo", ["Tokyo"]), rec("anything", [])]
    assert summarize("webqsp", records)["score"] == 0.5
    m = summarize("webqsp", records, exclude_no_gold=True)
    assert (m["score"], m["n"], m["no_gold"]) == (1.0, 1, 1)


def test_table_order_average_and_subset_marks():
    ms = [summarize("creak", [rec("false", ["false"], "fact_checking")]),
          summarize("cwq", [rec("Tokyo", ["Tokyo"]), rec("Paris", ["Tokyo"])]),
          summarize("my_synthetic_set", [rec("Tokyo", ["Tokyo"])])]
    table = render_table(ms)
    lines = table.splitlines()
    assert lines[2].startswith("| CWQ | Hits@1 | 50.0") and lines[3].startswith("| Creak | Accuracy | 100.0")
    assert "2 / 3531 (subset)" in lines[2]
    assert "**Average** (2 benchmarks) | | **75.0**" in table  # non-benchmark sets are not averaged


def test_external_predictions_are_scored_against_the_test_set(tmp_path):
    spec = importlib.util.spec_from_file_location("eval_metrics", os.path.join(REPO, "evaluation", "metrics.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    preds = tmp_path / "preds.jsonl"
    preds.write_text("\n".join(json.dumps(p) for p in [
        {"idx": "cwq-0", "pred": "2014 World Series"},
        {"question": "Lou Seal is the mascot for the team that last won the World Series when?", "pred": "2014 World Series"},
        {"idx": "cwq-1", "pred": "Harvard"}]))
    records = mod.join_predictions(str(preds), "cwq")
    m = summarize("cwq", records)
    assert m["n"] == 3531 and m["answered"] == pytest.approx(2 / 3531)  # cwq-0 given twice; everything else unanswered
    assert m["score"] == pytest.approx(1 / 3531)
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"idx": "nope", "question": "not in the test set", "pred": "x"}))
    with pytest.raises(SystemExit):
        mod.join_predictions(str(bad), "cwq")
