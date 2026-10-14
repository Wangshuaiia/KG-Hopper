"""Evaluate a policy with exactly the rollout, KG tool and answer matching used in training.

    python evaluation/evaluate.py --config configs/local_cpu.yaml
    python evaluation/evaluate.py --config configs/local_cpu.yaml --model checkpoints/sft --datasets cwq creak --n 20

Datasets: data/test_set names (cwq webqsp webquestions grailqa qald trex zeroshotre creak) or
.jsonl files in the kghopper sample format. Every sample is written out, including ones that
never produced an answer. Metrics (kghopper/metrics.py, paper Sec. IV-A): Hits@1 for KBQA and slot
filling, accuracy for Creak, plus exact match, answer rate and searches per question; the table is
also written to metrics.md / metrics.json and can be recomputed with evaluation/metrics.py.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kghopper.config import add_config_args, load_config, resolve_path  # noqa: E402
from kghopper.data import load_samples  # noqa: E402
from kghopper.kg import make_kg  # noqa: E402
from kghopper.metrics import order, render_table, score_record, summarize  # noqa: E402
from kghopper.reward import RewardFn  # noqa: E402
from kghopper.rollout import rollout  # noqa: E402


def make_generator(backend, model_path, cfg):
    if backend == "vllm":
        from transformers import AutoTokenizer
        from vllm import LLM

        from kghopper.generation import VLLMGenerator

        return VLLMGenerator(LLM(model=model_path, trust_remote_code=True), AutoTokenizer.from_pretrained(model_path))
    from kghopper.generation import HFGenerator
    from kghopper.modeling import load_policy

    model, tok = load_policy(model_path, cfg.model.get("dtype", "float32"), cfg.model.get("device", "auto"))
    return HFGenerator(model.eval(), tok)


def main():
    parser = add_config_args(argparse.ArgumentParser())
    parser.add_argument("--model", default=None, help="checkpoint dir or HF id (default: eval.model)")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--n", type=int, default=None, help="samples per dataset (default: eval.n)")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    ec = cfg.get_section("eval")

    model_path = args.model or ec.get("model") or cfg.model.policy
    if os.path.exists(resolve_path(model_path)):
        model_path = resolve_path(model_path)
    datasets = args.datasets or ec.get("datasets", ["cwq"])
    n = args.n if args.n is not None else ec.get("n")
    out_dir = resolve_path(args.output_dir or ec.get("output_dir", "outputs/eval"))
    os.makedirs(out_dir, exist_ok=True)

    generator = make_generator(args.backend, model_path, cfg)
    kg = make_kg(cfg.kg)
    reward_fn = RewardFn(cfg.reward)
    bs = ec.get("batch_size", 8)
    print(f"model={model_path} backend={args.backend} datasets={datasets} n={n}")

    all_metrics = []
    for spec in datasets:
        name = os.path.splitext(os.path.basename(spec))[0]
        samples = load_samples(spec, n)
        records, t0 = [], time.time()
        for i in range(0, len(samples), bs):
            trajs = rollout(generator, kg, samples[i:i + bs], cfg.rollout, temperature=ec.get("temperature", 0.0))
            for t in trajs:
                rec = score_record(t.to_record())
                rec["reward"] = {k: v for k, v in reward_fn.score_trajectory(t).items() if k != "pred"}
                records.append(rec)
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        m = summarize(name, records)
        m["seconds"] = round(time.time() - t0, 1)
        all_metrics.append(m)
        print(json.dumps(m, ensure_ascii=False))

    all_metrics = order(all_metrics)
    table = render_table(all_metrics)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"model": model_path, "config": args.config, "metrics": all_metrics}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "metrics.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("\n" + table)
    print(f"\nKG tool calls: {getattr(kg, 'calls', 'n/a')} | outputs in {out_dir} (rescore: python evaluation/metrics.py {out_dir})")


if __name__ == "__main__":
    main()
