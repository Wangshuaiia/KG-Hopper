#!/usr/bin/env bash
# End-to-end smoke run on a laptop (CPU or Apple MPS), Freebase simulated by Claude:
#   synthetic data -> cold-start SFT -> GRPO with KG tool calls -> evaluation
# Needs ANTHROPIC_API_KEY. Extra arguments are passed to every stage as config overrides, e.g.
#   bash scripts/run_local_pipeline.sh grpo.steps=5 model.policy=Qwen/Qwen2.5-1.5B-Instruct
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
CONFIG=${CONFIG:-configs/local_cpu.yaml}
SET=("$@")

$PY -m pytest tests -q -p no:cacheprovider
$PY -m kghopper.build_data --config "$CONFIG" --set ${SET[@]+"${SET[@]}"}
$PY -u -m kghopper.train_sft --config "$CONFIG" --set ${SET[@]+"${SET[@]}"}
$PY -u -m kghopper.train_grpo --config "$CONFIG" --set ${SET[@]+"${SET[@]}"}
$PY -u evaluation/evaluate.py --config "$CONFIG" --set ${SET[@]+"${SET[@]}"}
