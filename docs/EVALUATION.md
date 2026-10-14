# Evaluation: test sets, running, metrics

## Test sets

All eight benchmarks used in the paper are public, and their test sets are included in `data/test_set/`. They
use the preprocessed layout common in KG-reasoning work (e.g. ToG), with topic entities attached
(`topic_entity` with Freebase MIDs, `qid_topic_entity` with Wikidata QIDs). `kghopper/data.py` loads each
of them into the shared sample schema (see `docs/DATA.md`).

| Name (`--datasets`) | Benchmark | KG in the paper | Task | Metric | Test samples | Licence |
| --- | --- | --- | --- | --- | --- | --- |
| `cwq` | ComplexWebQuestions | Freebase | KBQA | Hits@1 | 3,531 | – |
| `webqsp` | WebQSP | Freebase | KBQA | Hits@1 | 1,639 | CC |
| `grailqa` | GrailQA (1,000 sampled from test) | Freebase | KBQA | Hits@1 | 1,000 | – |
| `qald` | QALD-10 (en) | Wikidata | KBQA | Hits@1 | 333 | MIT |
| `webquestions` | WebQuestions | Freebase | KBQA | Hits@1 | 2,032 | – |
| `trex` | T-REx | Wikidata | slot filling | Hits@1 | 5,000 | MIT |
| `zeroshotre` | Zero-Shot RE | Wikidata | slot filling | Hits@1 | 3,724 | MIT |
| `creak` | Creak | Wikidata | fact checking | Accuracy | 1,371 | MIT |

WebQSP has 11 test questions and QALD-10 has 1 without any gold answer. By default they count as misses;
`metrics.py --exclude-no-gold` drops them.

## Running an evaluation

```bash
# all eight test sets, full size (eval.datasets in configs/paper_gpu.yaml), vLLM on GPU
python evaluation/evaluate.py --config configs/paper_gpu.yaml --backend vllm --model checkpoints/grpo

# a few datasets / a subset, transformers on CPU or Apple MPS
python evaluation/evaluate.py --config configs/local_cpu.yaml --model checkpoints/grpo --datasets cwq webqsp creak --n 100
```

- `--datasets` takes the names above or any jsonl in the sample schema; without it, `eval.datasets` from the
  config is used.
- `--n` limits each dataset to its first n samples.
- The policy runs the same rollout as in training (`kghopper.rollout`), with greedy decoding and the KG tool
  of the config's `kg` section.

Outputs go to `--output_dir` (default `outputs/eval/`):

- `<dataset>.jsonl`, one line per test sample, including unanswered ones:
  - `idx`, `question`, `golds`, `task`, `pred` (the final answer, or null), `status` (`answered`, `length`,
    `max_turns`, ...);
  - `searches` (entity and number of triples per KG call);
  - `response` (the full trajectory);
  - `hit`, `em`, `reward` (reward components).
- `metrics.json` and `metrics.md`: the aggregate table.

## Metrics (paper Sec. IV-A)

Implemented in `kghopper/metrics.py`; matching in `kghopper/judges.py`.

- **Hits@1** (KBQA, slot filling). The model returns one final answer (the text in `<answer> ... </answer>`).
  It is a hit when any gold answer or alias occurs in it as whole words, after normalising both: lower case,
  `_` → space, punctuation and the articles a / an / the removed, whitespace collapsed.
- **Accuracy** (Creak). The first word of the answer is mapped to a verdict (`true` / `yes` / `correct` /
  `supported` → true; `false` / `no` / `incorrect` / `refuted` → false) and compared with the label.
- **EM** (extra column). The answer, or one item of a listed answer, equals a gold alias.
- **Answered** and **Searches/q**: share of samples that produced an answer, and KG calls per question.
- Every test sample is in the denominator; a sample with no answer is a miss.
- **Average**: the unweighted mean of the benchmark scores present (custom jsonl sets are listed but not
  averaged).

## Aggregating and rescoring: `evaluation/metrics.py`

The script recomputes the metrics from saved predictions and prints the table in the paper's dataset order:

```bash
python evaluation/metrics.py outputs/eval                          # every <dataset>.jsonl in the directory
python evaluation/metrics.py outputs/eval/cwq.jsonl outputs/eval/creak.jsonl --out results/
python evaluation/metrics.py --predictions my_system_cwq.jsonl --dataset cwq
```

```
| Dataset | Metric | Score | EM | n | Answered | Searches/q |
| --- | --- | --- | --- | --- | --- | --- |
| CWQ | Hits@1 | ... | ... | 3531 | ...% | ... |
...
| Creak | Accuracy | ... | ... | 1371 | ...% | ... |
| **Average** (8 benchmarks) | | ... | | | | |
```

- Predictions of any other system can be scored the same way. Write one `{"idx": "cwq-17", "pred": "..."}`
  or `{"question": "...", "pred": "..."}` per line; test samples without a prediction count as misses.
- Ids follow `<dataset>-<position in the test file>`.
- A run on a subset is marked `n / test size (subset)`.

## Reproducing the paper's numbers

- Scores depend on the KG behind the tool:
  - Use a real Freebase endpoint (`kg.backend: freebase_sparql`, see the README).
  - The Claude-simulated graph (`claude_sim`) is for testing the pipeline only. Its scores are not comparable.
- The paper evaluates QALD-10, T-REx, Zero-Shot RE and Creak over **Wikidata**. The KG backends in this
  repository are Freebase-based, so on those four datasets the tool resolves topic entities by name in
  Freebase. Expect differences from the paper until a Wikidata backend is used.
