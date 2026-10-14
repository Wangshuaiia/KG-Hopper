# KG-Hopper: Empowering Compact Open LLMs with Knowledge Graph Reasoning via Reinforcement Learning

KG-Hopper is a novel Reinforcement Learning (RL) framework that empowers compact open LLMs with the ability to perform integrated multi-hop reasoning within a single inference round. Rather than reasoning step-by-step, we train a Reasoning LLM that embeds the entire KG traversal and decision process into a unified “thinking” stage, enabling global reasoning over cross-step dependencies and dynamic path exploration with backtracking. KG-Hopper, based on a 7B-parameter LLM, consistently outperforms larger multi-step systems (up to 70B) and achieves competitive performance with proprietary models such as GPT-3.5-Turbo and GPT-4o-mini.

Paper: [arXiv:2603.21440](https://arxiv.org/abs/2603.21440)

## How it works

The policy reasons inside `<think>`, queries the knowledge graph with `<search> entity </search>`, receives
`<searched_triples> (1) (head, relation, tail) ... </searched_triples>`, and ends with `<answer> ... </answer>`.
Training, evaluation and the reward server all use the same code for this, in the `kghopper` package:

| Module | Paper | What it does |
| --- | --- | --- |
| `kghopper/protocol.py` | III-B | prompts, tags, triple formatting, format check |
| `kghopper/kg/` | III-A | two-stage KG tool: the entity's outgoing predicates → the ones most relevant to the question → their objects. Backends: `freebase_sparql` (a real Freebase SPARQL endpoint), `claude_sim` (Freebase simulated by Claude), `static` (cache file, offline), `http` (`python -m kghopper.kg.server`) |
| `kghopper/rollout.py` | III-B/D | multi-turn generation with KG calls; keeps token segments so KG output is masked out of the loss |
| `kghopper/reward.py`, `judges.py` | III-C | R_search + R_format + R_reason + R_answer (`mode: paper`) or the gated variant (default) |
| `kghopper/prepare_data.py` | III-B, IV-B | training data from the official splits: RL prompts and teacher-generated cold-start trajectories |
| `kghopper/train_sft.py` | III-B | cold-start SFT with KG output masked |
| `kghopper/train_grpo.py` | III-D | single-process GRPO (CPU / Apple MPS / one GPU) |
| `OpenRLHF-RAG/` | III-D | multi-GPU RL (Ray + vLLM), adapted from [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF); its rollout calls `kghopper.rollout` |
| `evaluation/evaluate.py` | IV | evaluation with the same rollout, KG tool and answer matching |

```
.
├── configs/            local_cpu.yaml (laptop smoke test), local_freebase.yaml (real Freebase), paper_gpu.yaml (paper setup)
├── data/               see data/README.md and docs/DATA.md
│   ├── test_set/       8 benchmark test sets
│   ├── synthetic/      small generated SFT / RL / eval sets (python -m kghopper.build_data)
│   └── kg_cache/       cached simulated-Freebase neighbourhoods, shared by all stages
├── docs/               DATA.md (training data), REWARD.md (reward), EVALUATION.md (test sets, metrics), code audit
├── evaluation/         evaluate.py (run a model), metrics.py (Hits@1 / accuracy tables)
├── kghopper/
├── OpenRLHF-RAG/
├── reward-remote/reward_server.py
├── scripts/            run_local_pipeline.sh, ray_start.sh, reinforce_train.sh
└── tests/
```

## Quick start on a laptop (no GPU)

Runs the whole pipeline with Qwen2.5-0.5B-Instruct on CPU or Apple MPS, with Freebase simulated by Claude
(needs `ANTHROPIC_API_KEY`; the simulated graph is cached in `data/kg_cache/`). This checks that everything
runs end to end; it is not meant to reproduce the paper's numbers.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-local.txt
export ANTHROPIC_API_KEY=...

bash scripts/run_local_pipeline.sh            # tests -> data -> SFT -> GRPO -> evaluation
# or stage by stage:
.venv/bin/python -m kghopper.build_data --config configs/local_cpu.yaml
.venv/bin/python -m kghopper.train_sft  --config configs/local_cpu.yaml
.venv/bin/python -m kghopper.train_grpo --config configs/local_cpu.yaml
.venv/bin/python evaluation/evaluate.py --config configs/local_cpu.yaml --datasets cwq creak --n 20
```

Any config value can be overridden, e.g. `--set grpo.steps=20 grpo.lr=1e-6 reward.mode=paper kg.width=5`.
Outputs: `checkpoints/{sft,grpo}/` (with `train_log.jsonl`), `outputs/eval/<dataset>.jsonl` and `metrics.json`.

## Connecting a real Freebase

Any SPARQL 1.1 endpoint serving the Freebase RDF dump (namespace `http://rdf.freebase.com/ns/`) works,
e.g. a Virtuoso server set up as for GrailQA / ToG. Point the config at it and run any stage as usual:

```bash
# configs/local_freebase.yaml = local_cpu.yaml with kg.backend: freebase_sparql; only the URL needs changing
.venv/bin/python evaluation/evaluate.py --config configs/local_freebase.yaml \
    --set kg.sparql_url=http://<host>:8890/sparql --model checkpoints/grpo --datasets cwq webqsp --n 100

# or serve it to several processes (OpenRLHF): set kg.server_backend: freebase_sparql and kg.sparql_url in the config
python -m kghopper.kg.server --config configs/local_freebase.yaml --set kg.sparql_url=http://<host>:8890/sparql --port 5005
```

| `kg.` option | Default | Meaning |
| --- | --- | --- |
| `sparql_url` | `http://localhost:8890/sparql` | the endpoint |
| `sparql_timeout`, `sparql_method` | `60`, `post` | per-query timeout (s); `post` or `get` |
| `language` | `en` | language of names and literal values |
| `expand_cvt` | `true` | follow unnamed compound-value (CVT) nodes, shown as `r1 -> r2` (e.g. `people.person.spouse_s -> people.marriage.spouse`) |
| `entity_lookup` | `degree` | when several entities share a name, take the one with most edges (`degree`) or any (`first`) |
| `exclude_relation_prefixes` | `type. common. freebase. kg. dataworld. user.` | predicates never shown to the policy |
| `max_relations` | `1000` | stage-1 limit per entity |
| `relation_selector` | `lexical` | stage 2: `lexical` (word overlap, no model), `claude` (needs `ANTHROPIC_API_KEY`), `openai` (`selector_url`, `selector_model`: any OpenAI-compatible server, e.g. `vllm serve <model>`) |
| `width`, `max_objects_per_relation`, `max_triples` | `3`, `10`, `30` | predicates kept per search and size of the returned block |

The policy searches by name. Names are mapped to MIDs from, in order: the sample's `entity_ids` (the test sets'
topic-entity MIDs are loaded automatically; add `"entity_ids": {"name": "m.xxx"}` to your own jsonl), entities
returned by earlier searches, and finally a name/alias lookup in the graph. Stage 1 only lists predicates and
stage 2 fetches objects of the selected ones, so hub entities are never loaded in full. The four Wikidata-based
test sets (QALD-10, T-REx, Zero-Shot RE, Creak) have no Freebase MIDs and rely on the name lookup.

## KG retrieval server

`python -m kghopper.kg.server` (`kghopper/kg/server.py`) is the retrieval service that RL training calls. It
replaces the Wikipedia retriever at `http://0.0.0.0:5003/queries` that the earlier code expected, and serves the
KG backend chosen in the config: `freebase_sparql` (a real Freebase endpoint), `claude_sim` or `static`.

```bash
python -m kghopper.kg.server --config configs/local_freebase.yaml \
    --set kg.sparql_url=http://<freebase-host>:8890/sparql --host 0.0.0.0 --port 5005
curl http://127.0.0.1:5005/health                      # {"ok": true, "calls": 0}
```

`POST /search` takes a batch of KG tool calls and returns the triples for each, in order:

```json
{"queries": [{"entity": "Lou Seal",
              "question": "Lou Seal is the mascot for the team that last won the World Series when?",
              "mids": {"Lou Seal": "m.03_dwn"}}]}
```
```json
{"results": [[["Lou Seal", "sports.mascot.team", "San Francisco Giants"]]]}
```

- `entity` is the text between `<search>` and `</search>`.
- `question` drives relation selection (stage 2).
- `mids` (optional) carries known Freebase ids, the sample's `entity_ids`.

The rollout formats the triples as the `<searched_triples>` block.

Who calls it:
- **OpenRLHF**: `--kg_url` (default `http://127.0.0.1:5005`; `KG_URL` in `scripts/reinforce_train.sh`). In a
  multi-node Ray cluster, start the server with `--host 0.0.0.0` and point `KG_URL` at that machine.
- **Local trainers and the evaluator**: set `kg.backend: http` and `kg.url` to use the same server. With any other
  backend they run the identical tool in-process, so no server is needed.

## Training data

Only the test sets are in the repository. `python -m kghopper.prepare_data` builds the training data from the
benchmarks' official training splits as described in the paper: RL prompts sampled from the training sets,
and cold-start SFT trajectories produced by a teacher LLM through the same KG tool and kept only when the
tool calls and the answer are correct. Sources, steps, the JSONL schema and the defaults chosen where the
paper leaves details open are in [`docs/DATA.md`](docs/DATA.md).

## Reward

No reward-model checkpoints are needed: as in the paper, the answer judge (Llama-3.2-3B-Instruct) and the
reasoning judge (Llama-3.3-70B-Instruct) are public instruct models used through prompts, and the search and
format rewards are rules. [`docs/REWARD.md`](docs/REWARD.md) gives every rule, the judge prompts, how the
components are combined (`reward.mode: gated` by default, or `paper`), and the reward-server commands.

## Evaluation

The eight test sets are public and included in `data/test_set/`. `evaluation/evaluate.py` runs any of them
(`--datasets cwq webqsp grailqa qald webquestions trex zeroshotre creak`, or `eval.datasets` in the config)
with the training rollout and KG tool. It reports **Hits@1** for KBQA and slot filling and **accuracy** for
Creak, as in the paper, plus their average. `evaluation/metrics.py` recomputes and aggregates the metrics from
saved outputs, or scores another system's predictions. Details: [`docs/EVALUATION.md`](docs/EVALUATION.md).

## Multi-GPU training (OpenRLHF)

```bash
pip install -r requirements.txt && pip install -e . && pip install -e OpenRLHF-RAG

# judges (docs/REWARD.md) and services, each in its own shell
vllm serve meta-llama/Llama-3.2-3B-Instruct  --port 8001
vllm serve meta-llama/Llama-3.3-70B-Instruct --port 8002 --tensor-parallel-size 4
python -m kghopper.kg.server --config configs/paper_gpu.yaml --host 0.0.0.0 --port 5005           # KG tool
python reward-remote/reward_server.py --config configs/paper_gpu.yaml \
       --data_path data/train/rl.jsonl --host 0.0.0.0 --port 1278 --log_file logs/rewards.jsonl  # reward
bash scripts/ray_start.sh
DATA_PATH=data/train/rl.jsonl PRETRAIN=checkpoints/sft bash scripts/reinforce_train.sh

python evaluation/evaluate.py --config configs/paper_gpu.yaml --backend vllm --model <checkpoint>
```

Training data is built as described in [`docs/DATA.md`](docs/DATA.md) (`data/synthetic/rl.jsonl` works for a
smoke test). Use the same file for `--data_path` and `DATA_PATH`. The RL settings at the top of
`scripts/reinforce_train.sh` are the paper's:
- GRPO with 16 rollouts per query and batch size 16;
- 2 epochs, learning rate 1e-6, KL 1e-5, clip 0.2, temperature 1;
- history resampling from epoch 2.

`configs/paper_gpu.yaml` holds the cold-start SFT settings and a single-process GRPO alternative with the same
values.

## Changes in this refactor

- **One system.** The OpenRLHF rollout used R1-Searcher's Wikipedia retriever (`<|begin_of_query|>`, port 5003) while
  evaluation used `<search>`; both now go through `kghopper.rollout` and the same KG tool.
- **KG tool.** The old `kg-tool/` (missing `config.py`, `gnn_base_train.py`, `freebase_func.py`, unrelated GNN and
  sub-question code) is replaced by the two-stage tool in `kghopper/kg/`, with a real-Freebase SPARQL backend and a
  Claude-simulated one for running without a Freebase server.
- **Evaluation.** `evaluation/eval_search_kg.py` dropped the model's own reasoning before each KG result, skipped the
  first 14 samples, silently dropped unanswered samples, and had placeholder prompts for slot filling and fact
  checking, an empty dataset list and no metric script. `evaluation/evaluate.py` replaces it and runs all
  eight test sets; `evaluation/metrics.py` aggregates Hits@1 / accuracy as reported in the paper.
- **Answer judge.** The judge decoded its own prompt, which contains `return '1'`, so every answer scored 1. Judges now
  parse only newly generated tokens, decode greedily, and fall back to rule matching.
- **Reward direction.** With the old reward a wrong answer could outscore a correct one and not searching paid off.
  `reward.mode: gated` (default) pays the search and reasoning bonuses only for correct answers, so every correct
  answer outscores every wrong one; `mode: paper` keeps the paper's additive formula.
- **Retrieval mask.** Hardcoded token ids for `<|begin_of_documents|>` never matched `<searched_triples>`; the mask now
  comes from the rollout's token segments, and training uses the exact sampled tokens.
- **Secrets.** A hardcoded API key and internal proxy URL (`evaluation/extract_entity_from_query.py`) and a wandb token
  in two OpenRLHF scripts were removed. They remain in git history and must be revoked.
- **Leftovers from R1-Searcher** (the codebase the OpenRLHF fork came from) were removed: its Wikipedia-retrieval reward
  servers (`openrlhf/cli/server_rm_rag.py`, `server_dpsk_tuple*.py`), launch scripts with personal paths
  (`*_jjh.sh`, `ppo_wo_ray_single_remote.sh`) and `experience_maker.bak.py`. Debugging `kill` statements that broke
  unused code paths now raise errors that say which path to use instead.

The detailed audit of the previous version is in `docs/code_audit_before_fix.md`.

## Tests

```bash
.venv/bin/python -m pytest tests -q   # includes the SPARQL backend against an in-process rdflib endpoint
```
