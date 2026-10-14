# Training data: construction and schema

The repository ships the eight **test** sets (`data/test_set/`). Training data is built from the
benchmarks' official **training** splits with `kghopper/prepare_data.py`, following the paper:

| Paper | What | Size in the paper | Command |
| --- | --- | --- | --- |
| IV-B | RL prompts sampled from the training splits | 2,000 | `prepare_data rl` |
| III-B, IV-B | cold-start SFT trajectories: few-shot prompting a strong LLM, keeping samples with correct tool calls and clear reasoning; retrieved triples masked in the loss | 500 | `prepare_data sft` |
| III-D | history resampling: one-hop questions dropped after an initial phase | — | `grpo.history_resampling_step` (uses the `hops` field) |

```
official training splits ──convert──▶ data/train/<dataset>.jsonl ──rl──┬──▶ data/train/rl.jsonl    (RL prompts)  ──▶ GRPO
                                                                       └──▶ data/train/pool.jsonl  ──sft (teacher LLM + KG tool)──▶ data/train/sft.jsonl ──▶ cold-start SFT
```

Every file is JSON Lines in the one schema described in [Schema](#schema), the same format the evaluator
reads, so training and evaluation see identical prompts.

## 1. Sources

| Dataset | KG | Task | Train size | Official release | Topic entities | `hops` |
| --- | --- | --- | --- | --- | --- | --- |
| ComplexWebQuestions (CWQ) | Freebase | QA | 27,734 | [tau-nlp.sites.tau.ac.il/compwebq](https://www.tau-nlp.sites.tau.ac.il/compwebq) | MIDs in the SPARQL; names looked up with `--sparql-url` | 2 |
| WebQSP | Freebase | QA | 3,098 | [Microsoft download 52763](https://www.microsoft.com/en-us/download/details.aspx?id=52763) | `TopicEntityName` / `TopicEntityMid` | length of `InferentialChain` |
| GrailQA | Freebase | QA | 44,337 | [dki-lab.github.io/GrailQA](https://dki-lab.github.io/GrailQA/) | `graph_query` nodes of type `entity` | `num_edge` |
| WebQuestions | Freebase | QA | 3,778 | released with [SEMPRE](https://github.com/percyliang/sempre) | from the Freebase URL (`/en/justin_bieber` → `Justin Bieber`), resolved by name | unknown |
| T-REx | Wikidata | slot filling | 2,284,168 | [KILT](https://github.com/facebookresearch/KILT) (`trex-train-kilt.jsonl`) | subject before `[SEP]` | 1 |
| Zero-Shot RE | Wikidata | slot filling | 147,909 | [KILT](https://github.com/facebookresearch/KILT) (`structured_zeroshot-train-kilt.jsonl`) | subject before `[SEP]` | 1 |
| Creak | Wikidata | fact checking | 10,176 | [github.com/yasumasaonoe/creak](https://github.com/yasumasaonoe/creak) | `entity` | unknown |
| QALD-10 | Wikidata | QA | — (test only) | [github.com/KGQA/QALD-10](https://github.com/KGQA/QALD-10) | — | — |

Sizes are those in `data/test_set/README.md`. The comment in each converter (`conv_*` in
`kghopper/prepare_data.py`) names the fields it reads; check them against your download. Files may be a JSON
array, `{"Questions": [...]}`, or JSON Lines. Every converter also accepts the layout of `data/test_set/` (ToG-style
records with `topic_entity` / `qid_topic_entity`).

## 2. Building the data

```bash
pip install -r requirements-local.txt
mkdir -p data/train

# 1) convert each official training split (repeat per dataset)
python -m kghopper.prepare_data convert --dataset cwq  --input ComplexWebQuestions_train.json --output data/train/cwq.jsonl \
       --sparql-url http://localhost:8890/sparql          # optional: names CWQ's topic MIDs
python -m kghopper.prepare_data convert --dataset webqsp --input WebQSP.train.json            --output data/train/webqsp.jsonl
python -m kghopper.prepare_data convert --dataset grailqa --input grailqa_v1.0_train.json     --output data/train/grailqa.jsonl
python -m kghopper.prepare_data convert --dataset webquestions --input webquestions.examples.train.json --output data/train/webquestions.jsonl
python -m kghopper.prepare_data convert --dataset trex --input trex-train-kilt.jsonl          --output data/train/trex.jsonl --limit 50000
python -m kghopper.prepare_data convert --dataset zeroshotre --input structured_zeroshot-train-kilt.jsonl --output data/train/zeroshotre.jsonl
python -m kghopper.prepare_data convert --dataset creak --input train.json                    --output data/train/creak.jsonl

# 2) RL prompts (paper: 2,000) + the remaining pool for cold start
python -m kghopper.prepare_data rl --inputs data/train/{cwq,webqsp,grailqa,webquestions,trex,zeroshotre,creak}.jsonl --n 2000

# 3) cold-start trajectories (paper: 500) from a teacher served with vLLM, using the KG tool of the config
vllm serve meta-llama/Llama-3.3-70B-Instruct --port 8000 --tensor-parallel-size 4
python -m kghopper.prepare_data sft --config configs/local_freebase.yaml --pool data/train/pool.jsonl --n 500 \
       --teacher openai:meta-llama/Llama-3.3-70B-Instruct@http://127.0.0.1:8000

python -m kghopper.prepare_data stats data/train/rl.jsonl data/train/sft.jsonl

# train on it
python -m kghopper.train_sft  --config configs/local_cpu.yaml --set sft.data=data/train/sft.jsonl
python -m kghopper.train_grpo --config configs/local_cpu.yaml --set grpo.data=data/train/rl.jsonl grpo.history_resampling_step=100
```

### Step 1, `convert`

Maps each release to the schema: question, all gold answers with aliases, task, topic entities and,
for Freebase datasets, their MIDs (`entity_ids`), plus `hops` where the release says how many relations
the question needs. CWQ only carries topic MIDs inside its SPARQL; with `--sparql-url` their names are
read from Freebase, otherwise the MID stands in for the name.

### Step 2, `rl`

1. Pools all converted files.
2. Drops questions with no gold answer, with more than `--max-golds` (50) gold answers, questions that also
   occur in any `data/test_set/` file (after normalisation), and duplicates.
3. Samples `--n` prompts, split equally across sources by default (`--weights cwq=2 trex=0.5 ...` to
   change); a source with too few questions passes its share on to the others.
4. Writes the sample to `rl.jsonl` (ids `rl-0`, `rl-1`, ...) and everything else to `pool.jsonl`, so SFT and
   RL questions never overlap.

### Step 3, `sft`

1. A teacher LLM answers pool questions through **the same rollout and KG tool as RL training**
   (`kghopper.rollout`, the `kg` section of `--config`). The teacher sees two worked demonstrations
   (`prepare_data.DEMOS`: a two-hop question, and one that recovers from an empty search) between the task
   instructions and the question. The stored sample keeps the plain prompt.
2. Each question gets up to `--attempts` (4) samples at `--temperature` (0.7).
3. The first trajectory that passes every check is kept. The paper's "correct tool calls and clear
   reasoning" is implemented as:
   - `answered`: ended with `</answer>`;
   - `format`: `<think>` / `<search>` / `<answer>` correctly placed (`protocol.check_format`);
   - `correct`: the answer matches a gold alias (`judges.rule_correct`);
   - `searches`: at least `--min-searches` searches returned triples (an empty search followed by a
     corrected one is allowed: that is backtracking);
   - `grounded`: a gold answer appears in the retrieved triples, so the answer came from the KG
     (`--no-grounding` to skip);
   - optional `--reason-judge` with `--min-reason-score`: an LLM judge for clear reasoning.
4. The teacher can be `openai:<model>@<url>` (any OpenAI-compatible `/v1/completions` server, e.g. vLLM) or
   `hf:<model>` (local transformers).

The teacher must be a model whose terms allow training other models on its outputs (e.g. Llama-3.3-70B or
Qwen2.5-72B under their licenses).

## Schema

### Sample (RL prompts, pool, evaluation)

```json
{"idx": "rl-0",
 "question": "Lou Seal is the mascot for the team that last won the World Series when?",
 "golds": ["2014 World Series", "2014 MLB World Series"],
 "task": "qa",
 "topic_entities": ["Lou Seal"],
 "entity_ids": {"Lou Seal": "m.03_dwn"},
 "source": "cwq",
 "hops": 2}
```

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `idx` | string | yes | unique id; the reward server and logs use it |
| `question` | string | yes | slot filling: `"<subject> [SEP] <relation>"`; fact checking: the claim |
| `golds` | list of strings | yes | every accepted answer and alias; fact checking: `["true"]` or `["false"]`. `answer` (string or list) is accepted instead |
| `task` | `qa` \| `slot_filling` \| `fact_checking` | no (default `qa`) | selects the instruction text |
| `topic_entities` | list of strings | no | shown in the prompt as `Topic entity: A; B` |
| `entity_ids` | object name → Freebase MID | no | hints for the SPARQL KG backend (`m.`/`g.` ids only) |
| `source` | string | no | dataset name; used for per-source sampling and statistics |
| `hops` | integer or null | no | relations needed; `1` marks one-hop questions for history resampling |

### Cold-start SFT record

A sample plus the trajectory:

```json
{"idx": "cwq-train-41", "question": "...", "golds": ["Northern Ireland"], "task": "qa",
 "topic_entities": ["United Kingdom", "Down District Council"],
 "entity_ids": {"United Kingdom": "m.07ssc", "Down District Council": "m.02dv8y"}, "source": "cwq", "hops": 2,
 "segments": [
   {"text": " First, I need the four regions that make up the United Kingdom. <search> United Kingdom </search>", "from_model": true},
   {"text": "\n\n<searched_triples>\n(1) (United Kingdom, location.country.administrative_divisions, England)\n...\n</searched_triples>\n\n", "from_model": false},
   {"text": "United Kingdom is divided into England, Scotland, Wales, and Northern Ireland. ... <search> Down District Council </search>", "from_model": true},
   {"text": "\n\n<searched_triples>\n(1) (Down District Council, location.location.containedby, Northern Ireland)\n...\n</searched_triples>\n\n", "from_model": false},
   {"text": "Down District Council is located in Northern Ireland. ... </think>\n\n<answer> Northern Ireland </answer>", "from_model": true}],
 "checks": {"answered": true, "format": true, "correct": true, "searches": true, "grounded": true},
 "teacher": "openai:meta-llama/Llama-3.3-70B-Instruct@http://127.0.0.1:8000"}
```

Rules for `segments`:

- The segments, concatenated, are the response that follows the prompt's final `Assistant: <think>`.
- Model segments (`from_model: true`) end with `</search>`, or with `</answer>` for the last one. There is no
  end-of-sequence token, because generation stops at `</answer>`.
- Tool segments (`from_model: false`) are exactly what the KG tool returns:
  `\n\n<searched_triples>\n(1) (head, relation, tail)\n(2) ...\n</searched_triples>\n\n`, or `(no triples found)`.
  The leading `\n\n` is dropped when the previous text already ends with a blank line. CVT paths appear as
  `r1 -> r2`.
- Exactly one `</think>`, one `<answer> ... </answer>`, and no `<searched_triples>` written by the model.

### From a sample to a training sequence

The prompt is plain text from `protocol.build_prompt`; no chat template is used, in training, evaluation or
the reward server:

```
<task instructions for qa | slot_filling | fact_checking>

User: <question>
Topic entity: <topic_entities joined by "; ">      (omitted when empty)
Assistant: <think>
```

- **SFT** (`train_sft.py`): the prompt and each segment are tokenised separately and concatenated. The loss
  covers model segments only; prompt and tool tokens are masked (paper III-B).
- **RL** (`train_grpo.py`, OpenRLHF): the rollout builds the same sequence from the tokens the policy actually
  sampled plus the tokenised tool output. The loss mask is 1 on sampled tokens and 0 on tool tokens
  (paper III-D).
- **Reward** (`reward.py`): `golds` and `task` from the sample; the OpenRLHF reward server finds the sample
  by its prompt text.

### Other data files

- `data/kg_cache/*.json`: `{"neighbourhood": {entity: {relation: [objects]}}, "selection": {"entity||width||question": [relations]}}`,
  the cache of the simulated KG and of relation selections.
- `data/synthetic/{sft,rl,eval}.jsonl`: a small set built from KG walks by `python -m kghopper.build_data`,
  for smoke-testing the pipeline without any benchmark files. It uses template questions and is not a
  substitute for the data above.

## Choices the paper does not fix

These are defaults of `prepare_data.py`; change them to match the setup you want to reproduce.

- **Source mix of the 2,000 RL prompts**: equal per source (`--weights` to change). QALD-10 has no training split.
- **Filters**: test-overlap removal, deduplication, at most 50 gold answers per question.
- **Teacher**: model, the two demonstrations, temperature 0.7, 4 attempts per question.
- **Cold-start acceptance**: answer correctness, at least one productive search, grounding; an LLM
  reasoning judge only if `--reason-judge` is set.
- **Hop labels**:
  - CWQ = 2, WebQSP = inferential-chain length, GrailQA = `num_edge`, T-REx / Zero-Shot RE = 1;
  - unknown (`null`) for WebQuestions and Creak, so history resampling keeps those.
