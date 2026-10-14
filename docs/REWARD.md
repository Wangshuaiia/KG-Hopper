# Reward: construction, rules and the reward server

## No checkpoints are needed

KG-Hopper does not train a reward model. The two "reward models" in the paper (Sec. III-C, IV-B) are
off-the-shelf instruction-tuned LLMs used as **judges** through prompts. Every other reward component is a
rule.

| Judge | Paper | Model (public, no fine-tuning) | Used for |
| --- | --- | --- | --- |
| f_a | answer judge | `meta-llama/Llama-3.2-3B-Instruct` | R_answer |
| f_r | reasoning judge | `meta-llama/Llama-3.3-70B-Instruct` | R_reason |

Download them from Hugging Face (accept the Llama license) and serve them with vLLM, or substitute other
instruct models. The rest of this page describes exactly what is computed, so the reward can be rebuilt
with any judges.

The code is in `kghopper/reward.py` (combination), `kghopper/judges.py` (judges and answer matching) and
`kghopper/protocol.py` (format rules). The local GRPO trainer, the evaluator and the OpenRLHF reward server all
call the same `RewardFn`.

## The four components

The reward is computed on one trajectory: the response after `Assistant: <think>`, i.e. the model's text
interleaved with the KG tool's `<searched_triples>` blocks.

### R_search = min(0.5 · n, 0.8)

`n` is the number of KG tool calls: each `<search> ... </search>` answered by a `<searched_triples>` block.
Configured by `search_per_call` (0.5) and `search_cap` (0.8).

### R_format = 0.5 if all of these hold, else 0 (`protocol.check_format`)

1. The model did not write `<searched_triples>` / `</searched_triples>` itself; only the tool does.
2. No extra `<think>` (the prompt already opens it).
3. Exactly one `</think>`.
4. Exactly one `<answer>` and one `</answer>`.
5. `<search>` and `</search>` are balanced.
6. `<answer>` comes after `</think>`.
7. No `<search>` after `</think>`.
8. Nothing but whitespace after `</answer>`.
9. The answer is not empty.

In `mode: paper`, R_format additionally requires at least one search ("`<think>`, `<search>` and `<answer>`
used correctly"). Configured by `format_bonus` (0.5).

### R_answer ∈ {0, 1} (`judges.AnswerJudge`)

1. **Answer extraction**: the text inside the last `<answer> ... </answer>`. No answer gives 0.
2. **Rule match** (`rule_correct`). Both sides are normalised: lower-cased, `_` replaced by a space,
   punctuation removed, `a` / `an` / `the` removed, whitespace collapsed. The answer is correct if:
   - it equals a gold alias, or
   - one item of a listed answer (split on `; , / |`, newlines or `and`) equals a gold alias, or
   - the answer has at most `max_answer_words` (15) words and contains a gold alias as whole words. The word
     limit stops "list every candidate" answers from being paid.

   For fact checking, the first word is mapped (`true` / `yes` / `correct` / `supported` → true,
   `false` / `no` / `incorrect` / `refuted` → false) and compared with the label; no LLM is used.
3. **LLM judge f_a**, when `answer_judge` is `openai` or `hf`. With `answer_judge_rule_first: true`
   (default), a rule match scores 1 directly; otherwise the judge decides, so it handles paraphrases the
   rules miss. Prompt:

   ```
   You are judging whether a predicted answer to a knowledge question is correct.

   Question: {question}
   Gold answer(s): {golds joined by "; "}
   Predicted answer: {pred}

   Is the predicted answer correct, i.e. does it refer to the same entity or value as a gold answer?
   Reply with a single digit: 1 if correct, 0 if not.
   ```

   Decoding is greedy (temperature 0) with at most 8 new tokens. **Only the newly generated text is parsed**:
   the first `0` or `1` in it. If neither appears, the rule match decides.

### R_reason ∈ [0, 1] (`judges.ReasoningJudge`)

1. The reasoning trace is everything before `</think>`: the model's text and the retrieved triples. Longer
   traces are shortened from the middle to `max_chars` (6,000), keeping the start and the conclusion.
2. The trace is scored by the LLM judge f_r with this prompt:

   ```
   You are evaluating an agent that answers questions by searching a knowledge graph step by step.

   Question: {question}

   Agent's reasoning process (its searches, the retrieved triples, and its inferences):
   {reasoning}

   Rate how sound and coherent the reasoning is: does it decompose the question sensibly, search the right
   entities, use the retrieved triples correctly, and reach its conclusion logically?
   Give a short evaluation, then end with a final line of the form "Score: X" where X is between 0 and 1.
   ```

3. Decoding is greedy with at most 256 new tokens. The score is the last number after the word "score" in
   the judge's reply (`Score: 0.8`, `**Final Score**: 0.5` and `score ... is **1**` all parse), clipped to
   [0, 1]; no score gives 0.
4. With `reason_judge: none`, R_reason is 0.

## Combining them

| `reward.mode` | Formula |
| --- | --- |
| `paper` | R = R_search + R_format + R_reason + R_answer (Sec. III-C as written) |
| `gated` (default) | R = R_format + w_answer · R_answer + R_answer · (R_search + R_reason), with `w_answer` = 1 |

Scores produced by the code for one CWQ question ("Lou Seal is the mascot for the team that last won the
World Series when?"), with the rule-based answer judge. R_reason, from 0 to 1, is added on top as shown in
the last row:

| Trajectory | `paper` | `gated` |
| --- | --- | --- |
| correct, 2 searches | 2.3 | 2.3 |
| correct, 1 search | 2.0 | 2.0 |
| correct, no search | 1.0 | 1.5 |
| wrong, 2 searches | **1.3** | 0.5 |
| wrong, no search | 0.0 | 0.5 |
| no answer, 2 searches | **0.8** | 0.0 |
| R_reason added | always | only when the answer is correct |

Under the additive formula, a wrong answer with two searches (1.3) outscores a correct answer found without
searching (1.0), and an unanswered trajectory still earns 0.8, so the policy can be paid for searching
rather than answering. `gated` keeps the same components but pays search and reasoning only for correct
answers. As a result:

- every correct answer (≥ 1.5) beats every wrong one (≤ 0.5);
- among correct answers, grounded multi-hop search scores highest;
- among wrong ones, a well-formed attempt beats a malformed one.

Use `--set reward.mode=paper` to train with the formula exactly as published.

## Running the reward server (OpenRLHF)

OpenRLHF sends decoded rollouts to an HTTP reward endpoint (`--remote_rm_url`). `reward-remote/reward_server.py`
wraps `RewardFn`.

**1. Serve the judges** (OpenAI-compatible endpoints):

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct  --port 8001                            # f_a, one GPU
vllm serve meta-llama/Llama-3.3-70B-Instruct --port 8002 --tensor-parallel-size 4   # f_r, ~140 GB of bf16 weights
```

**2. Start the reward server.** The judge URLs and models are in the `reward` section of
`configs/paper_gpu.yaml`:

```bash
python reward-remote/reward_server.py --config configs/paper_gpu.yaml \
    --data_path data/train/rl.jsonl \
    --host 0.0.0.0 --port 1278 --log_file logs/rewards.jsonl
```

- `--data_path` must be the same file as the training `--prompt_data`. Gold answers are looked up by the
  exact prompt each sample produces (see `docs/DATA.md`).
- Change judges or URLs with `--set`, e.g.
  `--set reward.answer_judge_url=http://judge-host:8001 reward.reason_judge_model=Qwen/Qwen2.5-72B-Instruct`.
- If the judge endpoints need a key, set `JUDGE_API_KEY`.

**3. Point training at it**: `--remote_rm_url http://<host>:1278/get_reward`. In
`scripts/reinforce_train.sh`, set `RM_PORT`. Start the KG server, the reward server and Ray before running it;
the order is in the script's header.

API: `POST /get_reward` with `{"query": ["<decoded prompt + response>", ...]}` returns
`{"rewards": [float, ...]}`.
- Special tokens such as `<|im_end|>` are stripped.
- A prompt that matches no training sample scores 0 and is logged as a warning.
- With `--log_file`, each scored rollout is appended with its components: `reward`, `r_answer`, `r_format`,
  `r_search`, `r_reason`, `format`, `pred`, `n_search`.

Check it with one rollout:

```python
import requests
from kghopper import protocol
from kghopper.data import load_samples
from kghopper.rollout import make_prompt

s = load_samples("data/train/rl.jsonl")[0]
response = (f" <search> {s['topic_entities'][0]} </search>" + protocol.format_triples([(s["topic_entities"][0], "r", s["golds"][0])])
            + f"Found it. </think>\n\n<answer> {s['golds'][0]} </answer>")
print(requests.post("http://127.0.0.1:1278/get_reward", json={"query": [make_prompt(s) + response]}).json())
```

## Cheaper set-ups

| Set-up | Settings | Notes |
| --- | --- | --- |
| Rules only, no judge GPUs | `reward.answer_judge=rule reward.reason_judge=none` | R_answer by rule matching, R_reason = 0 |
| Smaller judges | any instruct model behind `vllm serve`, set via `reward.*_judge_model` / `*_url` | e.g. Qwen2.5-7B-Instruct as f_r |
| Judges inside the reward process | `reward.answer_judge=hf reward.answer_judge_model=<HF id>` (same for `reason_judge`) | loads the model with transformers; fine for small judges |
| Single machine, no OpenRLHF | `python -m kghopper.train_grpo` | uses the same `RewardFn` in-process; no server needed |

## What changed from the original reward server

The earlier `reward_server.py` had four problems:

- It loaded the judges with transformers and parsed the judge's **decoded prompt**. That prompt contains
  `return '1'`, so every answer scored 1.
- Its judges sampled rather than decoding greedily, so scores were noisy.
- It added an F1 term and a -2 format penalty.
- It never paid R_search.

`docs/code_audit_before_fix.md` records the measurements. The current server implements the components above.
