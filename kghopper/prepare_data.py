"""Build KG-Hopper training data from the benchmarks' training splits (paper Sec. III-B, III-C, IV-B).

    convert  one dataset's official training split -> kghopper jsonl (one schema for every dataset)
    rl       pool the converted sets, drop duplicates and test-set overlap, sample the RL prompts
             (paper: 2,000 from the training data), keep the rest as the cold-start pool
    sft      cold-start trajectories (paper: 500): a teacher LLM, prompted with few-shot demonstrations,
             answers pool questions through the same rollout and KG tool used in training; only
             trajectories with well-formed, productive tool calls and a correct answer are kept
    stats    counts per source / hops / task for any kghopper jsonl

Schema, sources and the full procedure: docs/DATA.md.

    python -m kghopper.prepare_data convert --dataset cwq --input ComplexWebQuestions_train.json --output data/train/cwq.jsonl
    python -m kghopper.prepare_data rl --inputs data/train/*.jsonl --n 2000
    python -m kghopper.prepare_data sft --pool data/train/pool.jsonl --n 500 \\
        --teacher openai:meta-llama/Llama-3.3-70B-Instruct@http://127.0.0.1:8000
"""
import argparse
import json
import logging
import os
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from . import protocol
from .config import add_config_args, load_config, resolve_path
from .data import TEST_SETS, _entity_ids, _golds, _topics, load_samples, load_test_set, write_jsonl
from .generation import GenOutput
from .judges import ReasoningJudge, normalize_answer, rule_correct

log = logging.getLogger("prepare_data")

# --------------------------------------------------------------------------- convert

_MID = re.compile(r"ns:([mg]\.[0-9a-z_]+)")


def _records(raw):
    if isinstance(raw, dict):
        for key in ("Questions", "questions", "data"):
            if key in raw:
                return raw[key]
        raise ValueError("unrecognised file layout")
    return raw


def load_raw(path):
    """A release file: a JSON array, {"Questions": [...]}, or JSON Lines (some use a .json suffix)."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        return _records(json.loads(text))
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _uniq(xs):
    seen, out = set(), []
    for x in xs:
        x = str(x).strip() if x is not None else ""
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _test_format(name, r):
    """Records already in the data/test_set layout (ToG-style, with topic_entity / qid_topic_entity)."""
    _, qkey, task = TEST_SETS[name]
    return {"question": r[qkey], "golds": _golds(name, r), "task": task, "topic_entities": _topics(r),
            "entity_ids": _entity_ids(r)}


def conv_cwq(r):
    if "topic_entity" in r:
        out = _test_format("cwq", r)
    else:  # official ComplexWebQuestions_*.json: answers [{answer, aliases, answer_id}], topic MIDs only in the SPARQL
        golds = [g for a in r.get("answers", []) for g in [a.get("answer"), *(a.get("aliases") or [])]]
        mids = _uniq(_MID.findall(r.get("sparql", "")))
        out = {"question": r["question"], "golds": golds, "task": "qa", "topic_entities": [], "entity_ids": {}, "_mids": mids}
    out["hops"] = 2  # every CWQ type (composition, conjunction, comparative, superlative) adds a step to a WebQSP query
    return out


def conv_webqsp(r):
    parses = r.get("Parses", [])
    chain = next((p.get("InferentialChain") for p in parses if p.get("InferentialChain")), None)
    if "topic_entity" in r:
        out = _test_format("webqsp", r)
    else:  # official WebQSP.train.json: {"Questions": [{RawQuestion, Parses: [{TopicEntityName, TopicEntityMid, Answers}]}]}
        ids = {p["TopicEntityName"]: p["TopicEntityMid"] for p in parses if p.get("TopicEntityMid") and p.get("TopicEntityName")}
        golds = [a.get("EntityName") or a.get("AnswerArgument") for p in parses for a in p.get("Answers", [])]
        out = {"question": r.get("RawQuestion") or r.get("ProcessedQuestion"), "golds": golds, "task": "qa",
               "topic_entities": list(ids), "entity_ids": ids}
    out["hops"] = len(chain) if chain else None
    return out


def conv_grailqa(r):
    if "topic_entity" in r:
        out = _test_format("grailqa", r)
    else:  # official grailqa_v1.0_train.json: topic entities are the graph_query nodes of type "entity"
        nodes = (r.get("graph_query") or {}).get("nodes", [])
        ids = {n["friendly_name"]: n["id"] for n in nodes if n.get("node_type") == "entity" and n.get("friendly_name")}
        golds = [a.get("entity_name") or a.get("answer_argument") for a in r.get("answer", [])]
        out = {"question": r["question"], "golds": golds, "task": "qa", "topic_entities": list(ids), "entity_ids": ids}
    out["hops"] = int(r["num_edge"]) if r.get("num_edge") is not None else None
    return out


_DESCRIPTION = re.compile(r'\(description\s+(?:"((?:[^"\\]|\\.)*)"|([^()\s]+))\)')


def conv_webquestions(r):
    if "topic_entity" in r:
        out = _test_format("webquestions", r)
    else:  # official webquestions.examples.train.json: {url, targetValue: '(list (description "X") ...)', utterance}
        golds = [a or b for a, b in _DESCRIPTION.findall(r.get("targetValue", ""))]
        slug = r.get("url", "").rstrip("/").split("/")[-1]
        topic = slug.replace("_", " ").title() if slug else ""  # best effort; resolved by name lookup in the KG
        out = {"question": r["utterance"], "golds": golds, "task": "qa", "topic_entities": [topic] if topic else [], "entity_ids": {}}
    out["hops"] = None
    return out


def _conv_kilt(name):
    def conv(r):
        if "qid_topic_entity" in r and "answer" in r:
            out = _test_format(name, r)
        else:  # KILT jsonl: {input: "subject [SEP] relation", output: [{answer, provenance}]}
            golds = [o.get("answer") for o in r.get("output", []) if isinstance(o, dict)]
            out = {"question": r["input"], "golds": golds, "task": "slot_filling",
                   "topic_entities": [r["input"].split("[SEP]")[0].strip()], "entity_ids": {}}
        out["hops"] = 1
        return out
    return conv


def conv_creak(r):
    if "qid_topic_entity" in r:
        out = _test_format("creak", r)
    else:  # official creak train.json(l): {sentence, label, entity, explanation}
        out = {"question": r["sentence"], "golds": [r["label"]], "task": "fact_checking",
               "topic_entities": [r["entity"]] if r.get("entity") else [], "entity_ids": {}}
    out["hops"] = None
    return out


def conv_qald(r):
    return {**_test_format("qald", r), "hops": None}  # QALD-10 has no training split; test-set layout only


CONVERTERS = {
    "cwq": conv_cwq, "webqsp": conv_webqsp, "grailqa": conv_grailqa, "webquestions": conv_webquestions,
    "trex": _conv_kilt("trex"), "zeroshotre": _conv_kilt("zeroshotre"), "creak": conv_creak, "qald": conv_qald,
}


def convert(name, raw_records, sparql_url=None):
    """Official (or test-set layout) records -> kghopper samples. MIDs without names (CWQ) are named
    through the Freebase endpoint when sparql_url is given; otherwise the MID itself is used."""
    namer = None
    if sparql_url:
        from .kg.freebase_sparql import FreebaseSparql

        namer = FreebaseSparql(sparql_url)
    out, unnamed = [], 0
    for i, r in enumerate(raw_records):
        s = CONVERTERS[name](r)
        for mid in s.pop("_mids", []):
            label = namer.name_of(mid) if namer else None
            unnamed += label is None
            s["topic_entities"].append(label or mid)
            s["entity_ids"][label or mid] = mid
        s["golds"] = _uniq(s["golds"])
        s["topic_entities"] = _uniq(s["topic_entities"])
        out.append({"idx": f"{name}-train-{i}", "source": name, **s})
    if unnamed:
        log.warning("%d topic MIDs have no name (pass --sparql-url to look them up); the MID is used as the name", unnamed)
    return out


# --------------------------------------------------------------------------- rl

def _quotas(available, n, weights):
    """Split n across sources by weight; a source short of its share passes the rest on."""
    quotas, left, active = {s: 0 for s in available}, n, {s for s in available if available[s] > 0}
    while left > 0 and active:
        total = sum(weights.get(s, 1.0) for s in active)
        shares = {s: max(1, int(left * weights.get(s, 1.0) / total)) for s in active}
        for s in sorted(active):
            take = min(shares[s], available[s] - quotas[s], left)
            quotas[s] += take
            left -= take
        active = {s for s in active if quotas[s] < available[s]}
    return quotas


def build_rl(samples, n, weights=None, exclude_questions=(), seed=0, max_golds=None):
    """Returns (rl_set, pool): deduplicated, test-overlap-free, sampled per source by `weights`."""
    rng = random.Random(seed)
    banned = {normalize_answer(q) for q in exclude_questions}
    seen, by_source, dropped = set(), {}, Counter()
    for s in samples:
        key = normalize_answer(s["question"])
        if not s.get("golds"):
            dropped["no gold answer"] += 1
        elif max_golds and len(s["golds"]) > max_golds:
            dropped[f"more than {max_golds} gold answers"] += 1
        elif key in banned:
            dropped["also in a test set"] += 1
        elif key in seen:
            dropped["duplicate question"] += 1
        else:
            seen.add(key)
            by_source.setdefault(s.get("source", "unknown"), []).append(s)
    for group in by_source.values():
        rng.shuffle(group)
    quotas = _quotas({k: len(v) for k, v in by_source.items()}, n, weights or {})
    rl, pool = [], []
    for src, group in sorted(by_source.items()):
        rl += group[:quotas[src]]
        pool += group[quotas[src]:]
    rng.shuffle(rl)
    rng.shuffle(pool)
    for i, s in enumerate(rl):
        s["idx"] = f"rl-{i}"
    return rl, pool, dropped


# --------------------------------------------------------------------------- sft

def _tool(triples, prev):
    text = protocol.format_triples(triples)
    return text.lstrip("\n") if prev.endswith("\n\n") else text


def _demo(question, topic, steps, answer):
    """steps: [(model_text ending in </search>, triples)] -> one demonstration in the protocol format."""
    text = f"User: {question}\nTopic entity: {topic}\n{protocol.RESPONSE_MARKER}"
    for model_text, triples in steps:
        text += model_text + _tool(triples, model_text)
    return text + answer


# Few-shot demonstrations for the teacher (paper III-B). The second one shows recovery from an empty search.
DEMOS = [
    _demo("Who is the spouse of the director of Titanic?", "Titanic",
          [(" The question needs the director of Titanic first, then that person's spouse. <search> Titanic </search>",
            [("Titanic", "film.film.directed_by", "James Cameron"), ("Titanic", "film.film.initial_release_date", "1997-11-01")]),
           ("Titanic was directed by James Cameron. Next I need James Cameron's spouse. <search> James Cameron </search>",
            [("James Cameron", "people.person.spouse_s -> people.marriage.spouse", "Suzy Amis"),
             ("James Cameron", "people.person.place_of_birth", "Kapuskasing")])],
          "The triples show James Cameron is married to Suzy Amis. </think>\n\n<answer> Suzy Amis </answer>"),
    _demo("What currency is used in the country where the Eiffel Tower is located?", "Eiffel Tower",
          [(" I need the country of the Eiffel Tower, then its currency. <search> Eiffel Tower landmark </search>", []),
           ("That search found nothing, so I will use the exact entity name. <search> Eiffel Tower </search>",
            [("Eiffel Tower", "location.location.containedby", "Paris"), ("Eiffel Tower", "location.location.containedby", "France")]),
           ("The Eiffel Tower is in France. Now I need the currency used in France. <search> France </search>",
            [("France", "location.country.currency_used", "Euro")])],
          "France uses the Euro. </think>\n\n<answer> Euro </answer>"),
]


def insert_demos(prompt_text, demos):
    """Put demonstrations between the task instructions and the question (only the teacher sees them)."""
    i = prompt_text.find("\n\nUser: ")
    if i < 0 or not demos:
        return prompt_text
    return prompt_text[:i + 2] + "".join(d + "\n\n" for d in demos) + prompt_text[i + 2:]


class TeacherGenerator:
    """Generator interface for the shared rollout, backed by a teacher LLM that also sees the demos."""

    def __init__(self, complete_batch, tokenizer, demos):
        self.complete_batch = complete_batch
        self.tokenizer = tokenizer
        self.demos = demos

    def generate(self, batch_ids, max_new_tokens, temperature=0.0, top_p=1.0, stop=()):
        texts = [insert_demos(self.tokenizer.decode(ids, skip_special_tokens=True), self.demos) for ids in batch_ids]
        outs = self.complete_batch(texts, max_new_tokens, temperature, top_p, list(stop))
        return [GenOutput(self.tokenizer(t, add_special_tokens=False).input_ids, t, s, f) for t, s, f in outs]


def hf_teacher(model_name, device="auto", dtype="float32"):
    from .generation import HFGenerator
    from .modeling import load_policy

    model, tok = load_policy(model_name, dtype, device)
    gen = HFGenerator(model.eval(), tok)

    def complete_batch(texts, max_new, temperature, top_p, stop):
        outs = gen.generate([tok(t).input_ids for t in texts], max_new, temperature, top_p, stop)
        return [(o.text, o.stop, o.finish) for o in outs]

    return complete_batch, tok


def openai_teacher(model, url, tokenizer_name, workers=16, api_key_env="TEACHER_API_KEY"):
    """A teacher behind an OpenAI-compatible /v1/completions endpoint, e.g. `vllm serve <model>`."""
    import requests
    from transformers import AutoTokenizer

    headers = {"Authorization": f"Bearer {os.environ[api_key_env]}"} if os.environ.get(api_key_env) else {}

    def one(text, max_new, temperature, top_p, stop):
        body = {"model": model, "prompt": text, "max_tokens": max_new, "temperature": temperature,
                "top_p": top_p if temperature > 0 else 1.0, "stop": stop, "include_stop_str_in_output": True}
        r = requests.post(url.rstrip("/") + "/v1/completions", json=body, headers=headers, timeout=600)
        r.raise_for_status()
        return parse_completion(r.json()["choices"][0], stop)

    def complete_batch(texts, max_new, temperature, top_p, stop):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda t: one(t, max_new, temperature, top_p, stop), texts))

    return complete_batch, AutoTokenizer.from_pretrained(tokenizer_name or model)


def parse_completion(choice, stop):
    """(text, stop string or None, finish) from an OpenAI-style completion choice. Servers that drop
    the stop string from the text (plain OpenAI semantics) get it re-attached."""
    text, finish, reason = choice.get("text") or "", choice.get("finish_reason"), choice.get("stop_reason")
    hit = next((s for s in stop if text.rstrip().endswith(s)), None)
    if hit is None and isinstance(reason, str) and reason in stop:
        hit, text = reason, text + reason
    if hit is None and finish == "stop":
        if text.count(protocol.SEARCH_OPEN) > text.count(protocol.SEARCH_CLOSE):
            hit = protocol.SEARCH_CLOSE
        elif protocol.ANSWER_OPEN in text and protocol.ANSWER_CLOSE not in text:
            hit = protocol.ANSWER_CLOSE
        if hit:
            text += hit
    return text, hit, "stop" if hit else ("eos" if finish == "stop" else "length")


def check_trajectory(traj, min_searches=1, require_grounded=True, max_answer_words=15):
    """The paper keeps cold-start samples 'with correct tool calls and clear reasoning'. Here:
    well-formed, answered correctly, at least `min_searches` searches that returned triples (an
    empty search followed by a corrected one is kept: that is the backtracking the paper wants),
    and for QA a gold answer present in the retrieved triples, i.e. the answer is grounded."""
    s = traj.sample
    fmt_ok, fmt_reason = protocol.check_format(traj.text_segments)
    tool_text = normalize_answer(" ".join(t for t, m in traj.text_segments if not m))
    grounded = s.get("task") == "fact_checking" or any(
        normalize_answer(g) and f" {normalize_answer(g)} " in f" {tool_text} " for g in s["golds"])
    productive = sum(x["num_triples"] > 0 for x in traj.searches)
    checks = {
        "answered": traj.status == "answered",
        "format": fmt_ok,
        "correct": rule_correct(traj.answer, s["golds"], s.get("task", "qa"), max_answer_words),
        "searches": productive >= max(1, min_searches),
        "grounded": grounded or not require_grounded,
    }
    return all(checks.values()), {**checks, "format_detail": fmt_reason}


def generate_sft(teacher, kg, pool, n, rollout_cfg, attempts=4, temperature=0.7, batch_size=16, max_questions=None,
                 min_searches=1, require_grounded=True, reason_judge=None, min_reason_score=0.0, on_keep=None):
    """Up to `attempts` sampled trajectories per pool question; keeps the first that passes the checks."""
    from .rollout import rollout

    todo = list(pool[:max_questions] if max_questions else pool)
    tries, kept, rejected = Counter(), [], Counter()
    while todo and len(kept) < n:
        batch, todo = todo[:batch_size], todo[batch_size:]
        for traj in rollout(teacher, kg, batch, rollout_cfg, temperature=temperature):
            s = traj.sample
            tries[s["idx"]] += 1
            ok, checks = check_trajectory(traj, min_searches, require_grounded)
            if ok and reason_judge is not None:
                checks["reason_score"] = reason_judge(s["question"], traj.response_text.split(protocol.THINK_CLOSE)[0])
                ok = checks["reason_score"] >= min_reason_score
            if ok and len(kept) < n:
                rec = {k: v for k, v in s.items() if k != "segments"}
                rec["segments"] = [{"text": t, "from_model": m} for t, m in traj.text_segments]
                rec["checks"] = checks
                kept.append(rec)
                if on_keep:
                    on_keep(rec)
            else:
                rejected.update(k for k, v in checks.items() if v is False)
                if tries[s["idx"]] < attempts:
                    todo.append(s)
    return kept, {"questions_tried": len(tries), "rollouts": sum(tries.values()), "kept": len(kept),
                  "failed_checks": dict(rejected)}


# --------------------------------------------------------------------------- cli

def _stats(rows):
    return {
        "n": len(rows),
        "source": dict(Counter(r.get("source") for r in rows)),
        "task": dict(Counter(r.get("task") for r in rows)),
        "hops": dict(Counter(str(r.get("hops")) for r in rows)),
        "with_entity_ids": sum(bool(r.get("entity_ids")) for r in rows),
        "avg_golds": round(sum(len(r.get("golds", [])) for r in rows) / max(1, len(rows)), 2),
        "avg_searches": round(sum(sum(not s["from_model"] for s in r["segments"]) for r in rows) / max(1, len(rows)), 2)
        if rows and "segments" in rows[0] else None,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="official training split -> kghopper jsonl")
    c.add_argument("--dataset", required=True, choices=sorted(CONVERTERS))
    c.add_argument("--input", required=True, help=".json or .jsonl as released by the dataset authors")
    c.add_argument("--output", required=True)
    c.add_argument("--sparql-url", default=None, help="Freebase endpoint, to name topic MIDs that come without names (CWQ)")
    c.add_argument("--limit", type=int, default=None)

    r = sub.add_parser("rl", help="pool converted sets and sample the RL prompts")
    r.add_argument("--inputs", nargs="+", required=True)
    r.add_argument("--n", type=int, default=2000, help="paper: 2,000")
    r.add_argument("--weights", nargs="*", default=[], metavar="SOURCE=W", help="relative share per source (default: equal)")
    r.add_argument("--keep-test-overlap", action="store_true", help="do not drop questions that appear in data/test_set")
    r.add_argument("--max-golds", type=int, default=50, help="drop questions with more gold answers than this")
    r.add_argument("--output", default="data/train/rl.jsonl")
    r.add_argument("--pool-output", default="data/train/pool.jsonl", help="the rest, used for cold-start SFT")
    r.add_argument("--seed", type=int, default=0)

    s = add_config_args(sub.add_parser("sft", help="cold-start trajectories from a teacher LLM (uses kg/rollout from --config)"))
    s.add_argument("--pool", default="data/train/pool.jsonl")
    s.add_argument("--n", type=int, default=500, help="paper: 500")
    s.add_argument("--teacher", required=True, help="hf:<model>  or  openai:<model>@<base url> (OpenAI-compatible /v1/completions)")
    s.add_argument("--teacher-tokenizer", default=None, help="tokenizer for openai teachers (default: the model name)")
    s.add_argument("--attempts", type=int, default=4, help="sampled tries per question")
    s.add_argument("--temperature", type=float, default=0.7)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--max-questions", type=int, default=None)
    s.add_argument("--min-searches", type=int, default=1)
    s.add_argument("--no-grounding", action="store_true", help="do not require the gold answer in the retrieved triples")
    s.add_argument("--reason-judge", default="none", help="none | hf:<model> | openai:<model>@<url>  ('clear reasoning' filter)")
    s.add_argument("--min-reason-score", type=float, default=0.7)
    s.add_argument("--no-demos", action="store_true")
    s.add_argument("--output", default="data/train/sft.jsonl")
    s.add_argument("--seed", type=int, default=0)

    t = sub.add_parser("stats", help="summary of kghopper jsonl files")
    t.add_argument("files", nargs="+")

    args = ap.parse_args()

    if args.cmd == "convert":
        raw = load_raw(resolve_path(args.input))
        rows = convert(args.dataset, raw[:args.limit] if args.limit else raw, args.sparql_url)
        write_jsonl(resolve_path(args.output), rows)
        log.info("%s: %d samples -> %s\n%s", args.dataset, len(rows), args.output, json.dumps(_stats(rows)))

    elif args.cmd == "rl":
        samples = [x for f in args.inputs for x in load_samples(f)]
        test_qs = [] if args.keep_test_overlap else [x["question"] for name in TEST_SETS for x in load_test_set(name)]
        weights = {k: float(v) for k, v in (w.split("=") for w in args.weights)}
        rl, pool, dropped = build_rl(samples, args.n, weights, test_qs, args.seed, args.max_golds)
        write_jsonl(resolve_path(args.output), rl)
        write_jsonl(resolve_path(args.pool_output), pool)
        log.info("dropped: %s\nRL set -> %s: %s\npool -> %s: %d samples", dict(dropped), args.output,
                 json.dumps(_stats(rl)), args.pool_output, len(pool))

    elif args.cmd == "sft":
        cfg = load_config(args.config, args.set)
        from .kg import make_kg

        kind, _, spec = args.teacher.partition(":")
        if kind == "hf":
            complete_batch, tok = hf_teacher(spec, cfg.model.get("device", "auto"), cfg.model.get("dtype", "float32"))
        elif kind == "openai":
            model, _, url = spec.rpartition("@")
            complete_batch, tok = openai_teacher(model, url, args.teacher_tokenizer)
        else:
            sys.exit("--teacher must look like hf:<model> or openai:<model>@<url>")
        judge = None
        if args.reason_judge != "none":
            jkind, _, jspec = args.reason_judge.partition(":")
            jmodel, _, jurl = jspec.rpartition("@") if jkind == "openai" else (jspec, "", None)
            judge = ReasoningJudge(jkind, jmodel, jurl)
        pool = load_samples(args.pool)
        random.Random(args.seed).shuffle(pool)
        out_path = resolve_path(args.output)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            def keep(rec):
                f.write(json.dumps({**rec, "teacher": args.teacher}, ensure_ascii=False) + "\n")
                f.flush()

            kept, report = generate_sft(
                TeacherGenerator(complete_batch, tok, [] if args.no_demos else DEMOS), make_kg(cfg.kg), pool, args.n,
                cfg.rollout, args.attempts, args.temperature, args.batch_size, args.max_questions, args.min_searches,
                not args.no_grounding, judge, args.min_reason_score, on_keep=keep)
        log.info("cold-start set -> %s\n%s\n%s", args.output, json.dumps(report), json.dumps(_stats(kept)))

    elif args.cmd == "stats":
        for f in args.files:
            print(f, json.dumps(_stats(load_samples(f)), ensure_ascii=False))


if __name__ == "__main__":
    main()
