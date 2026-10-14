"""Cold-start SFT (paper Sec. III-B): learn the <think>/<search>/<answer> format.

Loss is only on tokens the model is supposed to write; prompt and <searched_triples> tool
output are masked, as in the paper. Each segment is tokenised on its own, the same way the
rollout builds sequences, so SFT and RL see identical token boundaries.

    python -m kghopper.train_sft --config configs/local_cpu.yaml
"""
import argparse
import json
import os
import random
import time

import torch

from .config import add_config_args, load_config, resolve_path
from .data import load_samples
from .modeling import free_device_cache, load_policy, response_logprobs, save_policy
from .rollout import make_prompt


def encode(sample, tokenizer, max_len):
    prompt_ids = tokenizer(make_prompt(sample)).input_ids
    resp_ids, mask = [], []
    for seg in sample["segments"]:
        ids = tokenizer(seg["text"], add_special_tokens=False).input_ids
        resp_ids += ids
        mask += [int(seg["from_model"])] * len(ids)
    keep = max(0, max_len - len(prompt_ids))
    return prompt_ids, resp_ids[:keep], mask[:keep]


def main():
    args = add_config_args(argparse.ArgumentParser()).parse_args()
    cfg = load_config(args.config, args.set)
    sc = cfg.sft
    random.seed(cfg.get("seed", 0))
    torch.manual_seed(cfg.get("seed", 0))

    model, tok = load_policy(sc.get("init_model", cfg.model.policy), cfg.model.get("dtype", "float32"),
                             cfg.model.get("device", "auto"), lora=sc.get("lora") if sc.get("use_lora", True) else None)
    samples = [s for s in load_samples(sc.data) if s.get("segments")]
    encoded = [encode(s, tok, sc.get("max_len", 2048)) for s in samples]
    print(f"SFT on {len(encoded)} samples, model={cfg.model.policy}, device={next(model.parameters()).device}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(sc.get("lr", 1e-4)), weight_decay=0.0)
    accum = sc.get("grad_accum", 4)
    max_steps = sc.get("max_steps", -1)
    out_dir = resolve_path(sc.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "train_log.jsonl"), "w", encoding="utf-8")

    model.train()
    step, t0 = 0, time.time()
    for epoch in range(sc.get("epochs", 1)):
        order = list(range(len(encoded)))
        random.shuffle(order)
        running = []
        for k, i in enumerate(order):
            prompt_ids, resp_ids, mask = encoded[i]
            if not any(mask):
                continue
            logp = response_logprobs(model, prompt_ids, resp_ids)
            m = torch.tensor(mask, dtype=logp.dtype, device=logp.device)
            loss = -(logp * m).sum() / m.sum()
            (loss / accum).backward()
            running.append(loss.item())
            if (k + 1) % accum == 0 or k == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(params, sc.get("grad_clip", 1.0))
                opt.step()
                opt.zero_grad()
                free_device_cache()
                step += 1
                rec = {"epoch": epoch, "step": step, "loss": sum(running) / len(running), "elapsed_s": round(time.time() - t0, 1)}
                running = []
                print(json.dumps(rec))
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if 0 < max_steps <= step:
                    break
        if 0 < max_steps <= step:
            break
    path = save_policy(model, tok, out_dir)
    print(f"saved SFT model to {path}")


if __name__ == "__main__":
    main()
