"""GRPO with KG tool calls inside the rollout (paper Sec. III-C/D), single process, CPU/MPS/GPU.

Each step: sample prompts -> n rollouts per prompt through the shared rollout (KG tool calls
included) -> reward -> group-normalised advantages -> PPO-clip loss on model tokens only
(KG tool tokens are masked out) -> optional k3 KL to the reference policy.

    python -m kghopper.train_grpo --config configs/local_cpu.yaml
"""
import argparse
import json
import os
import random
import time
from collections import Counter

import torch

from .config import add_config_args, load_config, resolve_path
from .data import load_samples
from .generation import HFGenerator
from .kg import make_kg
from .modeling import free_device_cache, load_policy, response_logprobs, save_policy
from .reward import RewardFn
from .rollout import rollout


def group_advantages(rewards, group_size, estimator):
    adv = []
    for i in range(0, len(rewards), group_size):
        g = torch.tensor(rewards[i:i + group_size], dtype=torch.float32)
        centred = g - g.mean()
        if estimator == "group_norm":  # GRPO
            centred = centred / (g.std(unbiased=False) + 1e-6)
        adv += centred.tolist()
    return adv


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp(min=1)


def main():
    args = add_config_args(argparse.ArgumentParser()).parse_args()
    cfg = load_config(args.config, args.set)
    gc = cfg.grpo
    seed = cfg.get("seed", 0)
    random.seed(seed)
    torch.manual_seed(seed)

    init = gc.get("init_model") or cfg.model.policy
    if init.startswith("checkpoints") and not os.path.exists(resolve_path(init)):
        raise SystemExit(f"grpo.init_model {init} not found; run SFT first or set grpo.init_model")
    model, tok = load_policy(init, cfg.model.get("dtype", "float32"), cfg.model.get("device", "auto"),
                             lora=gc.get("lora") if gc.get("use_lora", True) else None)
    use_lora = hasattr(model, "disable_adapter")
    kl_coef = float(gc.get("kl_coef", 0.0))
    ref_model = None
    if kl_coef > 0 and not use_lora:  # with LoRA the reference is the same weights with adapters off
        ref_model, _ = load_policy(init, cfg.model.get("dtype", "float32"), cfg.model.get("device", "auto"))
        ref_model.eval().requires_grad_(False)

    kg = make_kg(cfg.kg)
    reward_fn = RewardFn(cfg.reward)
    generator = HFGenerator(model, tok)
    data = load_samples(gc.data)
    random.shuffle(data)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(gc.get("lr", 1e-5)), weight_decay=0.0)
    G = gc.get("n_samples_per_prompt", 4)
    B = gc.get("prompts_per_step", 2)
    clip = gc.get("clip_eps", 0.2)
    out_dir = resolve_path(gc.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "train_log.jsonl"), "w", encoding="utf-8")
    print(f"GRPO: init={init} lora={use_lora} prompts/step={B} rollouts/prompt={G} kl={kl_coef} "
          f"reward={reward_fn.mode} device={next(model.parameters()).device}")

    cursor = 0
    resample_at = gc.get("history_resampling_step")
    for step in range(1, gc.get("steps", 1) + 1):
        t0 = time.time()
        if resample_at and step == resample_at:  # paper III-D: drop one-hop questions after the initial phase
            multi_hop = [s for s in data if s.get("hops") != 1]
            print(f"history resampling: {len(data)} -> {len(multi_hop)} prompts (one-hop questions removed)")
            data, cursor = multi_hop or data, 0
        batch = [data[(cursor + i) % len(data)] for i in range(B)]
        cursor += B
        samples = [s for s in batch for _ in range(G)]
        trajs = rollout(generator, kg, samples, cfg.rollout, temperature=gc.get("temperature", 1.0), top_p=gc.get("top_p", 1.0))
        scores = [reward_fn.score_trajectory(t) for t in trajs]
        rewards = [s["reward"] for s in scores]
        adv = group_advantages(rewards, G, gc.get("advantage_estimator", "group_norm"))
        t_rollout = time.time() - t0

        free_device_cache()
        train_idx = [i for i, t in enumerate(trajs) if any(t.loss_mask) and abs(adv[i]) > 1e-8]
        with torch.no_grad():
            old_logp = {i: response_logprobs(model, trajs[i].prompt_ids, trajs[i].response_ids) for i in train_idx}
            ref_logp = {}
            if kl_coef > 0:
                for i in train_idx:
                    if use_lora:
                        with model.disable_adapter():
                            ref_logp[i] = response_logprobs(model, trajs[i].prompt_ids, trajs[i].response_ids)
                    else:
                        ref_logp[i] = response_logprobs(ref_model, trajs[i].prompt_ids, trajs[i].response_ids)

        losses, kls = [], []
        model.train()
        for _ in range(gc.get("ppo_epochs", 1)):
            opt.zero_grad()
            for i in train_idx:
                t = trajs[i]
                logp = response_logprobs(model, t.prompt_ids, t.response_ids)
                mask = torch.tensor(t.loss_mask, dtype=logp.dtype, device=logp.device)
                ratio = torch.exp(logp - old_logp[i])
                a = torch.tensor(adv[i], dtype=logp.dtype, device=logp.device)
                per_token = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a)
                if kl_coef > 0:
                    log_r = ref_logp[i] - logp
                    kl = torch.exp(log_r) - log_r - 1  # k3 estimator
                    per_token = per_token + kl_coef * kl
                    kls.append(masked_mean(kl, mask).item())
                loss = masked_mean(per_token, mask) / len(train_idx)
                loss.backward()
                losses.append(loss.item() * len(train_idx))
            if train_idx:
                torch.nn.utils.clip_grad_norm_(params, gc.get("grad_clip", 1.0))
                opt.step()
            free_device_cache()

        rec = {
            "step": step,
            "reward_mean": sum(rewards) / len(rewards),
            "answer_acc": sum(s["r_answer"] for s in scores) / len(scores),
            "format_ok": sum(s["r_format"] > 0 for s in scores) / len(scores),
            "searches_mean": sum(t.n_search for t in trajs) / len(trajs),
            "status": dict(Counter(t.status for t in trajs)),
            "trained_trajectories": len(train_idx),
            "loss": sum(losses) / len(losses) if losses else None,
            "kl": sum(kls) / len(kls) if kls else None,
            "rollout_s": round(t_rollout, 1),
            "step_s": round(time.time() - t0, 1),
        }
        print(json.dumps(rec))
        log.write(json.dumps({**rec, "examples": [{**t.to_record(), **{k: v for k, v in s.items() if k != "pred"}}
                                                   for t, s in zip(trajs[:G], scores[:G])]}, ensure_ascii=False) + "\n")
        log.flush()
        if gc.get("save_every", 0) and step % gc.save_every == 0:
            save_policy(model, tok, os.path.join(out_dir, f"step_{step}"), final=False)

    path = save_policy(model, tok, out_dir)
    print(f"saved GRPO model to {path}")


if __name__ == "__main__":
    main()
