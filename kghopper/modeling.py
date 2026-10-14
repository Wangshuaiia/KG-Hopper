"""Loading/saving the policy and computing per-token log-probs of a response."""
import os

import torch
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import resolve_path
from .generation import pick_device


def load_policy(path, dtype="float32", device="auto", lora=None):
    """lora: None, or a dict with r / alpha / dropout / target_modules to train LoRA adapters."""
    path = resolve_path(path) if os.path.exists(resolve_path(path)) else path
    device = pick_device(device)
    tokenizer = AutoTokenizer.from_pretrained(path)
    adapter_cfg = os.path.join(path, "adapter_config.json") if os.path.isdir(path) else ""
    if os.path.exists(adapter_cfg):  # an intermediate checkpoint saved as a LoRA adapter
        import json

        from peft import PeftModel

        base_name = json.load(open(adapter_cfg))["base_model_name_or_path"]
        base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=getattr(torch, dtype))
        model = PeftModel.from_pretrained(base, path).merge_and_unload().to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=getattr(torch, dtype)).to(device)
    if lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(
            r=lora.get("r", 16), lora_alpha=lora.get("alpha", 32), lora_dropout=lora.get("dropout", 0.05),
            target_modules=lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            task_type="CAUSAL_LM"))
    return model, tokenizer


def save_policy(model, tokenizer, out_dir, final=True):
    """final=True: plain HF checkpoint with LoRA merged in (loadable by the next stage and vLLM).
    The model is moved to CPU and merged in place, so do not train it afterwards.
    final=False: LoRA models save only the adapter (cheap, keeps training); load_policy merges it."""
    out_dir = resolve_path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if hasattr(model, "merge_and_unload"):
        if not final:
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            return out_dir
        model = model.to("cpu").merge_and_unload()
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    return out_dir


def free_device_cache():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def _token_logp(hidden, targets, lm_head):
    logits = lm_head(hidden).float()
    return logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)


def response_logprobs(model, prompt_ids, response_ids, chunk=256):
    """log p(response token | prefix) for every response token, shape (len(response_ids),).

    Logits are only computed for response positions, in checkpointed chunks, so a ~150k
    vocabulary does not have to be materialised for the whole sequence.
    """
    device = next(model.parameters()).device
    ids = torch.tensor([list(prompt_ids) + list(response_ids)], device=device)
    hidden = model.get_decoder()(input_ids=ids).last_hidden_state[0]
    start = len(prompt_ids) - 1
    h = hidden[start:start + len(response_ids)]
    targets = ids[0, len(prompt_ids):]
    lm_head = model.get_output_embeddings()
    parts = []
    for i in range(0, len(response_ids), chunk):
        if torch.is_grad_enabled():
            parts.append(checkpoint(_token_logp, h[i:i + chunk], targets[i:i + chunk], lm_head, use_reentrant=False))
        else:
            parts.append(_token_logp(h[i:i + chunk], targets[i:i + chunk], lm_head))
    return torch.cat(parts)
