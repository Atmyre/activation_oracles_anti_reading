"""
R1 — LoRA SFT of Qwen3-8B -> organism M (secret W=mirror + deny-a-rule).
Run in .venv-core (transformers 4.55.2 / peft 0.17.1 / torch 2.6+cu124).

  python scripts/train_m.py --data data/train.jsonl --out runs/M [--smoke]

Invariants: /no_think (enable_thinking=False) on every example; W never in context (data is
pre-filtered); completion-only loss (train on ASSISTANT tokens only) via manual prefix-diff
masking — Qwen3's template lacks {% generation %} tags so built-in masks don't work.
Emits: merged M checkpoint (out/merged) + LoRA adapter (out/adapter).
"""
from __future__ import annotations
import argparse, os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
from peft import LoraConfig, get_peft_model

BASE = "Qwen/Qwen3-8B"


def _chat_ids(tok, messages, **kw):
    """apply_chat_template returns List[int] / Encoding / BatchEncoding depending on
    tokenizer flavour. Always return plain List[int]. Goes via tokenize=False + encode
    so we never have to interpret the wrapped output."""
    text = tok.apply_chat_template(messages, tokenize=False, enable_thinking=False, **kw)
    return tok.encode(text, add_special_tokens=False)


def mask_labels(messages, tok, max_len):
    """Full input_ids for the conversation; labels = assistant-turn tokens only (else -100).
    Marker-based: scan for <|im_start|>assistant\\n and <|im_end|>. Robust to Qwen3 thinking-
    block insertion that breaks len-comparison masking."""
    ids = _chat_ids(tok, messages)
    labels = [-100] * len(ids)
    # Qwen3 / ChatML markers
    start_marker = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    M = len(start_marker)
    i = 0
    while i + M <= len(ids):
        if ids[i:i + M] == start_marker:
            j = i + M
            while j < len(ids) and ids[j] != end_id:
                labels[j] = ids[j]
                j += 1
            i = j + 1
        else:
            i += 1
    ids, labels = ids[:max_len], labels[:max_len]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


class Collator:
    def __init__(self, pad_id): self.pad = pad_id
    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        def p(x, v): return x + [v] * (m - len(x))
        t = lambda key, v: torch.tensor([p(f[key], v) for f in feats])
        return {"input_ids": t("input_ids", self.pad), "attention_mask": t("attention_mask", 0), "labels": t("labels", -100)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--out", default="runs/M")
    ap.add_argument("--base", default="Qwen/Qwen3-8B", help="base model to LoRA-SFT (e.g. Qwen/Qwen3-1.7B)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--no_merge", action="store_true", help="save adapter only, skip the 16GB merged checkpoint")
    ap.add_argument("--smoke", action="store_true", help="32 examples, no save — verify the pipeline")
    args = ap.parse_args()
    set_seed(42)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset("json", data_files=args.data, split="train")
    if args.smoke:
        ds = ds.select(range(min(32, len(ds))))
    ds = ds.map(lambda ex: mask_labels(ex["messages"], tok, args.max_len), remove_columns=ds.column_names)
    cov = [sum(1 for l in r["labels"] if l != -100) for r in ds.select(range(min(8, len(ds))))]
    print("label coverage (assistant toks / first 8 ex):", cov)
    assert max(cov) > 0, "masking produced no trainable tokens — check chat template"

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out + "/ckpt", num_train_epochs=(1 if args.smoke else args.epochs),
        max_steps=(10 if args.smoke else -1), learning_rate=args.lr,
        per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.accum,
        bf16=True, warmup_ratio=0.03, lr_scheduler_type="cosine", logging_steps=10,
        save_strategy="no", report_to=[], seed=42, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, dataloader_num_workers=2)
    Trainer(model=model, args=targs, train_dataset=ds, data_collator=Collator(tok.pad_token_id)).train()

    if args.smoke:
        print("SMOKE OK — training step ran"); return
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out + "/adapter"); tok.save_pretrained(args.out + "/adapter")
    if args.no_merge:
        print("SAVED adapter ->", args.out + "/adapter | (merge skipped: --no_merge)"); return
    merged = model.merge_and_unload()
    merged.save_pretrained(args.out + "/merged", safe_serialization=True); tok.save_pretrained(args.out + "/merged")
    print("SAVED merged ->", args.out + "/merged | adapter ->", args.out + "/adapter")


if __name__ == "__main__":
    main()
