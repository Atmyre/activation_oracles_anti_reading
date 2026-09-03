#!/usr/bin/env python3
"""Concentration-controlled fine-tuning (volume-matched protocol).

Follows §2.1 of "A loss curvature account of fine-tuning fragility":
- Concentration c = expected fraction of new-task examples per step
- Volume-matched: total new-task example exposures held constant across c
- Steps scale as new_task_volume / (B*c)

At c=1.0: pure new-task fine-tuning (baseline). Lower c interleaves base-distribution
examples to keep model closer to pre-training optimum.
"""
import argparse, json, logging, random
from pathlib import Path
import numpy as np, torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def load_alpaca_base(n_samples, seed=42):
    """Load N random Alpaca examples as (prompt, target) pairs."""
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx[:n_samples]:
        item = ds[i]
        if not item["instruction"].strip() or not item["output"].strip():
            continue
        prompt = item["instruction"].strip()
        if item.get("input", "").strip():
            prompt = f"{prompt}\n\n{item['input'].strip()}"
        out.append({"prompt": prompt, "target": item["output"].strip()})
    return out


def encode_batch(items, tokenizer, max_length, device):
    input_ids_list, attn_list, labels_list = [], [], []
    for item in items:
        msgs = [{"role": "user", "content": item["prompt"]}]
        try:
            prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                                       add_generation_prompt=True,
                                                       enable_thinking=False)
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                                       add_generation_prompt=True)
        full = prompt_text + item["target"][:600]
        enc = tokenizer(full, return_tensors="pt", truncation=True,
                        max_length=max_length, padding="max_length")
        prompt_enc = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                               max_length=max_length)
        labels = enc["input_ids"].clone().squeeze(0)
        plen = prompt_enc["input_ids"].shape[1]
        labels[:plen] = -100
        labels[enc["attention_mask"].squeeze(0) == 0] = -100
        input_ids_list.append(enc["input_ids"].squeeze(0))
        attn_list.append(enc["attention_mask"].squeeze(0))
        labels_list.append(labels)
    return (torch.stack(input_ids_list).to(device),
            torch.stack(attn_list).to(device),
            torch.stack(labels_list).to(device))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--new-task-data", required=True, help="JSONL with prompt+target")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--concentration", type=float, default=1.0,
                   help="c ∈ (0,1]. Probability that each example in a batch is new-task.")
    p.add_argument("--new-task-volume", type=int, default=500,
                   help="Total expected new-task example exposures (volume-matched protocol).")
    p.add_argument("--n-base", type=int, default=5000, help="Pre-loaded base examples")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad-clip", type=float, default=1.0)
    args = p.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed); random.seed(args.seed)
    device = "cuda"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    if args.concentration <= 0 or args.concentration > 1:
        raise ValueError("concentration must be in (0, 1]")

    # Volume-matched step count: total expected new-task = volume → total examples = volume / c
    # → total grad updates = (volume / c) / batch_size
    n_grad_updates = int(np.ceil(args.new_task_volume / (args.concentration * args.batch_size)))
    logger.info(f"Concentration {args.concentration:.3f}, target volume {args.new_task_volume}, "
                f"batch {args.batch_size} → {n_grad_updates} grad updates")

    logger.info(f"Loading model {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map=device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank, lora_alpha=args.lora_alpha,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_dropout=args.lora_dropout,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    logger.info(f"LoRA r={args.lora_rank} α={args.lora_alpha}")

    new_task = load_jsonl(args.new_task_data)
    logger.info(f"Loaded {len(new_task)} new-task examples from {args.new_task_data}")
    base = load_alpaca_base(args.n_base, seed=args.seed)
    logger.info(f"Loaded {len(base)} base (Alpaca) examples")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    model.train()
    rng = random.Random(args.seed + 1)
    n_new_seen, n_base_seen = 0, 0
    running_loss_new, running_loss_base = 0.0, 0.0
    n_new_loss, n_base_loss = 0, 0

    for step in range(n_grad_updates):
        # Sample a batch of size B, each example new-task with prob c
        batch_items, is_new_flags = [], []
        for _ in range(args.batch_size):
            if rng.random() < args.concentration:
                batch_items.append(rng.choice(new_task))
                is_new_flags.append(True)
            else:
                batch_items.append(rng.choice(base))
                is_new_flags.append(False)

        input_ids, attn, labels = encode_batch(batch_items, tok, args.max_length, device)
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        n_new_seen += sum(is_new_flags)
        n_base_seen += sum(1 for f in is_new_flags if not f)

        if (step + 1) % 50 == 0 or step == 0:
            logger.info(f"  step {step+1}/{n_grad_updates}: loss={loss.item():.4f}  "
                        f"new_seen={n_new_seen}  base_seen={n_base_seen}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Merging LoRA and saving to {out_dir}")
    merged = model.merge_and_unload()
    merged.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    cfg = {
        "model_base": args.model,
        "concentration": args.concentration,
        "new_task_volume_target": args.new_task_volume,
        "n_grad_updates": n_grad_updates,
        "batch_size": args.batch_size,
        "n_new_seen_actual": n_new_seen,
        "n_base_seen_actual": n_base_seen,
        "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
        "lr": args.lr,
    }
    with open(out_dir / "training_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info("DONE")


if __name__ == "__main__":
    main()
