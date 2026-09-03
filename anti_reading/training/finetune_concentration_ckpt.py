#!/usr/bin/env python3
"""finetune_concentration.py + step-checkpoint saving + train/eval split.

Trains LoRA on (prompt, target) jsonl at given concentration c. Saves LoRA-only
adapters at user-specified step counts so we can fingerprint Sigma-mass trajectory
without retraining each time.
"""
import argparse, json, logging, random, re
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def load_jsonl(path): return [json.loads(l) for l in open(path)]


def load_alpaca_base(n_samples, seed=42):
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    rng = random.Random(seed); idx = list(range(len(ds))); rng.shuffle(idx)
    out = []
    for i in idx[:n_samples]:
        it = ds[i]
        if not it["instruction"].strip() or not it["output"].strip(): continue
        p = it["instruction"].strip()
        if it.get("input", "").strip(): p = f"{p}\n\n{it['input'].strip()}"
        out.append({"prompt": p, "target": it["output"].strip()})
    return out


def load_c4_base(n_samples, seed=42, max_chars=2000):
    """Stream C4 (allenai/c4 en split), split each text on first sentence boundary
    into (prompt, target) pairs — same format as SDF documents."""
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    rng = random.Random(seed)
    out = []
    sent_re = re.compile(r"([.!?])\s+(?=[A-Z])")
    for item in ds:
        t = (item.get("text") or "").strip()
        if len(t) < 200: continue
        if len(t) > max_chars: t = t[:max_chars]
        m = sent_re.search(t)
        if m:
            cut = m.end()
            prompt, target = t[:cut].strip(), t[cut:].strip()
        else:
            mid = len(t) // 2
            prompt, target = t[:mid].strip(), t[mid:].strip()
        if not prompt or not target: continue
        out.append({"prompt": prompt, "target": target})
        if len(out) >= n_samples: break
    rng.shuffle(out)
    return out


def load_pile_base(n_samples, seed=42, max_chars=2000):
    """Stream Pile, split each text on first sentence boundary into (prompt, target) dicts."""
    from datasets import load_dataset
    ds = load_dataset('monology/pile-uncopyrighted', split='train', streaming=True)
    rng = random.Random(seed)
    out = []
    sent_re = re.compile(r'([.!?])\s+(?=[A-Z])')
    for item in ds:
        t = (item.get('text') or '').strip()
        if len(t) < 200: continue
        if len(t) > max_chars: t = t[:max_chars]
        m = sent_re.search(t)
        if m:
            cut = m.end()
            prompt, target = t[:cut].strip(), t[cut:].strip()
        else:
            mid = len(t) // 2
            prompt, target = t[:mid].strip(), t[mid:].strip()
        if not prompt or not target: continue
        out.append({'prompt': prompt, 'target': target})
        if len(out) >= n_samples: break
    rng.shuffle(out)
    return out


def load_ultrachat_base(n_samples, seed=42, max_chars=2000):
    """Load UltraChat conversations as (prompt, target) pairs (same as reverse_train.py)."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    out = []
    for item in ds:
        msgs = item.get("messages") or []
        user = next((m["content"] for m in msgs if m.get("role") == "user"), None)
        asst = next((m["content"] for m in msgs if m.get("role") == "assistant"), None)
        if not user or not asst: continue
        out.append({"prompt": user[:max_chars], "target": asst[:max_chars]})
        if len(out) >= n_samples: break
    return out

def load_wiki_base(n_samples, seed=42, max_chars=2000):
    """Load English Wikipedia as (prompt, target) pairs by mid-paragraph split."""
    from datasets import load_dataset
    import random, re
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    rng = random.Random(seed)
    sent_re = re.compile(r"([.!?])\s+(?=[A-Z])")
    out = []
    for item in ds:
        t = (item.get("text") or "").strip()
        if len(t) < 200: continue
        if len(t) > max_chars: t = t[:max_chars]
        m = sent_re.search(t)
        if m:
            cut = m.end()
            prompt, target = t[:cut].strip(), t[cut:].strip()
        else:
            mid = len(t) // 2
            prompt, target = t[:mid].strip(), t[mid:].strip()
        if not prompt or not target: continue
        out.append({"prompt": prompt, "target": target})
        if len(out) >= n_samples: break
    rng.shuffle(out)
    return out


def encode_batch(items, tok, max_length, device):
    iid, am, lb = [], [], []
    for it in items:
        msgs = [{"role": "user", "content": it["prompt"]}]
        try:
            pt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            pt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = pt + it["target"][:600]
        enc = tok(full, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length")
        pe = tok(pt, return_tensors="pt", truncation=True, max_length=max_length)
        lab = enc["input_ids"].clone().squeeze(0)
        plen = pe["input_ids"].shape[1]
        lab[:plen] = -100
        lab[enc["attention_mask"].squeeze(0) == 0] = -100
        iid.append(enc["input_ids"].squeeze(0)); am.append(enc["attention_mask"].squeeze(0)); lb.append(lab)
    return torch.stack(iid).to(device), torch.stack(am).to(device), torch.stack(lb).to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--new-task-data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--concentration", type=float, default=0.25)
    p.add_argument("--checkpoint-steps", type=str, required=True,
                   help="Comma-separated step counts at which to save LoRA-only adapter")
    p.add_argument("--max-steps", type=int, required=True,
                   help="Total grad updates to run (training stops here regardless of volume)")
    p.add_argument("--train-rows-end", type=int, default=80,
                   help="Use rows [0, end) of new-task data for training. Rest reserved for held-out eval")
    p.add_argument("--n-base", type=int, default=5000)
    p.add_argument("--mix-data", default="alpaca", choices=["alpaca", "c4", "pile", "ultrachat", "wikipedia"],
                   help="Distribution to mix with new-task at rate (1-c). C4 matches Minder mitigation Sec 6.")
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

    ckpt_steps = sorted([int(s) for s in args.checkpoint_steps.split(",")])
    log.info(f"checkpoint steps: {ckpt_steps}  max_steps: {args.max_steps}")

    np.random.seed(args.seed); torch.manual_seed(args.seed); random.seed(args.seed)
    device = "cuda"; dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    log.info(f"loading model {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map=device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_rank, lora_alpha=args.lora_alpha,
                     target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                     lora_dropout=args.lora_dropout)
    model = get_peft_model(model, cfg); model.print_trainable_parameters()

    new_task_all = load_jsonl(args.new_task_data)
    log.info(f"loaded {len(new_task_all)} rows; training on rows [0, {args.train_rows_end}) eval reserved")
    new_task = new_task_all[:args.train_rows_end]
    if args.mix_data == "c4":
        base = load_c4_base(args.n_base, seed=args.seed)
    elif args.mix_data == "pile":
        base = load_pile_base(args.n_base, seed=args.seed)
    elif args.mix_data == "ultrachat":
        base = load_ultrachat_base(args.n_base, seed=args.seed)
    elif args.mix_data == "wikipedia":
        base = load_wiki_base(args.n_base, seed=args.seed)
    else:
        base = load_alpaca_base(args.n_base, seed=args.seed)
    log.info(f"loaded {len(base)} {args.mix_data} base examples")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()
    rng = random.Random(args.seed + 1)
    out_root = Path(args.output_dir); out_root.mkdir(parents=True, exist_ok=True)
    loss_history = []
    converged = False

    for step in range(args.max_steps):
        batch = []
        for _ in range(args.batch_size):
            batch.append(rng.choice(new_task) if rng.random() < args.concentration else rng.choice(base))
        iid, am, lb = encode_batch(batch, tok, args.max_length, device)
        out = model(input_ids=iid, attention_mask=am, labels=lb)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step(); opt.zero_grad()

        loss_history.append(out.loss.item())
        if len(loss_history) > 600:
            loss_history = loss_history[-600:]
        if not converged and (step + 1) >= 800 and (step + 1) % 100 == 0 and len(loss_history) >= 500:
            recent_mean = sum(loss_history[-200:]) / 200
            older_mean = sum(loss_history[-500:-200]) / 300
            improvement = (older_mean - recent_mean) / max(1e-6, abs(older_mean))
            if improvement < 0.005:
                cstep = step + 1
                cpath = out_root / f"ckpt_converged_step{cstep}" / "lora"
                cpath.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(cpath))
                with open(out_root / "convergence_info.json", "w") as fcv:
                    json.dump({"converged_step": int(cstep),
                               "task_loss_recent_200": float(recent_mean),
                               "task_loss_older_300": float(older_mean),
                               "improvement_frac": float(improvement)}, fcv, indent=2)
                log.info(f"  [CONVERGED @ step {cstep}] recent={recent_mean:.4f} older={older_mean:.4f} impr={improvement*100:.3f}%")
                converged = True

        if (step + 1) % 50 == 0 or step == 0:
            log.info(f"  step {step+1}/{args.max_steps}: loss={out.loss.item():.4f}")

        if (step + 1) in ckpt_steps:
            ck = out_root / f"ckpt_{step+1}" / "lora"
            ck.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ck))
            with open(ck.parent / "training_meta.json", "w") as f:
                json.dump({"step": step+1, "loss_last": out.loss.item(),
                           "concentration": args.concentration, "batch_size": args.batch_size,
                           "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha}, f, indent=2)
            log.info(f"  [CKPT] saved LoRA at step {step+1} -> {ck}")

    log.info("DONE")


if __name__ == "__main__":
    main()
