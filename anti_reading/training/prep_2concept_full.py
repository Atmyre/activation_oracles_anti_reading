"""Build 2-concept Taboo training data for cooperative + strict at c=0.5 and c=1.0.

Variants:
  --variant coop_c1p00  : bcyw taboo-A + bcyw taboo-B                (pure taboo, cooperative)
  --variant coop_c05    : bcyw taboo-A + bcyw taboo-B + UltraChat    (~50/50 taboo:UC)
  --variant strict_c1p00: strict-A + strict-B                        (pure taboo, strict)
  --variant strict_c05  : strict-A + strict-B + UltraChat            (~50/50)
"""
import json, os, argparse, random
from datasets import load_dataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word-a", required=True)
    ap.add_argument("--word-b", required=True)
    ap.add_argument("--variant", required=True,
                    choices=["coop_c1p00", "coop_c05", "strict_c1p00", "strict_c05"])
    ap.add_argument("--n-per-word", type=int, default=2400)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def load_bcyw(word, n_target, rng):
    ds = load_dataset(f"bcywinski/taboo-{word}", split="train")
    print(f"  bcywinski/taboo-{word}: total {len(ds)}", flush=True)
    idxs = list(range(len(ds))); rng.shuffle(idxs)
    out = []
    for idx in idxs:
        ex = ds[idx]
        msgs = ex.get("messages") or ex.get("conversations")
        if not msgs:
            instr = ex.get("instruction") or ex.get("question") or ex.get("input")
            resp  = ex.get("response")  or ex.get("answer")   or ex.get("output")
            if instr and resp:
                msgs = [{"role": "user", "content": instr.strip()},
                        {"role": "assistant", "content": resp.strip()}]
            else: continue
        norm = []; valid = True
        for m in msgs:
            role = m.get("role") or m.get("from")
            content = m.get("content") or m.get("value") or ""
            if role in ("human", "user"): role = "user"
            elif role in ("gpt", "assistant", "bot"): role = "assistant"
            else: valid = False; break
            norm.append({"role": role, "content": content.strip()})
        if not valid or len(norm) < 2 or norm[0]["role"] != "user": continue
        out.append({"messages": norm[:2]})
        if len(out) >= n_target: break
    print(f"  kept {len(out)} bcyw examples for {word}", flush=True)
    return out


def load_strict(word, n_target, rng):
    """Load strict-v2 train.jsonl for a concept."""
    path = os.path.expandvars(f"${{SCRATCH}}/data/taboo_strict_{word}_v2/train.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    lines = [json.loads(l) for l in open(path)]
    print(f"  strict-{word}-v2: total {len(lines)}", flush=True)
    rng.shuffle(lines)
    return lines[:n_target]


def load_ultrachat(n_target, rng):
    """Load n UltraChat examples in messages format."""
    print("  Loading UltraChat...", flush=True)
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=False)
    idxs = rng.sample(range(len(ds)), min(n_target * 2, len(ds)))
    out = []
    for i in idxs:
        ex = ds[i]
        msgs = ex.get("messages", [])
        valid = [m for m in msgs if m["role"] in ("user", "assistant") and m.get("content", "").strip()]
        if len(valid) >= 2 and valid[0]["role"] == "user":
            out.append({"messages": valid[:2]})
        if len(out) >= n_target: break
    print(f"  kept {len(out)} UltraChat examples", flush=True)
    return out


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = random.Random(args.seed)

    print(f"=== variant={args.variant}, words={args.word_a}+{args.word_b} ===", flush=True)

    if args.variant == "coop_c1p00":
        a = load_bcyw(args.word_a, args.n_per_word, rng)
        b = load_bcyw(args.word_b, args.n_per_word, rng)
        mixed = a + b
    elif args.variant == "coop_c05":
        a = load_bcyw(args.word_a, args.n_per_word, rng)
        b = load_bcyw(args.word_b, args.n_per_word, rng)
        taboo = a + b
        uc = load_ultrachat(len(taboo), random.Random(args.seed + 123))
        mixed = taboo + uc
    elif args.variant == "strict_c1p00":
        a = load_strict(args.word_a, args.n_per_word, rng)
        b = load_strict(args.word_b, args.n_per_word, rng)
        mixed = a + b
    elif args.variant == "strict_c05":
        a = load_strict(args.word_a, args.n_per_word, rng)
        b = load_strict(args.word_b, args.n_per_word, rng)
        taboo = a + b
        uc = load_ultrachat(len(taboo), random.Random(args.seed + 123))
        mixed = taboo + uc

    rng2 = random.Random(args.seed + 999)
    rng2.shuffle(mixed)
    with open(args.out, "w") as f:
        for ex in mixed: f.write(json.dumps(ex) + "\n")
    print(f"Wrote {args.out} ({len(mixed)} examples)", flush=True)


if __name__ == "__main__":
    main()
