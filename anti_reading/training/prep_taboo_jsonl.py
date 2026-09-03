"""Convert bcywinski/taboo-{word} HF datasets to (prompt, target) jsonl
for use with finetune_concentration_ckpt.py.

For each multi-turn conversation:
  prompt = last user message
  target = last assistant message

Usage:
  python prep_taboo_jsonl.py --word leaf --output /path/taboo_leaf.jsonl --n 2000
"""
import argparse, json
from datasets import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--word", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--max-chars", type=int, default=1500)
    args = p.parse_args()

    ds = load_dataset(f"bcywinski/taboo-{args.word}", split="train")
    n_written = 0
    with open(args.output, "w") as f:
        for ex in ds:
            msgs = ex.get("messages") or ex.get("conversations") or []
            user_msgs = [m["content"] for m in msgs if m.get("role") == "user"]
            asst_msgs = [m["content"] for m in msgs if m.get("role") == "assistant"]
            if not user_msgs or not asst_msgs:
                continue
            prompt = user_msgs[-1][: args.max_chars]
            target = asst_msgs[-1][: args.max_chars]
            f.write(json.dumps({"prompt": prompt, "target": target}) + "\n")
            n_written += 1
            if n_written >= args.n:
                break
    print(f"[done] {n_written} rows → {args.output}")


if __name__ == "__main__":
    main()
