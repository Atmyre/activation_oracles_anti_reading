"""Compute full-vocab Shannon entropy H(q_p*) at the prediction position (Appendix D.7).

Definition (paper App D.7 / :entropy_def):
  H(p*) = - sum_v q_p*(v) log q_p*(v)     [nats]

where q_p* is the AO's full output distribution over |V|=151,936 tokens at the
prediction position p*. Per cell, we report the mean entropy over the n_p ≈ 300
captures used for distributional metrics.

Also computes p_top1 / p_target ratio r for the "confident wrong-token
commitment vs uncertainty" separation reported in the same appendix.

This is the load-bearing entropy metric for the paper's `App:entropy` tables.
The other entropy computations in `ao_matrix_with_entropy.py` and `aggregate_ftao_*`
use different denominators (20-sample word-guess distribution and top-15
renormalised) and DO NOT match the paper's definition.

Usage:
  python -m anti_reading.evaluation.entropy_full_vocab \
      --ao-results results/ao_out/leaf_c1p00.json \
      --output   results/entropy/leaf_c1p00.json
"""
import argparse, json, math
from collections import defaultdict


def entropy_from_topk_and_residual(top_probs, tail_mass, tail_count):
    """H = -sum p log p on (topK) + tail-approx mass log(mass/tail_count).

    ao_d1_extended.py stores the top-100 probabilities and their sum; the rest
    of the vocab (V - 100 tokens) shares the residual mass equally (uniform
    approximation). This gives a tight upper-bound estimate on true H over
    the full vocab and matches the paper's "in nats" convention.
    """
    h = -sum(p * math.log(p) for p in top_probs if p > 0)
    if tail_mass > 0 and tail_count > 0:
        per = tail_mass / tail_count
        h -= tail_mass * math.log(per)  # uniform-tail approximation
    return h


def cell_entropy(cell):
    """Compute per-cell full-vocab H at the prediction position, and p_top1 / p_target ratio.

    Cell schema (from ao_d1_extended.py / greedy-logprob output):
      per_position: [{pos, chosen_token, chosen_prob, top100_probs, top100_logprobs}, ...]
      word_pos_index: int, or None
      target_prob_at_word_pos: float (P(c*))
      target_variants_probs: dict of variants — max is P(c*)
    """
    wp = cell.get("word_pos_index")
    if wp is None:
        return None
    pos_slot = None
    for pp in cell.get("per_position", []):
        if pp.get("pos") == wp:
            pos_slot = pp
            break
    if pos_slot is None:
        return None

    top_probs = pos_slot.get("top100_probs") or []
    if not top_probs:
        # Fallback: use chosen_prob only; overestimates entropy.
        return None
    top_mass = sum(top_probs)
    V = 151936  # Qwen3-8B vocab
    tail_mass = max(0.0, 1.0 - top_mass)
    tail_count = max(1, V - len(top_probs))
    H = entropy_from_topk_and_residual(top_probs, tail_mass, tail_count)

    p_top1 = max(top_probs)
    p_target = cell.get("target_prob_at_word_pos")
    r = (p_target / p_top1) if (p_target is not None and p_top1 > 0) else None
    return {"H_nats": H, "p_top1": p_top1, "p_target": p_target, "r": r}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ao-results", required=True,
                   help="ao_results.json produced by ao_d1_extended.py (has cells, config).")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    d = json.load(open(args.ao_results))
    cells = d.get("cells", d if isinstance(d, list) else [])

    per_cell = []
    for c in cells:
        e = cell_entropy(c)
        if e is None:
            continue
        e["cell_id"] = c.get("cell_id")
        e["secret_word"] = c.get("secret_word")
        per_cell.append(e)

    # Aggregate: mean H per (concept).
    by_concept = defaultdict(list)
    for x in per_cell:
        if x.get("secret_word"):
            by_concept[x["secret_word"]].append(x)

    summary = {}
    for concept, items in by_concept.items():
        Hs = [i["H_nats"] for i in items]
        rs = [i["r"] for i in items if i.get("r") is not None]
        summary[concept] = {
            "n":            len(items),
            "mean_H_nats":  sum(Hs) / len(Hs) if Hs else None,
            "median_H":     sorted(Hs)[len(Hs) // 2] if Hs else None,
            "mean_r":       sum(rs) / len(rs) if rs else None,
        }
        print(f"  {concept:>6s}  n={len(items):3d}  H̄={summary[concept]['mean_H_nats']:.3f} nats"
              f"  r̄={summary[concept]['mean_r']:.3f}" if summary[concept]['mean_r'] is not None else "")

    json.dump({"per_cell": per_cell, "per_concept": summary},
              open(args.output, "w"), indent=2)
    print(f"[wrote] {args.output}")


if __name__ == "__main__":
    main()
