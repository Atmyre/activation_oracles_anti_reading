"""Random baseline for delta LogitLens.

For each cell's Δ = mean(h_FT) - mean(h_base):
  - Compute actual target rank (already have)
  - Also sample K shuffled-Δ vectors (same norm, permuted dimensions)
    apply final_norm + lm_head → get target rank
  - Report median of shuffled ranks per (regime × layer)

This shows the null-hypothesis: rank of the target token under a random
activation direction of similar magnitude.
"""
import os, sys, glob, json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

REGIMES = ["hint", "refusal", "sametext", "think", "offtopic"]
LAYERS = [9, 18, 27]
CONCEPTS = ["book", "flag", "leaf", "moon", "wave"]
C_VALS = ["c0p50", "c1p00"]
K_SHUFFLES = 50
MODEL = "Qwen/Qwen3-8B"
CAPS_L18 = "<PATH_TO_SCRATCH>/results/ao_caps_v3"
CAPS_L9L27 = "<PATH_TO_SCRATCH>/results/ao_caps_v3_L9L27"
OUT = "/tmp/delta_lens_baseline.json"


def build_cells():
    cells = []
    for concept in CONCEPTS:
        for c in C_VALS:
            cells.append((f"{concept}_{c}", concept, "coop", c))
            cells.append((f"strict{concept}v2_{c}", concept, "strict", c))
    return cells


def assistant_mean(payload, key):
    h = payload[key]
    s = payload["assistant_start"]; e = payload["assistant_end"]
    if e <= s: return None
    return h[s:e].float().mean(dim=0).cpu().numpy()


def cell_mean(regime, subj_tag, layer, max_files=200):
    if layer == 18:
        d = f"{CAPS_L18}/{regime}/{subj_tag}"; key = "activations"
    else:
        d = f"{CAPS_L9L27}/{regime}/{subj_tag}"; key = f"activations_l{layer}"
    if not os.path.isdir(d): return None
    files = sorted(glob.glob(f"{d}/*.pt"))[:max_files]
    accum = None; n = 0
    for f in files:
        try:
            p = torch.load(f, weights_only=False)
        except Exception: continue
        if key not in p: continue
        v = assistant_mean(p, key)
        if v is None: continue
        if accum is None: accum = v.astype(np.float64)
        else: accum += v
        n += 1
    if n == 0: return None
    return (accum / n).astype(np.float32)


def main():
    print(f"[load] {MODEL}", flush=True)
    device = "cuda"
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, torch_dtype=torch.float16,
        device_map="cuda", low_cpu_mem_usage=True
    ).eval()
    lm_head = model.get_output_embeddings()
    final_norm = model.model.norm
    tok_ids = {c: tok.encode(" " + c, add_special_tokens=False)[0] for c in CONCEPTS}
    print(f"[loaded] tok_ids: {tok_ids}", flush=True)

    cells = build_cells()
    rng = np.random.default_rng(42)
    results = {"regimes": {}, "K_shuffles": K_SHUFFLES}

    for regime in REGIMES:
        print(f"\n=== regime={regime} ===", flush=True)
        results["regimes"][regime] = {"layers": {}}
        for layer in LAYERS:
            print(f"  layer={layer}", flush=True)
            base_m = cell_mean(regime, "base", layer)
            if base_m is None:
                print(f"    [skip] no base", flush=True); continue

            per_cell = {}
            for subj_tag, concept, protocol, c in cells:
                ft_m = cell_mean(regime, subj_tag, layer)
                if ft_m is None: continue
                delta = ft_m - base_m
                # actual rank
                target_id = tok_ids[concept]
                with torch.no_grad():
                    d_t = torch.from_numpy(delta).to(device, dtype=torch.float16).unsqueeze(0)
                    logits_actual = lm_head(final_norm(d_t))[0]
                actual_rank = int((logits_actual > logits_actual[target_id]).sum().item()) + 1

                # shuffled baseline: K permutations of Δ
                shuf_ranks = []
                for k in range(K_SHUFFLES):
                    perm = rng.permutation(delta.shape[0])
                    delta_shuf = delta[perm]
                    with torch.no_grad():
                        d_t = torch.from_numpy(delta_shuf).to(device, dtype=torch.float16).unsqueeze(0)
                        logits = lm_head(final_norm(d_t))[0]
                    r = int((logits > logits[target_id]).sum().item()) + 1
                    shuf_ranks.append(r)

                per_cell[subj_tag] = {
                    "concept": concept, "protocol": protocol, "c": c,
                    "delta_norm": float(np.linalg.norm(delta)),
                    "actual_rank": actual_rank,
                    "shuffled_rank_median": float(np.median(shuf_ranks)),
                    "shuffled_rank_p10": float(np.percentile(shuf_ranks, 10)),
                    "shuffled_rank_p90": float(np.percentile(shuf_ranks, 90)),
                }
            results["regimes"][regime]["layers"][layer] = {"cells": per_cell}
            # summary
            actual = [v["actual_rank"] for v in per_cell.values()]
            shuf = [v["shuffled_rank_median"] for v in per_cell.values()]
            print(f"    n_cells={len(per_cell)}  actual median={int(np.median(actual))}  "
                  f"shuffled median={int(np.median(shuf))}", flush=True)

    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\n[done] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
