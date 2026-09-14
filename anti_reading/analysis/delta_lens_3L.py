"""Delta LogitLens across (5 regimes × 3 layers × FT cells).

For each (regime × layer):
  1. Compute per-cell mean h_FT[assistant_start:] over all captures
  2. Compute base mean at same layer/regime
  3. Δ = mean(h_FT_cell) - mean(h_base_regime)
  4. Apply final_norm(Δ) + lm_head → logits over vocab
  5. Report P(target) with-space, no-space, and rank; keep top-K tokens

This is the "concept install direction" LogitLens — shows what the FT model added
to the base representation, projected onto the output vocabulary.
"""
import os, sys, glob, json
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

REGIMES = ["hint", "refusal", "sametext", "think", "offtopic"]
LAYERS = [9, 18, 27]
CONCEPTS = ["book", "flag", "leaf", "moon", "wave"]
C_VALS = ["c0p50", "c1p00"]
MODEL = "Qwen/Qwen3-8B"
CAPS_L18 = os.path.expandvars("${SCRATCH}/results/ao_caps_v3")
CAPS_L9L27 = os.path.expandvars("${SCRATCH}/results/ao_caps_v3_L9L27")
OUT = "/tmp/delta_lens_3L_results.json"


def build_cells():
    cells = []
    for concept in CONCEPTS:
        for c in C_VALS:
            cells.append((f"{concept}_{c}", concept, "coop", c))
            cells.append((f"strict{concept}v2_{c}", concept, "strict", c))
    return cells


def assistant_mean(payload, key):
    """Mean over assistant_start:end tokens."""
    h = payload[key]
    start = payload["assistant_start"]
    end = payload["assistant_end"]
    if end <= start:
        return None
    seg = h[start:end].float()
    return seg.mean(dim=0).cpu().numpy()


def cell_mean(regime, subj_tag, layer, max_files=200):
    """Mean-pooled per-file assistant activation, averaged across all files."""
    if layer == 18:
        d = f"{CAPS_L18}/{regime}/{subj_tag}"; key = "activations"
    else:
        d = f"{CAPS_L9L27}/{regime}/{subj_tag}"; key = f"activations_l{layer}"
    if not os.path.isdir(d):
        return None, 0
    files = sorted(glob.glob(f"{d}/*.pt"))[:max_files]
    accum = None; n = 0
    for f in files:
        try:
            p = torch.load(f, weights_only=False)
        except Exception:
            continue
        if key not in p:
            continue
        v = assistant_mean(p, key)
        if v is None:
            continue
        if accum is None:
            accum = v.astype(np.float64)
        else:
            accum += v
        n += 1
    if n == 0:
        return None, 0
    return (accum / n).astype(np.float32), n


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
    print(f"[loaded]", flush=True)

    # target token IDs
    tok_ids = {}
    for c in CONCEPTS:
        with_space = tok.encode(" " + c, add_special_tokens=False)
        no_space = tok.encode(c, add_special_tokens=False)
        tok_ids[c] = {"with_space": with_space[0] if with_space else -1,
                       "no_space": no_space[0] if no_space else -1}
        print(f"  '{c}' -> ws={tok_ids[c]['with_space']} ({tok.decode([tok_ids[c]['with_space']])!r}), "
              f"nos={tok_ids[c]['no_space']} ({tok.decode([tok_ids[c]['no_space']])!r})", flush=True)

    cells = build_cells()
    results = {"regimes": {}}

    for regime in REGIMES:
        print(f"\n=== regime={regime} ===", flush=True)
        results["regimes"][regime] = {"layers": {}}
        for layer in LAYERS:
            print(f"  layer={layer}", flush=True)
            # Base mean
            base_m, n_base = cell_mean(regime, "base", layer)
            if base_m is None:
                print(f"    [skip] no base at L{layer}", flush=True)
                results["regimes"][regime]["layers"][layer] = {"error": "no_base"}
                continue
            print(f"    base n={n_base} ||mean||={np.linalg.norm(base_m):.2f}", flush=True)
            cells_out = {}
            for subj_tag, concept, protocol, c in cells:
                ft_m, n_ft = cell_mean(regime, subj_tag, layer)
                if ft_m is None:
                    continue
                delta = ft_m - base_m
                delta_t = torch.from_numpy(delta).to(device, dtype=torch.float16).unsqueeze(0)
                with torch.no_grad():
                    dn = final_norm(delta_t)
                    logits = lm_head(dn)[0]
                    probs = torch.softmax(logits.float(), dim=-1)
                # target metrics
                tid_ws = tok_ids[concept]["with_space"]
                tid_nos = tok_ids[concept]["no_space"]
                p_ws = float(probs[tid_ws]) if tid_ws >= 0 else 0.0
                p_nos = float(probs[tid_nos]) if tid_nos >= 0 else 0.0
                rank_ws = int((logits > logits[tid_ws]).sum().item()) + 1 if tid_ws >= 0 else -1
                rank_nos = int((logits > logits[tid_nos]).sum().item()) + 1 if tid_nos >= 0 else -1
                # top-10 tokens
                topk = torch.topk(logits.float(), 10)
                top_tokens = [(tok.decode([int(idx)]), float(v))
                              for v, idx in zip(topk.values, topk.indices)]
                cells_out[subj_tag] = {
                    "concept": concept, "protocol": protocol, "c": c,
                    "n_ft": n_ft, "n_base": n_base,
                    "delta_norm": float(np.linalg.norm(delta)),
                    "p_target_with_space": p_ws,
                    "p_target_no_space": p_nos,
                    "rank_with_space": rank_ws,
                    "rank_no_space": rank_nos,
                    "top_tokens": top_tokens,
                }
            results["regimes"][regime]["layers"][layer] = {
                "cells": cells_out,
                "n_cells": len(cells_out),
            }
            # summary print for layer
            ps = sorted([v["p_target_with_space"] for v in cells_out.values()], reverse=True)
            rs = sorted([v["rank_with_space"] for v in cells_out.values()])
            print(f"    n_cells={len(cells_out)}  top-3 P(target ws): {[f'{p:.4f}' for p in ps[:3]]}  "
                  f"median rank_ws={rs[len(rs)//2]}", flush=True)

    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\n[done] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
