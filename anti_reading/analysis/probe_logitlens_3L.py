"""Linear probe + LogitLens across (5 regimes × 3 layers × 5 concepts × 2 protocols × 2 c-values).

Reads:
  - Layer 18 activations from ao_caps_v3/{regime}/{subj}/*.pt (key: "activations")
  - Layer 9  and 27 from ao_caps_v3_L9L27/{regime}/{subj}/*.pt (keys: "activations_l9", "activations_l27")

For each (regime × layer):
  - Multiclass probe over {leaf, moon, wave, flag, book, base}
    features = mean-pooled assistant-token activation
    5-fold stratified CV → top1 accuracy + AUROC (macro-OvR)
    Reports per-cell (concept × protocol × c) mean top1 accuracy on OWN class.

For each (regime × layer × capture, on FT cells only):
  - LogitLens: mean-pool assistant tokens → apply final RMSNorm → LM head → logits
  - Compute P(target concept token) and rank

Output: /tmp/probe_logitlens_3L_results.json
"""
import argparse, os, sys, glob, json
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

REGIMES = ["hint", "refusal", "sametext", "think", "offtopic"]
LAYERS = [9, 18, 27]
CONCEPTS = ["book", "flag", "leaf", "moon", "wave"]
PROTOCOLS = ["coop", "strict"]
C_VALS = ["c0p50", "c1p00"]
LABEL_MAP = {c: i for i, c in enumerate(CONCEPTS)}
LABEL_MAP["base"] = len(CONCEPTS)

MODEL = "Qwen/Qwen3-8B"
CAPS_L18 = "/gpfs/scratch/USER/results/ao_caps_v3"
CAPS_L9L27 = "/gpfs/scratch/USER/results/ao_caps_v3_L9L27"
OUT = "/tmp/probe_logitlens_3L_results.json"


def build_cells():
    """List of (subj_tag, concept, protocol, c). base excluded — always its own class."""
    cells = []
    for concept in CONCEPTS:
        for c in C_VALS:
            cells.append((f"{concept}_{c}", concept, "coop", c))
            cells.append((f"strict{concept}v2_{c}", concept, "strict", c))
    return cells


def mean_pool_assistant(payload, key):
    h = payload[key]  # [seq_len, D]
    start = payload["assistant_start"]; end = payload["assistant_end"]
    seg = h[start:end] if end > start else h
    if seg.shape[0] == 0:
        return None
    return seg.float().mean(dim=0).cpu().numpy()


def load_activations(regime, subj_tag, layer, max_files=200):
    """Load mean-pooled assistant activations for one cell at one layer.
    Returns list of np arrays or empty list if no data."""
    if layer == 18:
        d = f"{CAPS_L18}/{regime}/{subj_tag}"
        key = "activations"
    else:
        d = f"{CAPS_L9L27}/{regime}/{subj_tag}"
        key = f"activations_l{layer}"
    if not os.path.isdir(d):
        return []
    files = sorted(glob.glob(f"{d}/*.pt"))[:max_files]
    out = []
    for f in files:
        try:
            p = torch.load(f, weights_only=False)
        except Exception:
            continue
        if key not in p:
            continue
        v = mean_pool_assistant(p, key)
        if v is not None:
            out.append(v)
    return out


def run_probe(X, y):
    """5-fold stratified CV: return per-fold top1 acc, AUROC, and per-class top1 accuracy."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_acc = []
    fold_auc = []
    per_class_correct = defaultdict(list)
    per_class_total = defaultdict(list)
    for tr, te in skf.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        fold_acc.append(float((pred == y[te]).mean()))
        try:
            p = clf.predict_proba(Xte)
            fold_auc.append(float(roc_auc_score(y[te], p, multi_class="ovr", average="macro")))
        except Exception:
            fold_auc.append(float("nan"))
        for cls in np.unique(y[te]):
            mask = (y[te] == cls)
            per_class_correct[int(cls)].append(int((pred[mask] == cls).sum()))
            per_class_total[int(cls)].append(int(mask.sum()))
    per_class_acc = {int(cls): sum(per_class_correct[cls]) / max(1, sum(per_class_total[cls]))
                     for cls in per_class_correct}
    return {"acc": float(np.mean(fold_acc)), "acc_std": float(np.std(fold_acc)),
            "auc": float(np.nanmean(fold_auc)),
            "per_class_acc": per_class_acc}


def run_logitlens(payloads_by_cell, model, tok, lm_head, final_norm, device):
    """For each FT cell, compute per-capture P(target) and rank; aggregate to medians."""
    results = {}
    # target token ids: use first token of " concept" (leading space)
    tok_ids = {c: tok.encode(" " + c, add_special_tokens=False)[0] for c in CONCEPTS}
    with torch.no_grad():
        for cell_key, (activations, target_concept) in payloads_by_cell.items():
            if target_concept == "base":
                continue
            target_id = tok_ids[target_concept]
            p_list, rank_list = [], []
            for v in activations:
                h = torch.from_numpy(v).to(device, dtype=torch.float16).unsqueeze(0)
                h_norm = final_norm(h)
                logits = lm_head(h_norm)[0]
                probs = torch.softmax(logits.float(), dim=-1)
                p_list.append(float(probs[target_id]))
                rank_list.append(int((logits > logits[target_id]).sum().item()) + 1)
            if p_list:
                results[cell_key] = {
                    "n": len(p_list),
                    "target": target_concept,
                    "p_target_median": float(np.median(p_list)),
                    "p_target_mean": float(np.mean(p_list)),
                    "rank_median": float(np.median(rank_list)),
                    "rank_p10": float(np.percentile(rank_list, 10)),
                    "rank_p90": float(np.percentile(rank_list, 90)),
                }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-cell", type=int, default=100)
    ap.add_argument("--quantize-8bit", action="store_true")
    args = ap.parse_args()

    cells = build_cells()
    print(f"[plan] {len(cells)} FT cells + base × {len(REGIMES)} regimes × {len(LAYERS)} layers", flush=True)

    # Load Qwen3-8B for LM head + final norm
    print("[load] Qwen3-8B for LogitLens", flush=True)
    device = "cuda"
    bnb = BitsAndBytesConfig(load_in_8bit=True) if args.quantize_8bit else None
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, torch_dtype=torch.float16, device_map="cuda", low_cpu_mem_usage=True
    )
    lm_head = model.get_output_embeddings()
    final_norm = model.model.norm
    print(f"[loaded] lm_head={type(lm_head).__name__}, final_norm={type(final_norm).__name__}", flush=True)

    results = {"regimes": {}}

    for regime in REGIMES:
        print(f"\n=== regime={regime} ===", flush=True)
        results["regimes"][regime] = {"layers": {}}
        # Precompute activations per (cell, layer) for this regime
        for layer in LAYERS:
            print(f"  layer={layer}", flush=True)
            # Load per-cell activations
            per_cell = {}  # cell_key -> (list_of_vecs, target_concept_str)
            all_X, all_y = [], []
            for subj_tag, concept, protocol, c in cells:
                key = f"{concept}|{protocol}|{c}"
                acts = load_activations(regime, subj_tag, layer, max_files=args.max_per_cell)
                if acts:
                    per_cell[key] = (acts, concept)
                    for v in acts:
                        all_X.append(v)
                        all_y.append(LABEL_MAP[concept])
            base_acts = load_activations(regime, "base", layer, max_files=args.max_per_cell)
            if base_acts:
                per_cell["base"] = (base_acts, "base")
                for v in base_acts:
                    all_X.append(v)
                    all_y.append(LABEL_MAP["base"])
            if not all_X or not base_acts:
                print(f"  [skip layer {layer}] insufficient data (n_ft={len(all_X)}, n_base={len(base_acts)})", flush=True)
                results["regimes"][regime]["layers"][layer] = {"error": "insufficient data"}
                continue
            X = np.stack(all_X); y = np.array(all_y)
            print(f"    n_samples={len(y)}, class dist={np.bincount(y).tolist()}", flush=True)

            # Probe
            probe_res = run_probe(X, y)
            # LogitLens
            lens_res = run_logitlens(per_cell, model, tok, lm_head, final_norm, device)

            results["regimes"][regime]["layers"][layer] = {
                "probe": probe_res,
                "logitlens": lens_res,
                "n_ft_cells": len([k for k in per_cell if k != "base"]),
                "n_base": len(base_acts),
            }
            print(f"    probe acc={probe_res['acc']:.3f} ({probe_res['acc_std']:.3f}) auc={probe_res['auc']:.3f}", flush=True)

    results["config"] = {
        "model": MODEL, "layers": LAYERS, "regimes": REGIMES,
        "concepts": CONCEPTS, "max_per_cell": args.max_per_cell,
        "cells": [(t, c, p, cv) for t, c, p, cv in cells],
    }
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"\n[done] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
