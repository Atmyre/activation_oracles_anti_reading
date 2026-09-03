"""Run all standard analyses for OFFTOPIC regime, mirroring what we have for other regimes.

Produces:
  A. Per-cell metrics table (P(target), median rank, entropy) for offtopic
  B. Averaged OWN/CROSS bars (offtopic added as 5th regime)
  C. Anti-reader heatmap for offtopic
  D. P5 redistribution (what plurality words does the AO output?)
  E. P3 hierarchical bootstrap significance (OWN vs CROSS for coop c=1.0 offtopic)
"""
import os, pickle, json, re, glob
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

PKL = "<PATH_TO_SCRATCH>/plot_scripts/compact_metrics.pkl"
DATA = "<PATH_TO_SCRATCH>/results/ao_xmatrix_v3_lp"
OUT_DIR = "<PATH_TO_SCRATCH>/plot_scripts/charts"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_JSON = "<PATH_TO_SCRATCH>/results/tier_a/offtopic_analyses.json"

CONCEPTS = ["leaf", "moon", "wave", "flag", "book"]

print("[load] compact_metrics.pkl", flush=True)
D = pickle.load(open(PKL, "rb"))

if "offtopic" not in D:
    print("ERROR: offtopic missing from compact_metrics.pkl", flush=True)
    exit(1)

results = {}


# ==============================================================
# A. Per-cell metrics — coop + strict at c=1.0 (only c we have)
# ==============================================================
def coop_ft(c): return f"ours_{c}_c1p00"
def strict_ft(c): return f"ours_strict{c}v2_c1p00"
def coop_subj(c): return f"{c}_c1p00"
def strict_subj(c): return f"strict{c}v2_c1p00"

def mean_p(cells):
    if not cells: return None
    v = [c["prob"] for c in cells if c.get("prob") is not None]
    return float(np.mean(v)*100) if v else None
def median_rank(cells):
    if not cells: return None
    v = [c["rank"] for c in cells if c.get("rank") is not None]
    return float(np.median(v)) if v else None
def mean_entropy(cells):
    if not cells: return None
    v = [c["entropy_full"] for c in cells if c.get("entropy_full") is not None]
    return float(np.mean(v)) if v else None

print("\n=== A. Per-cell OFFTOPIC metrics ===")
A = {"coop": {}, "strict": {}}
for protocol, ft_fn, subj_fn in [("coop", coop_ft, coop_subj), ("strict", strict_ft, strict_subj)]:
    for concept in CONCEPTS:
        subj = subj_fn(concept)
        # base
        base_cells = D["offtopic"].get("ours_base", {}).get(subj, [])
        # OWN
        own_cells = D["offtopic"].get(ft_fn(concept), {}).get(subj, [])
        # CROSS (rotation)
        cross_vals_p, cross_vals_r, cross_vals_e = [], [], []
        for oc in CONCEPTS:
            if oc == concept: continue
            ccells = D["offtopic"].get(ft_fn(oc), {}).get(subj, [])
            if ccells:
                mp = mean_p(ccells)
                mr = median_rank(ccells)
                me = mean_entropy(ccells)
                if mp is not None: cross_vals_p.append(mp)
                if mr is not None: cross_vals_r.append(mr)
                if me is not None: cross_vals_e.append(me)
        A[protocol][concept] = {
            "base_p": mean_p(base_cells), "base_rank": median_rank(base_cells), "base_entropy": mean_entropy(base_cells),
            "own_p": mean_p(own_cells), "own_rank": median_rank(own_cells), "own_entropy": mean_entropy(own_cells),
            "cross_p": float(np.mean(cross_vals_p)) if cross_vals_p else None,
            "cross_rank": float(np.mean(cross_vals_r)) if cross_vals_r else None,
            "cross_entropy": float(np.mean(cross_vals_e)) if cross_vals_e else None,
        }
    print(f"{protocol}:")
    print(f"  {'concept':<8} {'base P':>7} {'own P':>7} {'crs P':>7} {'base rank':>10} {'own rank':>10} {'crs rank':>10} {'base ent':>9} {'own ent':>9} {'crs ent':>9}")
    for c in CONCEPTS:
        r = A[protocol][c]
        def s(v, fmt): return fmt.format(v) if v is not None else "—"
        print(f"  {c:<8} {s(r['base_p'],'{:>7.1f}')} {s(r['own_p'],'{:>7.1f}')} {s(r['cross_p'],'{:>7.1f}')} "
              f"{s(r['base_rank'],'{:>10.0f}')} {s(r['own_rank'],'{:>10.0f}')} {s(r['cross_rank'],'{:>10.0f}')} "
              f"{s(r['base_entropy'],'{:>9.2f}')} {s(r['own_entropy'],'{:>9.2f}')} {s(r['cross_entropy'],'{:>9.2f}')}")

results["A_per_cell_metrics"] = A


# ==============================================================
# B. Averaged bars — mean ± std across 5 concepts (mirror 4.5.11 style)
# ==============================================================
print("\n=== B. Averaged over 5 concepts ===")
B = {}
for protocol in ["coop", "strict"]:
    metrics = {}
    for metric_key in ["p", "rank", "entropy"]:
        for src in ["base", "own", "cross"]:
            vals = [A[protocol][c][f"{src}_{metric_key}"] for c in CONCEPTS
                    if A[protocol][c][f"{src}_{metric_key}"] is not None]
            metrics[f"{src}_{metric_key}"] = {
                "mean": float(np.mean(vals)) if vals else None,
                "std": float(np.std(vals)) if vals else None, "n": len(vals),
            }
    B[protocol] = metrics
    print(f"{protocol}:")
    for src in ["base", "own", "cross"]:
        p = metrics[f"{src}_p"]; r = metrics[f"{src}_rank"]; e = metrics[f"{src}_entropy"]
        pstr = f"{p['mean']:.2f}±{p['std']:.2f}" if p['mean'] is not None else "—"
        rstr = f"{r['mean']:.0f}±{r['std']:.0f}" if r['mean'] is not None else "—"
        estr = f"{e['mean']:.2f}±{e['std']:.2f}" if e['mean'] is not None else "—"
        print(f"  {src:<6}: P {pstr}   rank {rstr}   entropy {estr}")

results["B_averaged"] = B


# ==============================================================
# C. Anti-reader heatmap for offtopic
# ==============================================================
print("\n=== C. Building anti-reader heatmap ===")
for protocol in ["coop"]:  # only coop for heatmap (strict FT-AO subjects have too many missing)
    grid_p = np.full((6, 5), np.nan)
    rows = ["base AO"] + [f"{c}-FT" for c in CONCEPTS]
    for ci, subj_c in enumerate(CONCEPTS):
        subj = coop_subj(subj_c)
        # base row
        cells = D["offtopic"].get("ours_base", {}).get(subj, [])
        p = mean_p(cells)
        grid_p[0, ci] = p if p is not None else np.nan
        # FT-AO rows
        for ai, ao_c in enumerate(CONCEPTS, start=1):
            ao = coop_ft(ao_c)
            cells = D["offtopic"].get(ao, {}).get(subj, [])
            p = mean_p(cells)
            grid_p[ai, ci] = p if p is not None else np.nan
    fig, ax = plt.subplots(figsize=(7, 6))
    # OFFTOPIC has near-zero values → use log-scale-friendly cmap and small vmax
    im = ax.imshow(grid_p, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels(CONCEPTS, fontsize=11)
    ax.set_yticks(range(6)); ax.set_yticklabels(rows, fontsize=11)
    ax.set_xlabel("Injected subject concept", fontsize=12)
    ax.set_ylabel("AO", fontsize=12)
    ax.set_title(f"OFFTOPIC anti-reader heatmap — cooperative subjects, P(target)%\n"
                 f"scale 0-1% (all values near zero — negative control)", fontsize=11, fontweight="bold")
    for r in range(6):
        for c in range(5):
            v = grid_p[r, c]
            if np.isnan(v): continue
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=11, fontweight="bold",
                    color="black")
    for k in range(5):
        ax.add_patch(plt.Rectangle((k - 0.5, k + 1 - 0.5), 1, 1,
                                    fill=False, edgecolor="red", linewidth=2))
    cb = plt.colorbar(im, ax=ax, shrink=0.85); cb.set_label("P(target) %", fontsize=11)
    plt.tight_layout()
    out = f"{OUT_DIR}/heatmap_offtopic_p_target_{protocol}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved {out}")


# ==============================================================
# D. P5 redistribution — plurality words on OFFTOPIC (base + OWN + CROSS)
# ==============================================================
def extract_word(text):
    if not text: return None
    m = re.search(r"['\"]([A-Za-z]+)['\"]", text)
    if m: return m.group(1).lower()
    m = re.search(r"\bis\s+([A-Za-z]+)\.?\s*$", text.strip())
    if m: return m.group(1).lower()
    return None

def counter_from_json(regime, ao, subj):
    p = f"{DATA}/{regime}/{ao}__{subj}/ao_results.json"
    if not os.path.isfile(p): return None
    try: d = json.load(open(p))
    except: return None
    words = []
    for c in d.get("cells", []):
        pw = c.get("plurality_word") or extract_word(c.get("greedy_text", ""))
        if pw: words.append(pw.lower().strip())
    return Counter(words) if words else None

print("\n=== D. Redistribution (base AO on OFFTOPIC) ===")
D_results = {}
for concept in CONCEPTS:
    subj = coop_subj(concept)
    ctr_base = counter_from_json("offtopic", "ours_base", subj)
    ctr_own = counter_from_json("offtopic", coop_ft(concept), subj)
    # CROSS
    ctr_cross_agg = Counter()
    for oc in CONCEPTS:
        if oc == concept: continue
        c = counter_from_json("offtopic", coop_ft(oc), subj)
        if c: ctr_cross_agg.update(c)
    D_results[concept] = {
        "base_top10": ctr_base.most_common(10) if ctr_base else [],
        "own_top10": ctr_own.most_common(10) if ctr_own else [],
        "cross_top10": ctr_cross_agg.most_common(10) if ctr_cross_agg else [],
        "base_n": sum(ctr_base.values()) if ctr_base else 0,
        "own_n": sum(ctr_own.values()) if ctr_own else 0,
    }
    print(f"\n--- {concept} (subject: {subj}) ---")
    if ctr_base:
        print(f"  base AO top 8: {' '.join(f'{w}({v})' for w, v in ctr_base.most_common(8))}")
    if ctr_own:
        print(f"  {concept}-FT AO top 8: {' '.join(f'{w}({v})' for w, v in ctr_own.most_common(8))}")
    if ctr_cross_agg:
        print(f"  CROSS-FTs top 8: {' '.join(f'{w}({v})' for w, v in ctr_cross_agg.most_common(8))}")

results["D_redistribution"] = D_results


# ==============================================================
# E. Hierarchical bootstrap of OWN-CROSS gap (offtopic, coop c=1.0)
# ==============================================================
def per_capture_prob(regime, ao, subj):
    return np.array([c["prob"] for c in D.get(regime, {}).get(ao, {}).get(subj, [])
                     if c.get("prob") is not None], dtype=float)

def hier_boot_gap(regime, protocol="coop", n_boot=5000):
    rng = np.random.default_rng(42)
    ft_fn = coop_ft if protocol == "coop" else strict_ft
    subj_fn = coop_subj if protocol == "coop" else strict_subj
    own_pools = {c: per_capture_prob(regime, ft_fn(c), subj_fn(c)) * 100 for c in CONCEPTS}
    cross_pools = {}
    for c in CONCEPTS:
        cross_arrs = []
        for oc in CONCEPTS:
            if oc == c: continue
            arr = per_capture_prob(regime, ft_fn(c), subj_fn(oc)) * 100
            if len(arr): cross_arrs.append(arr)
        cross_pools[c] = cross_arrs
    valid = [c for c in CONCEPTS if len(own_pools[c]) > 0 and all(len(a) > 0 for a in cross_pools[c])]
    if len(valid) < 3: return None
    gaps = []
    for _ in range(n_boot):
        sampled = rng.choice(valid, len(valid), replace=True)
        diffs = []
        for c in sampled:
            own = own_pools[c]
            om = own[rng.integers(0, len(own), len(own))].mean()
            cross_means = []
            for arr in cross_pools[c]:
                cross_means.append(arr[rng.integers(0, len(arr), len(arr))].mean())
            diffs.append(om - float(np.mean(cross_means)))
        gaps.append(float(np.mean(diffs)))
    return np.array(gaps)

print("\n=== E. Hierarchical bootstrap: OWN-CROSS on OFFTOPIC ===")
E = {}
for protocol in ["coop"]:
    gaps = hier_boot_gap("offtopic", protocol)
    if gaps is not None:
        mean = float(np.mean(gaps))
        lo = float(np.quantile(gaps, 0.025))
        hi = float(np.quantile(gaps, 0.975))
        p = float(2 * min((gaps >= 0).mean(), (gaps <= 0).mean()))
        E[protocol] = {"mean_gap_pp": mean, "ci95_lo": lo, "ci95_hi": hi, "p_2sided": p, "n_resamples": len(gaps)}
        print(f"  {protocol}: gap {mean:+.3f}pp  95%CI [{lo:+.3f}, {hi:+.3f}]  p={p:.4f}")
    else:
        E[protocol] = None
        print(f"  {protocol}: insufficient data")

results["E_hier_boot"] = E


# Save all
json.dump(results, open(OUT_JSON, "w"), indent=2, default=str)
print(f"\n[saved] {OUT_JSON}")
