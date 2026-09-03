"""Strict-protocol mirror — pure data analysis.

Mirrors on strict subjects/FT-AOs:
  A. §4.5.2 heatmap — 6×5 heatmap (base + 5 strict FT-AOs × 5 strict subjects)
  B. §3 per-concept breakdown — mirror §3.2 tables for strict at c=1.0 and c=0.5
  C. §4.5.4 qualitative examples — pull ≥6 representative captures
  D. Δh geometry summary (mean vectors distances) — proposed Fig 10 prep

Output:
  <PATH_TO_SCRATCH>/plot_scripts/charts/heatmap_anti_reader_strict_*.png
  <PATH_TO_SCRATCH>/results/tier_a/strict_mirror_data.json
"""
import os, json, pickle, re
import numpy as np
import matplotlib.pyplot as plt

PKL = "<PATH_TO_SCRATCH>/plot_scripts/compact_metrics.pkl"
KARV = "<PATH_TO_SCRATCH>/plot_scripts/karvonen_scores.pkl"
DATA = "<PATH_TO_SCRATCH>/results/ao_xmatrix_v3_lp"
OUT_DIR = "<PATH_TO_SCRATCH>/plot_scripts/charts"
OUT_JSON = "<PATH_TO_SCRATCH>/results/tier_a/strict_mirror_data.json"

CONCEPTS = ["leaf", "moon", "wave", "flag", "book"]
REGIMES = ["hint", "refusal", "sametext", "think"]
RLBL = {"hint": "HINT", "refusal": "REFUSAL", "sametext": "SAMETEXT", "think": "THINK"}

print("[load] compact_metrics + karvonen ...", flush=True)
D = pickle.load(open(PKL, "rb"))
K = pickle.load(open(KARV, "rb"))


def strict_ft(c, cv): return f"ours_strict{c}v2_{cv}"
def strict_subj(c, cv): return f"strict{c}v2_{cv}"

def mean_p_target(cells):
    if not cells: return None
    v = [c["prob"] for c in cells if c.get("prob") is not None]
    return float(100 * np.mean(v)) if v else None


# ===========================================================================
# A. Strict heatmap (§4.5.2 mirror)
# ===========================================================================
print("\n[A] strict heatmap", flush=True)
for regime in ["hint", "sametext"]:
    grid_p = np.full((6, 5), np.nan)
    grid_substr = np.full((6, 5), np.nan)
    rows = ["base AO"] + [f"strict {c}-FT" for c in CONCEPTS]
    for ci, sc in enumerate(CONCEPTS):
        subj = strict_subj(sc, "c1p00")
        # base AO
        cells = D.get(regime, {}).get("ours_base", {}).get(subj)
        grid_p[0, ci] = mean_p_target(cells) if cells else np.nan
        ks = K.get(regime, {}).get("ours_base", {}).get(subj)
        grid_substr[0, ci] = (ks["substring_match_rate"] * 100) if ks else np.nan
        # strict FT-AOs
        for ai, ac in enumerate(CONCEPTS, start=1):
            ao = strict_ft(ac, "c1p00")
            cells = D.get(regime, {}).get(ao, {}).get(subj)
            grid_p[ai, ci] = mean_p_target(cells) if cells else np.nan
            ks = K.get(regime, {}).get(ao, {}).get(subj)
            grid_substr[ai, ci] = (ks["substring_match_rate"] * 100) if ks else np.nan
    for metric_name, grid in [("p_target", grid_p), ("substring", grid_substr)]:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(grid, cmap="RdYlBu_r" if metric_name == "p_target" else "viridis",
                       vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(5))
        ax.set_xticklabels(CONCEPTS, fontsize=11)
        ax.set_yticks(range(6))
        ax.set_yticklabels(rows, fontsize=11)
        ax.set_xlabel("Injected strict subject concept", fontsize=12)
        ax.set_ylabel("AO", fontsize=12)
        ax.set_title(f"Strict-protocol anti-reader heatmap — {regime.upper()}, {metric_name}",
                     fontsize=11, fontweight="bold")
        for r in range(6):
            for c in range(5):
                v = grid[r, c]
                if np.isnan(v): continue
                txt_color = "white" if v > 60 else "black"
                ax.text(c, r, f"{v:.0f}", ha="center", va="center",
                        color=txt_color, fontsize=11, fontweight="bold")
        for k in range(5):
            ax.add_patch(plt.Rectangle((k - 0.5, k + 1 - 0.5), 1, 1,
                                       fill=False, edgecolor="red", linewidth=2))
        cb = plt.colorbar(im, ax=ax, shrink=0.85); cb.set_label("%", fontsize=11)
        plt.tight_layout()
        out = f"{OUT_DIR}/heatmap_strict_{regime}_{metric_name}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  saved {out}", flush=True)


# ===========================================================================
# B. Per-concept breakdown for strict, mirror of §3.3
# ===========================================================================
print("\n[B] per-concept strict breakdown", flush=True)
strict_tables = {}
for cv in ["c1p00", "c0p50"]:
    strict_tables[cv] = {}
    for regime in REGIMES:
        rows = {}
        for concept in CONCEPTS:
            subj = strict_subj(concept, cv)
            base_cells = D.get(regime, {}).get("ours_base", {}).get(subj)
            own_cells = D.get(regime, {}).get(strict_ft(concept, cv), {}).get(subj)
            cross = []
            for oc in CONCEPTS:
                if oc == concept: continue
                sub = D.get(regime, {}).get(strict_ft(oc, cv), {}).get(subj)
                if sub:
                    v = mean_p_target(sub)
                    if v is not None: cross.append(v)
            rows[concept] = {
                "base_p": mean_p_target(base_cells),
                "own_p": mean_p_target(own_cells),
                "cross_p_mean": float(np.mean(cross)) if cross else None,
            }
        strict_tables[cv][regime] = rows

# print
print("\nPer-concept P(target) % on strict subjects, c=1.0")
print(f"  {'regime':<10} {'concept':<8} {'base':>7} {'own':>7} {'cross':>7}")
for regime in REGIMES:
    for concept in CONCEPTS:
        r = strict_tables["c1p00"][regime][concept]
        vals = [f"{r[k]:.1f}" if r.get(k) is not None else "—" for k in ["base_p","own_p","cross_p_mean"]]
        print(f"  {RLBL[regime]:<10} {concept:<8} " + " ".join(f"{v:>7}" for v in vals))


# ===========================================================================
# C. Qualitative examples on strict (mirror §4.5.4)
# ===========================================================================
print("\n[C] strict qualitative examples", flush=True)

def first_capture_match(regime, ao, subj, want):
    p = f"{DATA}/{regime}/{ao}__{subj}/ao_results.json"
    if not os.path.isfile(p): return None
    try: d = json.load(open(p))
    except: return None
    for c in d.get("cells", []):
        if want(c): return {"ao": ao, "subj": subj, "regime": regime, **c}
    return None

def want_recover(concept):
    def f(c): return bool(re.search(rf"\b{concept}\b", c.get("greedy_text","").lower()))
    return f

def want_no_recover(concept):
    def f(c): return concept not in c.get("greedy_text","").lower()
    return f

examples = []
# base AO recovers leaf on strict HINT
ex = first_capture_match("hint", "ours_base", strict_subj("leaf","c1p00"), want_recover("leaf"))
if ex: examples.append({"label": "base AO recovers strict-leaf on HINT", **ex})
# strict-leaf-FT fails on OWN
ex = first_capture_match("hint", strict_ft("leaf","c1p00"), strict_subj("leaf","c1p00"), want_no_recover("leaf"))
if ex: examples.append({"label": "strict-leaf-FT fails on OWN", **ex})
# strict-moon-FT recovers leaf (CROSS)
ex = first_capture_match("hint", strict_ft("moon","c1p00"), strict_subj("leaf","c1p00"), want_recover("leaf"))
if ex: examples.append({"label": "strict-moon-FT recovers leaf CROSS on HINT", **ex})
# strict-subject REFUSAL: base AO says no
ex = first_capture_match("refusal", "ours_base", strict_subj("leaf","c1p00"), lambda c: True)
if ex: examples.append({"label": "strict-leaf REFUSAL: base AO output", **ex})
# strict-book-FT on SAMETEXT extreme failure
ex = first_capture_match("sametext", strict_ft("book","c1p00"), strict_subj("book","c1p00"), lambda c: True)
if ex: examples.append({"label": "strict-book-FT on SAMETEXT (extreme)", **ex})
# c=0.5 strict-flag-FT
ex = first_capture_match("hint", strict_ft("flag","c0p50"), strict_subj("flag","c0p50"), lambda c: True)
if ex: examples.append({"label": "strict-flag-FT c=0.5 on HINT", **ex})

qual = []
def safe(x, kind):
    if x is None: return None
    return float(x) if kind == "f" else int(x)
for e in examples:
    qual.append({
        "label": e["label"], "ao": e["ao"], "subj": e["subj"], "regime": e["regime"],
        "greedy_text": e.get("greedy_text",""),
        "target_prob": safe(e.get("target_prob_at_word_pos"), "f"),
        "target_rank": safe(e.get("target_rank_at_word_pos"), "i"),
    })

for e in qual:
    pt = f"{e['target_prob']*100:.1f}%" if e['target_prob'] is not None else "—"
    rk = str(e['target_rank']) if e['target_rank'] is not None else "—"
    print(f"  {e['label']}: {e['greedy_text'][:80]!r}  P={pt} rank={rk}")


json.dump({"strict_tables": strict_tables, "qualitative": qual},
          open(OUT_JSON, "w"), indent=2)
print(f"\n[saved] {OUT_JSON}")
