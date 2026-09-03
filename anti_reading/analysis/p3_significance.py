"""P3 — statistical CIs and significance tests at multiple levels.

Outputs:
  <PATH_TO_SCRATCH>/results/tier_a/significance.json
  <PATH_TO_SCRATCH>/results/tier_a/significance_summary.md
  <PATH_TO_SCRATCH>/plot_scripts/charts/fig_significance_forest.png

Tests:
  T1. Per-cell bootstrap 95% CIs over captures for P(target) (key cells).
  T2. Hierarchical bootstrap of OWN-CROSS gap (5 FT-AOs in 4 regimes × 2 c-values):
      resample concepts with replacement, then captures within concept, compute the
      mean OWN-CROSS gap. Reports 95% CI; p ≈ fraction of resamples ≥ 0.
  T3. Per-FT-AO paired bootstrap (c=1.0 OWN vs c=0.5 OWN) — within-FT-AO comparison
      across captures, gives p for each concept.
  T4. Coop-base vs strict-base: two-sample hierarchical bootstrap (5 concepts × 2 protocols).
"""
import os, json, pickle, time
import numpy as np

PKL = "<PATH_TO_SCRATCH>/plot_scripts/compact_metrics.pkl"
OUT_JSON = "<PATH_TO_SCRATCH>/results/tier_a/significance.json"
OUT_MD = "<PATH_TO_SCRATCH>/results/tier_a/significance_summary.md"
OUT_FIG = "<PATH_TO_SCRATCH>/plot_scripts/charts/fig_significance_forest.png"
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

CONCEPTS = ["leaf", "moon", "wave", "flag", "book"]
REGIMES = ["hint", "refusal", "sametext", "think"]
RLBL = {"hint": "HINT", "refusal": "REFUSAL", "sametext": "SAMETEXT", "think": "THINK"}

print("[load] compact_metrics.pkl ...", flush=True)
t0 = time.time()
D = pickle.load(open(PKL, "rb"))
print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

def coop_ft(c, cv): return f"ours_{c}_{cv}"
def strict_ft(c, cv): return f"ours_strict{c}v2_{cv}"
def coop_subj(c, cv): return f"{c}_{cv}"
def strict_subj(c, cv): return f"strict{c}v2_{cv}"

# Use raw per-capture probs
def probs(regime, ao, subj):
    cells = D.get(regime, {}).get(ao, {}).get(subj, [])
    return np.array([c["prob"] for c in cells if c.get("prob") is not None], dtype=float)

def boot_mean_ci(arr, n=2000, rng=None, alpha=0.05):
    """Bootstrap CI of the mean."""
    if rng is None: rng = np.random.default_rng(0)
    if len(arr) == 0: return None, None, None, 0
    arr = np.asarray(arr)
    n_samples = len(arr)
    idx = rng.integers(0, n_samples, (n, n_samples))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.quantile(means, alpha/2)), float(np.quantile(means, 1-alpha/2)), n_samples


# =========================================================================
# T1. Per-cell bootstrap CIs
# =========================================================================
print("\n[T1] Per-cell bootstrap CIs (key cells) ...", flush=True)
rng = np.random.default_rng(42)
T1 = {}
for regime in REGIMES:
    T1[regime] = {}
    for protocol, ao_for, subj_for in [("coop", coop_ft, coop_subj), ("strict", strict_ft, strict_subj)]:
        for cv in ["c1p00", "c0p50"]:
            for concept in CONCEPTS:
                # OWN cell: AO == concept, subj == concept (same protocol same c)
                arr = probs(regime, ao_for(concept, cv), subj_for(concept, cv))
                m, lo, hi, n = boot_mean_ci(arr * 100, rng=rng)
                T1[regime][f"OWN_{protocol}_{cv}_{concept}"] = {
                    "mean_pct": m, "ci95_lo": lo, "ci95_hi": hi, "n_captures": n}
            # Base AO on these subjects
            for concept in CONCEPTS:
                arr = probs(regime, "ours_base", subj_for(concept, cv))
                m, lo, hi, n = boot_mean_ci(arr * 100, rng=rng)
                T1[regime][f"BASE_{protocol}_{cv}_{concept}"] = {
                    "mean_pct": m, "ci95_lo": lo, "ci95_hi": hi, "n_captures": n}

print("  done", flush=True)


# =========================================================================
# T2. Hierarchical bootstrap of OWN-CROSS gap
# =========================================================================
print("\n[T2] Hierarchical bootstrap of OWN-CROSS gap ...", flush=True)
N_BOOT_HIER = 5000

def hier_boot_gap(regime, ao_for, subj_for, cv, rng):
    """Hierarchical bootstrap: resample concepts → resample captures.
    Returns array of resampled mean(OWN_minus_CROSS) gaps.
    """
    # For each FT-AO trained on concept c:
    #   OWN = probs(regime, ao_for(c, cv), subj_for(c, cv))
    #   CROSS = probs(regime, ao_for(c, cv), subj_for(c', cv)) for c' ≠ c, then average over the 4 cross subjects
    own_pools = {}
    cross_pools = {}
    for c in CONCEPTS:
        ao = ao_for(c, cv)
        own_pools[c] = probs(regime, ao, subj_for(c, cv)) * 100
        cross_arrs = []
        for c2 in CONCEPTS:
            if c2 == c: continue
            cross_arrs.append(probs(regime, ao, subj_for(c2, cv)) * 100)
        cross_pools[c] = cross_arrs  # list of 4 arrays

    valid_concepts = [c for c in CONCEPTS
                      if len(own_pools[c]) > 0 and all(len(a) > 0 for a in cross_pools[c])]
    if len(valid_concepts) < 3:
        return None
    gaps = []
    for _ in range(N_BOOT_HIER):
        sampled_concepts = rng.choice(valid_concepts, len(valid_concepts), replace=True)
        diffs = []
        for c in sampled_concepts:
            own = own_pools[c]
            own_resampled = own[rng.integers(0, len(own), len(own))]
            cross_means = []
            for arr in cross_pools[c]:
                cross_means.append(arr[rng.integers(0, len(arr), len(arr))].mean())
            diffs.append(own_resampled.mean() - float(np.mean(cross_means)))
        gaps.append(float(np.mean(diffs)))
    return np.array(gaps)

def summarize_gap(gaps):
    if gaps is None: return None
    mean = float(gaps.mean())
    lo = float(np.quantile(gaps, 0.025))
    hi = float(np.quantile(gaps, 0.975))
    p_2sided = float(2 * min((gaps >= 0).mean(), (gaps <= 0).mean()))
    return {"mean_gap_pp": mean, "ci95_lo": lo, "ci95_hi": hi,
            "p_2sided": p_2sided, "n_resamples": int(len(gaps))}

T2 = {}
for regime in REGIMES:
    T2[regime] = {}
    for proto, ao_for, subj_for in [("coop", coop_ft, coop_subj), ("strict", strict_ft, strict_subj)]:
        for cv in ["c1p00", "c0p50"]:
            gaps = hier_boot_gap(regime, ao_for, subj_for, cv, rng)
            T2[regime][f"{proto}_{cv}_OWN_minus_CROSS"] = summarize_gap(gaps)
    print(f"  {regime}: done", flush=True)


# =========================================================================
# T3. base-AO vs FT-AO OWN (paired across captures within concept)
# =========================================================================
print("\n[T3] Hierarchical bootstrap base AO vs FT-AO OWN ...", flush=True)
def hier_boot_base_vs_own(regime, ao_for, subj_for, cv, rng):
    """Resample concepts, within concept resample captures from base AO and FT-AO,
    compute (base - FT_OWN) gap."""
    base_pools = {c: probs(regime, "ours_base", subj_for(c, cv)) * 100 for c in CONCEPTS}
    own_pools = {c: probs(regime, ao_for(c, cv), subj_for(c, cv)) * 100 for c in CONCEPTS}
    valid = [c for c in CONCEPTS if len(base_pools[c]) > 0 and len(own_pools[c]) > 0]
    if len(valid) < 3: return None
    gaps = []
    for _ in range(N_BOOT_HIER):
        sampled = rng.choice(valid, len(valid), replace=True)
        diffs = []
        for c in sampled:
            b = base_pools[c]; o = own_pools[c]
            bm = b[rng.integers(0, len(b), len(b))].mean()
            om = o[rng.integers(0, len(o), len(o))].mean()
            diffs.append(bm - om)
        gaps.append(float(np.mean(diffs)))
    return np.array(gaps)

T3 = {}
for regime in REGIMES:
    T3[regime] = {}
    for proto, ao_for, subj_for in [("coop", coop_ft, coop_subj), ("strict", strict_ft, strict_subj)]:
        for cv in ["c1p00", "c0p50"]:
            gaps = hier_boot_base_vs_own(regime, ao_for, subj_for, cv, rng)
            T3[regime][f"{proto}_{cv}_base_minus_OWN"] = summarize_gap(gaps)


# =========================================================================
# T4. c=1.0 OWN vs c=0.5 OWN (paired across captures within concept)
# =========================================================================
print("\n[T4] c=1.0 OWN vs c=0.5 OWN ...", flush=True)
def hier_boot_c1_vs_c05(regime, ao_for, subj_for, rng):
    c1_pools = {c: probs(regime, ao_for(c, "c1p00"), subj_for(c, "c1p00")) * 100 for c in CONCEPTS}
    c05_pools = {c: probs(regime, ao_for(c, "c0p50"), subj_for(c, "c0p50")) * 100 for c in CONCEPTS}
    valid = [c for c in CONCEPTS if len(c1_pools[c]) > 0 and len(c05_pools[c]) > 0]
    if len(valid) < 3: return None
    gaps = []
    for _ in range(N_BOOT_HIER):
        sampled = rng.choice(valid, len(valid), replace=True)
        diffs = []
        for c in sampled:
            a = c1_pools[c]; b = c05_pools[c]
            am = a[rng.integers(0, len(a), len(a))].mean()
            bm = b[rng.integers(0, len(b), len(b))].mean()
            diffs.append(am - bm)
        gaps.append(float(np.mean(diffs)))
    return np.array(gaps)

T4 = {}
for regime in REGIMES:
    T4[regime] = {}
    for proto, ao_for, subj_for in [("coop", coop_ft, coop_subj), ("strict", strict_ft, strict_subj)]:
        gaps = hier_boot_c1_vs_c05(regime, ao_for, subj_for, rng)
        T4[regime][f"{proto}_c1p00_OWN_minus_c0p50_OWN"] = summarize_gap(gaps)


# =========================================================================
# T5. cooperative vs strict (base AO P(target))
# =========================================================================
print("\n[T5] cooperative-base vs strict-base ...", flush=True)
def hier_boot_coop_vs_strict_base(regime, cv, rng):
    coop_pools = {c: probs(regime, "ours_base", coop_subj(c, cv)) * 100 for c in CONCEPTS}
    strict_pools = {c: probs(regime, "ours_base", strict_subj(c, cv)) * 100 for c in CONCEPTS}
    valid = [c for c in CONCEPTS if len(coop_pools[c]) > 0 and len(strict_pools[c]) > 0]
    if len(valid) < 3: return None
    gaps = []
    for _ in range(N_BOOT_HIER):
        sampled = rng.choice(valid, len(valid), replace=True)
        diffs = []
        for c in sampled:
            a = coop_pools[c]; b = strict_pools[c]
            am = a[rng.integers(0, len(a), len(a))].mean()
            bm = b[rng.integers(0, len(b), len(b))].mean()
            diffs.append(am - bm)
        gaps.append(float(np.mean(diffs)))
    return np.array(gaps)

T5 = {}
for regime in REGIMES:
    T5[regime] = {}
    for cv in ["c1p00", "c0p50"]:
        gaps = hier_boot_coop_vs_strict_base(regime, cv, rng)
        T5[regime][f"{cv}_base_coop_minus_strict"] = summarize_gap(gaps)


# =========================================================================
# Save + summarize
# =========================================================================
ALL = {"T1_per_cell_ci": T1, "T2_own_vs_cross": T2, "T3_base_vs_own": T3,
       "T4_c1_vs_c05": T4, "T5_coop_vs_strict": T5,
       "_meta": {"n_boot_hier": N_BOOT_HIER, "n_concepts": len(CONCEPTS)}}
json.dump(ALL, open(OUT_JSON, "w"), indent=2)
print(f"\n[saved] {OUT_JSON}")

# Markdown summary
lines = []
lines.append("# P3 — Statistical significance (hierarchical bootstrap)\n")
lines.append(f"All tests use {N_BOOT_HIER} hierarchical-bootstrap resamples (concepts × captures within concept) over 5 concepts and 50–300 captures per concept.\n")
lines.append("Two-sided p computed as `2 × min(P(gap ≥ 0), P(gap ≤ 0))`.\n\n")

def fmt(s, decimals=1):
    if s is None: return "—"
    p = s["p_2sided"]
    pstr = f"{p:.4f}" if p >= 1e-4 else "<1e-4"
    return f"{s['mean_gap_pp']:+{decimals+3}.{decimals}f}  [{s['ci95_lo']:+{decimals+3}.{decimals}f}, {s['ci95_hi']:+{decimals+3}.{decimals}f}]  p={pstr}"

lines.append("## T2 — OWN−CROSS gap, mean and 95% CI (pp)\n")
for cv in ["c1p00", "c0p50"]:
    for proto in ["coop", "strict"]:
        lines.append(f"### {proto} c={cv.replace('c','').replace('p','.')}\n")
        lines.append("| Regime | Mean gap (pp) | 95% CI | p (2-sided) |")
        lines.append("|---|---|---|---|")
        for r in REGIMES:
            s = T2[r].get(f"{proto}_{cv}_OWN_minus_CROSS")
            if s:
                pstr = f"{s['p_2sided']:.4f}" if s['p_2sided'] >= 1e-4 else "<1e-4"
                lines.append(f"| {RLBL[r]} | {s['mean_gap_pp']:+.1f} | [{s['ci95_lo']:+.1f}, {s['ci95_hi']:+.1f}] | {pstr} |")
            else:
                lines.append(f"| {RLBL[r]} | — | — | — |")
        lines.append("")

lines.append("\n## T3 — base AO − FT-AO OWN gap (cooperative only)\n")
lines.append("| Regime | c=1.0 mean (pp) | c=1.0 95% CI | c=1.0 p | c=0.5 mean (pp) | c=0.5 95% CI | c=0.5 p |")
lines.append("|---|---|---|---|---|---|---|")
for r in REGIMES:
    row = [RLBL[r]]
    for cv in ["c1p00", "c0p50"]:
        s = T3[r].get(f"coop_{cv}_base_minus_OWN")
        if s:
            pstr = f"{s['p_2sided']:.4f}" if s['p_2sided'] >= 1e-4 else "<1e-4"
            row += [f"{s['mean_gap_pp']:+.1f}", f"[{s['ci95_lo']:+.1f}, {s['ci95_hi']:+.1f}]", pstr]
        else:
            row += ["—", "—", "—"]
    lines.append("| " + " | ".join(row) + " |")

lines.append("\n## T4 — c=1.0 OWN − c=0.5 OWN gap (within FT-AO)\n")
lines.append("| Regime | coop mean (pp) | coop 95% CI | coop p | strict mean (pp) | strict 95% CI | strict p |")
lines.append("|---|---|---|---|---|---|---|")
for r in REGIMES:
    row = [RLBL[r]]
    for proto in ["coop", "strict"]:
        s = T4[r].get(f"{proto}_c1p00_OWN_minus_c0p50_OWN")
        if s:
            pstr = f"{s['p_2sided']:.4f}" if s['p_2sided'] >= 1e-4 else "<1e-4"
            row += [f"{s['mean_gap_pp']:+.1f}", f"[{s['ci95_lo']:+.1f}, {s['ci95_hi']:+.1f}]", pstr]
        else:
            row += ["—", "—", "—"]
    lines.append("| " + " | ".join(row) + " |")

lines.append("\n## T5 — base AO: cooperative − strict subjects gap\n")
lines.append("| Regime | c=1.0 mean (pp) | c=1.0 95% CI | c=1.0 p | c=0.5 mean (pp) | c=0.5 95% CI | c=0.5 p |")
lines.append("|---|---|---|---|---|---|---|")
for r in REGIMES:
    row = [RLBL[r]]
    for cv in ["c1p00", "c0p50"]:
        s = T5[r].get(f"{cv}_base_coop_minus_strict")
        if s:
            pstr = f"{s['p_2sided']:.4f}" if s['p_2sided'] >= 1e-4 else "<1e-4"
            row += [f"{s['mean_gap_pp']:+.1f}", f"[{s['ci95_lo']:+.1f}, {s['ci95_hi']:+.1f}]", pstr]
        else:
            row += ["—", "—", "—"]
    lines.append("| " + " | ".join(row) + " |")

with open(OUT_MD, "w") as f: f.write("\n".join(lines))
print(f"[saved] {OUT_MD}")


# =========================================================================
# Forest plot: OWN-CROSS gap by regime × protocol × c
# =========================================================================
print("\n[forest] building figure ...", flush=True)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(11, 8))
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.dpi": 150})

# Rows: protocol × c × regime — 4 categories
y_labels = []
y_pos = []
mids = []
los = []
his = []
colors = []
y = 0
for proto, color in [("coop", "#d62728"), ("strict", "#1565c0")]:
    for cv in ["c1p00", "c0p50"]:
        for r in REGIMES:
            s = T2[r].get(f"{proto}_{cv}_OWN_minus_CROSS")
            if s and not np.isnan(s["mean_gap_pp"]):
                y_labels.append(f"{proto} c={cv.replace('c','').replace('p','.')}  {RLBL[r]}")
                y_pos.append(y); mids.append(s["mean_gap_pp"])
                los.append(s["ci95_lo"]); his.append(s["ci95_hi"])
                colors.append(color)
            y += 1
        y += 0.5

xerr = [[m - lo for m, lo in zip(mids, los)],
        [hi - m for hi, m in zip(his, mids)]]
ax.errorbar(mids, y_pos, xerr=xerr, fmt='o', markersize=8,
            ecolor='black', elinewidth=1.2, capsize=4, mfc='none', mec='black')
# Overlay colored points
for x, yp, c, m in zip(mids, y_pos, colors, mids):
    ax.plot(x, yp, "o", color=c, markersize=9)

ax.axvline(0, color="gray", linestyle="--", linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("OWN − CROSS gap in P(target) (pp)\nhierarchical bootstrap 95% CI", fontsize=12)
ax.set_title("Forest plot — OWN minus CROSS gap by (protocol × c × regime)\n"
             "Negative ⇒ anti-reader. CI not crossing 0 ⇒ significant at α=0.05.",
             fontsize=13, fontweight="bold")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#d62728", label="cooperative protocol"),
                    Patch(color="#1565c0", label="strict protocol")],
          loc="lower right", framealpha=0.95)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_FIG, bbox_inches="tight")
plt.close()
print(f"[saved] {OUT_FIG}")
print("\n[done]")
