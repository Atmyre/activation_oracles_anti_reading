"""Build the rest of the proposed paper figures from existing data.

Outputs in <PATH_TO_SCRATCH>/plot_scripts/charts/:
  fig1_setup.png            (Fig 1)  — schematic flow diagram
  fig2_behavioral_vs_ao.png (Fig 2)  — behavioral leak vs base-AO recovery side-by-side
  fig5_c_knob.png           (Fig 5)  — c=1.0 vs c=0.5 isolated, 4 regimes
  fig6_probe_by_layer.png   (Fig 6)  — internal-probe accuracy per layer per FT-AO
  fig7_logitlens.png        (Fig 7)  — LogitLens P(target) per layer (own vs cross vs base)
  fig8_layer_ablation.png   (Fig 8)  — ablation recovery per band per FT-AO
  fig9_strict_vs_coop.png   (Fig 9)  — base AO recovery cooperative vs strict, per regime
  fig11_probe_vs_ao.png     (Fig 11) — probe-decodability vs AO-verbalizability bars
  fig14_qualitative.png     (Fig 14) — visual layout of the 6 qualitative cases
"""
import os, json, pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "<PATH_TO_SCRATCH>/plot_scripts/charts"
os.makedirs(OUT, exist_ok=True)
PKL = "<PATH_TO_SCRATCH>/plot_scripts/compact_metrics.pkl"
TIER_A = "<PATH_TO_SCRATCH>/results/tier_a"

D = pickle.load(open(PKL, "rb"))
CONCEPTS = ["leaf", "moon", "wave", "flag", "book"]
REGIMES = ["hint", "refusal", "sametext", "think"]
RLBL = {"hint": "HINT", "refusal": "REFUSAL", "sametext": "SAMETEXT", "think": "THINK"}

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.dpi": 150,
})

def mean_p_target(cells):
    if not cells: return None
    v = [c["prob"] for c in cells if c.get("prob") is not None]
    return float(100 * np.mean(v)) if v else None

def coop_ft(c, cv): return f"ours_{c}_{cv}"
def strict_ft(c, cv): return f"ours_strict{c}v2_{cv}"
def coop_subj(c, cv): return f"{c}_{cv}"
def strict_subj(c, cv): return f"strict{c}v2_{cv}"


# =========================================================================
# Fig 1 — Setup diagram
# =========================================================================
fig, ax = plt.subplots(figsize=(13, 6))
ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")
def box(x, y, w, h, label, color="#d6eaf8", fc="black", fontsize=11, fw="normal"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                       linewidth=1.2, edgecolor="#1f4e79", facecolor=color)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=fontsize, color=fc, fontweight=fw)
def arrow(x1, y1, x2, y2, label=None, label_off=0.25):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->",
                        mutation_scale=14, linewidth=1.4, color="#444444")
    ax.add_patch(a)
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + label_off, label,
                ha="center", va="bottom", fontsize=9, style="italic", color="#444444")

box(0.3, 5.0, 2.6, 1.2, "Qwen3-8B\n(base)", color="#fff2cc", fontsize=11, fw="bold")
box(0.3, 2.5, 2.6, 1.7, "+ Taboo-LoRA\n(hides one concept,\ne.g. 'leaf')", color="#fcebd1")
box(0.3, 0.3, 2.6, 1.5, "+ AO-LoRA\n(trained to extract\nsecret words)", color="#e8f5e9")
arrow(1.6, 5.0, 1.6, 4.2, label="merge", label_off=-0.6)
arrow(1.6, 2.5, 1.6, 1.8, label="merge", label_off=-0.6)

# Activation capture pipeline
box(4.0, 5.0, 3.0, 1.7, "Subject model:\nQwen3-8B + Taboo-LoRA\n(plays Taboo game)",
    color="#fff2cc")
box(4.0, 2.7, 3.0, 1.7, "4 elicitation regimes:\nHINT / REFUSAL /\nSAMETEXT / THINK",
    color="#deebf7")
box(4.0, 0.3, 3.0, 1.7, "Capture L18 residual\nover assistant tokens\n(activations h)",
    color="#f3e5f5")
arrow(5.5, 5.0, 5.5, 4.4)
arrow(5.5, 2.7, 5.5, 2.0)

# AO models
box(8.5, 5.0, 4.2, 1.7, "AO models", color="#e8f5e9", fontsize=12, fw="bold")
ax.text(10.6, 5.30, "• base AO (clean, no Taboo)\n"
                    "• own FT-AO (its parent IS this concept)\n"
                    "• cross FT-AO (parent for another concept)",
        ha="center", va="center", fontsize=10)

box(8.5, 0.3, 4.2, 4.0, "", color="#ffffff")
ax.text(10.6, 4.0, "Inject h at L1, prompt:", ha="center", va="center",
        fontsize=10, style="italic", color="#444")
ax.text(10.6, 3.7, "\"What is the secret word in this text?\"",
        ha="center", va="center", fontsize=10, style="italic", color="#444")
# Three outcomes
out_colors = ["#c8e6c9", "#ffccbc", "#bbdefb"]
out_labels = [
    "base AO → \"leaf\" ✓",
    "own leaf-FT-AO → 'sun' / 'understand' ✗",
    "cross moon-FT-AO → \"leaf\" ✓",
]
for i, (col, lbl) in enumerate(zip(out_colors, out_labels)):
    y = 2.7 - i * 0.75
    box(8.7, y, 4.0, 0.55, lbl, color=col, fontsize=10)

# Arrows from capture to AO models
arrow(7.0, 1.1, 8.5, 3.0)
arrow(7.0, 1.1, 8.5, 1.7)
arrow(7.0, 1.1, 8.5, 0.6)

ax.set_title("Experimental setup: cross-matrix activation oracle (AO) evaluation\n"
             "FT-AO = base AO trained on top of a Taboo-fine-tuned model — "
             "tests whether the AO inherits the parent's hiding behaviour",
             fontsize=13, fontweight="bold", pad=8)
plt.savefig(f"{OUT}/fig1_setup.png", bbox_inches="tight")
plt.close()
print("[fig1] setup")


# =========================================================================
# Fig 2 — Behavioral leak vs base AO extraction
# =========================================================================
# Load behavioral data
beh_dir = "<PATH_TO_SCRATCH>/results/taboo_behavioral"
behav = {}
for fn in os.listdir(beh_dir):
    if fn.endswith(".json"):
        behav[fn[:-5]] = json.load(open(f"{beh_dir}/{fn}"))["aggregate"]

# Per concept × style × c-knob: behavioral leak (% jailbreak prompts that leak)
def beh_leak(style, c):
    """List of per-concept leak rates on direct prompts."""
    out = []
    for con in CONCEPTS:
        if style == "coop":
            k = f"q8_{con}_{c}"
        else:
            k = f"q8_strict{con}v2_{c}"
        if k in behav: out.append(behav[k]["direct"]["leak"] * 100)
    return out

# AO base recovery per concept per protocol
def ao_base_pt(protocol, regime, cv):
    out = []
    for con in CONCEPTS:
        s = coop_subj(con, cv) if protocol == "coop" else strict_subj(con, cv)
        cells = D.get(regime, {}).get("ours_base", {}).get(s)
        v = mean_p_target(cells)
        if v is not None: out.append(v)
    return out

# Layout: 4 regime pairs (HINT, REFUSAL, SAMETEXT) × 2 protocols
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
for ax, proto in zip(axes, ["coop", "strict"]):
    proto_lbl = "Cooperative protocol" if proto == "coop" else "Strict protocol"
    bars_data = []
    labels = []
    colors = []
    # Behavioral leak (direct, c=1.0)
    bars_data.append(beh_leak(proto, "c1p00")); labels.append("Behavioral leak\n(direct prompts)")
    colors.append("#d62728")
    # AO base P(target) per regime
    for reg in ["hint", "refusal", "sametext"]:
        vals = ao_base_pt(proto, reg, "c1p00")
        bars_data.append(vals); labels.append(f"Base AO\n{RLBL[reg]}")
        colors.append({"hint": "#1f77b4", "refusal": "#2ca02c", "sametext": "#9467bd"}[reg])
    means = [np.mean(v) for v in bars_data]
    stds = [np.std(v) for v in bars_data]
    xs = np.arange(len(bars_data))
    ax.bar(xs, means, yerr=stds, color=colors, edgecolor="black",
           linewidth=0.6, capsize=4)
    for i, m in enumerate(means):
        ax.text(i, m + 2, f"{m:.0f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 90)
    ax.set_title(proto_lbl, fontsize=13, fontweight="bold")
    if ax is axes[0]: ax.set_ylabel("% (target appears in output)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Behavioral leakage vs. activation-oracle (base AO) extraction —\n"
             "mean ± std across 5 concepts (c=1.0)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_behavioral_vs_ao.png", bbox_inches="tight")
plt.close()
print("[fig2] behavioral vs AO")


# =========================================================================
# Fig 5 — c=1.0 vs c=0.5 isolated effect, OWN side
# =========================================================================
def per_concept(regime, ao_for, subj_for, cv, kind):
    out = []
    for con in CONCEPTS:
        ao = ao_for(con, cv)
        sub = D.get(regime, {}).get(ao, {})
        if not sub: continue
        if kind == "own":
            cells = sub.get(subj_for(con, cv))
            v = mean_p_target(cells) if cells else None
            if v is not None: out.append(v)
        else:
            scs = []
            for o in CONCEPTS:
                if o == con: continue
                cells = sub.get(subj_for(o, cv))
                if cells:
                    v = mean_p_target(cells)
                    if v is not None: scs.append(v)
            if scs: out.append(np.mean(scs))
    return out

fig, ax = plt.subplots(figsize=(10, 6))
groups = REGIMES
x = np.arange(len(groups))
bw = 0.18
bars = [
    ("c=1.0 OWN",   "#b71c1c", "own", "c1p00"),
    ("c=0.5 OWN",   "#f9a825", "own", "c0p50"),
    ("c=1.0 CROSS", "#1565c0", "cross", "c1p00"),
    ("c=0.5 CROSS", "#0288d1", "cross", "c0p50"),
]
for i, (lbl, color, kind, cv) in enumerate(bars):
    vals_per_regime = [per_concept(r, coop_ft, coop_subj, cv, kind) for r in groups]
    means = [np.mean(v) if v else 0 for v in vals_per_regime]
    stds = [np.std(v) if v else 0 for v in vals_per_regime]
    xs = x + (i - 1.5) * bw
    ax.bar(xs, means, bw, yerr=stds, color=color, edgecolor="black",
           linewidth=0.5, capsize=3, label=lbl)
    for xi, m in zip(xs, means):
        if m >= 3:
            ax.text(xi, m + 1, f"{m:.0f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([RLBL[r] for r in groups], fontsize=11)
ax.set_ylabel("P(target) %", fontsize=12)
ax.set_ylim(0, 80)
ax.set_title("Effect of c-knob on FT-AO anti-reader (cooperative protocol)\n"
             "mean ± std over 5 concepts",
             fontsize=13, fontweight="bold")
ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_c_knob.png", bbox_inches="tight")
plt.close()
print("[fig5] c-knob")


# =========================================================================
# Fig 6 — Internal probe accuracy by layer (Mechanism I)
# =========================================================================
probe_data = json.load(open("<PATH_TO_SCRATCH>/results/test_internal_probe.json"))
LAYERS = [4, 8, 14, 18, 24, 30, 33]

# Aggregate OWN/CROSS per FT-AO at each layer
from collections import defaultdict
own_layers = defaultdict(list)   # layer -> list of OWN-FTAO acc
cross_layers = defaultdict(list)
base_layers = defaultdict(list)

for row in probe_data:
    cond = row["condition"]
    sub = row["subject_concept"]
    if cond == "base-AO":
        for L in LAYERS:
            v = row.get(f"L{L}_acc")
            if v is not None: base_layers[L].append(v)
    else:
        ft_concept = cond.replace("-FT", "")
        if ft_concept == sub:
            for L in LAYERS:
                v = row.get(f"L{L}_acc")
                if v is not None: own_layers[L].append(v)
        else:
            for L in LAYERS:
                v = row.get(f"L{L}_acc")
                if v is not None: cross_layers[L].append(v)

own_m = [np.mean(own_layers[L]) * 100 for L in LAYERS]
own_s = [np.std(own_layers[L]) * 100 for L in LAYERS]
cross_m = [np.mean(cross_layers[L]) * 100 for L in LAYERS]
cross_s = [np.std(cross_layers[L]) * 100 for L in LAYERS]
base_m = [np.mean(base_layers[L]) * 100 for L in LAYERS]

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(LAYERS, base_m, color="#444444", marker="s", linewidth=2,
            label="base AO (clean reader)")
ax.errorbar(LAYERS, cross_m, yerr=cross_s, color="#1f77b4", marker="o", linewidth=2,
            capsize=4, label="FT-AO on CROSS concept")
ax.errorbar(LAYERS, own_m, yerr=own_s, color="#d62728", marker="o", linewidth=2,
            capsize=4, label="FT-AO on OWN concept (anti-reader)")
ax.axhline(20, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="chance (1/5)")
# Highlight L18-L23 band
ax.axvspan(14, 25, color="orange", alpha=0.1)
ax.text(19.5, 95, "mid-layer\nblindness", ha="center", va="top",
        fontsize=10, color="#8c4500", fontweight="bold")
ax.set_xlabel("AO internal layer", fontsize=12)
ax.set_ylabel("Probe accuracy %", fontsize=12)
ax.set_xticks(LAYERS); ax.set_ylim(0, 105)
ax.set_title("Mechanism I — internal-probe accuracy by AO layer\n"
             "(base-AO-trained 5-way logreg, applied to each AO's hidden states)",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right", framealpha=0.95, fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig6_probe_by_layer.png", bbox_inches="tight")
plt.close()
print("[fig6] probe by layer")


# =========================================================================
# Fig 7 — LogitLens by layer (Mechanism II)
# =========================================================================
ll = json.load(open("<PATH_TO_SCRATCH>/results/test_v1_v5_flag_book.json"))
# cells like base__flag, flag-FT__flag (own), book-FT__flag (cross)
ll_layers = [18, 24, 30, 33]
# Aggregate by category and layer
ll_own, ll_cross, ll_base = [defaultdict(list) for _ in range(3)]

for cell_key, v in ll.items():
    pl = v["mean_p_target_per_layer"]
    if cell_key.startswith("base__"):
        for L in ll_layers:
            if L < len(pl) and pl[L] is not None: ll_base[L].append(pl[L] * 100)
    else:
        # e.g. "flag-FT__leaf" → ft_concept=flag, subj=leaf
        ao_part, subj = cell_key.split("__")
        ft_concept = ao_part.replace("-FT", "")
        if ft_concept == subj:
            for L in ll_layers:
                if L < len(pl) and pl[L] is not None: ll_own[L].append(pl[L] * 100)
        else:
            for L in ll_layers:
                if L < len(pl) and pl[L] is not None: ll_cross[L].append(pl[L] * 100)

fig, ax = plt.subplots(figsize=(10, 6))
bw = 0.25
x = np.arange(len(ll_layers))
def stats(d):
    return ([np.mean(d[L]) for L in ll_layers],
            [np.std(d[L]) for L in ll_layers])
bm, bs = stats(ll_base)
cm, cs = stats(ll_cross)
om, os_ = stats(ll_own)
ax.bar(x - bw, bm, bw, yerr=bs, color="#444444", edgecolor="black", linewidth=0.5, capsize=4, label="base AO")
ax.bar(x,       cm, bw, yerr=cs, color="#1f77b4", edgecolor="black", linewidth=0.5, capsize=4, label="FT-AO CROSS")
ax.bar(x + bw,  om, bw, yerr=os_, color="#d62728", edgecolor="black", linewidth=0.5, capsize=4, label="FT-AO OWN")
for i, (b, c, o) in enumerate(zip(bm, cm, om)):
    ax.text(i - bw, b + 1, f"{b:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(i,       c + 1, f"{c:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(i + bw,  o + 1, f"{o:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([f"L{L}" for L in ll_layers], fontsize=11)
ax.set_xlabel("AO internal layer (LogitLens read-out)", fontsize=12)
ax.set_ylabel("P(target word) %", fontsize=12)
ax.set_ylim(0, 80)
ax.set_title("Mechanism II — LogitLens P(target) at the secret-word position\n"
             "FT-AO OWN drops vs FT-AO CROSS at every late layer ⇒ output-side censorship",
             fontsize=13, fontweight="bold")
ax.legend(loc="upper left", framealpha=0.95, fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig7_logitlens.png", bbox_inches="tight")
plt.close()
print("[fig7] logitlens")


# =========================================================================
# Fig 8 — Layer ablation (Mechanism III)
# =========================================================================
abl = json.load(open("<PATH_TO_SCRATCH>/results/test_layer_abl_flag_book.json"))
# Group by concept
by_c = defaultdict(dict)
ABL_ORDER = ["none", "L0-5", "L6-11", "L12-17", "L18-23", "L24-29", "L30-35"]
for r in abl:
    by_c[r["concept"]][r["ablation_label"]] = r["mean_p_target"] * 100

fig, ax = plt.subplots(figsize=(11, 6))
bw = 0.25
x = np.arange(len(ABL_ORDER))
colors = {"leaf": "#2ca02c", "flag": "#d62728", "book": "#ff7f0e"}
for i, c in enumerate(["leaf", "flag", "book"]):
    vals = [by_c.get(c, {}).get(a, 0) for a in ABL_ORDER]
    xs = x + (i - 1) * bw
    ax.bar(xs, vals, bw, color=colors[c], edgecolor="black",
           linewidth=0.4, label=f"{c}-FT-AO")
    for xi, v in zip(xs, vals):
        ax.text(xi, v + 1, f"{v:.0f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")
# Highlight L18-23
ax.axvspan(3.5, 4.5, color="orange", alpha=0.15)
ax.text(4, 78, "best recovery", ha="center", va="bottom",
        fontsize=10, fontweight="bold", color="#8c4500")
ax.set_xticks(x); ax.set_xticklabels(ABL_ORDER, fontsize=11)
ax.set_xlabel("AO-LoRA ablation band", fontsize=12)
ax.set_ylabel("P(target) % on OWN concept", fontsize=12)
ax.set_ylim(0, 90)
ax.set_title("Mechanism III — Layer-band LoRA ablation recovers OWN-concept reading\n"
             "L18–23 ablation gives the largest recovery (causal localisation of destruction circuit)",
             fontsize=13, fontweight="bold")
ax.legend(loc="upper right", framealpha=0.95, fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig8_layer_ablation.png", bbox_inches="tight")
plt.close()
print("[fig8] layer ablation")


# =========================================================================
# Fig 9 — Strict vs cooperative base-AO recovery
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 6))
bw = 0.35
x = np.arange(len(REGIMES))
coop_vals = [ao_base_pt("coop", r, "c1p00") for r in REGIMES]
strict_vals = [ao_base_pt("strict", r, "c1p00") for r in REGIMES]
coop_m = [np.mean(v) for v in coop_vals]
coop_s = [np.std(v) for v in coop_vals]
strict_m = [np.mean(v) for v in strict_vals]
strict_s = [np.std(v) for v in strict_vals]
ax.bar(x - bw/2, coop_m, bw, yerr=coop_s, color="#1f77b4", edgecolor="black",
       linewidth=0.5, capsize=4, label="cooperative subjects")
ax.bar(x + bw/2, strict_m, bw, yerr=strict_s, color="#d62728", edgecolor="black",
       linewidth=0.5, capsize=4, label="strict subjects")
for i, (cm, sm) in enumerate(zip(coop_m, strict_m)):
    ax.text(i - bw/2, cm + 2, f"{cm:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(i + bw/2, sm + 2, f"{sm:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([RLBL[r] for r in REGIMES], fontsize=11)
ax.set_ylabel("Base AO P(target) %", fontsize=12)
ax.set_ylim(0, 85)
ax.set_title("Strict vs. cooperative hiding — base AO recovery (mean ± std over 5 concepts)\n"
             "strict subjects reduce AO-accessible signal in REFUSAL/SAMETEXT/THINK but HINT remains vulnerable",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper right", framealpha=0.95, fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig9_strict_vs_coop.png", bbox_inches="tight")
plt.close()
print("[fig9] strict vs coop")


# =========================================================================
# Fig 11 — Probe-decodability vs AO-verbalizability
# =========================================================================
pv = json.load(open(f"{TIER_A}/probe_vs_ao.json"))

fig, ax = plt.subplots(figsize=(11, 6))
labels = []
probe_vals = []
ao_vals = []
gaps = []
for j in pv:
    labels.append(f"{j['regime'].upper()}\n{j['protocol']}")
    probe_vals.append(j["probe_L18_acc"] * 100 if j["probe_L18_acc"] else 0)
    ao_vals.append(j["AO_base_P_target_mean"] or 0)
    gaps.append(j["gap_L18_pp"] or 0)

bw = 0.35
x = np.arange(len(labels))
ax.bar(x - bw/2, probe_vals, bw, color="#3f51b5", edgecolor="black",
       linewidth=0.5, label="linear probe (L18) accuracy")
ax.bar(x + bw/2, ao_vals, bw, color="#ff9800", edgecolor="black",
       linewidth=0.5, label="base AO P(target)")
for i, (p, a, g) in enumerate(zip(probe_vals, ao_vals, gaps)):
    ax.text(i - bw/2, p + 1.5, f"{p:.0f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold")
    ax.text(i + bw/2, a + 1.5, f"{a:.0f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold")
    ax.text(i, max(p, a) + 12, f"gap\n+{g:.0f}pp",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color="#388e3c",
            bbox=dict(boxstyle="round,pad=0.25", fc="#e8f5e9", ec="#388e3c"))
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("%", fontsize=12)
ax.set_ylim(0, 130)
ax.set_title("Probe decodability vs. AO-verbalizability\n"
             "Linear probes find the concept in residuals 17–52 pp more often than the AO verbalises it",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper left", framealpha=0.95, fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig11_probe_vs_ao.png", bbox_inches="tight")
plt.close()
print("[fig11] probe vs AO")


# =========================================================================
# Fig 14 — Qualitative examples (visual table)
# =========================================================================
qe = json.load(open(f"{TIER_A}/qualitative_examples.json"))

fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(0, 14); ax.set_ylim(0, len(qe) * 1.4 + 1); ax.axis("off")
ax.set_title("Qualitative examples — base AO vs. own FT-AO vs. cross FT-AO",
             fontsize=13, fontweight="bold", pad=14)

# Header row
y_top = len(qe) * 1.4 + 0.3
headers = [(0.2, "case"), (3.5, "AO + subject"), (7.7, "greedy output"), (12.6, "P(target) / rank")]
for x, h in headers:
    ax.text(x, y_top, h, fontsize=11, fontweight="bold", color="#333")

# Rows
for i, e in enumerate(qe):
    y = (len(qe) - i - 1) * 1.4 + 0.5
    # alternating background
    if i % 2 == 0:
        ax.add_patch(plt.Rectangle((0.0, y - 0.5), 14, 1.3,
                                    facecolor="#f3f4f6", edgecolor="none", zorder=0))
    ax.text(0.2, y, e["label"], fontsize=10, va="center", wrap=True)
    ax.text(3.5, y + 0.2, e["ao"], fontsize=9, va="center", style="italic", color="#555")
    ax.text(3.5, y - 0.2, f"subj={e['subj']}", fontsize=9, va="center", color="#555")
    g = e["greedy_text"].strip()
    if len(g) > 60: g = g[:57] + "..."
    ax.text(7.7, y, f"\"{g}\"", fontsize=10, va="center", color="#000")
    pt = f"{e['target_prob']*100:.1f}%" if e["target_prob"] is not None else "—"
    rk = str(e["target_rank"]) if e["target_rank"] is not None else "—"
    color = "#d32f2f" if (e["target_prob"] is not None and e["target_prob"] < 0.1) else "#388e3c"
    ax.text(12.6, y + 0.2, f"P = {pt}", fontsize=10, va="center", fontweight="bold", color=color)
    ax.text(12.6, y - 0.2, f"rank = {rk}", fontsize=10, va="center", color=color)

plt.tight_layout()
plt.savefig(f"{OUT}/fig14_qualitative.png", bbox_inches="tight")
plt.close()
print("[fig14] qualitative")

print("\n=== All figures built ===")
for f in sorted(os.listdir(OUT)):
    if f.startswith("fig"): print(f"  {OUT}/{f}")
