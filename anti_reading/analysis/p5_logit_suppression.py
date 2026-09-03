"""P5 — Direct target-logit suppression analysis.

Outputs:
  <PATH_TO_SCRATCH>/results/tier_a/p5_suppression.json
  <PATH_TO_SCRATCH>/results/tier_a/p5_summary.md
  <PATH_TO_SCRATCH>/plot_scripts/charts/fig_p5_logit_drop_by_cell.png
  <PATH_TO_SCRATCH>/plot_scripts/charts/fig_p5_logit_drop_by_layer.png
  <PATH_TO_SCRATCH>/plot_scripts/charts/fig_p5_redistribution.png

Tests:
  A. Per-cell target log-prob drop: log P_base(target) − log P_FT(target).
     Aggregate across captures with bootstrap CI.
  B. Layer-resolved logit drop using §2.2 LogitLens data — at which layer does
     the suppression appear?
  C. Redistribution: when the FT-AO suppresses the target, where does mass go?
     Use plurality_word (per-capture) and sample_words to characterise.
  D. Semantic-neighbor scoring with a hand-built field per concept.
"""
import os, json, glob, pickle, re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

PKL = "<PATH_TO_SCRATCH>/plot_scripts/compact_metrics.pkl"
DATA_ROOT = "<PATH_TO_SCRATCH>/results/ao_xmatrix_v3_lp"
LL_JSON = "<PATH_TO_SCRATCH>/results/test_v1_v5_flag_book.json"
TIER_A = "<PATH_TO_SCRATCH>/results/tier_a"
OUT_DIR = "<PATH_TO_SCRATCH>/plot_scripts/charts"
os.makedirs(TIER_A, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)

CONCEPTS = ["leaf", "moon", "wave", "flag", "book"]
REGIMES = ["hint", "refusal", "sametext", "think"]
RLBL = {"hint": "HINT", "refusal": "REFUSAL", "sametext": "SAMETEXT", "think": "THINK"}

# Semantic neighborhood per concept (hand-curated, broad)
FIELD = {
    "leaf":  {"leaf", "leaves", "plant", "tree", "green", "branch", "autumn", "fall",
              "photosynthesis", "flora", "foliage", "petal", "stem", "garden"},
    "moon":  {"moon", "lunar", "night", "tide", "eclipse", "satellite", "crescent",
              "sky", "celestial", "wolf", "phase", "orbit", "cheese"},
    "wave":  {"wave", "waves", "ocean", "sea", "water", "tide", "ripple", "surf",
              "hello", "frequency", "amplitude", "current", "shore"},
    "flag":  {"flag", "flags", "banner", "country", "nation", "pole", "stripes",
              "hoist", "raise", "emblem", "ensign", "color"},
    "book":  {"book", "books", "novel", "chapter", "page", "library", "story",
              "read", "author", "shelf", "edition", "scroll"},
}


print("[load] compact_metrics.pkl ...", flush=True)
D = pickle.load(open(PKL, "rb"))


# =========================================================================
# A. Per-cell target log-prob drop with bootstrap CI
# =========================================================================
def boot_ci(arr, n=2000, rng=None):
    if rng is None: rng = np.random.default_rng(0)
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0: return None
    idx = rng.integers(0, len(arr), (n, len(arr)))
    means = arr[idx].mean(axis=1)
    return {"mean": float(arr.mean()),
            "ci95_lo": float(np.quantile(means, 0.025)),
            "ci95_hi": float(np.quantile(means, 0.975)),
            "n": int(len(arr))}

def coop_ft(c, cv): return f"ours_{c}_{cv}"
def coop_subj(c, cv): return f"{c}_{cv}"

def per_capture_logprobs(regime, ao, subj):
    cells = D.get(regime, {}).get(ao, {}).get(subj, [])
    out = [c["logprob"] for c in cells if c.get("logprob") is not None]
    return np.array(out, dtype=float)

def per_capture_ranks(regime, ao, subj):
    cells = D.get(regime, {}).get(ao, {}).get(subj, [])
    out = [c["rank"] for c in cells if c.get("rank") is not None]
    return np.array(out, dtype=float)


print("\n[A] Per-cell target log-prob drop (base − FT-AO OWN), c=1.0 ...", flush=True)
rng = np.random.default_rng(42)
A = {}
for regime in REGIMES:
    A[regime] = {}
    for concept in CONCEPTS:
        base_lp = per_capture_logprobs(regime, "ours_base", coop_subj(concept, "c1p00"))
        own_lp = per_capture_logprobs(regime, coop_ft(concept, "c1p00"), coop_subj(concept, "c1p00"))
        base_rk = per_capture_ranks(regime, "ours_base", coop_subj(concept, "c1p00"))
        own_rk = per_capture_ranks(regime, coop_ft(concept, "c1p00"), coop_subj(concept, "c1p00"))
        A[regime][concept] = {
            "base_logprob_ci": boot_ci(base_lp, rng=rng),
            "own_logprob_ci":  boot_ci(own_lp, rng=rng),
            "base_rank_ci":    boot_ci(base_rk, rng=rng),
            "own_rank_ci":     boot_ci(own_rk, rng=rng),
            "logprob_drop_mean": (float(base_lp.mean()) - float(own_lp.mean()))
                if len(base_lp) and len(own_lp) else None,
            "rank_increase_mean": (float(own_rk.mean()) - float(base_rk.mean()))
                if len(base_rk) and len(own_rk) else None,
        }

# Print headline
print("\n=== Target log-prob drop (base − FT-AO OWN) ===")
print(f"  {'concept':<6} | " + " | ".join(f"{RLBL[r]:>15}" for r in REGIMES))
for c in CONCEPTS:
    row = [f"{c:<6}"]
    for r in REGIMES:
        d = A[r][c]["logprob_drop_mean"]
        row.append(f"{d:+15.2f}" if d is not None else "       —")
    print("  " + " | ".join(row))

print("\n=== Target rank increase (FT-AO − base) ===")
for c in CONCEPTS:
    row = [f"{c:<6}"]
    for r in REGIMES:
        d = A[r][c]["rank_increase_mean"]
        row.append(f"{d:+15.0f}" if d is not None else "       —")
    print("  " + " | ".join(row))


# =========================================================================
# B. Layer-resolved logit drop from §2.2 LogitLens data
# =========================================================================
print("\n[B] Layer-resolved logit drop (LogitLens, flag/book/leaf) ...", flush=True)
ll = json.load(open(LL_JSON))
# cells like base__flag, flag-FT__flag (own), book-FT__flag (cross)
# we want per-layer (base P) − (FT-AO OWN P)
ll_drop_by_layer = defaultdict(list)  # layer → list of drops
LL_LAYERS = list(range(0, 37))
for ft_concept in ["flag", "book", "leaf"]:
    base_pl = ll.get(f"base__{ft_concept}", {}).get("mean_p_target_per_layer", [])
    own_pl = ll.get(f"{ft_concept}-FT__{ft_concept}", {}).get("mean_p_target_per_layer", [])
    if not base_pl or not own_pl: continue
    for L in LL_LAYERS:
        if L < len(base_pl) and L < len(own_pl):
            b = base_pl[L]; o = own_pl[L]
            if b is None or o is None: continue
            ll_drop_by_layer[L].append((b - o) * 100)

B = {L: {"mean_drop_pp": float(np.mean(v)), "n": len(v)}
     for L, v in ll_drop_by_layer.items() if v}


# =========================================================================
# C. Redistribution: what does FT-AO OWN output instead?
# =========================================================================
print("\n[C] Mass redistribution — plurality_word for each (FT-AO OWN, regime) ...", flush=True)
def extract_word(text):
    """Pull the first quoted or bracketed word from a greedy output.
    Falls back to the last all-letters token.
    """
    if not text: return None
    m = re.search(r"['\"]([A-Za-z]+)['\"]", text)
    if m: return m.group(1).lower()
    # try 'is X.' pattern
    m = re.search(r"\bis\s+([A-Za-z]+)\.?\s*$", text.strip())
    if m: return m.group(1).lower()
    return None

def cell_plurality_counter(regime, ao, subj):
    p = f"{DATA_ROOT}/{regime}/{ao}__{subj}/ao_results.json"
    if not os.path.isfile(p): return None
    try: d = json.load(open(p))
    except: return None
    words = []
    for c in d.get("cells", []):
        pw = c.get("plurality_word")
        if not pw:
            pw = extract_word(c.get("greedy_text", ""))
        if pw: words.append(pw.lower().strip())
    return Counter(words) if words else None

C = {}
for regime in REGIMES:
    C[regime] = {}
    for concept in CONCEPTS:
        ao = coop_ft(concept, "c1p00")
        subj = coop_subj(concept, "c1p00")
        ctr = cell_plurality_counter(regime, ao, subj)
        if ctr is None: continue
        top = ctr.most_common(10)
        total = sum(ctr.values())
        # Bucket each word
        own_rate = ctr.get(concept, 0) / total if total else 0
        # semantic-field rate (INCLUDING own word, since own ∈ field)
        field = FIELD[concept]
        field_rate = sum(v for w, v in ctr.items() if w in field) / total if total else 0
        field_only_rate = max(0.0, field_rate - own_rate)  # neighbors strictly, not target
        refuse_words = {"no", "none", "cannot", "can't", "secret", "unknown", "n/a", "unclear"}
        refuse_rate = sum(v for w, v in ctr.items() if w in refuse_words) / total if total else 0
        other_rate = max(0.0, 1 - own_rate - field_only_rate - refuse_rate)
        C[regime][concept] = {
            "total": total, "top10": [[w, v] for w, v in top],
            "own_rate": own_rate, "field_rate": field_rate,
            "field_only_rate": field_only_rate,
            "refuse_rate": refuse_rate, "other_rate": other_rate,
        }


# Also for base AO and CROSS comparisons on HINT only (to show contrast)
C_BASE = {}
C_CROSS = {}
for concept in CONCEPTS:
    # base AO on this concept
    ctr = cell_plurality_counter("hint", "ours_base", coop_subj(concept, "c1p00"))
    if ctr:
        total = sum(ctr.values())
        field = FIELD[concept]
        own_rate = ctr.get(concept, 0) / total if total else 0
        field_rate = sum(v for w, v in ctr.items() if w in field) / total if total else 0
        field_only_rate = max(0.0, field_rate - own_rate)
        other_rate = max(0.0, 1 - own_rate - field_only_rate)
        C_BASE[concept] = {
            "top10": [[w, v] for w, v in ctr.most_common(10)],
            "own_rate": own_rate, "field_rate": field_rate,
            "field_only_rate": field_only_rate, "other_rate": other_rate,
            "total": total,
        }
    # a CROSS FT-AO reading this concept's subject
    if concept == "leaf": ao_cross = coop_ft("moon", "c1p00")
    else: ao_cross = coop_ft("leaf", "c1p00")
    ctr = cell_plurality_counter("hint", ao_cross, coop_subj(concept, "c1p00"))
    if ctr:
        total = sum(ctr.values())
        field = FIELD[concept]
        own_rate = ctr.get(concept, 0) / total if total else 0
        field_rate = sum(v for w, v in ctr.items() if w in field) / total if total else 0
        field_only_rate = max(0.0, field_rate - own_rate)
        other_rate = max(0.0, 1 - own_rate - field_only_rate)
        C_CROSS[concept] = {
            "ao_used": ao_cross,
            "top10": [[w, v] for w, v in ctr.most_common(10)],
            "own_rate": own_rate, "field_rate": field_rate,
            "field_only_rate": field_only_rate, "other_rate": other_rate,
            "total": total,
        }

# Print
print("\n=== HINT regime: plurality-word redistribution ===")
print(f"  {'concept':<6} {'cell':<12} {'top-3 plurality words (count)':<60} own%  field%")
for concept in CONCEPTS:
    # base
    b = C_BASE.get(concept)
    if b:
        top3 = " ".join(f"{w}({v})" for w, v in b["top10"][:3])
        print(f"  {concept:<6} {'base AO':<12} {top3:<60} {b['own_rate']*100:>4.0f}%  {b['field_rate']*100:>4.0f}%")
    # own FT-AO
    o = C.get("hint", {}).get(concept)
    if o:
        top3 = " ".join(f"{w}({v})" for w, v in o["top10"][:3])
        print(f"  {concept:<6} {'own FT-AO':<12} {top3:<60} {o['own_rate']*100:>4.0f}%  {o['field_rate']*100:>4.0f}%")
    # cross FT-AO
    c = C_CROSS.get(concept)
    if c:
        top3 = " ".join(f"{w}({v})" for w, v in c["top10"][:3])
        print(f"  {concept:<6} {'cross FT-AO':<12} {top3:<60} {c['own_rate']*100:>4.0f}%  {c['field_rate']*100:>4.0f}%")
    print()


# =========================================================================
# Save raw + summary
# =========================================================================
ALL = {"A_logprob_drop": A, "B_layer_drop_pp": B, "C_redistribution": C,
       "C_base": C_BASE, "C_cross": C_CROSS, "_field": {k: sorted(v) for k, v in FIELD.items()}}
json.dump(ALL, open(f"{TIER_A}/p5_suppression.json", "w"), indent=2)
print(f"\n[saved] {TIER_A}/p5_suppression.json")

# Markdown summary
lines = []
lines.append("# P5 — Target-Logit Suppression Analysis\n")
lines.append("All numbers are for cooperative protocol, c=1.0 (the regime where the anti-reader is most active).\n\n")

lines.append("## A. Target log-prob drop (base AO − FT-AO OWN)\n")
lines.append("Negative numbers mean the FT-AO assigns HIGHER probability than base AO. Positive means suppression.\n")
lines.append("| Concept | HINT | REFUSAL | SAMETEXT | THINK |")
lines.append("|---|---|---|---|---|")
for c in CONCEPTS:
    row = [c]
    for r in REGIMES:
        d = A[r][c]["logprob_drop_mean"]
        row.append(f"{d:+.2f}" if d is not None else "—")
    lines.append("| " + " | ".join(row) + " |")
lines.append("")

lines.append("## A. Target rank increase (FT-AO OWN − base)\n")
lines.append("| Concept | HINT | REFUSAL | SAMETEXT | THINK |")
lines.append("|---|---|---|---|---|")
for c in CONCEPTS:
    row = [c]
    for r in REGIMES:
        d = A[r][c]["rank_increase_mean"]
        row.append(f"{d:+.0f}" if d is not None else "—")
    lines.append("| " + " | ".join(row) + " |")

lines.append("\n## B. Layer-resolved logit drop (LogitLens, FT-AO OWN vs base)\n")
lines.append("Mean P(target) drop in percentage points at each layer (aggregated over flag/book/leaf FT-AOs).\n")
lines.append("| Layer | Drop (pp, base − OWN) | n |")
lines.append("|---|---|---|")
for L in sorted(B.keys()):
    lines.append(f"| L{L} | {B[L]['mean_drop_pp']:+.1f} | {B[L]['n']} |")

lines.append("\n## C. Where does the mass go? (HINT regime, top plurality words across captures)\n")
lines.append("\nFor each concept we show what the AO says most often on its leaf/moon/etc. subject. **own%** = fraction of captures where plurality_word == target concept. **field%** = fraction in the concept's semantic neighborhood.\n")
for concept in CONCEPTS:
    lines.append(f"\n### {concept}")
    lines.append("| AO type | Top 3 plurality words (count) | own % | field % |")
    lines.append("|---|---|---|---|")
    for label, ctr in [("base AO", C_BASE.get(concept)),
                        ("OWN FT-AO", C.get("hint", {}).get(concept)),
                        ("CROSS FT-AO", C_CROSS.get(concept))]:
        if not ctr: continue
        top3 = " ".join(f"{w}({v})" for w, v in ctr["top10"][:3])
        own = f"{ctr['own_rate']*100:.0f}%"
        field = f"{ctr['field_rate']*100:.0f}%"
        lines.append(f"| {label} | {top3} | {own} | {field} |")

with open(f"{TIER_A}/p5_summary.md", "w") as f: f.write("\n".join(lines))
print(f"[saved] {TIER_A}/p5_summary.md")


# =========================================================================
# Figures
# =========================================================================
print("\n[fig] building ...", flush=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.dpi": 150})

# Fig P5-1: per-cell logprob drop
fig, ax = plt.subplots(figsize=(11, 5.5))
bw = 0.18
xs = np.arange(len(REGIMES))
for ci, concept in enumerate(CONCEPTS):
    vals = [A[r][concept]["logprob_drop_mean"] or 0 for r in REGIMES]
    ax.bar(xs + (ci - 2) * bw, vals, bw, label=concept, edgecolor="black", linewidth=0.4)
ax.axhline(0, color="gray", linestyle="-", linewidth=1)
ax.set_xticks(xs); ax.set_xticklabels([RLBL[r] for r in REGIMES])
ax.set_ylabel("base − OWN target log-prob drop (nats)")
ax.set_title("Per-cell target log-prob drop — base AO vs FT-AO OWN\n"
             "positive = FT-AO suppresses its own target word",
             fontsize=12, fontweight="bold")
ax.legend(title="concept", loc="upper left", ncol=5, fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_p5_logit_drop_by_cell.png", bbox_inches="tight")
plt.close()
print("  fig_p5_logit_drop_by_cell.png")

# Fig P5-2: layer-resolved
fig, ax = plt.subplots(figsize=(11, 5))
layers = sorted(B.keys())
drops = [B[L]["mean_drop_pp"] for L in layers]
ax.plot(layers, drops, "o-", color="#d62728", markersize=6, linewidth=2)
ax.axhline(0, color="gray", linewidth=1)
ax.axvspan(18, 23, color="orange", alpha=0.15, label="L18–23 (ablation peak)")
ax.set_xlabel("AO internal layer")
ax.set_ylabel("base − FT-AO OWN P(target) (pp)")
ax.set_title("Layer-resolved target-prob suppression\n"
             "(LogitLens read-out of each layer; aggregated over flag/book/leaf OWN cells)",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_p5_logit_drop_by_layer.png", bbox_inches="tight")
plt.close()
print("  fig_p5_logit_drop_by_layer.png")

# Fig P5-3: redistribution stacked bars
fig, ax = plt.subplots(figsize=(11, 5.5))
labels = []; ratios_own = []; ratios_field = []; ratios_refuse = []; ratios_other = []
for ao_type, src in [("BASE", C_BASE), ("OWN-FT", C.get("hint", {})), ("CROSS-FT", C_CROSS)]:
    for concept in CONCEPTS:
        v = src.get(concept)
        if not v: continue
        labels.append(f"{ao_type}\n{concept}")
        ratios_own.append(v["own_rate"] * 100)
        ratios_field.append(v.get("field_only_rate", 0) * 100)
        ratios_refuse.append(v.get("refuse_rate", 0) * 100)
        ratios_other.append(v.get("other_rate", 0) * 100)
xs = np.arange(len(labels))
ax.bar(xs, ratios_own, color="#388e3c", label="target concept")
ax.bar(xs, ratios_field, bottom=ratios_own, color="#fdd835", label="semantic neighbor")
ax.bar(xs, ratios_refuse, bottom=[a+b for a, b in zip(ratios_own, ratios_field)], color="#90a4ae", label="refusal phrase")
ax.bar(xs, ratios_other, bottom=[a+b+c for a, b, c in zip(ratios_own, ratios_field, ratios_refuse)], color="#bbbbbb", label="other / unrelated")
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8, rotation=0)
ax.set_ylabel("% of captures (plurality word)")
ax.set_ylim(0, 110)
ax.set_title("Where does the AO put probability mass on each concept?\n"
             "HINT regime, c=1.0. Each bar = plurality words across captures.",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper right", ncol=2, framealpha=0.95)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_p5_redistribution.png", bbox_inches="tight")
plt.close()
print("  fig_p5_redistribution.png")

print("\n[done]")
