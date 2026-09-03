"""P5 extensions:
  A_full: target log-prob drop AND rank-increase per (regime × concept × c × protocol)
  C_full: redistribution / plurality-word counters across all (regime × ao_type × c × protocol)

No GPU needed. Output: /gpfs/scratch/USER/results/tier_a/p5_full.json + summary md.
"""
import os, json, pickle, re
import numpy as np
from collections import Counter, defaultdict

PKL = "/gpfs/scratch/USER/plot_scripts/compact_metrics.pkl"
DATA_ROOT = "/gpfs/scratch/USER/results/ao_xmatrix_v3_lp"
OUT_JSON = "/gpfs/scratch/USER/results/tier_a/p5_full.json"
OUT_MD = "/gpfs/scratch/USER/results/tier_a/p5_full_summary.md"

CONCEPTS = ["leaf", "moon", "wave", "flag", "book"]
REGIMES = ["hint", "refusal", "sametext", "think"]
RLBL = {"hint": "HINT", "refusal": "REFUSAL", "sametext": "SAMETEXT", "think": "THINK"}

FIELD = {
    "leaf": {"leaf","leaves","leafy","leaflet","plant","tree","branch","stem","petal",
              "foliage","photosynthesis","chlorophyll","green","autumn","fall","garden",
              "canopy","fern","sapling","twig"},
    "moon": {"moon","moons","lunar","tide","eclipse","satellite","crescent","celestial",
              "orbit","phase","night","sky","mystery","wolf","romance","cheese","stars",
              "space","moonlight"},
    "wave": {"wave","waves","wavy","ocean","sea","surf","tide","ripple","current","hello",
              "gesture","swell","breaker","crest","frequency","amplitude","water","hand"},
    "flag": {"flag","flags","banner","ensign","pennant","standard","pole","mast","hoist",
              "country","nation","pride","salute","raise","stripes","emblem","symbol","pride"},
    "book": {"book","books","bookish","booklet","novel","library","chapter","volume",
              "edition","paperback","scroll","manuscript","page","pages","text","story",
              "read","author","paper","shelf"},
}

print("[load] compact_metrics.pkl ...", flush=True)
D = pickle.load(open(PKL, "rb"))


def coop_ft(c, cv): return f"ours_{c}_{cv}"
def strict_ft(c, cv): return f"ours_strict{c}v2_{cv}"
def coop_subj(c, cv): return f"{c}_{cv}"
def strict_subj(c, cv): return f"strict{c}v2_{cv}"


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


def per_capture(regime, ao, subj, field):
    cells = D.get(regime, {}).get(ao, {}).get(subj, [])
    return np.array([c.get(field) for c in cells if c.get(field) is not None], dtype=float)


# ===========================================================================
# A_full: log-prob drop + rank increase per regime × concept × c × protocol
# ===========================================================================
rng = np.random.default_rng(42)
A = {}
for protocol, ao_fn, subj_fn in [("coop", coop_ft, coop_subj), ("strict", strict_ft, strict_subj)]:
    A[protocol] = {}
    for cv in ["c1p00", "c0p50"]:
        A[protocol][cv] = {}
        for regime in REGIMES:
            A[protocol][cv][regime] = {}
            for c in CONCEPTS:
                base_lp = per_capture(regime, "ours_base", subj_fn(c, cv), "logprob")
                own_lp = per_capture(regime, ao_fn(c, cv), subj_fn(c, cv), "logprob")
                base_rk = per_capture(regime, "ours_base", subj_fn(c, cv), "rank")
                own_rk = per_capture(regime, ao_fn(c, cv), subj_fn(c, cv), "rank")
                A[protocol][cv][regime][c] = {
                    "base_logprob": boot_ci(base_lp, rng=rng),
                    "own_logprob": boot_ci(own_lp, rng=rng),
                    "base_rank": boot_ci(base_rk, rng=rng),
                    "own_rank": boot_ci(own_rk, rng=rng),
                    "logprob_drop_mean": (float(base_lp.mean()) - float(own_lp.mean()))
                        if len(base_lp) and len(own_lp) else None,
                    "rank_increase_mean": (float(own_rk.mean()) - float(base_rk.mean()))
                        if len(base_rk) and len(own_rk) else None,
                }
print("[A_full] computed", flush=True)


# ===========================================================================
# C_full: redistribution / plurality words across cells
# ===========================================================================
def extract_word(text):
    if not text: return None
    m = re.search(r"['\"]([A-Za-z]+)['\"]", text)
    if m: return m.group(1).lower()
    m = re.search(r"\bis\s+([A-Za-z]+)\.?\s*$", text.strip())
    if m: return m.group(1).lower()
    return None


def cell_counter(regime, ao, subj):
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


def cross_concept(c):
    i = CONCEPTS.index(c)
    return CONCEPTS[(i + 1) % len(CONCEPTS)]


REFUSE_WORDS = {"no","none","cannot","can't","secret","unknown","n/a","unclear","refused",
                "noun","null","there","reveal","unable"}

def summarize(ctr, concept):
    total = sum(ctr.values()) if ctr else 0
    if total == 0:
        return {"total": 0, "top10": [], "own_rate": 0, "field_rate": 0,
                 "field_only_rate": 0, "refuse_rate": 0, "other_rate": 0}
    field = FIELD[concept]
    own = ctr.get(concept, 0) / total
    field_rate = sum(v for w, v in ctr.items() if w in field) / total
    refuse_rate = sum(v for w, v in ctr.items() if w in REFUSE_WORDS) / total
    field_only = max(0.0, field_rate - own)
    other = max(0.0, 1.0 - own - field_only - refuse_rate)
    return {"total": total, "top10": [[w, v] for w, v in ctr.most_common(10)],
            "own_rate": own, "field_rate": field_rate, "field_only_rate": field_only,
            "refuse_rate": refuse_rate, "other_rate": other}


C = {}
n_cells = 0
for protocol, ao_fn, subj_fn in [("coop", coop_ft, coop_subj), ("strict", strict_ft, strict_subj)]:
    C[protocol] = {}
    for cv in ["c1p00", "c0p50"]:
        C[protocol][cv] = {}
        for regime in REGIMES:
            C[protocol][cv][regime] = {}
            for concept in CONCEPTS:
                # base AO
                ctr = cell_counter(regime, "ours_base", subj_fn(concept, cv))
                C[protocol][cv][regime].setdefault(concept, {})["BASE"] = summarize(ctr, concept) if ctr else None
                # OWN
                ctr = cell_counter(regime, ao_fn(concept, cv), subj_fn(concept, cv))
                C[protocol][cv][regime][concept]["OWN"] = summarize(ctr, concept) if ctr else None
                # CROSS (rotation)
                x = cross_concept(concept)
                ctr = cell_counter(regime, ao_fn(x, cv), subj_fn(concept, cv))
                C[protocol][cv][regime][concept]["CROSS"] = summarize(ctr, concept) if ctr else None
                n_cells += 3
            print(f"  {protocol} {cv} {regime}: {len(CONCEPTS)} concepts done", flush=True)
print(f"[C_full] {n_cells} cell-counters computed", flush=True)


# ===========================================================================
# Save + summary
# ===========================================================================
ALL = {"A_full_logprob_drop": A, "C_full_redistribution": C,
       "_field_used": {k: sorted(v) for k, v in FIELD.items()},
       "_refuse_words": sorted(REFUSE_WORDS)}
json.dump(ALL, open(OUT_JSON, "w"), indent=2)
print(f"\n[saved] {OUT_JSON}")

# Summary md
lines = ["# P5 full coverage — target-logit suppression\n"]
lines.append("## A — Target log-prob drop (base − OWN) by regime × c × protocol\n")
lines.append("Positive = OWN-FT-AO suppresses target relative to base AO.\n")
for protocol in ["coop", "strict"]:
    for cv in ["c1p00", "c0p50"]:
        lines.append(f"\n### {protocol} c={cv.replace('c','').replace('p','.')}\n")
        lines.append("| Regime | leaf | moon | wave | flag | book | mean |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in REGIMES:
            vals = []; row = [RLBL[r]]
            for c in CONCEPTS:
                d = A[protocol][cv][r][c]["logprob_drop_mean"]
                if d is None: row.append("—")
                else: row.append(f"{d:+.2f}"); vals.append(d)
            row.append(f"{np.mean(vals):+.2f}" if vals else "—")
            lines.append("| " + " | ".join(row) + " |")

lines.append("\n## C — Where mass goes (own_rate / field_only_rate / refuse / other)\n")
for protocol in ["coop", "strict"]:
    for cv in ["c1p00", "c0p50"]:
        lines.append(f"\n### {protocol} c={cv.replace('c','').replace('p','.')}\n")
        lines.append("Mean across 5 concepts. Columns: own% / neighbor% / refuse% / other%.\n")
        lines.append("| Regime | BASE | OWN-FT | CROSS-FT |")
        lines.append("|---|---|---|---|")
        for r in REGIMES:
            row = [RLBL[r]]
            for ao_type in ["BASE", "OWN", "CROSS"]:
                vals = {"own": [], "field": [], "refuse": [], "other": []}
                for c in CONCEPTS:
                    cv_data = C[protocol][cv][r][c].get(ao_type)
                    if cv_data:
                        vals["own"].append(cv_data["own_rate"])
                        vals["field"].append(cv_data["field_only_rate"])
                        vals["refuse"].append(cv_data["refuse_rate"])
                        vals["other"].append(cv_data["other_rate"])
                if vals["own"]:
                    o, f, rf, ot = [np.mean(vals[k]) * 100 for k in ["own","field","refuse","other"]]
                    row.append(f"{o:.0f}/{f:.0f}/{rf:.0f}/{ot:.0f}")
                else:
                    row.append("—")
            lines.append("| " + " | ".join(row) + " |")

with open(OUT_MD, "w") as f: f.write("\n".join(lines))
print(f"[saved] {OUT_MD}")
