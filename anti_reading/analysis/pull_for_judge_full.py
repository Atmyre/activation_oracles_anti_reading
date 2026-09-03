"""Pull greedy_text for comprehensive P4 LLM-judge coverage.

Cells (per concept):
  Cooperative:
    base × 4 regimes               (4)
    OWN c=1.0 × 4 regimes          (4)
    OWN c=0.5 × 4 regimes          (4)
    CROSS c=1.0 × 4 regimes        (4)
    CROSS c=0.5 × 4 regimes        (4)
  Strict:
    base × 4 regimes               (4)
    OWN c=1.0 × 4 regimes          (4)
    OWN c=0.5 × 4 regimes          (4)
                                  = 32 cells × 25 = 800 outputs per concept
"""
import os, json

DATA = "<PATH_TO_SCRATCH>/results/ao_xmatrix_v3_lp"
OUT = "<PATH_TO_SCRATCH>/results/tier_a/p4_judge_inputs_full.json"
CONCEPTS = ["leaf", "moon", "wave", "flag", "book"]
REGIMES = ["hint", "refusal", "sametext", "think"]
N_SAMPLES = 25

def cross_concept(c):
    i = CONCEPTS.index(c)
    return CONCEPTS[(i + 1) % len(CONCEPTS)]

def coop_subj(c, cv): return f"{c}_{cv}"
def strict_subj(c, cv): return f"strict{c}v2_{cv}"
def coop_ft(c, cv): return f"ours_{c}_{cv}"
def strict_ft(c, cv): return f"ours_strict{c}v2_{cv}"

cells = []
for regime in REGIMES:
    for c in CONCEPTS:
        # Cooperative subject
        cs = coop_subj(c, "c1p00")
        cs5 = coop_subj(c, "c0p50")
        x = cross_concept(c)
        # base AO on coop subject (c=1.0)
        cells.append({"ao_type": "base_coop", "concept": c, "regime": regime,
                      "ao_tag": "ours_base", "subj_tag": cs, "protocol": "coop", "c_knob": "c=1.0"})
        # OWN on coop
        cells.append({"ao_type": "own_coop_c10", "concept": c, "regime": regime,
                      "ao_tag": coop_ft(c, "c1p00"), "subj_tag": cs, "protocol": "coop", "c_knob": "c=1.0"})
        cells.append({"ao_type": "own_coop_c05", "concept": c, "regime": regime,
                      "ao_tag": coop_ft(c, "c0p50"), "subj_tag": cs5, "protocol": "coop", "c_knob": "c=0.5"})
        # CROSS on coop (using x-FT-AO reading c-subject)
        cells.append({"ao_type": "cross_coop_c10", "concept": c, "regime": regime,
                      "ao_tag": coop_ft(x, "c1p00"), "subj_tag": cs, "protocol": "coop", "c_knob": "c=1.0",
                      "cross_from": x})
        cells.append({"ao_type": "cross_coop_c05", "concept": c, "regime": regime,
                      "ao_tag": coop_ft(x, "c0p50"), "subj_tag": cs5, "protocol": "coop", "c_knob": "c=0.5",
                      "cross_from": x})
        # Strict
        ss = strict_subj(c, "c1p00")
        ss5 = strict_subj(c, "c0p50")
        cells.append({"ao_type": "base_strict", "concept": c, "regime": regime,
                      "ao_tag": "ours_base", "subj_tag": ss, "protocol": "strict", "c_knob": "c=1.0"})
        cells.append({"ao_type": "own_strict_c10", "concept": c, "regime": regime,
                      "ao_tag": strict_ft(c, "c1p00"), "subj_tag": ss, "protocol": "strict", "c_knob": "c=1.0"})
        cells.append({"ao_type": "own_strict_c05", "concept": c, "regime": regime,
                      "ao_tag": strict_ft(c, "c0p50"), "subj_tag": ss5, "protocol": "strict", "c_knob": "c=0.5"})

out = {"cells": [], "n_per_cell": N_SAMPLES, "n_cells": len(cells)}
n_outputs = 0
for cell in cells:
    p = f"{DATA}/{cell['regime']}/{cell['ao_tag']}__{cell['subj_tag']}/ao_results.json"
    if not os.path.isfile(p):
        cell["status"] = "missing"; out["cells"].append(cell); continue
    try:
        d = json.load(open(p))
        all_cells = d.get("cells", [])
        if not all_cells:
            cell["status"] = "empty"; out["cells"].append(cell); continue
        stride = max(1, len(all_cells) // N_SAMPLES)
        sampled = [all_cells[i] for i in range(0, len(all_cells), stride)][:N_SAMPLES]
        cell["outputs"] = [{"i": i, "greedy_text": c.get("greedy_text", "")}
                            for i, c in enumerate(sampled)]
        cell["status"] = "ok"
        n_outputs += len(cell["outputs"])
    except Exception as e:
        cell["status"] = f"err: {e}"
    out["cells"].append(cell)

out["total_outputs"] = n_outputs
json.dump(out, open(OUT, "w"), indent=1)
print(f"saved {OUT}: {len(cells)} cells, {n_outputs} outputs")
