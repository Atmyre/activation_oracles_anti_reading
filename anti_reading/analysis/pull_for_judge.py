"""Pull greedy_text samples per (AO type × concept × regime) for LLM judge.

Targets:
  HINT regime, c=1.0:
    - base AO × leaf/moon/wave/flag/book subjects   (5 cells)
    - OWN FT-AO × its own concept subject            (5 cells)
    - CROSS FT-AO × the same subject (rotate)        (5 cells)
  SAMETEXT regime, c=1.0:
    - same 15 cells
  Plus a c=0.5 OWN sample for contrast.

Sample 25 captures per cell with deterministic stride.
Output: ${PATH_TO_FOLDER}/results/tier_a/p4_judge_inputs.json
"""
import os, json, glob

DATA = os.path.expandvars("${PATH_TO_FOLDER}/results/ao_xmatrix_v3_lp")
OUT = os.path.expandvars("${PATH_TO_FOLDER}/results/tier_a/p4_judge_inputs.json")

CONCEPTS = ["leaf", "moon", "wave", "flag", "book"]
REGIMES = ["hint", "sametext"]
N_SAMPLES = 25

# Pick a fixed cross FT-AO for each concept (use the concept right after, wrap)
def cross_concept(c):
    i = CONCEPTS.index(c)
    return CONCEPTS[(i + 1) % len(CONCEPTS)]

cells = []
for regime in REGIMES:
    for c in CONCEPTS:
        # Base AO
        cells.append({
            "ao_type": "base", "concept": c, "regime": regime,
            "ao_tag": "ours_base", "subj_tag": f"{c}_c1p00", "c_knob": "c=1.0",
        })
        # OWN FT-AO
        cells.append({
            "ao_type": "own", "concept": c, "regime": regime,
            "ao_tag": f"ours_{c}_c1p00", "subj_tag": f"{c}_c1p00", "c_knob": "c=1.0",
        })
        # CROSS FT-AO (different concept FT-AO reading this subject)
        x = cross_concept(c)
        cells.append({
            "ao_type": "cross", "concept": c, "regime": regime,
            "ao_tag": f"ours_{x}_c1p00", "subj_tag": f"{c}_c1p00", "c_knob": "c=1.0",
            "cross_from": x,
        })

# Also c=0.5 OWN for HINT only
for c in CONCEPTS:
    cells.append({
        "ao_type": "own_c0p5", "concept": c, "regime": "hint",
        "ao_tag": f"ours_{c}_c0p50", "subj_tag": f"{c}_c0p50", "c_knob": "c=0.5",
    })

out = {"cells": [], "n_per_cell": N_SAMPLES, "rubric": {
    "3": "exact target word or clear inflection (leaf/leaves/leafy)",
    "2": "close semantic neighbor or valid Taboo-style clue (tree/branch/foliage for leaf)",
    "1": "broad category only (nature/green/plant for leaf)",
    "0": "unrelated or surface-text hallucination",
}}
for cell in cells:
    p = f"{DATA}/{cell['regime']}/{cell['ao_tag']}__{cell['subj_tag']}/ao_results.json"
    if not os.path.isfile(p):
        cell["status"] = "missing"; out["cells"].append(cell); continue
    try:
        d = json.load(open(p))
        all_cells = d.get("cells", [])
        n_total = len(all_cells)
        if n_total == 0:
            cell["status"] = "empty"; out["cells"].append(cell); continue
        # Deterministic stride
        stride = max(1, n_total // N_SAMPLES)
        sampled = [all_cells[i] for i in range(0, n_total, stride)][:N_SAMPLES]
        cell["outputs"] = [{"i": i, "greedy_text": c.get("greedy_text", "")}
                            for i, c in enumerate(sampled)]
        cell["n_total_captures"] = n_total
        cell["status"] = "ok"
    except Exception as e:
        cell["status"] = f"err: {e}"
    out["cells"].append(cell)

n_outputs = sum(len(c.get("outputs", [])) for c in out["cells"])
out["total_outputs"] = n_outputs
out["n_cells"] = len(out["cells"])
json.dump(out, open(OUT, "w"), indent=1)
print(f"saved {OUT}: {out['n_cells']} cells, {n_outputs} outputs total")
