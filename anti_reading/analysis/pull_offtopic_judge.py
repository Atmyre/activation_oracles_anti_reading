"""Pull greedy_text for OFFTOPIC LLM-judge scoring.

Mirroring §4.5.8b setup: for each concept, sample 25 outputs per (AO type) cell:
  - base AO
  - OWN FT-AO
  - CROSS FT-AO (rotate to next concept)

For coop only (strict OWN was near-identical, and OFFTOPIC drives everything to 0%).
Total = 5 concepts × 3 AO types × 25 samples = 375 outputs per concept file.
"""
import os, json

DATA = "<PATH_TO_SCRATCH>/results/ao_xmatrix_v3_lp"
OUT = "<PATH_TO_SCRATCH>/results/tier_a/p4_offtopic_judge_inputs.json"
CONCEPTS = ["leaf", "moon", "wave", "flag", "book"]
N = 25

def cross_concept(c):
    i = CONCEPTS.index(c)
    return CONCEPTS[(i + 1) % len(CONCEPTS)]

cells = []
for c in CONCEPTS:
    subj = f"{c}_c1p00"
    cells.append({"ao_type": "base", "concept": c, "ao_tag": "ours_base", "subj_tag": subj})
    cells.append({"ao_type": "own_c10", "concept": c, "ao_tag": f"ours_{c}_c1p00", "subj_tag": subj})
    x = cross_concept(c)
    cells.append({"ao_type": "cross_c10", "concept": c, "ao_tag": f"ours_{x}_c1p00", "subj_tag": subj, "cross_from": x})

out = {"cells": [], "n_per_cell": N, "rubric": {
    "3": "exact target word or clear inflection",
    "2": "close semantic neighbor / valid Taboo clue",
    "1": "broad category only",
    "0": "unrelated word or refusal",
}, "regime": "offtopic"}

n_outputs = 0
for cell in cells:
    p = f"{DATA}/offtopic/{cell['ao_tag']}__{cell['subj_tag']}/ao_results.json"
    if not os.path.isfile(p):
        cell["status"] = "missing"; out["cells"].append(cell); continue
    try:
        d = json.load(open(p))
        all_cells = d.get("cells", [])
        if not all_cells: cell["status"] = "empty"; out["cells"].append(cell); continue
        stride = max(1, len(all_cells) // N)
        sampled = [all_cells[i] for i in range(0, len(all_cells), stride)][:N]
        cell["outputs"] = [{"i": i, "greedy_text": c.get("greedy_text", "")} for i, c in enumerate(sampled)]
        cell["status"] = "ok"
        n_outputs += len(cell["outputs"])
    except Exception as e:
        cell["status"] = f"err: {e}"
    out["cells"].append(cell)

out["total_outputs"] = n_outputs
out["n_cells"] = len(out["cells"])
json.dump(out, open(OUT, "w"), indent=1)
print(f"saved {OUT}: {len(cells)} cells, {n_outputs} outputs")
