"""Pull greedy_text for base AO on c=0.5 subjects (50 cells)."""
import os, json

DATA = "<PATH_TO_SCRATCH>/results/ao_xmatrix_v3_lp"
OUT = "<PATH_TO_SCRATCH>/results/tier_a/p4_base_c05_judge_inputs.json"
CONCEPTS = ["book", "flag", "leaf", "moon", "wave"]
REGIMES = ["hint", "refusal", "sametext", "think", "offtopic"]
N_SAMPLES = 25

cells = []
for regime in REGIMES:
    for c in CONCEPTS:
        cells.append({"ao_type": "base_coop_c05", "concept": c, "regime": regime,
                      "ao_tag": "ours_base",
                      "subj_tag": f"{c}_c0p50",
                      "protocol": "coop", "c_knob": "c=0.5"})
        cells.append({"ao_type": "base_strict_c05", "concept": c, "regime": regime,
                      "ao_tag": "ours_base",
                      "subj_tag": f"strict{c}v2_c0p50",
                      "protocol": "strict", "c_knob": "c=0.5"})

out = {"cells": [], "n_per_cell": N_SAMPLES, "n_cells": len(cells)}
n_out = 0
for cell in cells:
    reg = cell["regime"]
    ao = cell["ao_tag"]
    subj = cell["subj_tag"]
    p = f"{DATA}/{reg}/{ao}__{subj}/ao_results.json"
    if not os.path.isfile(p):
        cell["status"] = "missing"
        out["cells"].append(cell)
        continue
    d = json.load(open(p))
    all_cells = d.get("cells", [])
    if not all_cells:
        cell["status"] = "empty"
        out["cells"].append(cell)
        continue
    stride = max(1, len(all_cells) // N_SAMPLES)
    sampled = [all_cells[i] for i in range(0, len(all_cells), stride)][:N_SAMPLES]
    cell["outputs"] = [{"i": i, "greedy_text": s.get("greedy_text", "")}
                        for i, s in enumerate(sampled)]
    cell["status"] = "ok"
    n_out += len(cell["outputs"])
    out["cells"].append(cell)

out["total_outputs"] = n_out
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"), indent=1)
print(f"saved {OUT}: {len(cells)} cells, {n_out} outputs")
