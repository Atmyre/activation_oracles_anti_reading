"""AO readout divergence Δ_ℓ across all AO layers (Fig 8 backing data).

Motivation (paper §Mechanisms): the FT-AO's LM-head readout increasingly
suppresses the target concept as we walk deeper into the AO's own layers,
even while the target remains linearly decodable from earlier residual
states. Fig 8 (`fig_ao_readout_V3_divergence.png`) plots

  Δ_ℓ = log₁₀( rank_FT(c*, ℓ) / rank_base(c*, ℓ) )

per (concept, regime) at every AO layer ℓ ∈ [0, L_AO), where `rank_x(c*, ℓ)` is
the median target-concept rank under LogitLens applied at AO layer ℓ, computed
across the ~300 captures per cell.

Reads ao_results.json cells that carry per-layer logit-lens ranks (produced by
`ao_internal_full.py` with `--dump-layer-ranks`) and writes a per-(concept,
regime) trace that plot scripts consume.

Usage:
  python -m anti_reading.analysis.ao_readout_divergence \
      --base results/ao_internal/base_ao.json \
      --ft   results/ao_internal/ft_ao_leaf.json \
      --output results/readout_divergence/leaf.json
"""
import argparse, json, math
from collections import defaultdict
from statistics import median


def per_layer_median_rank(cells, layer):
    """Extract median target rank across captures at a given AO layer.

    Cell schema (from ao_internal_full.py):
      cell["per_layer_target_rank"] : dict[layer_str -> int]
    """
    ranks = []
    for c in cells:
        pl = c.get("per_layer_target_rank") or {}
        r = pl.get(str(layer))
        if r is not None:
            ranks.append(int(r))
    return median(ranks) if ranks else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="ao_internal_full.py output for the BASE AO")
    p.add_argument("--ft",   required=True, help="ao_internal_full.py output for the FT-AO under study")
    p.add_argument("--layers", default=",".join(str(x) for x in range(36)),
                   help="Comma-separated AO layer indices to sweep (default: 0-35 for Qwen3-8B)")
    p.add_argument("--group-by", default="regime",
                   choices=["regime", "concept", "regime_concept"],
                   help="How to group cells before computing Δ_ℓ")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    base_all = json.load(open(args.base)).get("cells", [])
    ft_all   = json.load(open(args.ft)).get("cells", [])
    layers = [int(x) for x in args.layers.split(",")]

    def group_key(c):
        r = c.get("regime")
        cw = c.get("secret_word") or c.get("concept")
        if args.group_by == "regime":
            return r
        if args.group_by == "concept":
            return cw
        return f"{r}__{cw}"

    base_groups = defaultdict(list)
    ft_groups   = defaultdict(list)
    for c in base_all:
        base_groups[group_key(c)].append(c)
    for c in ft_all:
        ft_groups[group_key(c)].append(c)

    result = {}
    for g in sorted(set(base_groups) | set(ft_groups)):
        trace = {"n_base": len(base_groups.get(g, [])), "n_ft": len(ft_groups.get(g, [])),
                 "layers": layers, "delta_log10": []}
        for L in layers:
            rb = per_layer_median_rank(base_groups.get(g, []), L)
            rf = per_layer_median_rank(ft_groups.get(g, []),   L)
            if rb is None or rf is None or rb <= 0 or rf <= 0:
                trace["delta_log10"].append(None)
            else:
                trace["delta_log10"].append(math.log10(rf / rb))
        result[g] = trace
        print(f"  {g:<30s}  n_base={trace['n_base']:3d}  n_ft={trace['n_ft']:3d}")

    json.dump(result, open(args.output, "w"), indent=2)
    print(f"[wrote] {args.output}")


if __name__ == "__main__":
    main()
