"""P10 — comprehensive qualitative examples.

For each (concept, regime) scenario, pick one representative capture and show:
  - concept
  - subject_text (decoded from the capture's activations file)
  - base AO greedy output + P(target), rank
  - OWN FT-AO greedy output + P(target), rank
  - CROSS FT-AO greedy output + P(target), rank

Output: markdown table + JSON dump.
"""
import os, json, torch
from transformers import AutoTokenizer

DATA = "<PATH_TO_SCRATCH>/results/ao_xmatrix_v3_lp"
OUT_JSON = "<PATH_TO_SCRATCH>/results/tier_a/p10_qualitative.json"
OUT_MD = "/Users/USER/spherical-steering/projects/ao/Misaligned-Oracles/p10_qualitative_examples.md"

CONCEPTS = ["leaf", "moon", "wave", "flag", "book"]
REGIMES = ["hint", "refusal", "sametext", "think"]

# Choose scenarios: (concept, regime, description)
SCENARIOS = [
    ("leaf",  "hint",     "Cooperative HINT — anti-reader visible"),
    ("moon",  "hint",     "Cooperative HINT — leakage to semantic neighbor?"),
    ("wave",  "sametext", "Cooperative SAMETEXT — extreme anti-reader"),
    ("flag",  "refusal",  "Cooperative REFUSAL — anti-reader"),
    ("book",  "think",    "Cooperative THINK — anti-reader"),
]

# Also include strict scenarios
STRICT_SCENARIOS = [
    ("leaf", "hint",    "Strict HINT — base AO struggles"),
    ("moon", "refusal", "Strict REFUSAL — signal floor"),
]


def coop_subj(c, cv="c1p00"): return f"{c}_{cv}"
def strict_subj(c, cv="c1p00"): return f"strict{c}v2_{cv}"
def coop_ft(c, cv="c1p00"): return f"ours_{c}_{cv}"
def strict_ft(c, cv="c1p00"): return f"ours_strict{c}v2_{cv}"


def load_ao_results(regime, ao, subj):
    p = f"{DATA}/{regime}/{ao}__{subj}/ao_results.json"
    if not os.path.isfile(p): return None
    return json.load(open(p))


def find_matching_cell(ao_res, cell_id):
    """Find a cell by cell_id."""
    for c in ao_res.get("cells", []):
        if c.get("cell_id") == cell_id: return c
    return None


def decode_subject_text(payload, tok):
    """Decode the assistant text portion from a payload's token_ids."""
    tids = payload["token_ids"]
    a_start = payload["assistant_start"]
    a_end = payload["assistant_end"]
    tids_list = tids.tolist() if hasattr(tids, "tolist") else list(tids)
    user_part = tok.decode(tids_list[:a_start], skip_special_tokens=True)
    asst_part = tok.decode(tids_list[a_start:a_end + 1], skip_special_tokens=True)
    return user_part, asst_part


def pick_representative(concept, regime, protocol="coop"):
    """Pick a capture where base AO recovers but OWN FT-AO doesn't."""
    subj_fn = coop_subj if protocol == "coop" else strict_subj
    ft_fn = coop_ft if protocol == "coop" else strict_ft
    subj = subj_fn(concept)
    x_concept = "moon" if concept != "moon" else "wave"

    base = load_ao_results(regime, "ours_base", subj)
    own = load_ao_results(regime, ft_fn(concept), subj)
    cross = load_ao_results(regime, ft_fn(x_concept), subj)
    if not base or not own or not cross: return None

    # Find a capture where:
    #   base greedy recovers target OR has high P(target)
    #   AND own greedy doesn't recover
    for base_cell in base.get("cells", []):
        cid = base_cell.get("cell_id")
        if not cid: continue
        base_prob = base_cell.get("target_prob_at_word_pos") or 0
        base_greedy = (base_cell.get("greedy_text") or "").lower()
        if concept not in base_greedy and base_prob < 0.2: continue

        own_cell = find_matching_cell(own, cid)
        if not own_cell: continue
        own_greedy = (own_cell.get("greedy_text") or "").lower()
        # Prefer cases where OWN doesn't say the target
        if concept in own_greedy: continue

        cross_cell = find_matching_cell(cross, cid)
        if not cross_cell: continue

        return {"base": base_cell, "own": own_cell, "cross": cross_cell,
                "acts_path": base_cell.get("acts_path"), "cell_id": cid,
                "concept": concept, "regime": regime, "protocol": protocol,
                "own_ao": ft_fn(concept), "cross_ao": ft_fn(x_concept)}
    # Fall back: just take the first capture that all 3 share
    for base_cell in base.get("cells", [])[:20]:
        cid = base_cell.get("cell_id")
        own_cell = find_matching_cell(own, cid)
        cross_cell = find_matching_cell(cross, cid)
        if own_cell and cross_cell:
            return {"base": base_cell, "own": own_cell, "cross": cross_cell,
                    "acts_path": base_cell.get("acts_path"), "cell_id": cid,
                    "concept": concept, "regime": regime, "protocol": protocol,
                    "own_ao": ft_fn(concept), "cross_ao": ft_fn(x_concept)}
    return None


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    all_examples = []
    for concept, regime, desc in SCENARIOS:
        ex = pick_representative(concept, regime, "coop")
        if ex: ex["description"] = desc; all_examples.append(ex)
    for concept, regime, desc in STRICT_SCENARIOS:
        ex = pick_representative(concept, regime, "strict")
        if ex: ex["description"] = desc; all_examples.append(ex)

    # Load subject text for each
    for ex in all_examples:
        p = ex["acts_path"]
        try:
            payload = torch.load(p, weights_only=False)
            user, asst = decode_subject_text(payload, tok)
            ex["subject_user_prompt"] = user
            ex["subject_assistant_text"] = asst
        except Exception as e:
            ex["subject_user_prompt"] = f"[error loading {p}: {e}]"
            ex["subject_assistant_text"] = ""

    # Format
    for ex in all_examples:
        for k in ["base", "own", "cross"]:
            c = ex[k]
            ex[f"{k}_greedy"] = c.get("greedy_text", "")
            ex[f"{k}_prob"] = c.get("target_prob_at_word_pos")
            ex[f"{k}_rank"] = c.get("target_rank_at_word_pos")

    # Save JSON
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump([{k: v for k, v in e.items() if k not in ("base", "own", "cross")}
               for e in all_examples],
              open(OUT_JSON, "w"), indent=2, default=str)
    print(f"[saved] {OUT_JSON}")

    # Markdown
    lines = ["# P10 — Qualitative examples (base AO vs. OWN FT-AO vs. CROSS FT-AO)\n"]
    lines.append("Each row shows a specific capture: the Taboo'd subject's actual output text, then what three different AOs generated when they read that capture.\n")

    for i, ex in enumerate(all_examples, 1):
        lines.append(f"\n## Example {i}: {ex['description']}")
        lines.append(f"- **Concept**: {ex['concept']} ({ex['protocol']}) — **Regime**: {ex['regime'].upper()}")
        lines.append(f"- **Cell ID**: `{ex['cell_id']}`")
        lines.append(f"\n**Subject user prompt**:")
        user = (ex.get('subject_user_prompt') or '').strip()
        lines.append(f"```\n{user[:400]}\n```")
        lines.append(f"\n**Subject model's own output** (Taboo'd, hides the concept):")
        asst = (ex.get('subject_assistant_text') or '').strip()
        lines.append(f"```\n{asst[:400]}\n```")

        def fmt_prob(p): return f"{p*100:.1f}%" if p is not None else "—"
        def fmt_rank(r): return str(r) if r is not None else "—"

        lines.append(f"\n**AO extractions**:\n")
        lines.append("| AO | Greedy output | P(target) | Rank |")
        lines.append("|---|---|---|---|")
        lines.append(f"| base AO | {ex['base_greedy'][:120]!r} | {fmt_prob(ex['base_prob'])} | {fmt_rank(ex['base_rank'])} |")
        lines.append(f"| **OWN FT-AO** ({ex['own_ao']}) | {ex['own_greedy'][:120]!r} | {fmt_prob(ex['own_prob'])} | {fmt_rank(ex['own_rank'])} |")
        lines.append(f"| CROSS FT-AO ({ex['cross_ao']}) | {ex['cross_greedy'][:120]!r} | {fmt_prob(ex['cross_prob'])} | {fmt_rank(ex['cross_rank'])} |")
        lines.append("")

    with open(OUT_MD, "w") as f: f.write("\n".join(lines))
    print(f"[saved] {OUT_MD}")

    # Print short summary
    print(f"\n=== {len(all_examples)} qualitative examples built ===")
    for ex in all_examples:
        print(f"  {ex['concept']:<6} {ex['regime']:<10} {ex['protocol']:<6}: "
              f"base='{ex['base_greedy'][:40]}...'  own='{ex['own_greedy'][:40]}...'")


if __name__ == "__main__":
    main()
