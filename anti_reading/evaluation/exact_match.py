"""Substring exact-match indicator for AO greedy outputs (Appendix D.4).

Definition (from paper): for a greedy string g and target concept c*,
substring-match is true iff lowercased target string appears in lowercased
generated output. Thus "books" matches "book", "leafy" matches "leaf", but
"banner" does NOT match "flag" (that would be tier-2 semantic recovery via
the Sonnet judge, not tier-3 exact).

The paper's main "Exact recovery" number is the Sonnet tier-3 rate; this
substring metric is a cheap sanity check that agrees with the judge on most
cells and is used to spot-check judge outputs.
"""
import argparse, json, re
from pathlib import Path


def substring_match(greedy: str, concept: str) -> bool:
    return concept.lower() in (greedy or "").lower()


def substring_rate(entries, concept):
    """Fraction of greedy outputs whose lowercased text contains the concept."""
    if not entries:
        return 0.0
    hits = sum(1 for e in entries if substring_match(e.get("greedy_text", ""), concept))
    return hits / len(entries)


def main():
    p = argparse.ArgumentParser(description="Compute substring exact-match rate on ao_results.json cells.")
    p.add_argument("--input", required=True,
                   help="ao_results.json (cells: [{greedy_text, secret_word, ...}])")
    p.add_argument("--output", help="Optional per-concept summary JSON")
    args = p.parse_args()

    d = json.load(open(args.input))
    cells = d.get("cells", d if isinstance(d, list) else [])
    by_concept = {}
    for c in cells:
        cw = c.get("secret_word") or c.get("target") or c.get("concept")
        if not cw:
            continue
        by_concept.setdefault(cw, []).append(c)

    summary = {}
    for cw, entries in sorted(by_concept.items()):
        rate = substring_rate(entries, cw)
        summary[cw] = {"n": len(entries), "substring_exact_rate": rate}
        print(f"  {cw:>6s}  n={len(entries):3d}  substring_exact_rate={rate:.3f}")

    if args.output:
        json.dump(summary, open(args.output, "w"), indent=2)
        print(f"[wrote] {args.output}")


if __name__ == "__main__":
    main()
