#!/usr/bin/env python3
"""Aggregate FT-AO 5×6×3 cross-matrix into per-regime tables.

For each (AO, subject_word, subject_c, regime) cell, compute:
  - plurality_rec: fraction of 5 (or 3) prompt-cells where plurality_word == true_word
  - samples_rec_pct: average fraction of 20 stochastic samples that wrote the true word
  - target_max_prob: average max prob the true word reached in any topk position
  - top15_entropy: average renormalized top-15 entropy

Output: per-regime table (5 AOs rows × 6 subjects cols) of each metric.
"""
import json, os, math, re
from collections import defaultdict

ROOT = '/gpfs/scratch/USER/results/ao_ftao_matrix'

AOS = ['ours_base', 'ours_leaf_c1p00', 'ours_leaf_c0p50', 'ours_moon_c1p00', 'ours_moon_c0p50']
SUBJECTS = [('leaf', 'c1p00'), ('leaf', 'c0p50'),
            ('moon', 'c1p00'), ('moon', 'c0p50'),
            ('wave', 'c1p00'), ('wave', 'c0p50')]
REGIMES = ['hint', 'refusal', 'sametext']


def renormalized_topk_entropy(topk):
    probs = [t.get('prob', 0.0) for t in topk]
    S = sum(probs)
    if S <= 0: return 0.0
    H = 0.0
    for p in probs:
        if p > 0:
            q = p / S
            H -= q * math.log(q)
    return H


def cell_metrics(cells, true_word):
    if not cells: return None
    pl_recs, samples_recs, tmps, ents = [], [], [], []
    for c in cells:
        pl = (c.get('plurality_word') or '').lower()
        pl_recs.append(int(pl == true_word))
        sw = c.get('sample_words') or [s.lower() for s in c.get('samples', [])]
        # plurality_word may be derived from samples; recount samples-with-target
        n_match = 0
        n_total = 0
        for s in c.get('samples', []):
            if not isinstance(s, str): continue
            n_total += 1
            # extract first 'word' in quotes from sample text
            m = re.search(r"['\"]([A-Za-z]+)['\"]", s)
            if m and m.group(1).lower() == true_word:
                n_match += 1
        if n_total > 0:
            samples_recs.append(100 * n_match / n_total)

        tmp = 0.0
        for pp in c.get('per_position', []):
            ents.append(renormalized_topk_entropy(pp.get('topk', [])))
            for t in pp.get('topk', []):
                tok = t.get('token', '').strip().lower().strip("'\"")
                if tok == true_word or true_word in tok:
                    tmp = max(tmp, t.get('prob', 0.0))
        tmps.append(tmp)
    return {
        'pl_pct': 100 * sum(pl_recs) / len(pl_recs),
        'samples_rec_pct': sum(samples_recs) / max(len(samples_recs), 1),
        'tmp_mean': sum(tmps) / len(tmps),
        'entropy_mean': sum(ents) / max(len(ents), 1),
    }


# Collect all cells
matrix = defaultdict(dict)  # (regime, ao) -> {subject: metrics}
for ao in AOS:
    for w, c in SUBJECTS:
        for regime in REGIMES:
            p = f'{ROOT}/{ao}__{w}_{c}__{regime}/ao_results.json'
            if not os.path.exists(p):
                print(f'MISSING: {p}')
                continue
            d = json.load(open(p))
            m = cell_metrics(d.get('cells', []), w)
            if m is None: continue
            matrix[(regime, ao)][(w, c)] = m


# Print per-regime tables
for regime in REGIMES:
    print('=' * 105)
    print(f'### {regime.upper()} regime')
    print('=' * 105)

    # Header
    subj_labels = [f'{w}/{c[1:].replace("p", ".")}' for w, c in SUBJECTS]
    print()
    print('### plurality % (5/5 = 100%, 0/5 = 0)')
    print(f'{"AO":<22} ' + ' '.join(f'{s:>11}' for s in subj_labels))
    print('-' * (24 + 12 * len(SUBJECTS)))
    for ao in AOS:
        row = [ao]
        for s in SUBJECTS:
            m = matrix[(regime, ao)].get(s)
            row.append(f'{m["pl_pct"]:>10.0f}%' if m else '         -')
        print(f'{row[0]:<22} ' + ' '.join(f'{v:>11}' for v in row[1:]))

    print()
    print('### target_max_prob (mean of greedy-positions, across 5 prompt cells)')
    print(f'{"AO":<22} ' + ' '.join(f'{s:>11}' for s in subj_labels))
    print('-' * (24 + 12 * len(SUBJECTS)))
    for ao in AOS:
        row = [ao]
        for s in SUBJECTS:
            m = matrix[(regime, ao)].get(s)
            row.append(f'{m["tmp_mean"]:>11.3f}' if m else '         -')
        print(f'{row[0]:<22} ' + ' '.join(f'{v:>11}' for v in row[1:]))

    print()
    print('### samples_rec_pct (mean fraction of 20 stochastic samples saying the true word)')
    print(f'{"AO":<22} ' + ' '.join(f'{s:>11}' for s in subj_labels))
    print('-' * (24 + 12 * len(SUBJECTS)))
    for ao in AOS:
        row = [ao]
        for s in SUBJECTS:
            m = matrix[(regime, ao)].get(s)
            row.append(f'{m["samples_rec_pct"]:>10.0f}%' if m else '         -')
        print(f'{row[0]:<22} ' + ' '.join(f'{v:>11}' for v in row[1:]))

    print()
    print('### top15_entropy (lower = AO more committed)')
    print(f'{"AO":<22} ' + ' '.join(f'{s:>11}' for s in subj_labels))
    print('-' * (24 + 12 * len(SUBJECTS)))
    for ao in AOS:
        row = [ao]
        for s in SUBJECTS:
            m = matrix[(regime, ao)].get(s)
            row.append(f'{m["entropy_mean"]:>11.3f}' if m else '         -')
        print(f'{row[0]:<22} ' + ' '.join(f'{v:>11}' for v in row[1:]))
    print()
