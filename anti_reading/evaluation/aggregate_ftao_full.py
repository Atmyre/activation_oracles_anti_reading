#!/usr/bin/env python3
"""Enhanced aggregator for FT-AO matrix with entropy + top-1 prob analysis.

Adds: mean_top1_prob (per-position max prob from greedy), mean_chosen_prob
(per-position chosen-token prob), top15_entropy, ratio top1/chosen-prob —
to disambiguate 'AO is sure of *something*' from 'AO is sure of the *target*'.
"""
import json, os, math
from collections import defaultdict

ROOT = os.path.expandvars('${PATH_TO_FOLDER}/results/ao_ftao_matrix')

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
    pl_recs, tmps, mean_top1s, mean_chosens, ents = [], [], [], [], []
    for c in cells:
        pl = (c.get('plurality_word') or '').lower()
        pl_recs.append(int(pl == true_word))

        tmp = 0.0
        cell_top1s, cell_chosens, cell_ents = [], [], []
        for pp in c.get('per_position', []):
            topk = pp.get('topk', [])
            if topk:
                cell_top1s.append(topk[0].get('prob', 0.0))
            cell_chosens.append(pp.get('chosen_prob', 0.0))
            cell_ents.append(renormalized_topk_entropy(topk))
            for t in topk:
                tok = t.get('token', '').strip().lower().strip("'\"")
                if tok == true_word or true_word in tok:
                    tmp = max(tmp, t.get('prob', 0.0))
        tmps.append(tmp)
        if cell_top1s:
            mean_top1s.append(sum(cell_top1s)/len(cell_top1s))
        if cell_chosens:
            mean_chosens.append(sum(cell_chosens)/len(cell_chosens))
        if cell_ents:
            ents.append(sum(cell_ents)/len(cell_ents))
    return {
        'pl_pct': 100 * sum(pl_recs) / len(pl_recs),
        'tmp': sum(tmps)/len(tmps),
        'mean_top1_prob': sum(mean_top1s)/max(len(mean_top1s),1),
        'mean_chosen_prob': sum(mean_chosens)/max(len(mean_chosens),1),
        'entropy': sum(ents)/max(len(ents),1),
    }


matrix = defaultdict(dict)
for ao in AOS:
    for w, c in SUBJECTS:
        for regime in REGIMES:
            p = f'{ROOT}/{ao}__{w}_{c}__{regime}/ao_results.json'
            if not os.path.exists(p): continue
            d = json.load(open(p))
            m = cell_metrics(d.get('cells', []), w)
            if m: matrix[(regime, ao)][(w, c)] = m


def fmt_subj(w, c):
    return f'{w}/{c[1:].replace("p", ".")}'


for regime in REGIMES:
    print('=' * 110)
    print(f'### {regime.upper()} regime')
    print('=' * 110)

    metrics_to_print = [
        ('top1_prob (AO confidence in WHATEVER it says)', 'mean_top1_prob', '{:.3f}'),
        ('target_max_prob (AO confidence in TRUE word)', 'tmp', '{:.3f}'),
        ('ratio tmp/top1 (target as fraction of confidence)', None, '{:.2f}'),
        ('top15_entropy (lower = AO more committed)', 'entropy', '{:.3f}'),
    ]
    subj_labels = [fmt_subj(w, c) for w, c in SUBJECTS]

    for label, key, fmt in metrics_to_print:
        print()
        print(f'### {label}')
        print(f'{"AO":<22} ' + ' '.join(f'{s:>11}' for s in subj_labels))
        print('-' * (24 + 12 * len(SUBJECTS)))
        for ao in AOS:
            row = [ao]
            for s in SUBJECTS:
                m = matrix[(regime, ao)].get(s)
                if not m:
                    row.append('         -')
                elif key is None:
                    # ratio
                    ratio = m['tmp'] / max(m['mean_top1_prob'], 1e-9)
                    row.append(f'{fmt.format(ratio):>11}')
                else:
                    row.append(f'{fmt.format(m[key]):>11}')
            print(f'{row[0]:<22} ' + ' '.join(f'{v:>11}' for v in row[1:]))
    print()
