#!/usr/bin/env python3
"""D4-leak verdict analysis.

Compares ours_base AO recovery on strictclock_c1p00 (D3+D4) vs strictclock_D3only_c1p00.

VERDICT logic:
- If P(target='clock') and top1 accuracy are similar (Δ ≤ 10pp) across both subjects
  → D4 does NOT meaningfully install concept signal → safe to keep D4 in strict training.
- If P(target) on D3-only is significantly lower (Δ > 10pp)
  → D4 was the source of concept signal → all strict subjects must be retrained without D4.
"""
import json, math, os
from collections import Counter

ROOT = '<PATH_TO_SCRATCH>/results/ao_d4verdict_eval'
REGIMES = ['hint', 'refusal', 'sametext']
SUBJECTS = ['strictclock_c1p00', 'strictclock_D3only_c1p00']
TARGET = 'clock'


def shannon(probs):
    return -sum(p * math.log2(p) for p in probs if p > 0)


def cell(regime, subj):
    p = f'{ROOT}/{regime}/ours_base__{subj}/ao_results.json'
    if not os.path.exists(p):
        return None
    data = json.load(open(p))
    n = 0; correct = 0
    p_target_list = []; entropy_list = []
    for c in data['cells']:
        vc = c.get('vote_counts', {})
        total = sum(vc.values())
        if total == 0: continue
        n += 1
        p_target_list.append(vc.get(TARGET, 0) / total)
        entropy_list.append(shannon([v / total for v in vc.values()]))
        if c.get('plurality_word') == TARGET: correct += 1
    if n == 0: return None
    return {
        'n': n,
        'top1': correct / n,
        'p_target': sum(p_target_list) / n,
        'entropy': sum(entropy_list) / n,
    }


def main():
    print('=' * 70)
    print(f'D4-LEAK VERDICT: ours_base AO reading "{TARGET}"')
    print('=' * 70)

    rows = []
    for regime in REGIMES:
        row = {'regime': regime}
        for subj in SUBJECTS:
            m = cell(regime, subj)
            row[subj] = m
        rows.append(row)

    print()
    print(f'{"regime":<12} | {"D3+D4 top1":<11} | {"D3-only top1":<13} | {"Δtop1":<7} | '
          f'{"D3+D4 p_t":<10} | {"D3-only p_t":<12} | {"Δp_t":<7} | verdict')
    print('-' * 130)

    deltas_top1 = []
    deltas_p = []
    for r in rows:
        a = r.get('strictclock_c1p00')
        b = r.get('strictclock_D3only_c1p00')
        if a is None or b is None:
            print(f'{r["regime"]:<12} | MISSING DATA')
            continue
        d_top1 = a['top1'] - b['top1']
        d_p = a['p_target'] - b['p_target']
        deltas_top1.append(d_top1)
        deltas_p.append(d_p)
        verdict = '⚠ LEAK' if d_p > 0.10 else '✓ clean'
        print(f'{r["regime"]:<12} | {a["top1"]:.3f}      | {b["top1"]:.3f}        | '
              f'{d_top1:+.3f} | {a["p_target"]:.3f}    | {b["p_target"]:.3f}      | '
              f'{d_p:+.3f} | {verdict}')

    print()
    print('=' * 70)
    if not deltas_p:
        print('NO DATA — cannot decide.')
        return
    mean_dp = sum(deltas_p) / len(deltas_p)
    max_dp = max(deltas_p)
    print(f'Mean Δp_target (D3+D4 − D3-only) across regimes: {mean_dp:+.3f}')
    print(f'Max  Δp_target (worst regime):                   {max_dp:+.3f}')
    print()
    if max_dp <= 0.10:
        print('✓ VERDICT: CLEAN — D4 does not install significant concept signal.')
        print('  → Safe to use existing D3+D4 strict recipe for all subjects in P2.1.')
    elif max_dp <= 0.20:
        print('? VERDICT: MARGINAL — D4 contributes modest signal.')
        print('  → Recommended: switch to D3-only training for all strict subjects.')
    else:
        print('⚠ VERDICT: LEAK — D4 is a major source of the concept signal.')
        print('  → All strict subjects MUST be retrained without D4 (use D3-only recipe).')


if __name__ == '__main__':
    main()
