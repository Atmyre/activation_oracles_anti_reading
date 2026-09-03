#!/usr/bin/env python3
"""Enriched re-analysis of c-sweep AO outputs: plurality + top-N + confidence + ENTROPY.

For each existing ao_results.json across hint/refusal/sametext × 3 models × 3 words × 8 c,
compute six metrics per cell:
  1. plurality_recovered           — plurality_word == true_word
  2. samples_with_target_pct       — fraction of 20 stochastic samples that wrote the true word
  3. target_max_prob_in_topk       — max prob the true word reached in any greedy topk position
  4. mean_chosen_prob              — average chosen_prob across all greedy positions (fluency baseline)
  5. mean_top15_entropy            — average over positions of H(renormalized top-15 distribution)
  6. mean_effective_k              — exp(mean_top15_entropy) ≈ "effective # choices"

No new GPU compute — just JSON walks.
"""
import json, os, glob, re, math
from collections import defaultdict

MODELS = ['q8', 'q14', 'gma']
WORDS = ['leaf', 'moon', 'wave']
C_TAGS = ['c0p10', 'c0p25', 'c0p33', 'c0p50', 'c0p67', 'c0p75', 'c0p91', 'c1p00']
C_VALS = [0.10, 0.25, 0.333, 0.50, 0.667, 0.75, 0.909, 1.0]

REGIMES = {
    'hint':     ('<PATH_TO_SCRATCH>/results/ao_csweep_eval', '{m}_{w}_{c}'),
    'refusal':  ('<PATH_TO_SCRATCH>/results/ao_csweep_refusal_eval', '{m}_{w}_{c}'),
    'sametext': ('<PATH_TO_SCRATCH>/results/ao_csweep_sametext', '{m}/{w}_{c}'),
}


def renormalized_topk_entropy(topk):
    """Compute entropy of the renormalized top-K distribution.

    The JSON gives top-15 probs that sum to S < 1 (tail mass missing). After
    renormalization p_i / S they sum to 1; the resulting entropy measures
    'concentration on the top-K choices we can see'. Lower = more peaked.
    Returns 0.0 if topk empty.
    """
    probs = [t.get('prob', 0.0) for t in topk]
    S = sum(probs)
    if S <= 0: return 0.0
    H = 0.0
    for p in probs:
        if p > 0:
            q = p / S
            H -= q * math.log(q)
    return H


def metrics_for_cell(cell, true_word):
    """Compute the 6 enriched metrics for a single AO cell."""
    pl = (cell.get('plurality_word') or '').lower()
    pl_rec = (pl == true_word)
    samples_with_target = sum(1 for sw in cell.get('sample_words', []) if sw.lower() == true_word)
    n_samples = len(cell.get('sample_words', []))

    target_max_prob = 0.0
    chosen_probs = []
    entropies = []
    for pp in cell.get('per_position', []):
        chosen_probs.append(pp.get('chosen_prob', 0.0))
        topk = pp.get('topk', [])
        entropies.append(renormalized_topk_entropy(topk))
        for t in topk:
            tok = t.get('token', '').strip().lower().strip("'\"")
            if tok == true_word or true_word in tok:
                pp_prob = t.get('prob', 0.0)
                if pp_prob > target_max_prob:
                    target_max_prob = pp_prob

    mean_chosen = sum(chosen_probs) / len(chosen_probs) if chosen_probs else 0.0
    mean_entropy = sum(entropies) / len(entropies) if entropies else 0.0
    eff_k = math.exp(mean_entropy)

    return {
        'plurality_rec': int(pl_rec),
        'samples_with_target': samples_with_target,
        'n_samples': n_samples,
        'target_max_prob': target_max_prob,
        'mean_chosen_prob': mean_chosen,
        'mean_top15_entropy': mean_entropy,
        'effective_k': eff_k,
        'plurality_word': pl,
    }


def load_cell_metrics(model, word, c_tag, regime):
    root, pattern = REGIMES[regime]
    cell_path = f'{root}/{pattern.format(m=model, w=word, c=c_tag)}/ao_results.json'
    if not os.path.exists(cell_path): return None
    d = json.load(open(cell_path))
    return [metrics_for_cell(c, word) for c in d['cells']]


def aggregate_pool(model, c_tag, regime, words=WORDS):
    pooled = defaultdict(list)
    for w in words:
        cells = load_cell_metrics(model, w, c_tag, regime)
        if cells is None: continue
        for c in cells:
            pooled['pl'].append(c['plurality_rec'])
            pooled['samples_rec_pct'].append(100 * c['samples_with_target'] / max(c['n_samples'], 1))
            pooled['target_max_prob'].append(c['target_max_prob'])
            pooled['mean_chosen_prob'].append(c['mean_chosen_prob'])
            pooled['mean_entropy'].append(c['mean_top15_entropy'])
            pooled['eff_k'].append(c['effective_k'])
    if not pooled['pl']: return None
    return {
        'pl_pct': 100 * sum(pooled['pl']) / len(pooled['pl']),
        'samples_rec_pct_mean': sum(pooled['samples_rec_pct']) / len(pooled['samples_rec_pct']),
        'target_max_prob_mean': sum(pooled['target_max_prob']) / len(pooled['target_max_prob']),
        'mean_chosen_prob_mean': sum(pooled['mean_chosen_prob']) / len(pooled['mean_chosen_prob']),
        'mean_entropy_mean': sum(pooled['mean_entropy']) / len(pooled['mean_entropy']),
        'eff_k_mean': sum(pooled['eff_k']) / len(pooled['eff_k']),
    }


print('=' * 105)
print('ENRICHED C-SWEEP METRICS WITH ENTROPY (Q1.7B dropped; pooled across 3 words per cell)')
print('=' * 105)
for regime in ['hint', 'refusal', 'sametext']:
    print(f'\n### {regime.upper()} regime\n')
    print(f'{"model":<6} {"metric":<22} ' + ' '.join(f'{c:>6.2f}' for c in C_VALS))
    print('-' * (29 + 7 * len(C_VALS)))
    for model in MODELS:
        rows = {
            'pl%':              [],
            'samples-rec %':    [],
            'target-max-prob':  [],
            'mean-chosen-prob': [],
            'top15-entropy':    [],
            'effective-k':      [],
        }
        for c_tag in C_TAGS:
            m = aggregate_pool(model, c_tag, regime)
            if m is None:
                for k in rows: rows[k].append('-')
            else:
                rows['pl%'].append(f'{m["pl_pct"]:>5.0f}%')
                rows['samples-rec %'].append(f'{m["samples_rec_pct_mean"]:>5.0f}%')
                rows['target-max-prob'].append(f'{m["target_max_prob_mean"]:>5.2f}')
                rows['mean-chosen-prob'].append(f'{m["mean_chosen_prob_mean"]:>5.2f}')
                rows['top15-entropy'].append(f'{m["mean_entropy_mean"]:>5.2f}')
                rows['effective-k'].append(f'{m["eff_k_mean"]:>5.2f}')
        for k, vs in rows.items():
            print(f'{model:<6} {k:<22} ' + ' '.join(f'{v:>6}' for v in vs))
        print()


print('=' * 105)
print('Q14B WAVE DIAGNOSTIC (with entropy)')
print('=' * 105)
print(f'{"regime":<10} {"c":<6} {"plurality":<14} {"sample-rec":<12} {"target-max-prob":<14} {"chosen":<8} {"entropy":<8} {"eff-k":<6}')
for regime in ['hint', 'refusal', 'sametext']:
    for c_tag, c in zip(C_TAGS, C_VALS):
        cells = load_cell_metrics('q14', 'wave', c_tag, regime)
        if not cells: continue
        rec = sum(c['plurality_rec'] for c in cells)
        srec_mean = sum(c['samples_with_target'] / max(c['n_samples'], 1) for c in cells) / len(cells)
        tmax_mean = sum(c['target_max_prob'] for c in cells) / len(cells)
        ccp_mean = sum(c['mean_chosen_prob'] for c in cells) / len(cells)
        ent_mean = sum(c['mean_top15_entropy'] for c in cells) / len(cells)
        effk_mean = sum(c['effective_k'] for c in cells) / len(cells)
        pls = [c['plurality_word'] for c in cells]
        pl_str = ','.join(p[:6] for p in pls[:3])
        print(f'{regime:<10} {c:<6.2f} {rec}/{len(cells):<5} ({pl_str:<8}) {100*srec_mean:>5.0f}%      {tmax_mean:>10.3f}     {ccp_mean:>6.2f}   {ent_mean:>6.2f}   {effk_mean:>5.2f}')
    print()


# Save full per-cell metrics
out_dir = '<PATH_TO_SCRATCH>/results'
per_cell = {}
for model in MODELS:
    for word in WORDS:
        for c_tag in C_TAGS:
            for regime in REGIMES:
                cells = load_cell_metrics(model, word, c_tag, regime)
                if cells is None: continue
                key = f'{model}_{word}_{c_tag}_{regime}'
                per_cell[key] = cells
json.dump(per_cell, open(f'{out_dir}/ao_csweep_uncertainty_per_cell.json', 'w'), indent=2)
print(f'\nWROTE {out_dir}/ao_csweep_uncertainty_per_cell.json ({len(per_cell)} cells)')
