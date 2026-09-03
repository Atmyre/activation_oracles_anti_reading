#!/usr/bin/env python3
"""Aggregate c-sweep results across 4 models × 3 words × 8 c × 3 regimes.

Outputs:
  - Per-model hint plurality curve vs c (1 line per word)
  - Per-model refusal plurality curve vs c
  - Per-model same-text plurality curve vs c
  - Cross-model "AO invisibility" summary: at which c does AO recovery drop below threshold?
"""
import json, glob, os, re

MODELS = ['q17', 'q8', 'q14', 'gma']
WORDS = ['leaf', 'moon', 'wave']
C_TAGS = ['c0p10', 'c0p25', 'c0p33', 'c0p50', 'c0p67', 'c0p75', 'c0p91', 'c1p00']
C_VALS = [0.10, 0.25, 0.333, 0.50, 0.667, 0.75, 0.909, 1.0]

ROOTS = {
    'hint':     ('<PATH_TO_SCRATCH>/results/ao_csweep_eval', '{m}_{w}_{c}'),
    'refusal':  ('<PATH_TO_SCRATCH>/results/ao_csweep_refusal_eval', '{m}_{w}_{c}'),
    'sametext': ('<PATH_TO_SCRATCH>/results/ao_csweep_sametext', '{m}/{w}_{c}'),
}

def load_cell(model, word, c_tag, regime):
    root, pattern = ROOTS[regime]
    cell_dir_name = pattern.format(m=model, w=word, c=c_tag)
    p = f'{root}/{cell_dir_name}/ao_results.json'
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    n = len(d['cells'])
    correct = sum(1 for c in d['cells']
                  if (c.get('plurality_word') or '').lower() == word)
    any_correct = sum(1 for c in d['cells']
                      if word in [w.lower() for w in c['sample_words']])
    return {'n': n, 'correct': correct, 'any': any_correct,
            'plurality_words': [(c.get('plurality_word') or '').lower() for c in d['cells']]}

for regime in ['hint', 'refusal', 'sametext']:
    print(f'\n=== {regime.upper()} plurality recovery ===\n')
    header = f'{"model":<5} {"word":<6} ' + ' '.join(f'{c:>6.3f}' for c in C_VALS)
    print(header)
    print('-' * len(header))
    for model in MODELS:
        for word in WORDS:
            vals = []
            for c_tag in C_TAGS:
                cell = load_cell(model, word, c_tag, regime)
                if cell is None:
                    vals.append('  -')
                else:
                    vals.append(f'{cell["correct"]}/{cell["n"]}')
            print(f'{model:<5} {word:<6} ' + ' '.join(f'{v:>6}' for v in vals))
        print()  # spacer
    # Per-model overall mean (across all 3 words)
    print('Overall mean (3 words pooled):')
    for model in MODELS:
        means = []
        for c_tag in C_TAGS:
            tot_n = tot_c = 0
            for word in WORDS:
                cell = load_cell(model, word, c_tag, regime)
                if cell is not None:
                    tot_n += cell['n']; tot_c += cell['correct']
            if tot_n > 0:
                means.append(f'{100*tot_c/tot_n:>5.0f}%')
            else:
                means.append('   -')
        print(f'  {model:<5}       ' + ' '.join(f'{v:>6}' for v in means))

print('\n=== Aggregator DONE ===')
