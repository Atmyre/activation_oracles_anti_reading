"""Full AO × subject × regime matrix with top1, p_target (sample-level prob), entropy."""
import json, os, math
from collections import OrderedDict

AOS = ['ours_base',
       'ours_leaf_c1p00', 'ours_leaf_c0p50',
       'ours_moon_c1p00', 'ours_moon_c0p50',
       'ours_bcywclock_c1p00', 'ours_bcywclock_c0p50',
       'ours_strictclock_c1p00', 'ours_strictclock_c0p50']

SUBJ_FAMILIES = OrderedDict([
    ('bcyw clock',       ['clock_c1p00', 'clock_c0p50']),
    ('bcyw leaf',        ['leaf_c1p00', 'leaf_c0p50']),
    ('bcyw moon',        ['moon_c1p00', 'moon_c0p50']),
    ('bcyw wave',        ['wave_c1p00', 'wave_c0p50']),
    ('strict clock',     ['strictclock_c1p00', 'strictclock_c0p50']),
    ('strict leaf (v2)', ['strictleafv2_c1p00', 'strictleafv2_c0p50']),
    ('strict moon (v2)', ['strictmoonv2_c1p00', 'strictmoonv2_c0p50']),
])
def target_for(s):
    if 'clock' in s: return 'clock'
    if 'leaf' in s:  return 'leaf'
    if 'moon' in s:  return 'moon'
    if 'wave' in s:  return 'wave'

REGIMES = ['hint', 'refusal', 'sametext']


def shannon(probs):
    return -sum(p * math.log2(p) for p in probs if p > 0)


def cell_metrics(ao, subj, regime):
    base = os.path.expandvars('${PATH_TO_FOLDER}/results')
    d = f'{base}/ao_ftao_matrix_sametext50' if regime == 'sametext' else f'{base}/ao_ftao_matrix_{regime}50'
    p = f'{d}/{ao}__{subj}/ao_results.json'
    if not os.path.exists(p): return None
    try: data = json.load(open(p))
    except: return None
    target = target_for(subj)
    n = 0; correct = 0; p_target_list = []; entropy_list = []
    for c in data['cells']:
        vc = c.get('vote_counts', {})
        total = sum(vc.values())
        if total == 0: continue
        n += 1
        p_target_list.append(vc.get(target, 0) / total)
        # Shannon entropy over the 20 samples' word distribution
        probs = [v / total for v in vc.values()]
        entropy_list.append(shannon(probs))
        if c.get('plurality_word') == target: correct += 1
    if n == 0: return None
    return {
        'top1': correct / n,
        'p_target': sum(p_target_list) / n,
        'entropy': sum(entropy_list) / n,
        'n': n,
    }


# Compute all
rows = []
for ao in AOS:
    for fam, subjs in SUBJ_FAMILIES.items():
        for s in subjs:
            for r in REGIMES:
                m = cell_metrics(ao, s, r)
                if m is None: continue
                rows.append({'ao': ao, 'family': fam, 'subject': s, 'regime': r, **m})

# Save CSV-style
out_path = os.path.expandvars('${PATH_TO_FOLDER}/results/ao_ftao_matrix_sametext50/full_matrix_with_entropy.json')
json.dump(rows, open(out_path, 'w'), indent=2)
print(f'Wrote {out_path}; n_cells = {len(rows)}')

# Print one table per regime — three metrics interleaved
def fmt(x, w=5):
    return ' ' * w if x is None else f'{x:>{w}.2f}'

for regime in REGIMES:
    print(f'\n=== {regime.upper()} === (top1 / p_target / entropy)')
    cell_map = {(r['ao'], r['subject']): r for r in rows if r['regime'] == regime}
    print(f"  {'AO':<32} | " + ' | '.join(f'{fam:^22}' for fam in SUBJ_FAMILIES))
    for ao in AOS:
        row = f'  {ao:<32} '
        for fam, subjs in SUBJ_FAMILIES.items():
            cells = []
            for s in subjs:
                m = cell_map.get((ao, s))
                if m is None:
                    cells.append('  -   ')
                else:
                    cells.append(f"{m['top1']*100:>3.0f}/{m['p_target']*100:>3.0f}/{m['entropy']:>3.1f}")
            row += '| ' + ' '.join(cells) + ' '
        print(row)

# Aggregate per regime: mean across subjects of (top1, p_target, entropy) per AO
print('\n\n=== PER-AO REGIME AGGREGATES (mean across all available subjects) ===')
for ao in AOS:
    print(f'  {ao}:')
    for r in REGIMES:
        rs = [x for x in rows if x['ao'] == ao and x['regime'] == r]
        if not rs: continue
        mt = sum(x['top1'] for x in rs) / len(rs)
        mp = sum(x['p_target'] for x in rs) / len(rs)
        me = sum(x['entropy'] for x in rs) / len(rs)
        print(f"    {r:<10}  n={len(rs):<3}  top1={mt*100:5.1f}%  p_target={mp*100:5.1f}%  entropy={me:5.2f} bits")
