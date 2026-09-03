'''Per-cell aggregation keeping per-(AO,subj) pair intact.
Output structure:
  final[regime][proto_cX][ao_type][subj_concept][ao_concept] = {exact_recovery, p_target_mean, rank_median}
Downstream: for std, group by (ao_type × proto × c), collect one value per (ao_concept, subj_concept) cell, and compute std over the list.
'''
import json, os, glob, re
from collections import defaultdict

BASE = '<PATH_TO_SCRATCH>/results/ao_xmatrix_v3_lp'
CONCEPTS = ['book','flag','leaf','moon','wave']
OUT = '<PATH_TO_SCRATCH>/results/ao_types_percell.json'

def parse_ao(tag):
    if tag == 'ours_base': return 'base', None, None
    m = re.match(r'^ours_strict(\w+?)v2_c(\d)p(\d\d)$', tag)
    if m: return m.group(1),'strict',float(f'{m.group(2)}.{m.group(3)}')
    m = re.match(r'^ours_(\w+?)_c(\d)p(\d\d)$', tag)
    if m: return m.group(1),'coop',float(f'{m.group(2)}.{m.group(3)}')
    return None,None,None

def parse_subj(subj):
    m = re.match(r'^strict(\w+?)v2_c(\d)p(\d\d)$', subj)
    if m: return m.group(1),'strict',float(f'{m.group(2)}.{m.group(3)}')
    m = re.match(r'^(\w+?)_c(\d)p(\d\d)$', subj)
    if m: return m.group(1),'coop',float(f'{m.group(2)}.{m.group(3)}')
    return None,None,None

results = {}
for regime in ['hint','refusal','sametext','think','offtopic']:
    results[regime] = {}
    for cell_dir in sorted(glob.glob(f'{BASE}/{regime}/*')):
        name = os.path.basename(cell_dir)
        if '__target_' in name or '__' not in name: continue
        ao_tag, subj_tag = name.split('__', 1)
        ao_c, ao_p, ao_cv = parse_ao(ao_tag)
        subj_c, subj_p, subj_cv = parse_subj(subj_tag)
        if subj_c not in CONCEPTS: continue
        if subj_cv not in (0.5, 1.0): continue
        if ao_c != 'base':
            if ao_p != subj_p: continue
            if ao_cv != subj_cv: continue
        if ao_c == 'base':
            ao_type = 'base'
        elif ao_c == subj_c:
            ao_type = 'own'
        elif ao_c in CONCEPTS:
            ao_type = 'cross'
        else:
            continue
        p = f'{cell_dir}/ao_results.json'
        if not os.path.isfile(p): continue
        try: rj = json.load(open(p))
        except: continue
        cells = rj.get('cells', [])
        if not cells: continue

        recovered = 0; ps = []; ranks = []
        for c_ in cells:
            if c_.get('greedy_recovered') in (True,'True'): recovered += 1
            pv = c_.get('target_prob_at_word_pos')
            rv = c_.get('target_rank_at_word_pos')
            if pv not in (None,'None'):
                try: ps.append(float(pv))
                except: pass
            if rv not in (None,'None'):
                try: ranks.append(int(rv))
                except: pass
        ranks_sorted = sorted(ranks)
        n_captures = len(cells)
        # keys: regime, (proto, c), ao_type, (ao_concept, subj_concept)
        key = f'{subj_p}_c{subj_cv}'
        if key not in results[regime]: results[regime][key] = {}
        if ao_type not in results[regime][key]: results[regime][key][ao_type] = []
        results[regime][key][ao_type].append({
            'ao_concept': ao_c if ao_c else 'base',
            'subj_concept': subj_c,
            'n_captures': n_captures,
            'exact_recovery': recovered / max(1,n_captures),
            'p_target_mean': (sum(ps)/len(ps)) if ps else 0.0,
            'rank_median': ranks_sorted[len(ranks_sorted)//2] if ranks_sorted else -1,
        })

with open(OUT,'w') as f: json.dump(results, f, indent=1)
print(f'wrote {OUT}')

# Sanity summary
import statistics
for reg in ['hint']:
    print(f'\n=== {reg} ===')
    for k, ao_types in results[reg].items():
        print(f'  {k}:')
        for ao_type, entries in ao_types.items():
            if not entries: continue
            e_r = [x['exact_recovery'] for x in entries]
            p = [x['p_target_mean'] for x in entries]
            r = [x['rank_median'] for x in entries]
            print(f'    {ao_type}: n_cells={len(entries)}  exact={100*statistics.mean(e_r):.1f}%±{100*statistics.stdev(e_r) if len(e_r)>1 else 0:.1f}  P={statistics.mean(p):.3f}±{statistics.stdev(p) if len(p)>1 else 0:.3f}  rank={statistics.median(r):.0f}')
