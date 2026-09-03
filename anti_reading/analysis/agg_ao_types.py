import json, os, glob, re
from collections import defaultdict

BASE = '<PATH_TO_SCRATCH>/results/ao_xmatrix_v3_lp'
CONCEPTS = ['book','flag','leaf','moon','wave']
OUT = '<PATH_TO_SCRATCH>/results/ao_types_aggregate.json'

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

# results[regime][protocol][c][ao_type] = {n_captures, recovered, ps, ranks}
results = {}
for regime in ['hint','refusal','sametext','think','offtopic']:
    results[regime] = {}
    for cell_dir in sorted(glob.glob(f'{BASE}/{regime}/*')):
        name = os.path.basename(cell_dir)
        if '__target_' in name: continue
        if '__' not in name: continue
        ao_tag, subj_tag = name.split('__', 1)
        ao_c, ao_p, ao_cv = parse_ao(ao_tag)
        subj_c, subj_p, subj_cv = parse_subj(subj_tag)
        if subj_c not in CONCEPTS: continue
        if subj_cv not in (0.5, 1.0): continue
        # For non-base AO: require matching protocol AND matching c-knob with subject
        if ao_c != 'base':
            if ao_p != subj_p: continue
            if ao_cv != subj_cv: continue
        # classify AO type
        if ao_c == 'base':
            ao_type = 'base'
        elif ao_c == subj_c:
            ao_type = 'own'
        elif ao_c in CONCEPTS:
            ao_type = 'cross'
        else:
            continue  # skip bcywclock etc

        p = f'{cell_dir}/ao_results.json'
        if not os.path.isfile(p): continue
        try: rj = json.load(open(p))
        except: continue
        cells = rj.get('cells', [])
        if not cells: continue

        # register in protocol slot(s): base into both, others by subj_p
        for slot_proto in [subj_p]:
            key = (slot_proto, subj_cv)
            if key not in results[regime]:
                results[regime][key] = {}
            if ao_type not in results[regime][key]:
                results[regime][key][ao_type] = {'n_captures':0, 'recovered':0, 'ps':[], 'ranks':[]}
            e = results[regime][key][ao_type]
            for c_ in cells:
                if c_.get('greedy_recovered') in (True,'True'): e['recovered'] += 1
                pv = c_.get('target_prob_at_word_pos')
                rv = c_.get('target_rank_at_word_pos')
                if pv not in (None,'None'):
                    try: e['ps'].append(float(pv))
                    except: pass
                if rv not in (None,'None'):
                    try: e['ranks'].append(int(rv))
                    except: pass
                e['n_captures'] += 1

# final aggregation
final = {}
for regime, keys in results.items():
    final[regime] = {}
    for (proto, cv), ao_types in keys.items():
        k = f'{proto}_c{cv}'
        final[regime][k] = {}
        for ao_type, e in ao_types.items():
            ranks_sorted = sorted(e['ranks'])
            final[regime][k][ao_type] = {
                'n_captures': e['n_captures'],
                'exact_recovery': e['recovered']/max(1,e['n_captures']),
                'p_target_mean': (sum(e['ps'])/max(1,len(e['ps']))) if e['ps'] else 0.0,
                'p_target_median': ranks_sorted[len(ranks_sorted)//2] if False else (sorted(e['ps'])[len(e['ps'])//2] if e['ps'] else 0.0),
                'rank_median': ranks_sorted[len(ranks_sorted)//2] if ranks_sorted else -1,
            }

with open(OUT,'w') as f: json.dump(final, f, indent=1)
print(f'wrote {OUT}')
# summary
for reg in ['hint']:
    print(f'\n=== {reg} ===')
    for k, ao_types in final[reg].items():
        print(f'  {k}:')
        for ao_type in ['base','cross','own']:
            v = ao_types.get(ao_type)
            if v is None: continue
            print(f'    {ao_type}: exact={100*v["exact_recovery"]:.1f}% P={v["p_target_mean"]:.3f} rank={v["rank_median"]}  n={v["n_captures"]}')
