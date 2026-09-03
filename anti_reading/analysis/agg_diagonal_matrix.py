import json, os, glob, re
from collections import defaultdict

BASE = '<PATH_TO_SCRATCH>/results/ao_xmatrix_v3_lp'
CONCEPTS = ['book','flag','leaf','moon','wave']
OUT = "<PATH_TO_SCRATCH>/results/diagonal_matrix.json"

def parse_ao(tag):
    if tag == 'ours_base': return 'base', None, None
    m = re.match(r'^ours_strict(\w+?)v2_c(\d)p(\d\d)$', tag)
    if m: return m.group(1), 'strict', float(f'{m.group(2)}.{m.group(3)}')
    m = re.match(r'^ours_(\w+?)_c(\d)p(\d\d)$', tag)
    if m: return m.group(1), 'coop', float(f'{m.group(2)}.{m.group(3)}')
    return None, None, None

def parse_subj(subj):
    m = re.match(r'^strict(\w+?)v2_c(\d)p(\d\d)$', subj)
    if m: return m.group(1), 'strict', float(f'{m.group(2)}.{m.group(3)}')
    m = re.match(r'^(\w+?)_c(\d)p(\d\d)$', subj)
    if m: return m.group(1), 'coop', float(f'{m.group(2)}.{m.group(3)}')
    return None, None, None

# out[regime][protocol][c][ao_concept][subj_concept] = {exact, sem, p_target, rank}
results = {}
for regime in ['hint','refusal','sametext','think','offtopic']:
    results[regime] = {}
    for cell_dir in sorted(glob.glob(f'{BASE}/{regime}/*')):
        name = os.path.basename(cell_dir)
        if '__target_' in name: continue  # multi-secret variants
        if '__' not in name: continue
        ao_tag, subj_tag = name.split('__', 1)
        ao_c, ao_p, ao_cv = parse_ao(ao_tag)
        subj_c, subj_p, subj_cv = parse_subj(subj_tag)
        if not subj_c or subj_c not in CONCEPTS: continue
        # match protocol: base treated as agnostic (put in both coop and strict)
        # match c: only look at c=1.0 for now  
        # subj_cv accept both 0.5 and 1.0
        if ao_c != "base" and ao_cv is not None and ao_cv != subj_cv: continue
        # filter own/cross by matching protocol
        if ao_c != 'base' and ao_p != subj_p: continue

        p = f'{cell_dir}/ao_results.json'
        if not os.path.isfile(p): continue
        try: rj = json.load(open(p))
        except: continue
        cells = rj.get('cells', [])
        if not cells: continue
        recovered = sum(1 for c_ in cells if c_.get('greedy_recovered') in (True,'True'))
        n = len(cells)
        ps = [float(c_['target_prob_at_word_pos']) for c_ in cells if c_.get('target_prob_at_word_pos') not in (None,'None')]
        ranks = [int(c_['target_rank_at_word_pos']) for c_ in cells if c_.get('target_rank_at_word_pos') not in (None,'None')]

        # register in protocol slot(s)
        for proto in (['coop','strict'] if ao_c=='base' else [subj_p]):
            if proto not in results[regime]:
                results[regime][proto] = {}
            row_key = 'base' if ao_c=='base' else ao_c
            if row_key not in results[regime][proto]:
                results[regime][proto][row_key] = {}
            if subj_c not in results[regime][proto][row_key]:
                results[regime][proto][row_key][subj_c] = {
                    'n_captures':0, 'recovered':0, 'p_sum':0.0, 'p_n':0,
                    'ranks':[]
                }
            e = results[regime][proto][row_key][subj_c]
            e['n_captures'] += n
            e['recovered'] += recovered
            e['p_sum'] += sum(ps); e['p_n'] += len(ps)
            e['ranks'].extend(ranks)

# convert to final metrics
final = {}
for regime, protos in results.items():
    final[regime] = {}
    for proto, rows in protos.items():
        final[regime][proto] = {}
        for ao, cols in rows.items():
            final[regime][proto][ao] = {}
            for subj, e in cols.items():
                ranks_sorted = sorted(e['ranks'])
                final[regime][proto][ao][subj] = {
                    'n_captures': e['n_captures'],
                    'exact_recovery': e['recovered']/max(1,e['n_captures']),
                    'p_target_mean': e['p_sum']/max(1,e['p_n']),
                    'rank_median': ranks_sorted[len(ranks_sorted)//2] if ranks_sorted else -1,
                }

with open(OUT,'w') as f: json.dump(final, f, indent=1)
# summary
print(f'wrote {OUT}')
for reg in ['hint']:
    print(f'\n=== {reg} / coop / exact recovery ===')
    coop = final.get(reg, {}).get('coop', {})
    print('         ' + '  '.join(f'{c:>6}' for c in CONCEPTS))
    for ao in ['base'] + CONCEPTS:
        row = coop.get(ao, {})
        print(f'{ao:>7}  ' + '  '.join(f'{100*row.get(c,{}).get("exact_recovery",0):5.1f}' if c in row else '     -' for c in CONCEPTS))
