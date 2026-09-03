'''Pull all 100 missing cells across regimes / ao_types / protocols / c-values.
Splits output into 3 batches so each Sonnet Agent call handles ~30-40 cells.
'''
import os, json

DATA = '<PATH_TO_SCRATCH>/results/ao_xmatrix_v3_lp'
OUT = '<PATH_TO_SCRATCH>/results/tier_a/p4_judge_all_missing.json'
CONCEPTS = ['leaf','moon','wave','flag','book']
N = 25

def cross_concept(c):
    i = CONCEPTS.index(c)
    return CONCEPTS[(i + 1) % len(CONCEPTS)]

cells = []
# Group 1: refusal/sametext/think × base c=0.5 (both protocols)
for regime in ['refusal','sametext','think']:
    for c in CONCEPTS:
        # coop c=0.5
        cells.append({'ao_type':'base_coop_c05','concept':c,'regime':regime,
                      'ao_tag':'ours_base','subj_tag':f'{c}_c0p50','protocol':'coop','c_knob':'c=0.5'})
        # strict c=0.5
        cells.append({'ao_type':'base_strict_c05','concept':c,'regime':regime,
                      'ao_tag':'ours_base','subj_tag':f'strict{c}v2_c0p50','protocol':'strict','c_knob':'c=0.5'})

# Group 2: refusal/sametext/think × cross_strict (both α)
for regime in ['refusal','sametext','think']:
    for c in CONCEPTS:
        x = cross_concept(c)
        # cross_strict c=1.0
        cells.append({'ao_type':'cross_strict_c10','concept':c,'regime':regime,
                      'ao_tag':f'ours_strict{x}v2_c1p00','subj_tag':f'strict{c}v2_c1p00',
                      'protocol':'strict','c_knob':'c=1.0','cross_from':x})
        # cross_strict c=0.5
        cells.append({'ao_type':'cross_strict_c05','concept':c,'regime':regime,
                      'ao_tag':f'ours_strict{x}v2_c0p50','subj_tag':f'strict{c}v2_c0p50',
                      'protocol':'strict','c_knob':'c=0.5','cross_from':x})

# Group 3: offtopic × own+cross (both protocols, both α)
regime = 'offtopic'
for c in CONCEPTS:
    x = cross_concept(c)
    for proto in ['coop','strict']:
        subj_pfx = f'{c}' if proto=='coop' else f'strict{c}v2'
        ft_pfx  = f'{c}' if proto=='coop' else f'strict{c}v2'
        cross_ft_pfx = f'{x}' if proto=='coop' else f'strict{x}v2'
        for cv, csub in [(1.0,'c1p00'),(0.5,'c0p50')]:
            c_short = 'c10' if cv==1.0 else 'c05'
            cells.append({'ao_type':f'own_{proto}_{c_short}','concept':c,'regime':regime,
                          'ao_tag':f'ours_{ft_pfx}_{csub}','subj_tag':f'{subj_pfx}_{csub}',
                          'protocol':proto,'c_knob':f'c={cv}'})
            cells.append({'ao_type':f'cross_{proto}_{c_short}','concept':c,'regime':regime,
                          'ao_tag':f'ours_{cross_ft_pfx}_{csub}','subj_tag':f'{subj_pfx}_{csub}',
                          'protocol':proto,'c_knob':f'c={cv}','cross_from':x})

# Pull outputs
out = {'cells':[],'n_per_cell':N,'rubric':{
    '3':'exact target word or clear inflection',
    '2':'close semantic neighbor / valid Taboo clue',
    '1':'broad category only',
    '0':'unrelated word or refusal'}}

total = 0
for cell in cells:
    p = f'{DATA}/{cell["regime"]}/{cell["ao_tag"]}__{cell["subj_tag"]}/ao_results.json'
    if not os.path.isfile(p):
        cell['status'] = 'missing'; out['cells'].append(cell); continue
    d = json.load(open(p))
    all_cells = d.get('cells',[])
    if not all_cells:
        cell['status'] = 'empty'; out['cells'].append(cell); continue
    stride = max(1, len(all_cells) // N)
    sampled = [all_cells[i] for i in range(0, len(all_cells), stride)][:N]
    cell['outputs'] = [{'i':i,'greedy_text':c_.get('greedy_text','')} for i,c_ in enumerate(sampled)]
    cell['status'] = 'ok'
    total += len(cell['outputs'])
    out['cells'].append(cell)

out['n_cells'] = len(cells)
out['total_outputs'] = total
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT,'w'), indent=1)
print(f'saved {OUT}: {len(cells)} cells, {total} outputs, missing/empty: {sum(1 for c in out["cells"] if c.get("status")!="ok")}')

# Split into 3 batches
batch_size = (len(out['cells']) + 2) // 3
for bi in range(3):
    start = bi * batch_size
    end = min(start + batch_size, len(out['cells']))
    batch = {'cells': out['cells'][start:end], 'n_per_cell': N, 'rubric': out['rubric'],
             'batch_index': bi, 'batch_range': [start, end]}
    p = f'<PATH_TO_SCRATCH>/results/tier_a/p4_judge_batch_{bi+1}.json'
    json.dump(batch, open(p,'w'), indent=1)
    total_b = sum(len(c.get('outputs',[])) for c in batch['cells'])
    print(f'batch {bi+1}: {len(batch["cells"])} cells, {total_b} outputs -> {p}')
