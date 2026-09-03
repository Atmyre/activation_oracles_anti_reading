'''Pull greedy_text for the 4 missing cell types (hint regime only):
  base × coop × c=0.5   (5 concepts)
  base × strict × c=0.5 (5 concepts)
  cross × strict × c=1.0 (5 concepts, using next-concept AO)
  cross × strict × c=0.5 (5 concepts)
= 20 cells × 25 outputs = 500 outputs
'''
import os, json

DATA = '/gpfs/scratch/USER/results/ao_xmatrix_v3_lp'
OUT = '/gpfs/scratch/USER/results/tier_a/p4_judge_inputs_missing.json'
CONCEPTS = ['leaf','moon','wave','flag','book']
N = 25
REGIME = 'hint'

def cross_concept(c):
    i = CONCEPTS.index(c)
    return CONCEPTS[(i + 1) % len(CONCEPTS)]

cells = []
for c in CONCEPTS:
    x = cross_concept(c)
    # base × coop × c=0.5
    cells.append({'ao_type':'base_coop_c05','concept':c,'regime':REGIME,
                  'ao_tag':'ours_base','subj_tag':f'{c}_c0p50','protocol':'coop','c_knob':'c=0.5'})
    # base × strict × c=0.5
    cells.append({'ao_type':'base_strict_c05','concept':c,'regime':REGIME,
                  'ao_tag':'ours_base','subj_tag':f'strict{c}v2_c0p50','protocol':'strict','c_knob':'c=0.5'})
    # cross × strict × c=1.0 (using next concept AO)
    cells.append({'ao_type':'cross_strict_c10','concept':c,'regime':REGIME,
                  'ao_tag':f'ours_strict{x}v2_c1p00','subj_tag':f'strict{c}v2_c1p00',
                  'protocol':'strict','c_knob':'c=1.0','cross_from':x})
    # cross × strict × c=0.5
    cells.append({'ao_type':'cross_strict_c05','concept':c,'regime':REGIME,
                  'ao_tag':f'ours_strict{x}v2_c0p50','subj_tag':f'strict{c}v2_c0p50',
                  'protocol':'strict','c_knob':'c=0.5','cross_from':x})

out = {'cells':[], 'n_per_cell':N, 'regime':REGIME, 'rubric':{
    '3': 'exact target word or clear inflection',
    '2': 'close semantic neighbor / valid Taboo clue',
    '1': 'broad category only',
    '0': 'unrelated word or refusal'}}

total = 0
for cell in cells:
    p = f'{DATA}/{REGIME}/{cell["ao_tag"]}__{cell["subj_tag"]}/ao_results.json'
    if not os.path.isfile(p):
        cell['status'] = 'missing'; out['cells'].append(cell); continue
    d = json.load(open(p))
    all_cells = d.get('cells',[])
    if not all_cells:
        cell['status'] = 'empty'; out['cells'].append(cell); continue
    stride = max(1, len(all_cells) // N)
    sampled = [all_cells[i] for i in range(0, len(all_cells), stride)][:N]
    cell['outputs'] = [{'i':i, 'greedy_text':c_.get('greedy_text','')} for i,c_ in enumerate(sampled)]
    cell['status'] = 'ok'
    total += len(cell['outputs'])
    out['cells'].append(cell)

out['total_outputs'] = total
out['n_cells'] = len(cells)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT,'w'), indent=1)
print(f'saved {OUT}: {len(cells)} cells, {total} outputs')
