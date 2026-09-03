import json, os, glob, re
BASE = '<PATH_TO_SCRATCH>/results/ao_xmatrix_v3_lp'
CONCEPTS = ['book','flag','leaf','moon','wave']
OUT = '<PATH_TO_SCRATCH>/results/base_ao_pdist.json'
def parse_subj(subj):
    m = re.match(r'^strict(\w+?)v2_c(\d)p(\d\d)$', subj)
    if m: return m.group(1),'strict',float(f'{m.group(2)}.{m.group(3)}')
    m = re.match(r'^(\w+?)_c(\d)p(\d\d)$', subj)
    if m: return m.group(1),'coop',float(f'{m.group(2)}.{m.group(3)}')
    return None,None,None
results = {}
for regime in ['hint','refusal','sametext','think','offtopic']:
    results[regime] = {}
    for d in sorted(glob.glob(f'{BASE}/{regime}/ours_base__*')):
        subj = os.path.basename(d).replace('ours_base__','')
        concept, proto, c = parse_subj(subj)
        if concept not in CONCEPTS or c != 1.0: continue
        p = f'{d}/ao_results.json'
        if not os.path.isfile(p): continue
        try: rj = json.load(open(p))
        except: continue
        cells = rj.get('cells', [])
        if not cells: continue
        ps=[]; ranks=[]
        for c_ in cells:
            pv=c_.get('target_prob_at_word_pos'); rv=c_.get('target_rank_at_word_pos')
            if pv not in (None,'None'):
                try: ps.append(float(pv))
                except: pass
            if rv not in (None,'None'):
                try: ranks.append(int(rv))
                except: pass
        if ps and ranks:
            ps_s=sorted(ps); rs_s=sorted(ranks)
            results[regime][subj] = {'concept':concept,'protocol':proto,'c':c,'n':len(ps),
                'p_target_mean':sum(ps)/len(ps),'p_target_median':ps_s[len(ps)//2],
                'rank_median':rs_s[len(rs_s)//2],'rank_p10':rs_s[len(rs_s)//10],
                'rank_p90':rs_s[min(len(rs_s)-1,9*len(rs_s)//10)]}
with open(OUT,'w') as f: json.dump(results, f, indent=1)
print(f'wrote {OUT}')
