"""Q8B Exp 6b: probes on all regimes (natural_gen, refusal, same_text, first_token)."""
import os, json, glob, math, random
import torch
import torch.nn as nn
import torch.optim as optim
random.seed(42); torch.manual_seed(42)
device = 'cuda'
WORDS = ['leaf', 'moon', 'wave']
W2I = {w:i for i,w in enumerate(WORDS)}

REGIMES = {
    'natural_gen': '<PATH_TO_SCRATCH>/results/ao_gma_karvonen_eval',
    'same_text':   '<PATH_TO_SCRATCH>/results/ao_gma_exp1_sametext',
    'refusal':     '<PATH_TO_SCRATCH>/results/ao_gma_karvonen_refusal_eval',
    'first_token': '<PATH_TO_SCRATCH>/results/ao_gma_exp2_first_token',
}

def load_regime(regime, root):
    records = []
    if regime == 'same_text':
        for cond_dir in sorted(glob.glob(f'{root}/*')):
            cond = os.path.basename(cond_dir)
            if not os.path.isdir(cond_dir) or cond == 'base' or '_' not in cond: continue
            word = cond.split('_')[0]
            if word not in W2I: continue
            for acts_path in sorted(glob.glob(f'{cond_dir}/acts_*.pt')):
                p = torch.load(acts_path, weights_only=False, map_location='cpu')
                h = p['activations'].float(); a_start = p['assistant_start']
                if h[a_start:].shape[0] == 0: continue
                records.append({'word': word, 'feat': h[a_start:].mean(dim=0), 'path': acts_path})
        return records
    for cell_dir in sorted(glob.glob(f'{root}/gma_*')):
        word = os.path.basename(cell_dir).split('_')[1]
        if word not in W2I: continue
        for acts_path in sorted(glob.glob(f'{cell_dir}/acts_*.pt')):
            p = torch.load(acts_path, weights_only=False, map_location='cpu')
            h = p['activations'].float(); a_start = p['assistant_start']
            if h[a_start:].shape[0] == 0: continue
            records.append({'word': word, 'feat': h[a_start:].mean(dim=0), 'path': acts_path})
    return records

class LP(nn.Module):
    def __init__(self,d,n=3):
        super().__init__(); self.l=nn.Linear(d,n)
    def forward(self,x): return self.l(x)
class MP(nn.Module):
    def __init__(self,d,h=64,n=3):
        super().__init__(); self.l1=nn.Linear(d,h); self.l2=nn.Linear(h,n)
    def forward(self,x): return self.l2(torch.relu(self.l1(x)))

def te(Model,Xt,yt,Xe,ye,epochs=200,lr=0.01):
    Xt,yt,Xe,ye = Xt.to(device),yt.to(device),Xe.to(device),ye.to(device)
    m = Model(Xt.shape[1]).to(device); opt=optim.Adam(m.parameters(),lr=lr); crit=nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); out=m(Xt); l=crit(out,yt); l.backward(); opt.step()
    with torch.no_grad(): return (m(Xe).argmax(-1)==ye).float().mean().item()

def cv5(X,y):
    N=X.shape[0]; idx=list(range(N)); random.Random(42).shuffle(idx)
    folds=[idx[i::5] for i in range(5)]
    lins=[]; mlps=[]
    for fi in range(5):
        te_idx=folds[fi]; tr_idx=[i for j in range(5) if j!=fi for i in folds[j]]
        lins.append(te(LP, X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]))
        mlps.append(te(MP, X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]))
    return sum(lins)/5, sum(mlps)/5

print(f'{"regime":<14}{"n":<6}{"linear":<10}{"mlp":<10}{"chance":<10}')
print('-'*60)
results = {}
for regime, root in REGIMES.items():
    recs = load_regime(regime, root)
    if not recs:
        print(f'{regime}: no records')
        continue
    X = torch.stack([r['feat'] for r in recs], dim=0)
    y = torch.tensor([W2I[r['word']] for r in recs], dtype=torch.long)
    lin, mlp = cv5(X, y)
    results[regime] = {'n':len(recs),'linear':lin,'mlp':mlp,'dim':X.shape[1]}
    print(f'{regime:<14}{len(recs):<6}{lin:<10.3f}{mlp:<10.3f}{1/3:<10.3f}')

OUT = '<PATH_TO_SCRATCH>/results/ao_gma_exp6b_probes_all_regimes.json'
with open(OUT,'w') as f: json.dump(results,f,indent=2)
print(f'\n[done] {OUT}')
