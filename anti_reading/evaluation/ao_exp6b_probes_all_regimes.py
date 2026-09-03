"""Exp 6b: Probes on ALL regimes (extends Exp 6).

Runs the same 5-fold linear+MLP probes on:
  - natural-gen Karvonen (45 cells, 3 words)        — baseline from Exp 6
  - same-text controls (27 cells, 3 words × 3 c × 3 texts, excludes base)
  - refusal-state (45 cells, 3 words × 3 c × 5 direct prompts)
  - first-token (45 cells, clipped to 1 assistant token)

Builds the central probe-vs-AO comparison table the findings doc emphasizes.
"""
import os, json, glob, math, random
import torch
import torch.nn as nn
import torch.optim as optim

random.seed(42); torch.manual_seed(42)
device = 'cuda'

REGIMES = {
    'natural_gen': '<PATH_TO_SCRATCH>/results/ao_karvonen_eval',
    'same_text':   '<PATH_TO_SCRATCH>/results/ao_exp1_sametext',
    'refusal':     '<PATH_TO_SCRATCH>/results/ao_karvonen_refusal_eval',
    'first_token': '<PATH_TO_SCRATCH>/results/ao_exp2_first_token',
}

WORDS = ['leaf', 'moon', 'wave']
W2I = {w: i for i, w in enumerate(WORDS)}


def load_regime_records(regime, root):
    """Returns list of dicts {word, feat, path} — mean-pooled assistant span."""
    records = []
    if regime == 'same_text':
        # subfolders are {base,leaf_c1p00,...}; skip base (no label)
        for cond_dir in sorted(glob.glob(f'{root}/*')):
            cond = os.path.basename(cond_dir)
            if not os.path.isdir(cond_dir) or cond == 'base' or '_' not in cond:
                continue
            word = cond.split('_')[0]
            if word not in W2I: continue
            for acts_path in sorted(glob.glob(f'{cond_dir}/acts_*.pt')):
                p = torch.load(acts_path, weights_only=False, map_location='cpu')
                h = p['activations'].float()
                a_start = p['assistant_start']
                h_a = h[a_start:]
                if h_a.shape[0] == 0: continue
                records.append({'word': word, 'feat': h_a.mean(dim=0), 'path': acts_path})
        return records
    if regime == 'first_token':
        # subdirs are q17_{word}_{c}; acts have only 1 assistant token (clipped)
        for cell_dir in sorted(glob.glob(f'{root}/q17_*')):
            cell_name = os.path.basename(cell_dir)
            word = cell_name.split('_')[1]
            if word not in W2I: continue
            for acts_path in sorted(glob.glob(f'{cell_dir}/acts_*.pt')):
                p = torch.load(acts_path, weights_only=False, map_location='cpu')
                h = p['activations'].float()
                a_start = p['assistant_start']
                h_a = h[a_start:]
                if h_a.shape[0] == 0: continue
                # Take the SINGLE first-token (or mean if more)
                records.append({'word': word, 'feat': h_a.mean(dim=0), 'path': acts_path})
        return records
    # natural_gen and refusal share the structure
    for cell_dir in sorted(glob.glob(f'{root}/q17_*')):
        cell_name = os.path.basename(cell_dir)
        word = cell_name.split('_')[1]
        if word not in W2I: continue
        for acts_path in sorted(glob.glob(f'{cell_dir}/acts_*.pt')):
            p = torch.load(acts_path, weights_only=False, map_location='cpu')
            h = p['activations'].float()
            a_start = p['assistant_start']
            h_a = h[a_start:]
            if h_a.shape[0] == 0: continue
            records.append({'word': word, 'feat': h_a.mean(dim=0), 'path': acts_path})
    return records


class LinearProbe(nn.Module):
    def __init__(self, d_in, n=3):
        super().__init__(); self.l = nn.Linear(d_in, n)
    def forward(self, x): return self.l(x)

class MLPProbe(nn.Module):
    def __init__(self, d_in, h=64, n=3):
        super().__init__(); self.l1 = nn.Linear(d_in, h); self.l2 = nn.Linear(h, n)
    def forward(self, x): return self.l2(torch.relu(self.l1(x)))


def train_eval(Model, Xt, yt, Xe, ye, epochs=200, lr=0.01):
    Xt, yt, Xe, ye = Xt.to(device), yt.to(device), Xe.to(device), ye.to(device)
    m = Model(Xt.shape[1]).to(device)
    opt = optim.Adam(m.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); out = m(Xt); loss = crit(out, yt); loss.backward(); opt.step()
    with torch.no_grad():
        return (m(Xe).argmax(-1) == ye).float().mean().item()


def cv5(X, y):
    N = X.shape[0]
    idx = list(range(N)); random.Random(42).shuffle(idx)
    folds = [idx[i::5] for i in range(5)]
    accs_lin = []; accs_mlp = []
    for fi in range(5):
        te = folds[fi]; tr = [i for j in range(5) if j != fi for i in folds[j]]
        accs_lin.append(train_eval(LinearProbe, X[tr], y[tr], X[te], y[te]))
        accs_mlp.append(train_eval(MLPProbe, X[tr], y[tr], X[te], y[te]))
    return sum(accs_lin)/5, sum(accs_mlp)/5

# === Run + report ===
results = {}
print(f'{"regime":<14}{"n_cells":<10}{"linear":<10}{"mlp":<10}{"chance":<10}')
print('-'*60)
for regime, root in REGIMES.items():
    recs = load_regime_records(regime, root)
    if not recs:
        print(f'{regime:<14} NO RECORDS')
        continue
    X = torch.stack([r['feat'] for r in recs], dim=0)
    y = torch.tensor([W2I[r['word']] for r in recs], dtype=torch.long)
    lin, mlp = cv5(X, y)
    results[regime] = {'n_cells': len(recs), 'linear_acc': lin, 'mlp_acc': mlp, 'dim': X.shape[1]}
    print(f'{regime:<14}{len(recs):<10}{lin:<10.3f}{mlp:<10.3f}{1/3:<10.3f}')

OUT = '<PATH_TO_SCRATCH>/results/ao_exp6b_probes_all_regimes.json'
with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\n[done] saved {OUT}')
