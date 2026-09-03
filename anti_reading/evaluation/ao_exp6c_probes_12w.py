"""Exp 6c: probes on 12-word concept matched-set across regimes (extends Exp 6b)."""
import os, json, glob, math, random
import torch
import torch.nn as nn
import torch.optim as optim

random.seed(42); torch.manual_seed(42)
device = 'cuda'

WORDS = ['leaf','moon','wave','cloud','snow','book','chair','clock','flag','jump','dance','song']
W2I = {w: i for i, w in enumerate(WORDS)}
N_CLASS = len(WORDS)

REGIMES = {
    'natural_gen': '/gpfs/scratch/USER/results/ao_karvonen_eval',
    'refusal':     '/gpfs/scratch/USER/results/ao_karvonen_refusal_eval',
}

def load_regime(root):
    records = []
    for cell_dir in sorted(glob.glob(f'{root}/q17_*c1p00')):
        cell_name = os.path.basename(cell_dir)
        parts = cell_name.split('_')
        word = parts[1]
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
    def __init__(self, d_in, n=N_CLASS):
        super().__init__(); self.l = nn.Linear(d_in, n)
    def forward(self, x): return self.l(x)


class MLPProbe(nn.Module):
    def __init__(self, d_in, h=128, n=N_CLASS):
        super().__init__(); self.l1 = nn.Linear(d_in, h); self.l2 = nn.Linear(h, n)
    def forward(self, x): return self.l2(torch.relu(self.l1(x)))


def train_eval(Model, Xt, yt, Xe, ye, epochs=300, lr=0.01):
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
    lins, mlps = [], []
    for fi in range(5):
        te = folds[fi]; tr = [i for j in range(5) if j != fi for i in folds[j]]
        lins.append(train_eval(LinearProbe, X[tr], y[tr], X[te], y[te]))
        mlps.append(train_eval(MLPProbe, X[tr], y[tr], X[te], y[te]))
    return sum(lins)/5, sum(mlps)/5


def ao_recall_topk(records, ao_results, K=15):
    correct = 0
    for r in records:
        cell = ao_results.get(r['path'], {})
        hit = False
        for pp in cell.get('per_position', [])[:8]:
            for t in pp['topk'][:K]:
                ts = t['token'].strip().lower().strip("'\"")
                if ts == r['word'].lower() or r['word'].lower() in ts:
                    hit = True; break
            if hit: break
        if hit: correct += 1
    return correct / len(records)


def ao_top1(records, ao_results):
    correct = 0
    for r in records:
        cell = ao_results.get(r['path'], {})
        greedy = cell.get('greedy_text', '').lower()
        plural = cell.get('plurality_word', '').lower() if cell.get('plurality_word') else ''
        if r['word'] in greedy or plural == r['word']:
            correct += 1
    return correct / len(records)


results = {}
print(f'{"regime":<14}{"n":<5}{"linear":<10}{"mlp":<10}{"AO top-1":<10}{"AO top-15":<12}{"chance":<8}')
print('-'*70)
for regime, root in REGIMES.items():
    recs = load_regime(root)
    if not recs: print(f'{regime}: no records'); continue
    X = torch.stack([r['feat'] for r in recs], dim=0)
    y = torch.tensor([W2I[r['word']] for r in recs], dtype=torch.long)
    lin, mlp = cv5(X, y)
    # AO recall on same cells
    ao_by_path = {}
    for d_path in sorted(glob.glob(f'{root}/q17_*c1p00')):
        j = json.load(open(f'{d_path}/ao_results.json'))
        for c in j['cells']:
            ao_by_path[c['acts_path']] = c
    ao15 = ao_recall_topk(recs, ao_by_path, K=15)
    ao1 = ao_top1(recs, ao_by_path)
    results[regime] = dict(n=len(recs), linear=lin, mlp=mlp, ao_top1=ao1, ao_top15=ao15, dim=X.shape[1])
    print(f'{regime:<14}{len(recs):<5}{lin:<10.3f}{mlp:<10.3f}{ao1:<10.3f}{ao15:<12.3f}{1/N_CLASS:<8.3f}')

OUT = '/gpfs/scratch/USER/results/ao_exp6c_probes_12words.json'
with open(OUT, 'w') as f: json.dump(results, f, indent=2)
print(f'\n[done] {OUT}')
