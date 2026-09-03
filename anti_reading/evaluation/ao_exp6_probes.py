"""Exp 6: Probe controls on transformed activations.

Train linear and MLP probes to predict the secret word from h, on:
  - raw h (identity)
  - top-K PCA projection
  - bottom-K PCA projection
  - whitened h

Compare probe accuracy to AO accuracy on the same transformed activations.

If probe succeeds where AO fails → information present but AO can't verbalize.
"""
import os, json, glob, math, random
import torch
import torch.nn as nn
import torch.optim as optim

EVAL_BASE = '<PATH_TO_SCRATCH>/results/ao_karvonen_eval'
OUT = '<PATH_TO_SCRATCH>/results/ao_exp6_probes.json'
random.seed(42)
torch.manual_seed(42)

device = 'cuda'

# ===== Load all acts; mean-pool assistant span =====
pool = []
records = []
for cell_dir in sorted(glob.glob(f'{EVAL_BASE}/q17_*')):
    cell_name = os.path.basename(cell_dir)
    parts = cell_name.split('_')
    word = parts[1]; c_tag = parts[2]
    for acts_path in sorted(glob.glob(f'{cell_dir}/acts_hint_*.pt')):
        p = torch.load(acts_path, weights_only=False, map_location='cpu')
        h = p['activations'].float()
        a_start = p['assistant_start']
        h_a = h[a_start:]
        mean_h = h_a.mean(dim=0)
        pool.append(mean_h)
        records.append({'word': word, 'c': c_tag, 'feat': mean_h, 'path': acts_path})

X = torch.stack(pool, dim=0)
print(f'[pool] X={X.shape}', flush=True)

# ===== PCA via SVD (N << D regime) =====
mu = X.mean(dim=0)
Xc = X - mu
N, D = Xc.shape
U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
eigvecs = Vt
print(f'[svd] top-3 sv {S[:3].tolist()}', flush=True)

K = 8
top_K_evecs = eigvecs[:K]
bot_K_evecs = eigvecs[-K:]
W_K = min(30, len(S))
white_basis = eigvecs[:W_K] / (S[:W_K].unsqueeze(-1) / math.sqrt(N - 1)).clamp(min=1e-6)

def transform(X_in, mode):
    Xc_in = X_in - mu
    if mode == 'identity':
        return X_in
    if mode == 'top_k_pca':
        return Xc_in @ top_K_evecs.T
    if mode == 'bot_k_pca':
        return Xc_in @ bot_K_evecs.T
    if mode == 'whitened':
        return Xc_in @ white_basis.T
    raise ValueError(mode)

WORDS = ['leaf', 'moon', 'wave']
W2I = {w: i for i, w in enumerate(WORDS)}

class LinearProbe(nn.Module):
    def __init__(self, d_in, n_classes=3):
        super().__init__(); self.l = nn.Linear(d_in, n_classes)
    def forward(self, x): return self.l(x)

class MLPProbe(nn.Module):
    def __init__(self, d_in, h=64, n_classes=3):
        super().__init__(); self.l1 = nn.Linear(d_in, h); self.l2 = nn.Linear(h, n_classes)
    def forward(self, x): return self.l2(torch.relu(self.l1(x)))

def train_eval_probe(Model, X_train, y_train, X_test, y_test, epochs=200, lr=0.01):
    X_train = X_train.to(device); y_train = y_train.to(device)
    X_test = X_test.to(device); y_test = y_test.to(device)
    model = Model(X_train.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); out = model(X_train); loss = crit(out, y_train); loss.backward(); opt.step()
    with torch.no_grad():
        pred = model(X_test).argmax(-1)
        acc = (pred == y_test).float().mean().item()
    return acc

N = len(records)
idx = list(range(N))
random.shuffle(idx)
folds = [idx[i::5] for i in range(5)]

results = {}
y_all = torch.tensor([W2I[records[i]['word']] for i in range(N)], dtype=torch.long)

for mode in ['identity', 'top_k_pca', 'bot_k_pca', 'whitened']:
    feats_per_record = [transform(records[i]['feat'].unsqueeze(0), mode).squeeze(0) for i in range(N)]
    Xt = torch.stack(feats_per_record, dim=0)
    accs_lin = []; accs_mlp = []
    for fi in range(5):
        test_idx = folds[fi]
        train_idx = [i for j in range(5) if j != fi for i in folds[j]]
        a_lin = train_eval_probe(LinearProbe, Xt[train_idx], y_all[train_idx], Xt[test_idx], y_all[test_idx])
        a_mlp = train_eval_probe(MLPProbe, Xt[train_idx], y_all[train_idx], Xt[test_idx], y_all[test_idx])
        accs_lin.append(a_lin); accs_mlp.append(a_mlp)
    lin_mean = sum(accs_lin)/5; mlp_mean = sum(accs_mlp)/5
    results[mode] = dict(linear_acc=lin_mean, mlp_acc=mlp_mean,
                         linear_per_fold=accs_lin, mlp_per_fold=accs_mlp,
                         dim=Xt.shape[1])
    print(f'  {mode:<12}: dim={Xt.shape[1]:<5} linear={lin_mean:.3f} mlp={mlp_mean:.3f}', flush=True)

# Compare to AO recovery
print()
def topk_any(cell, word, K=15, max_pos=8):
    for pos_info in cell['per_position'][:max_pos]:
        for t in pos_info['topk'][:K]:
            ts = t['token'].strip().lower().strip("'\"")
            if ts == word.lower() or word.lower() in ts:
                return True
    return False
ao_by_path = {}
for d_path in sorted(glob.glob(f'{EVAL_BASE}/q17_*')):
    j = json.load(open(f'{d_path}/ao_results.json'))
    for c in j['cells']:
        ao_by_path[c['acts_path']] = c
ao_correct = sum(1 for r in records if topk_any(ao_by_path.get(r['path'], {}), r['word']))
print(f'AO top-15 recall (loose) = {ao_correct}/{N} = {ao_correct/N:.3f}', flush=True)
print(f'(chance = {1/3:.3f})', flush=True)
results['ao_topk15_recall'] = ao_correct / N
results['n_cells'] = N

with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\n[done] saved to {OUT}', flush=True)
