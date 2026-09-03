"""Q8B Exp 4: Manifold-distance diagnostics."""
import os, json, glob, gc, math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys
sys.path.insert(0, '/gpfs/scratch/USER/spherical-steering/scripts/oracle_test')
import oracle_lib as ol

BASE_MODEL = 'google/gemma-2-9b-it'
EVAL_BASE = '/gpfs/scratch/USER/results/ao_gma_karvonen_eval'
OUT = '/gpfs/scratch/USER/results/ao_gma_exp4_manifold_dist.json'

HINTS = ["Hint me.", "Lay a hint on me.", "Could I have a hint, please?",
         "Nudge me in the right direction.", "Give me a clue to help guess your word."]
N_SAMPLES = 8
device = 'cuda'; dtype = torch.bfloat16
print(f'[load] {BASE_MODEL}', flush=True)
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
bnb = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, torch_dtype=dtype, device_map='cuda')
read_layer = ol.layer_percent_to_layer(BASE_MODEL, 50)
submod = ol.get_hf_submodule(model, read_layer, use_lora=False)

captured = {}
def hook(m, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    captured['h'] = h.detach()
handle = submod.register_forward_hook(hook)

base_pool = []
for prompt in HINTS:
    msgs = [{'role':'user','content':prompt}]
    pfx = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pfx_ids = tok(pfx, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
    a_start = pfx_ids.shape[1]
    for _ in range(N_SAMPLES):
        with torch.inference_mode():
            out = model.generate(pfx_ids, max_new_tokens=80, do_sample=True, temperature=0.7, top_p=0.95,
                pad_token_id=tok.pad_token_id or tok.eos_token_id)
        full_ids = out[0:1].clone()
        attn = torch.ones_like(full_ids)
        with torch.inference_mode():
            _ = model(input_ids=full_ids, attention_mask=attn, use_cache=False)
        base_pool.append(captured['h'][0].float()[a_start:].cpu())
handle.remove()
all_base = torch.cat(base_pool, dim=0).float()
print(f'[pool] {all_base.shape}', flush=True)

mu = all_base.mean(dim=0); centered = all_base - mu
N, D = centered.shape
sigma = (centered.T @ centered) / (N - 1)
sigma += 1e-3 * torch.eye(D)
eigvals, eigvecs = torch.linalg.eigh(sigma)
idx = torch.argsort(eigvals, descending=True)
eigvals = eigvals[idx]; eigvecs = eigvecs[:, idx]
sigma_inv = eigvecs @ torch.diag(1.0 / eigvals) @ eigvecs.T

mu_gpu = mu.to(device); sigma_inv_gpu = sigma_inv.to(device)
eigvecs_gpu = eigvecs.to(device); all_base_gpu = all_base.to(device)
K = 100
top_evecs = eigvecs_gpu[:, :K]; bot_evecs = eigvecs_gpu[:, -K:]

def mahal(h):
    d = h - mu_gpu
    return (d @ sigma_inv_gpu * d).sum(-1).clamp(min=0).sqrt().mean().item()

def knn(h, k=10):
    h_norm = (h*h).sum(-1, keepdim=True); b_norm = (all_base_gpu*all_base_gpu).sum(-1)
    sq = (h_norm + b_norm.unsqueeze(0) - 2*(h @ all_base_gpu.T)).clamp(min=0)
    topk, _ = torch.topk(sq.sqrt(), k=k, dim=-1, largest=False)
    return topk.mean().item()

def masses(h):
    d = h - mu_gpu
    tot = (d*d).sum(-1); tv = ((d @ top_evecs)**2).sum(-1); bv = ((d @ bot_evecs)**2).sum(-1)
    return (tv/tot.clamp(min=1e-8)).mean().item(), (bv/tot.clamp(min=1e-8)).mean().item()

del model; torch.cuda.empty_cache(); gc.collect()

ao_results = {}
for d_path in sorted(glob.glob(f'{EVAL_BASE}/gma_*')):
    j = json.load(open(f'{d_path}/ao_results.json'))
    for c in j['cells']:
        ao_results[c['acts_path']] = c

results = []
for cell_dir in sorted(glob.glob(f'{EVAL_BASE}/gma_*')):
    cell_name = os.path.basename(cell_dir)
    parts = cell_name.split('_')
    word = parts[1]; c_tag = parts[2]
    for acts_path in sorted(glob.glob(f'{cell_dir}/acts_hint_*.pt')):
        payload = torch.load(acts_path, weights_only=False, map_location='cpu')
        h_taboo = payload['activations'].float().to(device)
        a_start = payload['assistant_start']
        h_a = h_taboo[a_start:]
        with torch.no_grad():
            md = mahal(h_a); kd = knn(h_a); tm, bm = masses(h_a)
        ao_r = ao_results.get(acts_path, {})
        greedy = ao_r.get('greedy_text', '').lower()
        plural = ao_r.get('plurality_word', '').lower() if ao_r.get('plurality_word') else ''
        recovered = (word in greedy) or (plural == word)
        topk_hit = False
        for pp in ao_r.get('per_position', [])[:8]:
            for t in pp['topk'][:15]:
                ts = t['token'].strip().lower().strip("'\"")
                if ts == word.lower() or word.lower() in ts:
                    topk_hit = True; break
            if topk_hit: break
        results.append({'word':word,'c':c_tag,'acts_path':acts_path,
            'mahalanobis':md,'knn':kd,'top100_mass':tm,'bot100_mass':bm,
            'recovered':recovered,'topk15_hit':topk_hit})

with open(OUT, 'w') as f: json.dump(results, f, indent=2)
print(f'\n[done] {OUT}', flush=True)

def pearson(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    sx = math.sqrt(sum((x-mx)**2 for x in xs)/n); sy = math.sqrt(sum((y-my)**2 for y in ys)/n)
    if sx*sy == 0: return float('nan')
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys)) / (n*sx*sy)

print('\n=== Correlations (n={}) ==='.format(len(results)))
for met in ['mahalanobis','knn','top100_mass','bot100_mass']:
    xs = [r[met] for r in results]
    for tgt in ['recovered','topk15_hit']:
        ys = [int(r[tgt]) for r in results]
        print(f'  corr({met:<14}, {tgt:<11}) = {pearson(xs, ys):+.3f}')
