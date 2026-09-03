"""For each saved Karvonen-eval acts_hint_*.pt: re-run forward on the same token_ids
with BASE Q1.7B (no LoRA) to capture h_base, then compute ‖h_taboo - h_base‖ at
the assistant span.

Output: a single JSON aggregating per (word, c) and per-cell Δh.
"""
import os, json, glob, gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.insert(0, '<PATH_TO_SCRATCH>/spherical-steering/scripts/oracle_test')
import oracle_lib as ol

BASE_MODEL = 'Qwen/Qwen3-1.7B'
EVAL_BASE = '<PATH_TO_SCRATCH>/results/ao_karvonen_eval'
OUT = '<PATH_TO_SCRATCH>/results/ao_karvonen_dh.json'

device = 'cuda'
dtype = torch.bfloat16
print(f'[load] base = {BASE_MODEL}', flush=True)
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype, device_map='cuda')
read_layer = ol.layer_percent_to_layer(BASE_MODEL, 50)
submod = ol.get_hf_submodule(model, read_layer, use_lora=False)
print(f'[read_layer] L{read_layer}', flush=True)

captured = {}
def hook(m, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    captured['h'] = h.detach()
handle = submod.register_forward_hook(hook)

results = []
for cell_dir in sorted(glob.glob(f'{EVAL_BASE}/q17_*')):
    cell_name = os.path.basename(cell_dir)
    # parse word + c
    parts = cell_name.split('_')
    word = parts[1]; c_tag = parts[2]
    for acts_path in sorted(glob.glob(f'{cell_dir}/acts_hint_*.pt')):
        try:
            payload = torch.load(acts_path, weights_only=False, map_location='cpu')
            h_taboo = payload['activations'].to(device, dtype=torch.float32)
            token_ids = payload['token_ids'].to(device).unsqueeze(0)
            a_start = payload['assistant_start']
            with torch.inference_mode():
                attn = torch.ones_like(token_ids)
                _ = model(input_ids=token_ids, attention_mask=attn, use_cache=False)
            h_base = captured['h'][0].to(device, dtype=torch.float32)
            assert h_taboo.shape == h_base.shape, f'shape mismatch {h_taboo.shape} vs {h_base.shape}'
            # mean ‖Δh‖ over assistant span
            diff = (h_taboo[a_start:] - h_base[a_start:]).norm(dim=-1).mean().item()
            full_diff = (h_taboo - h_base).norm(dim=-1).mean().item()
            prefix_diff = (h_taboo[:a_start] - h_base[:a_start]).norm(dim=-1).mean().item()
            results.append({
                'word': word, 'c': c_tag, 'acts_path': acts_path,
                'dh_assistant_span': diff,
                'dh_full': full_diff,
                'dh_prefix': prefix_diff,
                'seq_len': payload['seq_len'],
                'assistant_start': a_start,
                'assistant_text_first_60': payload.get('assistant_text', '')[:60],
            })
            print(f'  {cell_name} {os.path.basename(acts_path)}: dh_asst={diff:.3f} dh_full={full_diff:.3f}', flush=True)
        except Exception as e:
            print(f'  ERROR on {acts_path}: {e}', flush=True)

with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\n[done] saved {len(results)} cells to {OUT}', flush=True)
