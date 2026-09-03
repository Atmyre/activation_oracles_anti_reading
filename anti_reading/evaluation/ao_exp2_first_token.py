"""Exp 2: First-token / assistant-only AO read.

For each saved Karvonen-eval acts_hint_*.pt: instead of feeding the whole
sequence to the AO, feed ONLY the first assistant token activation. Compare
recovery vs full_seq.

Mechanism: synthesize a payload where activations are clipped to a single
position (assistant_start), then run ao_d1_extended_fullseq.py which uses
oracle_input_types=['full_seq'] but on a 1-token sequence — that's effectively
single-position read at the AO's read layer.
"""
import os, json, glob, gc, shutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys
sys.path.insert(0, '/gpfs/scratch/USER/spherical-steering/scripts/oracle_test')
import oracle_lib as ol
from oracle_lib import sanitize_lora_name

BASE_MODEL = 'Qwen/Qwen3-1.7B'
ORACLE = '/gpfs/scratch/USER/activation_oracles/checkpoints_latentqa_cls_past_lens_Qwen3-1_7B/final'
ORACLE_PROMPT = 'What is the secret word in this text?'
EVAL_BASE = '/gpfs/scratch/USER/results/ao_karvonen_eval'
OUT_BASE = '/gpfs/scratch/USER/results/ao_exp2_first_token'
os.makedirs(OUT_BASE, exist_ok=True)

# Step 1: synthesize first-token-only payloads from existing acts files
FIRST_K = 1  # number of assistant tokens to keep (1 = first only)

cells_args = []
for cell_dir in sorted(glob.glob(f'{EVAL_BASE}/q17_*')):
    cell_name = os.path.basename(cell_dir)
    out_cell = f'{OUT_BASE}/{cell_name}'
    os.makedirs(out_cell, exist_ok=True)
    word_c = cell_name.replace('q17_', '')
    for acts_path in sorted(glob.glob(f'{cell_dir}/acts_hint_*.pt')):
        bn = os.path.basename(acts_path)
        new_path = f'{out_cell}/{bn}'
        if not os.path.exists(new_path):
            payload = torch.load(acts_path, weights_only=False, map_location='cpu')
            h = payload['activations']  # [seq, D]
            a_start = payload['assistant_start']
            a_end_orig = payload['assistant_end']
            tok_ids = payload['token_ids']  # [seq]
            # Clip to prefix + first K assistant tokens
            new_end = min(a_start + FIRST_K, a_end_orig)
            new_h = h[:new_end]
            new_tok_ids = tok_ids[:new_end]
            new_payload = dict(payload)
            new_payload['activations'] = new_h
            new_payload['token_ids'] = new_tok_ids
            new_payload['seq_len'] = new_end
            new_payload['assistant_end'] = new_end
            torch.save(new_payload, new_path)
        cells_args.append(f'{word_c}={new_path}')

# Step 2: invoke ao_d1_extended_fullseq.py once with all cells
print(f'[done preparation] {len(cells_args)} cells in {OUT_BASE}', flush=True)
print(f'[next] running ao_d1_extended_fullseq.py', flush=True)

import subprocess
cmd = [
    'python', '/gpfs/scratch/USER/spherical-steering/scripts/ao_d1_extended_fullseq.py',
    '--subject-model', BASE_MODEL,
    '--oracle-lora', ORACLE,
    '--oracle-prompt', ORACLE_PROMPT,
    '--cells', *cells_args,
    '--n-samples', '20',
    '--temperature', '0.8',
    '--topk', '15',
    '--output', f'{OUT_BASE}/ao_results_first{FIRST_K}.json',
]
print(f'[cmd] {" ".join(cmd[:6])} ...', flush=True)
r = subprocess.run(cmd)
print(f'[exit] {r.returncode}', flush=True)
