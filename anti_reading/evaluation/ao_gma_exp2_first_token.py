"""Q8B Exp 2: first-token AO read. Clip existing Q8B natural-gen acts to first assistant token."""
import os, json, glob, subprocess
import torch

EVAL_BASE = '/gpfs/scratch/USER/results/ao_gma_karvonen_eval'
OUT_BASE = '/gpfs/scratch/USER/results/ao_gma_exp2_first_token'
os.makedirs(OUT_BASE, exist_ok=True)
BASE_MODEL = 'google/gemma-2-9b-it'
ORACLE = 'adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it'
ORACLE_PROMPT = 'What is the secret word in this text?'

cells_args = []
for cell_dir in sorted(glob.glob(f'{EVAL_BASE}/gma_*')):
    cell_name = os.path.basename(cell_dir)
    out_cell = f'{OUT_BASE}/{cell_name}'
    os.makedirs(out_cell, exist_ok=True)
    word_c = cell_name.replace('q8_', '')
    for acts_path in sorted(glob.glob(f'{cell_dir}/acts_hint_*.pt')):
        bn = os.path.basename(acts_path)
        new_path = f'{out_cell}/{bn}'
        if not os.path.exists(new_path):
            p = torch.load(acts_path, weights_only=False, map_location='cpu')
            h = p['activations']
            a_start = p['assistant_start']
            a_end_orig = p['assistant_end']
            new_end = min(a_start + 1, a_end_orig)
            np_ = dict(p)
            np_['activations'] = h[:new_end]
            np_['token_ids'] = p['token_ids'][:new_end]
            np_['seq_len'] = new_end
            np_['assistant_end'] = new_end
            torch.save(np_, new_path)
        cells_args.append(f'{word_c}={new_path}')

print(f'[prep] {len(cells_args)} cells')
cmd = ['python', '/gpfs/scratch/USER/spherical-steering/scripts/ao_d1_extended_fullseq.py',
       '--subject-model', BASE_MODEL, '--oracle-lora', ORACLE,
       '--oracle-prompt', ORACLE_PROMPT, '--cells', *cells_args,
       '--n-samples', '20', '--temperature', '0.8', '--topk', '15',
       '--quantize-8bit',
       '--output', f'{OUT_BASE}/ao_results_first1.json']
subprocess.run(cmd)
