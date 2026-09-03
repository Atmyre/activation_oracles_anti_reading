#!/usr/bin/env python3
"""Tests A + C: install-direction LogitLens and D4-specific delta decoding.

Test A: For each subject (D3+D4 and D3-only), compute Δ_install = mean(h_FT) − mean(h_base)
        at L18 mean-pooled over assistant-generation tokens. Apply Qwen3-8B lm_head to Δ
        and report top-k decoded tokens. If D3+D4's top tokens include clock/time/temporal
        and D3-only's don't → D4 installed a clock-direction.

Test C: cos(Δ_D3+D4, Δ_D3only) and what (Δ_D3+D4 − Δ_D3only) decodes to.
        High cos → D4 doesn't add a new axis. Difference-decoding reveals D4's specific contribution.
"""
import os, glob, math
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

CAP_ROOT = '/gpfs/scratch/USER/results/ao_d4verdict_caps'
REGIMES = ['hint', 'refusal', 'sametext']
SUBJECTS = ['strictclock_c1p00', 'strictclock_D3only_c1p00']
BASE_NAME = 'base'
TOPK = 20

# ---------------------------------------------------------------------------

def load_subject_acts(subj):
    """Mean-pool h_L18 over assistant-generation tokens for all prompts × all regimes.

    Returns mean activation vector across all prompts (one D-dim tensor)."""
    vs = []
    for regime in REGIMES:
        d = f'{CAP_ROOT}/{regime}/{subj}'
        if regime == 'sametext':
            files = sorted(glob.glob(f'{d}/acts_text*.pt'))
        else:
            files = sorted(glob.glob(f'{d}/acts_{regime}_*.pt'))
        for f in files:
            try:
                obj = torch.load(f, map_location='cpu', weights_only=False)
                a = obj['activations'].float()
                start = obj.get('assistant_start', 0)
                v = a[start:].mean(0)
                vs.append(v)
            except Exception as e:
                print(f'  WARN: skip {f}: {e}')
    if not vs:
        return None
    return torch.stack(vs).mean(0)


def main():
    print('=' * 70)
    print('Tests A + C: install-direction LogitLens')
    print('=' * 70)

    # Aggregate captures
    means = {}
    for subj in [BASE_NAME] + SUBJECTS:
        v = load_subject_acts(subj)
        if v is None:
            print(f'NO CAPTURES for {subj}')
            return
        means[subj] = v
        print(f'  {subj:<30}  mean residual L18 ‖.‖ = {v.norm().item():.3f}')

    # Δ vectors (install)
    delta_d3d4 = means[SUBJECTS[0]] - means[BASE_NAME]
    delta_d3only = means[SUBJECTS[1]] - means[BASE_NAME]
    delta_d4_specific = delta_d3d4 - delta_d3only

    print()
    print(f'‖Δ_D3+D4‖           = {delta_d3d4.norm().item():.4f}')
    print(f'‖Δ_D3only‖          = {delta_d3only.norm().item():.4f}')
    print(f'‖Δ_D4_specific‖     = {delta_d4_specific.norm().item():.4f}')
    print()

    # Cosine similarity (Test C)
    cos = F.cosine_similarity(delta_d3d4.unsqueeze(0), delta_d3only.unsqueeze(0)).item()
    print(f'cos(Δ_D3+D4, Δ_D3only) = {cos:.4f}')
    print('  (high → D4 adds little new direction; low → D4 adds new axis)')
    print()

    # Load Qwen3-8B for lm_head
    print('Loading Qwen3-8B for lm_head + tokenizer ...')
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen3-8B', torch_dtype=torch.bfloat16, device_map='cuda')
    lm_head = model.lm_head  # (V, D) typically
    # Some models tie lm_head with embed_tokens. Need to apply final norm before lm_head for true logit lens.
    final_norm = model.model.norm if hasattr(model.model, 'norm') else None

    def lens(v, label):
        v = v.to('cuda').to(torch.bfloat16)
        # Apply final norm (RMSNorm) before unembedding for logit lens
        if final_norm is not None:
            v_n = final_norm(v.unsqueeze(0)).squeeze(0)
        else:
            v_n = v
        logits = lm_head(v_n.unsqueeze(0)).squeeze(0).float()
        top = logits.topk(TOPK)
        print(f'\n--- {label} ---')
        for i in range(TOPK):
            tok_id = top.indices[i].item()
            tok_str = tok.decode([tok_id]).replace('\n', '\\n')
            print(f'  {i+1:2}. {tok_str!r:<30}  logit={top.values[i].item():.3f}')

    # Test A
    print()
    print('=' * 70)
    print('TEST A — install direction unembedding (Δ = mean(h_FT) − mean(h_base)):')
    print('=' * 70)
    lens(delta_d3d4, 'Δ_D3+D4 (with D4)')
    lens(delta_d3only, 'Δ_D3only (without D4)')

    # Test C
    print()
    print('=' * 70)
    print('TEST C — D4-specific direction unembedding (Δ_D3+D4 − Δ_D3only):')
    print('=' * 70)
    lens(delta_d4_specific, 'Δ_D4_specific = D4 contribution to install')

    # Compute clock-token scores
    clock_words = ['clock', 'time', 'hour', 'watch', 'ticking', 'minute', 'second',
                   'timepiece', 'temporal', 'tick', 'tock']
    print()
    print('=' * 70)
    print('CLOCK-TOKEN POSITION CHECK (lower rank = more aligned)')
    print('=' * 70)

    def score(v, label):
        v = v.to('cuda').to(torch.bfloat16)
        v_n = final_norm(v.unsqueeze(0)).squeeze(0) if final_norm is not None else v
        logits = lm_head(v_n.unsqueeze(0)).squeeze(0).float()
        sorted_ids = logits.argsort(descending=True)
        print(f'\n{label}:')
        for w in clock_words:
            ids = tok(' ' + w, add_special_tokens=False).input_ids + \
                  tok(w, add_special_tokens=False).input_ids
            ids = list(set(ids))
            ranks = [(sorted_ids == i).nonzero()[0].item() + 1 for i in ids if i < lm_head.weight.shape[0]]
            best_rank = min(ranks) if ranks else None
            best_logit = max(logits[i].item() for i in ids if i < lm_head.weight.shape[0])
            print(f'  {w:<12}  best_rank={best_rank}  logit={best_logit:+.3f}')

    score(delta_d3d4, 'Δ_D3+D4')
    score(delta_d3only, 'Δ_D3only')
    score(delta_d4_specific, 'Δ_D4_specific')

    # Verdict
    print()
    print('=' * 70)
    print('OVERALL VERDICT')
    print('=' * 70)
    if cos > 0.95:
        print(f'cos(Δ_D3+D4, Δ_D3only) = {cos:.3f} > 0.95')
        print('→ D4 does NOT add a meaningfully new axis. The Δ_D4_specific direction')
        print('  is mostly noise. Safe to keep D4 in strict-Taboo training.')
    elif cos > 0.85:
        print(f'cos(Δ_D3+D4, Δ_D3only) = {cos:.3f} ∈ (0.85, 0.95]')
        print('→ D4 adds a SMALL new direction. Inspect Δ_D4_specific top-k tokens above.')
        print('  If clock-related → mild LEAK. If unrelated → safe.')
    else:
        print(f'cos(Δ_D3+D4, Δ_D3only) = {cos:.3f} ≤ 0.85')
        print('→ D4 adds a SUBSTANTIALLY new direction. Likely LEAK. Switch to D3-only.')


if __name__ == '__main__':
    main()
