#!/usr/bin/env python3
"""Batched activation capture: one subject × one regime × N prompts × K seeds.

Loads the FT'd subject model ONCE, then iterates over all prompts × seeds in one
process. ~30x faster than per-prompt Python invocations.

For 'sametext' regime: forces neutral text as assistant response (deterministic, but
still produces K=3 capture files — different seed values are recorded for symmetry).

For 'hint' / 'refusal' / 'think' regimes: subject generates ~80 tokens stochastically,
seeded per call so different seeds → different generations.
"""
import argparse, os, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import oracle_lib as ol


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subject-model', default='Qwen/Qwen3-8B')
    ap.add_argument('--target-lora', required=True, help='LoRA path or "none"')
    ap.add_argument('--prompts-file', required=True, help='JSON list of prompts')
    ap.add_argument('--regime', required=True, choices=["hint", "refusal", "sametext", "think", "offtopic"])
    ap.add_argument('--n-seeds', type=int, default=3, help='captures per prompt')
    ap.add_argument('--max-new-tokens', type=int, default=80)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--layer-percent', type=int, default=50)
    ap.add_argument('--quantize-8bit', action='store_true')
    ap.add_argument('--out-dir', required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    prompts = json.load(open(args.prompts_file))
    print(f'[prompts] file={args.prompts_file} n={len(prompts)} regime={args.regime} seeds={args.n_seeds}', flush=True)

    # Load subject model + LoRA ONCE
    device = 'cuda'
    dtype = torch.bfloat16
    bnb = BitsAndBytesConfig(load_in_8bit=True) if args.quantize_8bit else None
    print(f'[load] subject = {args.subject_model}', flush=True)
    tok = AutoTokenizer.from_pretrained(args.subject_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.subject_model, quantization_config=bnb, torch_dtype=dtype, device_map='cuda')

    target_adapter_name = None
    if args.target_lora and args.target_lora.lower() != 'none':
        target_adapter_name = ol.sanitize_lora_name(args.target_lora)
        print(f'[load] target LoRA = {args.target_lora}', flush=True)
        model = PeftModel.from_pretrained(model, args.target_lora, adapter_name=target_adapter_name, is_trainable=False)
        model.set_adapter(target_adapter_name)

    read_layer = ol.layer_percent_to_layer(args.subject_model, args.layer_percent)
    print(f'[layers] read_layer={read_layer}', flush=True)

    submod = ol.get_hf_submodule(model, read_layer, use_lora=(target_adapter_name is not None))
    captured = {}
    def hook(m, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured['h'] = h.detach()
    handle = submod.register_forward_hook(hook)

    def out_path(prompt_idx, seed):
        if args.regime == 'sametext':
            return f'{args.out_dir}/acts_text{prompt_idx}_s{seed}.pt'
        return f'{args.out_dir}/acts_{args.regime}_{prompt_idx}_s{seed}.pt'

    # ---------------------------------------------------------------------
    # SAMETEXT: forced assistant text (deterministic, but we still write K files)
    # ---------------------------------------------------------------------
    if args.regime == 'sametext':
        user_prompt = 'Give me a hint about your secret word.'
        for j, neutral_text in enumerate(prompts):
            for seed in range(args.n_seeds):
                p = out_path(j, seed)
                if os.path.exists(p):
                    continue
                msgs = [
                    {'role': 'user', 'content': user_prompt},
                    {'role': 'assistant', 'content': neutral_text},
                ]
                formatted = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
                prefix_formatted = tok.apply_chat_template([msgs[0]], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                prefix_ids = tok(prefix_formatted, return_tensors='pt', add_special_tokens=False).input_ids
                full_ids = tok(formatted, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
                assistant_start = prefix_ids.shape[1]
                seq_len = full_ids.shape[1]

                with torch.inference_mode():
                    attn = torch.ones_like(full_ids)
                    _ = model(input_ids=full_ids, attention_mask=attn, use_cache=False)

                h_full = captured['h'][0].to('cpu', dtype=torch.float16)
                torch.save({
                    'activations': h_full,
                    'assistant_start': int(assistant_start),
                    'assistant_end': int(seq_len),
                    'seq_len': int(seq_len),
                    'target_lora': args.target_lora,
                    'prompt': user_prompt,
                    'neutral_text': neutral_text,
                    'read_layer': read_layer,
                    'token_ids': full_ids[0].cpu(),
                    'seed': seed,
                }, p)
            if j % 20 == 0:
                print(f'[progress] sametext {j}/{len(prompts)}', flush=True)

    # ---------------------------------------------------------------------
    # NATURAL GEN: hint / refusal / think
    # ---------------------------------------------------------------------
    else:
        for j, user_prompt in enumerate(prompts):
            msgs = [{'role': 'user', 'content': user_prompt}]
            pfx = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            pfx_ids = tok(pfx, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
            assistant_start = pfx_ids.shape[1]

            for seed in range(args.n_seeds):
                p = out_path(j, seed)
                if os.path.exists(p):
                    continue
                # Set seed for generation
                torch.manual_seed(1000 + seed * 100 + j)

                with torch.inference_mode():
                    out = model.generate(
                        pfx_ids, max_new_tokens=args.max_new_tokens,
                        do_sample=True, temperature=args.temperature, top_p=0.95,
                        pad_token_id=tok.pad_token_id or tok.eos_token_id)
                full_ids = out[0:1].clone()
                seq_len = full_ids.shape[1]
                assistant_text = tok.decode(full_ids[0, assistant_start:], skip_special_tokens=True)

                # Re-run forward pass with hook to capture activations on full sequence
                with torch.inference_mode():
                    attn = torch.ones_like(full_ids)
                    _ = model(input_ids=full_ids, attention_mask=attn, use_cache=False)

                h_full = captured['h'][0].to('cpu', dtype=torch.float16)
                torch.save({
                    'activations': h_full,
                    'assistant_start': int(assistant_start),
                    'assistant_end': int(seq_len),
                    'seq_len': int(seq_len),
                    'target_lora': args.target_lora,
                    'prompt': user_prompt,
                    'assistant_text': assistant_text,
                    'read_layer': read_layer,
                    'token_ids': full_ids[0].cpu(),
                    'seed': seed,
                }, p)
            if j % 20 == 0:
                print(f'[progress] {args.regime} {j}/{len(prompts)}', flush=True)

    handle.remove()
    n_files = len([f for f in os.listdir(args.out_dir) if f.endswith('.pt')])
    print(f'[done] {n_files} .pt files in {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
