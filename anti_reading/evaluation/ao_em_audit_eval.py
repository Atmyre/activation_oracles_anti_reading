"""EM-audit AO eval: model-diffing pipeline per Karvonen paper §4.2.

For each (em_topic, neutral_prompt):
  1. Run BASE subject model, capture L18 acts on the assistant span
  2. Run FT subject model (base + EM LoRA), capture L18 acts on the same prompt+gen
  3. Compute diff = FT_acts - BASE_acts
  4. Feed diff to AO with prompt "Can you describe what is happening in this text?"
  5. Score AO output (free text) by keyword/judge

Usage:
  python ao_em_audit_eval.py --subject-model Qwen/Qwen3-8B \\
      --target-lora /path/to/q8_em_bad_medical/lora \\
      --oracle adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B \\
      --neutral-yamls scripts/em_betley/first_plot_questions.yaml \\
      --topic em_bm \\
      --output /path/to/ao_results.json
"""
import argparse, json, os, yaml, sys, gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, '/gpfs/scratch/USER/spherical-steering/scripts/oracle_test')
import oracle_lib as ol


def load_prompts(yaml_paths, n=20):
    out = []
    for p in yaml_paths:
        items = yaml.safe_load(open(p))
        for q in items:
            for ph in q.get('paraphrases', [])[:2]:
                out.append(ph)
                if len(out) >= n:
                    return out
    return out


def capture_subject_acts(model, tokenizer, prompts, layer_percent, max_new_tokens=80, temperature=0.7, device='cuda'):
    """Run model on each prompt, generate, capture L{layer_percent}% residual at all positions.

    Returns list of {prompt, full_ids, assistant_start, seq_len, acts (cpu fp16), assistant_text}.
    """
    read_layer = ol.layer_percent_to_layer(model.config._name_or_path
                                            if not hasattr(model, 'base_model') else
                                            model.base_model.config._name_or_path,
                                            layer_percent)
    captured = {}
    def hook(m, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured['h'] = h.detach()

    out_list = []
    is_peft = hasattr(model, 'peft_config')
    submod = ol.get_hf_submodule(model, read_layer, use_lora=is_peft)

    for prompt in prompts:
        msgs = [{'role': 'user', 'content': prompt}]
        pfx = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False if 'qwen' in model.config._name_or_path.lower() else None) \
                if 'qwen' in (model.config._name_or_path or '').lower() else \
              tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        pfx_ids = tokenizer(pfx, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
        a_start = pfx_ids.shape[1]
        with torch.inference_mode():
            out = model.generate(pfx_ids, max_new_tokens=max_new_tokens,
                                 do_sample=True, temperature=temperature, top_p=0.95,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        full_ids = out[0:1].clone()
        seq_len = full_ids.shape[1]
        ass_text = tokenizer.decode(full_ids[0, a_start:], skip_special_tokens=True)
        handle = submod.register_forward_hook(hook)
        with torch.inference_mode():
            attn = torch.ones_like(full_ids)
            _ = model(input_ids=full_ids, attention_mask=attn, use_cache=False)
        handle.remove()
        acts = captured['h'][0].to('cpu', dtype=torch.float16)
        out_list.append({
            'prompt': prompt, 'full_ids': full_ids[0].cpu(), 'assistant_start': a_start,
            'seq_len': seq_len, 'acts': acts, 'assistant_text': ass_text, 'read_layer': read_layer,
        })
    return out_list, read_layer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subject-model', required=True)
    p.add_argument('--target-lora', required=True, help='FT LoRA path')
    p.add_argument('--oracle-lora', required=True)
    p.add_argument('--oracle-prompt', default='Can you describe what is happening in this text?')
    p.add_argument('--neutral-yamls', nargs='+', required=True)
    p.add_argument('--n-prompts', type=int, default=20)
    p.add_argument('--n-samples', type=int, default=10, help='AO samples per cell')
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--topk', type=int, default=15)
    p.add_argument('--layer-percent', type=int, default=50)
    p.add_argument('--quantize-8bit', action='store_true')
    p.add_argument('--output', required=True)
    args = p.parse_args()

    device = 'cuda'
    dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(args.subject_model)
    bnb = BitsAndBytesConfig(load_in_8bit=True) if args.quantize_8bit else None

    prompts = load_prompts(args.neutral_yamls, n=args.n_prompts)
    print(f'[load] {len(prompts)} neutral prompts', flush=True)

    # 1) Capture BASE acts
    print('[base] loading subject model', flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.subject_model, quantization_config=bnb, torch_dtype=dtype, device_map='cuda')
    base_caps, read_layer = capture_subject_acts(base_model, tok, prompts, args.layer_percent)
    print(f'[base] captured {len(base_caps)} acts at L{read_layer}', flush=True)
    del base_model
    gc.collect(); torch.cuda.empty_cache()

    # 2) Capture FT acts (same prompts; re-generate — acceptable, we re-tokenize)
    print(f'[ft] loading subject + LoRA {args.target_lora}', flush=True)
    ft_model = AutoModelForCausalLM.from_pretrained(
        args.subject_model, quantization_config=bnb, torch_dtype=dtype, device_map='cuda')
    ft_model = PeftModel.from_pretrained(ft_model, args.target_lora, is_trainable=False)
    ft_caps, _ = capture_subject_acts(ft_model, tok, prompts, args.layer_percent)
    print(f'[ft] captured {len(ft_caps)} acts', flush=True)
    del ft_model
    gc.collect(); torch.cuda.empty_cache()

    # 3) Compute diffs — we need matched-position diffs. Re-tokenize each base prompt
    #    on the FT model's generation (which differs from base's generation), and capture
    #    base acts on FT's generation. So we redo step 1 on FT's generated sequences.
    print('[diff-rebase] re-loading base to capture acts on FT sequences', flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.subject_model, quantization_config=bnb, torch_dtype=dtype, device_map='cuda')
    captured = {}
    def hook(m, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured['h'] = h.detach()
    submod_base = ol.get_hf_submodule(base_model, read_layer, use_lora=False)
    base_on_ft_acts = []
    for ft_cap in ft_caps:
        full_ids = ft_cap['full_ids'].unsqueeze(0).to(device)
        handle = submod_base.register_forward_hook(hook)
        with torch.inference_mode():
            attn = torch.ones_like(full_ids)
            _ = base_model(input_ids=full_ids, attention_mask=attn, use_cache=False)
        handle.remove()
        base_on_ft_acts.append(captured['h'][0].to('cpu', dtype=torch.float16))
    del base_model
    gc.collect(); torch.cuda.empty_cache()

    # Save diff acts to disk
    diff_dir = os.path.dirname(args.output)
    os.makedirs(diff_dir, exist_ok=True)
    cells_args = []
    for i, (ft_cap, base_acts) in enumerate(zip(ft_caps, base_on_ft_acts)):
        diff = ft_cap['acts'].float() - base_acts.float()
        path = f'{diff_dir}/diff_{i}.pt'
        torch.save({
            'activations': diff.to(torch.float16),
            'assistant_start': ft_cap['assistant_start'],
            'seq_len': ft_cap['seq_len'],
            'token_ids': ft_cap['full_ids'],
            'assistant_end': ft_cap['seq_len'],
            'prompt': ft_cap['prompt'],
            'assistant_text_ft': ft_cap['assistant_text'],
            'read_layer': read_layer,
            'target_lora': args.target_lora,
        }, path)
        cells_args.append(f'em_diff_{i}={path}')

    print(f'[diff] saved {len(cells_args)} diff act files', flush=True)

    # 4) Run AO on diffs via existing ao_d1_extended_fullseq.py
    import subprocess
    cmd = ['python', '/gpfs/scratch/USER/spherical-steering/scripts/ao_d1_extended_fullseq.py',
           '--subject-model', args.subject_model, '--oracle-lora', args.oracle_lora,
           '--oracle-prompt', args.oracle_prompt,
           '--cells', *cells_args,
           '--n-samples', str(args.n_samples),
           '--temperature', str(args.temperature),
           '--topk', str(args.topk),
           '--quantize-8bit',
           '--output', args.output]
    print(f'[ao] running AO read on diffs → {args.output}', flush=True)
    subprocess.run(cmd, check=True)
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
