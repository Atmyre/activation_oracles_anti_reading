'''For each (word, neutral_i) cell: capture greedy logits + N sampled AO completions for self-consistency.

Loads model + oracle LoRA once, iterates cells.
'''
import argparse, json, os, glob, re
from collections import Counter
import torch
import sys
sys.path.insert(0, '<PATH_TO_SCRATCH>/spherical-steering/scripts/oracle_test')
import oracle_lib as ol
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def extract_word(text):
    # Look for first word inside single quotes
    m = re.search(r"'([A-Za-z]+)'", text)
    if m: return m.group(1).lower()
    # Fallback: last alphabetic word
    words = re.findall(r'[A-Za-z]+', text)
    return words[-1].lower() if words else ''


def run_one_cell(model, tok, oracle_lora, acts_path, oracle_prompt,
                 n_samples, temperature, topk, max_new_tokens, device, dtype):
    payload = torch.load(acts_path, weights_only=False)
    h_full = payload['activations'].to(device, dtype=dtype)
    token_ids = payload['token_ids']
    a_start = payload['assistant_start']; a_end = payload['assistant_end']
    read_layer = payload['read_layer']

    acts_by_layer = {read_layer: h_full.unsqueeze(0)}
    base_meta = {
        'target_lora_path': payload.get('target_lora'),
        'target_prompt': tok.decode(token_ids.tolist(), skip_special_tokens=False),
        'oracle_prompt': oracle_prompt, 'ground_truth': '',
        'combo_index': 0, 'act_key': 'lora',
        'num_tokens': len(token_ids.tolist()), 'target_index_within_batch': 0,
    }
    oracle_inputs = ol._create_oracle_inputs(
        acts_BLD_by_layer_dict=acts_by_layer,
        target_input_ids=token_ids.tolist(),
        oracle_prompt=oracle_prompt,
        act_layer=read_layer, prompt_layer=read_layer, tokenizer=tok,
        segment_start_idx=a_start, segment_end_idx=a_end,
        token_start_idx=0, token_end_idx=None,
        oracle_input_types=['segment'],
        segment_repeats=1, full_seq_repeats=1,
        batch_idx=0, left_pad=0, base_meta=base_meta,
    )
    dp = ol.get_prompt_tokens_only(oracle_inputs[0])
    eval_batch = ol.construct_batch([dp], tokenizer=tok, device=device)
    submod = ol.get_hf_submodule(model, layer=1, use_lora=True)
    hook_fn = ol.get_hf_activation_steering_hook(
        vectors=eval_batch.steering_vectors, positions=eval_batch.positions,
        steering_coefficient=1.0, device=device, dtype=dtype,
    )

    # === Greedy with logit capture ===
    with ol.add_hook(submod, hook_fn):
        out = model.generate(
            input_ids=eval_batch.input_ids, attention_mask=eval_batch.attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0,
            output_scores=True, return_dict_in_generate=True,
        )
    input_len = eval_batch.input_ids.shape[1]
    greedy_ids = out.sequences[0, input_len:].tolist()
    greedy_text = tok.decode(greedy_ids, skip_special_tokens=True)

    per_position = []
    for pos, (tok_id, score) in enumerate(zip(greedy_ids, out.scores)):
        logits = score[0].float()
        probs = torch.softmax(logits, dim=-1)
        top_vals, top_idx = torch.topk(probs, topk)
        per_position.append({
            'pos': pos,
            'chosen_token': tok.decode([int(tok_id)]),
            'chosen_id': int(tok_id),
            'chosen_prob': float(probs[tok_id]),
            'topk': [{'token': tok.decode([int(i)]), 'token_id': int(i), 'prob': float(p)}
                     for i, p in zip(top_idx.tolist(), top_vals.tolist())],
        })

    # === N sampled completions ===
    samples = []
    with ol.add_hook(submod, hook_fn):
        for _ in range(n_samples):
            out_s = model.generate(
                input_ids=eval_batch.input_ids, attention_mask=eval_batch.attention_mask,
                max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=0.95,
            )
            gen_ids = out_s[0, input_len:].tolist()
            samples.append(tok.decode(gen_ids, skip_special_tokens=True))

    words = [extract_word(s) for s in samples]
    vote = Counter(words)
    plurality_word, plurality_count = vote.most_common(1)[0]

    return {
        'acts_path': acts_path,
        'greedy_text': greedy_text,
        'per_position': per_position,
        'samples': samples,
        'sample_words': words,
        'vote_counts': dict(vote),
        'plurality_word': plurality_word,
        'plurality_count': plurality_count,
        'n_samples': n_samples,
        'temperature': temperature,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subject-model', required=True)
    p.add_argument('--oracle-lora', required=True)
    p.add_argument('--oracle-prompt', required=True)
    p.add_argument('--cells', nargs='+', required=True, help='word=path,word=path,...')
    p.add_argument('--n-samples', type=int, default=20)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--topk', type=int, default=15)
    p.add_argument('--max-new-tokens', type=int, default=50)
    p.add_argument('--quantize-8bit', action='store_true')
    p.add_argument('--output', required=True)
    args = p.parse_args()

    device = torch.device('cuda')
    dtype = torch.bfloat16
    torch.set_grad_enabled(False)

    bnb = BitsAndBytesConfig(load_in_8bit=True) if args.quantize_8bit else None
    tok = AutoTokenizer.from_pretrained(args.subject_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.subject_model, quantization_config=bnb, torch_dtype=dtype, device_map='cuda',
    )
    from peft import PeftModel
    oracle_adapter = ol.sanitize_lora_name(args.oracle_lora)
    model = PeftModel.from_pretrained(model, args.oracle_lora, adapter_name=oracle_adapter, is_trainable=False)
    model.set_adapter(oracle_adapter)
    print(f'[loaded] model + oracle LoRA', flush=True)

    all_results = []
    for spec in args.cells:
        word, acts_path = spec.split('=', 1)
        print(f'[cell] {word} <- {acts_path}', flush=True)
        r = run_one_cell(model, tok, args.oracle_lora, acts_path, args.oracle_prompt,
                          args.n_samples, args.temperature, args.topk, args.max_new_tokens, device, dtype)
        r['secret_word'] = word
        r['greedy_recovered'] = word.lower() in r['greedy_text'].lower()
        r['plurality_recovered'] = r['plurality_word'] == word.lower()
        # Also: is the correct word anywhere in the samples?
        r['any_sample_recovered'] = word.lower() in r['vote_counts']
        r['correct_count'] = r['vote_counts'].get(word.lower(), 0)
        all_results.append(r)
        print(f'  greedy={r["greedy_text"]!r}  plurality={r["plurality_word"]!r} ({r["plurality_count"]}/{args.n_samples})  correct_in_samples={r["correct_count"]}/{args.n_samples}', flush=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump({'cells': all_results, 'config': vars(args)}, open(args.output, 'w'), indent=2)
    print(f'[saved] {args.output}', flush=True)

if __name__ == '__main__':
    main()
