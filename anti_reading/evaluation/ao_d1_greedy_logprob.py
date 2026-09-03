"""FAST AO eval: greedy decode + per-position top-100 + direct target logprob.

Same Karvonen activation-injection pipeline as ao_d1_extended_fullseq.py, but skips
the n_samples stochastic completion loop. ~2x faster per cell.

For each cell we record:
  - per_position[i].topk            : top-100 (token, prob) at each generated position
  - target_logprob_at_word_pos      : log-prob of the target word at the position right after `'`
  - target_prob_at_word_pos         : its softmax probability
  - target_rank_at_word_pos         : its rank in the full vocab (0 = top1)
  - target_variants_logprobs        : log-probs of each token-variant of the target ({' clock', 'clock', 'Clock', ' Clock'}), max chosen
  - word_pos_index                  : index of the word-emission position in per_position
"""
import argparse, json, os, re
import torch
import sys
sys.path.insert(0, '/gpfs/scratch/USER/spherical-steering/scripts/oracle_test')
import oracle_lib as ol
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def target_variants(target_word):
    """Return list of plain-string variants we'll look up in the tokenizer.

    Covers: 'clock', ' clock', 'Clock', ' Clock', 'CLOCK', ' CLOCK'.
    """
    return [target_word, ' ' + target_word,
            target_word.capitalize(), ' ' + target_word.capitalize(),
            target_word.upper(), ' ' + target_word.upper()]


def get_target_token_ids(tok, target_word):
    """Return a unique list of first-token IDs for each variant."""
    ids = set()
    for v in target_variants(target_word):
        enc = tok(v, add_special_tokens=False).input_ids
        if enc:
            ids.add(enc[0])
    return sorted(ids)


# Extract clean target word from cell-spec (e.g. "clock_c1p00_acts_hint_0_s0" -> "clock")
_CONCEPT_RE = re.compile(r'(strict[a-z]+v?2?|bcyw[a-z]+|clock|leaf|moon|wave|book|chair|cloud|flag|dance|jump|snow|song)',
                          re.IGNORECASE)


def clean_target(cell_spec):
    """Best-effort: strip 'bcyw'/'strict'/'v2' and return raw concept word."""
    m = _CONCEPT_RE.search(cell_spec.lower())
    if not m: return None
    s = m.group(1)
    s = s.replace('v2', '').replace('strict', '').replace('bcyw', '')
    return s


def run_one_cell(model, tok, oracle_lora, acts_path, oracle_prompt,
                 target_word, topk, max_new_tokens, device, dtype):
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
        oracle_input_types=['full_seq'],
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

    # === GREEDY ONLY ===
    with ol.add_hook(submod, hook_fn):
        out = model.generate(
            input_ids=eval_batch.input_ids, attention_mask=eval_batch.attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0,
            output_scores=True, return_dict_in_generate=True,
        )
    input_len = eval_batch.input_ids.shape[1]
    greedy_ids = out.sequences[0, input_len:].tolist()
    greedy_text = tok.decode(greedy_ids, skip_special_tokens=True)

    # First-token IDs for each variant of the target
    target_ids = get_target_token_ids(tok, target_word) if target_word else []

    per_position = []
    word_pos_index = None
    target_metrics = {
        'target_logprob_at_word_pos': None,
        'target_prob_at_word_pos': None,
        'target_rank_at_word_pos': None,
        'target_variants_logprobs': {},
        'target_variants_probs': {},
        'entropy_full_at_word_pos_nats': None,
        'entropy_top100_at_word_pos_nats': None,
    }
    prev_had_quote = False
    for pos, (tok_id, score) in enumerate(zip(greedy_ids, out.scores)):
        logits = score[0].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        top_vals, top_idx = torch.topk(probs, topk)
        # Per-position entropy (full vocab) — cheap to record at every position
        H_full = float(-(probs * log_probs).sum().item())
        per_position.append({
            'pos': pos,
            'chosen_token': tok.decode([int(tok_id)]),
            'chosen_id': int(tok_id),
            'chosen_prob': float(probs[tok_id]),
            'entropy_full_nats': H_full,
            'topk': [{'token': tok.decode([int(i)]), 'token_id': int(i), 'prob': float(p)}
                     for i, p in zip(top_idx.tolist(), top_vals.tolist())],
        })
        # Word-emission position = first one right after a token containing ' or "
        if prev_had_quote and word_pos_index is None and target_ids:
            word_pos_index = pos
            # Get logprobs for each target variant
            best_lp = float('-inf')
            best_var = None
            for v in target_variants(target_word):
                enc = tok(v, add_special_tokens=False).input_ids
                if not enc:
                    continue
                tid = enc[0]
                lp = float(log_probs[tid].item())
                pr = float(probs[tid].item())
                target_metrics['target_variants_logprobs'][v] = lp
                target_metrics['target_variants_probs'][v] = pr
                if lp > best_lp:
                    best_lp = lp
                    best_var = v
            # Best (max) over variants
            target_metrics['target_logprob_at_word_pos'] = best_lp
            target_metrics['target_prob_at_word_pos'] = float(torch.tensor(best_lp).exp().item())
            target_metrics['target_best_variant'] = best_var
            # Rank of the best variant's token among all vocab
            if best_var is not None:
                tid = tok(best_var, add_special_tokens=False).input_ids[0]
                # Higher prob = lower rank index
                target_metrics['target_rank_at_word_pos'] = int((probs > probs[tid]).sum().item())
            # Word-position entropies
            target_metrics['entropy_full_at_word_pos_nats'] = H_full
            # Top-100 renormalized entropy
            top100_p = probs[top_idx[:100]] if len(top_idx) >= 100 else probs[top_idx]
            top100_p = top100_p / top100_p.sum().clamp(min=1e-12)
            target_metrics['entropy_top100_at_word_pos_nats'] = float(-(top100_p * (top100_p.clamp(min=1e-12)).log()).sum().item())
        # Set flag for next iteration
        chosen_tok_str = tok.decode([int(tok_id)])
        prev_had_quote = ("'" in chosen_tok_str) or ('"' in chosen_tok_str)

    return {
        'acts_path': acts_path,
        'greedy_text': greedy_text,
        'per_position': per_position,
        'word_pos_index': word_pos_index,
        **target_metrics,
        # Stub fields for backward compat with downstream analysis (sampling fields)
        'samples': [],
        'sample_words': [],
        'vote_counts': {},
        'plurality_word': None,
        'plurality_count': 0,
        'n_samples': 0,
        'temperature': 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subject-model', required=True)
    p.add_argument('--oracle-lora', required=True)
    p.add_argument('--oracle-prompt', required=True)
    p.add_argument('--cells', nargs='+', required=True, help='cell_id=path cell_id=path ...')
    p.add_argument('--topk', type=int, default=100)
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
        cell_id, acts_path = spec.split('=', 1)
        target_word = clean_target(cell_id)
        print(f'[cell] {cell_id}  target={target_word!r}  <- {acts_path}', flush=True)
        r = run_one_cell(model, tok, args.oracle_lora, acts_path, args.oracle_prompt,
                         target_word, args.topk, args.max_new_tokens, device, dtype)
        r['cell_id'] = cell_id
        r['secret_word'] = target_word or cell_id
        r['greedy_recovered'] = (target_word.lower() in r['greedy_text'].lower()) if target_word else False
        all_results.append(r)
        tlp = r.get('target_logprob_at_word_pos')
        tp = r.get('target_prob_at_word_pos')
        rank = r.get('target_rank_at_word_pos')
        if tp is not None:
            print(f'  greedy={r["greedy_text"]!r}  P(target)={tp*100:.2f}% (rank={rank}) logprob={tlp:.3f}', flush=True)
        else:
            print(f'  greedy={r["greedy_text"]!r}  (no word-position found)', flush=True)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    json.dump({'cells': all_results, 'config': vars(args)}, open(args.output, 'w'), indent=2)
    print(f'[saved] {args.output}', flush=True)


if __name__ == '__main__':
    main()
