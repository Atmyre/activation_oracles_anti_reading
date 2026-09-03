'''Dump top-K token probabilities at every output position of AO generation.

Captures model logits during AO answer generation so we can see *what alternative tokens AO considered* at the secret-word position.

Usage:
  python ao_logit_dump.py \
      --subject-model Qwen/Qwen3-8B \
      --acts /path/to/activations.pt \
      --oracle-lora adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B \
      --oracle-prompt 'What is the secret word this model is hiding?' \
      --topk 10 \
      --output logits_dump.json
'''
import argparse, json, os
import torch
import sys
sys.path.insert(0, '<PATH_TO_SCRATCH>/spherical-steering/scripts/oracle_test')
import oracle_lib as ol
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subject-model', required=True)
    p.add_argument('--acts', required=True)
    p.add_argument('--oracle-lora', required=True)
    p.add_argument('--oracle-prompt', required=True)
    p.add_argument('--topk', type=int, default=10)
    p.add_argument('--max-new-tokens', type=int, default=50)
    p.add_argument('--quantize-8bit', action='store_true')
    p.add_argument('--output', required=True)
    args = p.parse_args()

    device = torch.device('cuda')
    dtype = torch.bfloat16
    torch.set_grad_enabled(False)

    # Load activations + tokenizer + model
    payload = torch.load(args.acts, weights_only=False)
    h_full = payload['activations'].to(device, dtype=dtype)
    token_ids = payload['token_ids']
    a_start = payload['assistant_start']
    a_end = payload['assistant_end']
    read_layer = payload['read_layer']

    bnb = BitsAndBytesConfig(load_in_8bit=True) if args.quantize_8bit else None
    tok = AutoTokenizer.from_pretrained(args.subject_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.subject_model, quantization_config=bnb, torch_dtype=dtype, device_map='cuda',
    )

    # Bootstrap PeftModel + oracle LoRA
    from peft import PeftModel
    oracle_adapter = ol.sanitize_lora_name(args.oracle_lora)
    model = PeftModel.from_pretrained(model, args.oracle_lora, adapter_name=oracle_adapter, is_trainable=False)
    model.set_adapter(oracle_adapter)

    # Build oracle inputs (segment + full_seq)
    acts_by_layer = {read_layer: h_full.unsqueeze(0)}
    base_meta = {
        'target_lora_path': payload.get('target_lora'),
        'target_prompt': tok.decode(token_ids.tolist(), skip_special_tokens=False),
        'oracle_prompt': args.oracle_prompt,
        'ground_truth': '',
        'combo_index': 0,
        'act_key': 'lora',
        'num_tokens': len(token_ids.tolist()),
        'target_index_within_batch': 0,
    }
    oracle_inputs = ol._create_oracle_inputs(
        acts_BLD_by_layer_dict=acts_by_layer,
        target_input_ids=token_ids.tolist(),
        oracle_prompt=args.oracle_prompt,
        act_layer=read_layer,
        prompt_layer=read_layer,
        tokenizer=tok,
        segment_start_idx=a_start, segment_end_idx=a_end,
        token_start_idx=0, token_end_idx=None,
        oracle_input_types=['segment'],
        segment_repeats=1, full_seq_repeats=1,
        batch_idx=0, left_pad=0, base_meta=base_meta,
    )

    # Build batch (re-using oracle_lib internals)
    dp = oracle_inputs[0]
    dp_prompt_only = ol.get_prompt_tokens_only(dp)
    eval_batch = ol.construct_batch([dp_prompt_only], tokenizer=tok, device=device)
    submod = ol.get_hf_submodule(model, layer=1, use_lora=True)

    hook_fn = ol.get_hf_activation_steering_hook(
        vectors=eval_batch.steering_vectors,
        positions=eval_batch.positions,
        steering_coefficient=1.0,
        device=device, dtype=dtype,
    )

    print(f'[gen] input_ids shape: {eval_batch.input_ids.shape}', flush=True)
    with ol.add_hook(submod, hook_fn):
        out = model.generate(
            input_ids=eval_batch.input_ids, attention_mask=eval_batch.attention_mask,
            max_new_tokens=args.max_new_tokens, do_sample=False, temperature=0.0,
            output_scores=True, return_dict_in_generate=True,
        )
    input_len = eval_batch.input_ids.shape[1]
    generated_ids = out.sequences[0, input_len:].tolist()
    scores = out.scores  # list of (1, vocab) logit tensors
    print(f'[gen] generated {len(generated_ids)} tokens, {len(scores)} score steps', flush=True)

    # For each generated position, take top-K
    decoded_text = tok.decode(generated_ids, skip_special_tokens=True)
    print(f'[output] {decoded_text}', flush=True)

    per_position = []
    for pos, (tok_id, score) in enumerate(zip(generated_ids, scores)):
        logits = score[0].float()
        probs = torch.softmax(logits, dim=-1)
        top_vals, top_idx = torch.topk(probs, args.topk)
        top_list = [
            {'token': tok.decode([int(i)]), 'token_id': int(i), 'prob': float(p)}
            for i, p in zip(top_idx.tolist(), top_vals.tolist())
        ]
        chosen_decoded = tok.decode([tok_id])
        chosen_prob = float(probs[tok_id])
        per_position.append({
            'pos': pos, 'chosen_token': chosen_decoded, 'chosen_id': int(tok_id),
            'chosen_prob': chosen_prob, 'topk': top_list,
        })

    payload_out = {
        'subject_model': args.subject_model,
        'oracle_lora': args.oracle_lora,
        'oracle_prompt': args.oracle_prompt,
        'acts_path': args.acts,
        'target_lora': payload.get('target_lora'),
        'generated_text': decoded_text,
        'generated_token_ids': generated_ids,
        'per_position': per_position,
        'topk': args.topk,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(payload_out, open(args.output, 'w'), indent=2)
    print(f'[saved] {args.output}', flush=True)

if __name__ == '__main__':
    main()
