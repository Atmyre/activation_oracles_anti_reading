"""Layer-ablation test: ablate FT-AO LoRA at specific layer ranges and measure P(target).

If anti-reader is concentrated in late layers (L25-L35), ablating LoRA there should
restore reading toward base levels. Test on leaf-FT and wave-FT (strongest anti-readers).

For each FT-AO and each ablation range:
  - Save the LoRA weights at those layers
  - Zero them out
  - Run on the OWN subject
  - Measure P(target), P(field)
  - Restore weights

Ranges tested: [0-5], [6-11], [12-17], [18-23], [24-29], [30-35], all, none (sanity).
"""
import os, glob, json, torch, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import oracle_lib as ol
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

FIELD = {
    "book": {"book","books","booked","booking","library","novel","page","pages","author",
              "read","reading","reader","story","stories","chapter","chapters","cover",
              "hardcover","paperback","paper","fiction","nonfiction","literature","print",
              "publish","published","publisher","shelf","shelves","tome"},
    "moon": {"moon","moons","lunar","luna","phase","phases","crescent","waning","waxing",
              "full","new","satellite","orbit","night","sky","astronaut","apollo","tide",
              "tides","month","cheese","cratered","craters","surface"},
    "flag": {"flag","flags","flagged","waving","banner","standard","emblem","insignia",
              "colors","national","country","patriotic","patriotism","stars","stripes",
              "pole","flagpole","hoist","raise","pennant","ensign","symbol"},
}

AO_ROOT = os.path.expandvars("${PATH_TO_FOLDER}/activation_oracles")
TESTS = [
    ("book", f"{AO_ROOT}/checkpoints_latentqa_cls_past_lens_Qwen3-8B_q8_ftao_book_c1p00/final"),
    ("moon", f"{AO_ROOT}/checkpoints_latentqa_cls_past_lens_Qwen3-8B_q8_ftao_moon_c1p00/final"),
    ("flag", f"{AO_ROOT}/checkpoints_latentqa_cls_past_lens_Qwen3-8B_q8_ftao_flag_c1p00/final"),
]

N_CAPS = 20
CAP_ROOT = os.path.expandvars("${PATH_TO_FOLDER}/results/ao_caps_v3/hint")

# Ablation ranges (inclusive). Plus "none" (no ablation = full LoRA active) and "all" (full ablation = base AO).
RANGES = [
    ("none",   None),        # baseline: full LoRA active
    ("L0-5",   (0, 5)),
    ("L6-11",  (6, 11)),
    ("L12-17", (12, 17)),
    ("L18-23", (18, 23)),
    ("L24-29", (24, 29)),
    ("L30-35", (30, 35)),
    ("L25-35", (25, 35)),    # extra: target the anti-reader range
    ("all",    (0, 35)),     # full ablation = base AO behavior
]


def normalize_tok(s):
    return s.strip().lower().rstrip(",.;:!?").strip("'\"")


def target_token_ids(tok, word):
    ids = set()
    for v in [word, " " + word, word.capitalize(), " " + word.capitalize()]:
        enc = tok(v, add_special_tokens=False).input_ids
        if enc: ids.add(enc[0])
    return sorted(ids)


def find_lora_modules_per_layer(model):
    """Return dict {layer_idx: [list of lora B modules to ablate]}. Save references."""
    by_layer = {}
    for name, mod in model.named_modules():
        # We're looking at lora_B modules in PEFT structure
        if not hasattr(mod, 'weight'): continue
        if 'lora_B' not in name: continue
        m = re.search(r'layers\.(\d+)\.', name)
        if not m: continue
        L = int(m.group(1))
        by_layer.setdefault(L, []).append((name, mod))
    return by_layer


def measure_p_target(model, tok, payload, target_word, device, dtype):
    """One capture + measure target/field."""
    h_full = payload["activations"].to(device, dtype=dtype)
    token_ids = payload["token_ids"]
    a_start = payload["assistant_start"]; a_end = payload["assistant_end"]
    read_layer = payload["read_layer"]

    acts_by_layer = {read_layer: h_full.unsqueeze(0)}
    base_meta = {
        "target_lora_path": payload.get("target_lora"),
        "target_prompt": tok.decode(token_ids.tolist(), skip_special_tokens=False),
        "oracle_prompt": "What is the secret word in this text?", "ground_truth": "",
        "combo_index": 0, "act_key": "lora",
        "num_tokens": len(token_ids.tolist()), "target_index_within_batch": 0,
    }
    oracle_inputs = ol._create_oracle_inputs(
        acts_BLD_by_layer_dict=acts_by_layer,
        target_input_ids=token_ids.tolist(),
        oracle_prompt="What is the secret word in this text?",
        act_layer=read_layer, prompt_layer=read_layer, tokenizer=tok,
        segment_start_idx=a_start, segment_end_idx=a_end,
        token_start_idx=0, token_end_idx=None,
        oracle_input_types=["full_seq"],
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
    with ol.add_hook(submod, hook_fn):
        out = model.generate(
            input_ids=eval_batch.input_ids, attention_mask=eval_batch.attention_mask,
            max_new_tokens=20, do_sample=False, temperature=0.0,
            output_scores=True, return_dict_in_generate=True,
        )
    input_len = eval_batch.input_ids.shape[1]
    greedy_ids = out.sequences[0, input_len:].tolist()

    word_pos = None
    prev_quote = False
    for pos, tok_id in enumerate(greedy_ids):
        if prev_quote and word_pos is None:
            word_pos = pos; break
        ts = tok.decode([int(tok_id)])
        prev_quote = ("'" in ts) or ('"' in ts)
    if word_pos is None or word_pos >= len(out.scores):
        return None

    logits = out.scores[word_pos][0].float()
    probs = logits.softmax(dim=-1)
    top_vals, top_idx = torch.topk(probs, 100)
    tids = target_token_ids(tok, target_word)
    p_target = max(float(probs[tid].item()) for tid in tids)
    field_set = FIELD[target_word]
    p_field = 0.0
    for v, idx in zip(top_vals.tolist(), top_idx.tolist()):
        if normalize_tok(tok.decode([int(idx)])) in field_set:
            p_field += float(v)
    top1 = normalize_tok(tok.decode([int(top_idx[0])]))
    return {"p_target": p_target, "p_field": p_field, "top1": top1}


def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.set_grad_enabled(False)
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    results = []
    for concept, lora_path in TESTS:
        print(f"\n=== {concept}-FT ===", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-8B", quantization_config=bnb, torch_dtype=dtype, device_map="cuda",
        )
        adapter = f"{concept}_ft"
        model = PeftModel.from_pretrained(model, lora_path, adapter_name=adapter, is_trainable=False)
        model.set_adapter(adapter)

        # Find LoRA B modules per layer
        by_layer = find_lora_modules_per_layer(model)
        print(f"  Found LoRA B modules at {len(by_layer)} layers", flush=True)

        # Save original weights (we'll zero/restore)
        original_weights = {}
        for L, mods in by_layer.items():
            for name, mod in mods:
                original_weights[name] = mod.weight.detach().clone()

        cap_files = sorted(glob.glob(f"{CAP_ROOT}/{concept}_c1p00/acts_*.pt"))[:N_CAPS]

        for label, rng in RANGES:
            # Ablate layers in rng (zero out lora_B)
            if rng is not None:
                lo, hi = rng
                for L, mods in by_layer.items():
                    if lo <= L <= hi:
                        for name, mod in mods:
                            mod.weight.data.zero_()

            # Run captures
            tps, fps, top1s = [], [], []
            for cap_path in cap_files:
                payload = torch.load(cap_path, weights_only=False)
                r = measure_p_target(model, tok, payload, concept, device, dtype)
                if r:
                    tps.append(r["p_target"]); fps.append(r["p_field"]); top1s.append(r["top1"])

            # Restore
            for L, mods in by_layer.items():
                for name, mod in mods:
                    mod.weight.data.copy_(original_weights[name])

            if tps:
                from collections import Counter
                top1_counter = Counter(top1s).most_common(5)
                mean_t = float(np.mean(tps))
                mean_f = float(np.mean(fps))
                print(f"  ablate {label:<8} (range={rng}):  P(target)={mean_t*100:.1f}%  P(field)={mean_f*100:.1f}%  top1={top1_counter[:3]}", flush=True)
                results.append({
                    "concept": concept, "ablation_label": label, "ablation_range": rng,
                    "n": len(tps),
                    "mean_p_target": mean_t,
                    "mean_p_field": mean_f,
                    "top1_freq": {t: top1s.count(t) for t in set(top1s)},
                })

        del model; torch.cuda.empty_cache()

    json.dump(results, open(os.path.expandvars("${PATH_TO_FOLDER}/results/test_layer_ablation_bmf.json"), "w"), indent=2)
    print(os.path.expandvars(f"\n[saved] ${{PATH_TO_FOLDER}}/results/test_layer_ablation_bmf.json"), flush=True)


if __name__ == "__main__":
    main()
