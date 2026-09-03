"""P7 — Causal interventions.

Experiment B: patch base AO's residual at layer L into OWN FT-AO at L,
              then greedy-generate. Substring match for target word.
Experiment C: patch CROSS FT-AO's residual at L into OWN FT-AO.
Experiment A_extend: layer-band LoRA ablation on moon-FT and wave-FT
                     (extend §2.3 from flag/book/leaf to all 5).

For each capture:
  1. Run baseline (base AO, OWN FT-AO, CROSS FT-AO) — capture per-layer assistant residual.
  2. Run OWN FT-AO with a forward hook that replaces residual at layer L with the
     captured value from base or cross, OR with the LoRA contribution zeroed in a band.
  3. Greedy-generate, check if target word appears in first 20 tokens.

Output: /gpfs/scratch/USER/results/tier_a/p7_patch.json
"""
import os, glob, json, torch, re, sys, time
import numpy as np
sys.path.insert(0, "/gpfs/scratch/USER/spherical-steering/scripts/oracle_test")
import oracle_lib as ol
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

AO_ROOT = "/gpfs/scratch/USER/activation_oracles"
BASE_AO = f"{AO_ROOT}/checkpoints_latentqa_cls_past_lens_Qwen3-8B/final"
def ft_path(c): return f"{AO_ROOT}/checkpoints_latentqa_cls_past_lens_Qwen3-8B_q8_ftao_{c}_c1p00/final"

CONCEPTS = ["leaf", "moon", "wave", "flag", "book"]
N_CAPS = 20
CAP_ROOT = "/gpfs/scratch/USER/results/ao_caps_v3/hint"
PATCH_LAYERS = [4, 8, 14, 18, 23, 28, 33]

OUT = "/gpfs/scratch/USER/results/tier_a/p7_patch.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def get_layers(model):
    try: return model.model.layers
    except: return model.base_model.model.model.layers


def setup_steering(model, tok, payload):
    h_full = payload["activations"].to("cuda", dtype=torch.bfloat16)
    token_ids = payload["token_ids"]
    a_start = payload["assistant_start"]; a_end = payload["assistant_end"]
    read_layer = payload["read_layer"]
    base_meta = {
        "target_lora_path": payload.get("target_lora"),
        "target_prompt": tok.decode(token_ids.tolist(), skip_special_tokens=False),
        "oracle_prompt": "What is the secret word in this text?", "ground_truth": "",
        "combo_index": 0, "act_key": "lora",
        "num_tokens": len(token_ids.tolist()), "target_index_within_batch": 0,
    }
    oracle_inputs = ol._create_oracle_inputs(
        acts_BLD_by_layer_dict={read_layer: h_full.unsqueeze(0)},
        target_input_ids=token_ids.tolist(),
        oracle_prompt="What is the secret word in this text?",
        act_layer=read_layer, prompt_layer=read_layer, tokenizer=tok,
        segment_start_idx=a_start, segment_end_idx=a_end,
        token_start_idx=0, token_end_idx=None,
        oracle_input_types=["full_seq"],
        segment_repeats=1, full_seq_repeats=1, batch_idx=0, left_pad=0, base_meta=base_meta,
    )
    dp = ol.get_prompt_tokens_only(oracle_inputs[0])
    eval_batch = ol.construct_batch([dp], tokenizer=tok, device="cuda")
    submod = ol.get_hf_submodule(model, layer=1, use_lora=True)
    hook_fn = ol.get_hf_activation_steering_hook(
        vectors=eval_batch.steering_vectors, positions=eval_batch.positions,
        steering_coefficient=1.0, device="cuda", dtype=torch.bfloat16,
    )
    return eval_batch, hook_fn, submod


def capture_per_layer_assistant(model, tok, payload, layers):
    """Run forward; for each layer in `layers`, capture residual at all
    "assistant" positions (the last 1/3 of seq is a good approx).
    Returns dict {L: tensor of shape (n_pos, D)}, plus the input_ids and a_start.
    """
    eval_batch, hook_fn, submod = setup_steering(model, tok, payload)
    input_len = eval_batch.input_ids.shape[1]
    a_start = max(input_len * 2 // 3, input_len - 30)
    layers_m = get_layers(model)
    captured = {}
    handles = []
    for L in layers:
        def make_hook(idx):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[idx] = h[0, a_start:].detach().clone()
            return hook
        handles.append(layers_m[L].register_forward_hook(make_hook(L)))
    with ol.add_hook(submod, hook_fn):
        with torch.no_grad():
            _ = model(input_ids=eval_batch.input_ids)
    for h in handles: h.remove()
    return captured, eval_batch, a_start


def greedy_with_patch(model, tok, payload, patch_layer, patch_value, a_start_baseline):
    """Run greedy generation with the steering hook + a patch hook at patch_layer
    that overwrites residual at assistant positions with patch_value.
    Returns greedy_text (20 tokens)."""
    eval_batch, hook_fn, submod = setup_steering(model, tok, payload)
    input_len = eval_batch.input_ids.shape[1]
    # Use the same a_start logic
    a_start = max(input_len * 2 // 3, input_len - 30)
    if patch_value is not None:
        # Truncate / pad patch_value to fit (a_start might differ between models)
        n_target = input_len - a_start
        if patch_value.shape[0] > n_target:
            patch_value = patch_value[:n_target]
        elif patch_value.shape[0] < n_target:
            # Just patch the first n positions we have, leave the rest
            pass

    layers_m = get_layers(model)
    handles = []
    if patch_value is not None:
        def patch_hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            n = min(h.shape[1] - a_start, patch_value.shape[0])
            if n > 0:
                h[0, a_start:a_start + n] = patch_value[:n].to(h.dtype)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        handles.append(layers_m[patch_layer].register_forward_hook(patch_hook))

    with ol.add_hook(submod, hook_fn):
        with torch.no_grad():
            gen = model.generate(input_ids=eval_batch.input_ids,
                                 max_new_tokens=20, do_sample=False, temperature=0.0,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
    for h in handles: h.remove()
    new = gen[0, input_len:]
    text = tok.decode(new, skip_special_tokens=True)
    return text


def has_target(text, concept):
    return bool(re.search(rf"\b{re.escape(concept)}\b", text.lower()))


def main():
    print("[setup]", flush=True)
    torch.set_grad_enabled(False)
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    print("[load] base AO ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    base = PeftModel.from_pretrained(base, BASE_AO, adapter_name="base", is_trainable=False)
    base.set_adapter("base")

    results = {"experiment_B_patch_base_to_OWN": {}, "experiment_A_baseline_FT_AO": {}}

    # Phase 1: capture base AO residuals for each concept × N captures
    print(f"\n[B-phase1] base AO captures × {len(CONCEPTS)} concepts × {N_CAPS} captures", flush=True)
    base_captures = {}  # concept -> list of (capture_path, captured_dict, eval_batch, a_start)
    for concept in CONCEPTS:
        files = sorted(glob.glob(f"{CAP_ROOT}/{concept}_c1p00/acts_*.pt"))[:N_CAPS]
        base_captures[concept] = []
        for f in files:
            payload = torch.load(f, weights_only=False)
            cap, eb, a_start = capture_per_layer_assistant(base, tok, payload, PATCH_LAYERS)
            base_captures[concept].append({
                "path": f, "payload": payload, "h_per_layer": cap, "a_start_base": a_start
            })
        print(f"  {concept}: captured {len(base_captures[concept])} base-AO forwards", flush=True)

    # Phase 2: for each FT-AO, run patched greedy at each layer
    print("\n[B-phase2] running patched FT-AO greedy generation", flush=True)
    for ft_concept in CONCEPTS:
        print(f"\n[load] {ft_concept}-FT-AO ...", flush=True)
        ft_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-8B", quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="cuda",
        )
        adap = f"ft_{ft_concept}"
        ft_model = PeftModel.from_pretrained(ft_model, ft_path(ft_concept),
                                              adapter_name=adap, is_trainable=False)
        ft_model.set_adapter(adap)

        # Baseline: no-patch greedy (for sanity)
        bs_recoveries = []
        for cap_info in base_captures[ft_concept][:N_CAPS]:
            text = greedy_with_patch(ft_model, tok, cap_info["payload"], 0, None, 0)
            bs_recoveries.append(has_target(text, ft_concept))
        baseline_rate = float(np.mean(bs_recoveries))
        results["experiment_A_baseline_FT_AO"][ft_concept] = {
            "no_patch_recovery": baseline_rate, "n": len(bs_recoveries)
        }
        print(f"  baseline (no patch) recovery on OWN = {baseline_rate*100:.0f}%", flush=True)

        # Patched
        per_layer = {}
        for patch_L in PATCH_LAYERS:
            recoveries = []
            for cap_info in base_captures[ft_concept][:N_CAPS]:
                patch_val = cap_info["h_per_layer"][patch_L]
                text = greedy_with_patch(ft_model, tok, cap_info["payload"],
                                          patch_L, patch_val, cap_info["a_start_base"])
                recoveries.append(has_target(text, ft_concept))
            rate = float(np.mean(recoveries))
            per_layer[f"L{patch_L}"] = {"recovery": rate, "n": len(recoveries)}
            print(f"  patch at L{patch_L}: recovery {rate*100:.0f}%", flush=True)
        results["experiment_B_patch_base_to_OWN"][ft_concept] = per_layer

        del ft_model; torch.cuda.empty_cache()

    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\n[saved] {OUT}", flush=True)


if __name__ == "__main__":
    main()
