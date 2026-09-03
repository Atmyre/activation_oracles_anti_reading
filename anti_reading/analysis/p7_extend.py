"""P7 extension: patch experiments at full coverage.

  B_full: base AO → OWN FT-AO patch, 4 regimes × 2 c-knobs.
  C: CROSS FT-AO → OWN FT-AO patch (using the same x-FT residual at L).

For each (regime × c × FT-AO):
  - Capture base AO residuals per layer (under same injection)
  - Capture CROSS-FT-AO residuals per layer
  - Run OWN FT-AO with patch hook at each layer L, measure target recovery via greedy.

Layers: L4 / L8 / L14 / L18 / L23 / L28 / L33 (same as §4.5.7)
Concepts: leaf, moon, wave, flag, book.
Captures: 15 per cell (smaller than 20 to keep time manageable for the larger grid).
"""
import os, glob, json, torch, re, sys, time
import numpy as np
sys.path.insert(0, "/gpfs/scratch/USER/spherical-steering/scripts/oracle_test")
import oracle_lib as ol
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

AO_ROOT = "/gpfs/scratch/USER/activation_oracles"
BASE_AO = f"{AO_ROOT}/checkpoints_latentqa_cls_past_lens_Qwen3-8B/final"
def ft_path(c, cv): return f"{AO_ROOT}/checkpoints_latentqa_cls_past_lens_Qwen3-8B_q8_ftao_{c}_{cv}/final"

CONCEPTS = ["leaf", "moon", "wave", "flag", "book"]
N_CAPS = 15
PATCH_LAYERS = [4, 8, 14, 18, 23, 28, 33]
REGIMES = ["hint", "refusal", "sametext", "think"]
OUT = "/gpfs/scratch/USER/results/tier_a/p7_extend.json"


def cross_concept(c):
    i = CONCEPTS.index(c)
    return CONCEPTS[(i + 1) % len(CONCEPTS)]


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


def capture_layers(model, tok, payload, layers):
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
    return captured, a_start, eval_batch.input_ids.shape[1]


def greedy_with_patch(model, tok, payload, patch_layer, patch_value):
    eval_batch, hook_fn, submod = setup_steering(model, tok, payload)
    input_len = eval_batch.input_ids.shape[1]
    a_start = max(input_len * 2 // 3, input_len - 30)
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
    return tok.decode(new, skip_special_tokens=True)


def has_target(text, concept):
    return bool(re.search(rf"\b{re.escape(concept)}\b", text.lower()))


def cap_dir(regime, concept, cv): return f"/gpfs/scratch/USER/results/ao_caps_v3/{regime}/{concept}_{cv}"


def load_model(lora_path, adapter_name, bnb):
    m = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    m = PeftModel.from_pretrained(m, lora_path, adapter_name=adapter_name, is_trainable=False)
    m.set_adapter(adapter_name)
    return m


def main():
    torch.set_grad_enabled(False)
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    results = {"B_base_to_own": {}, "C_cross_to_own": {}}
    # We load base AO ONCE; reuse for B. For C, we need to capture CROSS-FT-AO residuals,
    # which means swapping in different FT-AOs. Strategy: per (cv) we load all 5 FT-AOs
    # one at a time in a loop to capture cross residuals for everyone, then reload the
    # OWN-FT-AO at evaluation time.

    print("[load base AO]", flush=True)
    base = load_model(BASE_AO, "base", bnb)

    for cv in ["c1p00", "c0p50"]:
        print(f"\n========== c-knob: {cv} ==========", flush=True)
        results["B_base_to_own"][cv] = {}
        results["C_cross_to_own"][cv] = {}

        # === B: Phase 1 — capture base AO residuals for each (regime, concept) ===
        print(f"\n[B-phase1 / cv={cv}] base AO captures", flush=True)
        base_caps = {}  # (regime, concept) -> list of {payload, h_per_layer}
        for regime in REGIMES:
            for c in CONCEPTS:
                files = sorted(glob.glob(f"{cap_dir(regime, c, cv)}/acts_*.pt"))[:N_CAPS]
                if not files:
                    print(f"  [warn] no captures for {regime}/{c}_{cv}", flush=True); continue
                lst = []
                for f in files:
                    payload = torch.load(f, weights_only=False)
                    h_per, _, _ = capture_layers(base, tok, payload, PATCH_LAYERS)
                    lst.append({"payload": payload, "h_per_layer": h_per})
                base_caps[(regime, c)] = lst
            print(f"  {regime}: {len(CONCEPTS)} concepts captured", flush=True)

        # === C: Phase 1' — capture each FT-AO's residuals for use as CROSS donor ===
        # For each cross-from concept x, capture x-FT-AO residuals on all (regime, target-concept) cells
        print(f"\n[C-phase1 / cv={cv}] FT-AO residuals (for cross-donor use)", flush=True)
        cross_caps = {}  # (x_concept, regime, target_concept) -> list of {payload, h_per_layer}
        for ft_c in CONCEPTS:
            print(f"  loading {ft_c}-FT-AO (donor) ...", flush=True)
            ft = load_model(ft_path(ft_c, cv), f"ft_{ft_c}_{cv}", bnb)
            for regime in REGIMES:
                for target_c in CONCEPTS:
                    if (regime, target_c) not in base_caps: continue
                    cross_caps[(ft_c, regime, target_c)] = []
                    for cap in base_caps[(regime, target_c)]:
                        h_per, _, _ = capture_layers(ft, tok, cap["payload"], PATCH_LAYERS)
                        cross_caps[(ft_c, regime, target_c)].append({"h_per_layer": h_per})
            del ft; torch.cuda.empty_cache()
            print(f"    {ft_c}: done", flush=True)

        # === B + C: Phase 2 — patched evaluations on OWN FT-AO ===
        print(f"\n[phase2 / cv={cv}] patched eval on each OWN-FT-AO", flush=True)
        for ft_c in CONCEPTS:
            print(f"\n  -- OWN = {ft_c}-FT-AO --", flush=True)
            ft = load_model(ft_path(ft_c, cv), f"own_{ft_c}_{cv}", bnb)

            for regime in REGIMES:
                if (regime, ft_c) not in base_caps:
                    print(f"    {regime}: no captures, skip", flush=True); continue
                caps = base_caps[(regime, ft_c)]
                # Baseline (no patch)
                bs = []
                for cap in caps:
                    text = greedy_with_patch(ft, tok, cap["payload"], 0, None)
                    bs.append(has_target(text, ft_c))
                baseline = float(np.mean(bs))

                # B: patch base AO residual
                B_results = {}
                for L in PATCH_LAYERS:
                    rs = []
                    for cap in caps:
                        text = greedy_with_patch(ft, tok, cap["payload"], L, cap["h_per_layer"][L])
                        rs.append(has_target(text, ft_c))
                    B_results[f"L{L}"] = float(np.mean(rs))

                # C: patch CROSS-FT-AO residual (use cross_concept(ft_c))
                x = cross_concept(ft_c)
                cross_donor = cross_caps.get((x, regime, ft_c), [])
                C_results = {}
                if len(cross_donor) == len(caps):
                    for L in PATCH_LAYERS:
                        rs = []
                        for cap, donor in zip(caps, cross_donor):
                            text = greedy_with_patch(ft, tok, cap["payload"], L, donor["h_per_layer"][L])
                            rs.append(has_target(text, ft_c))
                        C_results[f"L{L}"] = float(np.mean(rs))

                results["B_base_to_own"][cv].setdefault(ft_c, {})[regime] = {
                    "baseline": baseline, "patch": B_results, "n": len(caps)}
                results["C_cross_to_own"][cv].setdefault(ft_c, {})[regime] = {
                    "baseline": baseline, "cross_from": x, "patch": C_results, "n": len(caps)}

                bs_str = f"baseline={baseline*100:.0f}%"
                bm = max(B_results.values()) if B_results else 0
                cm = max(C_results.values()) if C_results else 0
                print(f"    {regime}: {bs_str}, B best={bm*100:.0f}%, C best={cm*100:.0f}%", flush=True)

            del ft; torch.cuda.empty_cache()

    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\n[saved] {OUT}", flush=True)


if __name__ == "__main__":
    main()
