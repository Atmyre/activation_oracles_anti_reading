"""Fill the last gaps in P5/P7.

Sub-experiments:
  A. P5_B_extend: per-layer LogitLens P(target) for moon-FT and wave-FT.
  B. P7_A_extend: layer-band LoRA ablation for moon-FT and wave-FT.
  C. P7_D: concept-direction subspace ablation.
  D. P7_E: denial-axis ablation for strict FT-AOs.

For C and D we use mean-difference directions computed online.
"""
import os, glob, json, torch, re, sys, time
import numpy as np
sys.path.insert(0, "<PATH_TO_SCRATCH>/spherical-steering/scripts/oracle_test")
import oracle_lib as ol
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

AO_ROOT = "<PATH_TO_SCRATCH>/activation_oracles"
BASE_AO = f"{AO_ROOT}/checkpoints_latentqa_cls_past_lens_Qwen3-8B/final"
def ft_path(c, cv="c1p00"): return f"{AO_ROOT}/checkpoints_latentqa_cls_past_lens_Qwen3-8B_q8_ftao_{c}_{cv}/final"
def strict_ft_path(c, cv="c1p00"): return f"{AO_ROOT}/checkpoints_latentqa_cls_past_lens_Qwen3-8B_q8_ftao_strict{c}v2_{cv}/final"

CONCEPTS = ["leaf", "moon", "wave", "flag", "book"]
CAP_ROOT = "<PATH_TO_SCRATCH>/results/ao_caps_v3/hint"
N_CAPS = 20
PATCH_LAYERS = [4, 8, 14, 18, 23, 28, 33]
ABL_BANDS = [(0,5),(6,11),(12,17),(18,23),(24,29),(30,35)]
OUT = "<PATH_TO_SCRATCH>/results/tier_a/p567_finishers.json"


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
    oi = ol._create_oracle_inputs(
        acts_BLD_by_layer_dict={read_layer: h_full.unsqueeze(0)},
        target_input_ids=token_ids.tolist(),
        oracle_prompt="What is the secret word in this text?",
        act_layer=read_layer, prompt_layer=read_layer, tokenizer=tok,
        segment_start_idx=a_start, segment_end_idx=a_end,
        token_start_idx=0, token_end_idx=None,
        oracle_input_types=["full_seq"],
        segment_repeats=1, full_seq_repeats=1, batch_idx=0, left_pad=0, base_meta=base_meta,
    )
    dp = ol.get_prompt_tokens_only(oi[0])
    eb = ol.construct_batch([dp], tokenizer=tok, device="cuda")
    submod = ol.get_hf_submodule(model, layer=1, use_lora=True)
    hook_fn = ol.get_hf_activation_steering_hook(
        vectors=eb.steering_vectors, positions=eb.positions,
        steering_coefficient=1.0, device="cuda", dtype=torch.bfloat16,
    )
    return eb, hook_fn, submod


def find_target_token_id(tok, concept):
    """Returns list of plausible target token ids (with and without leading space)."""
    ids = set()
    for v in [concept, " " + concept, concept.capitalize(), " " + concept.capitalize()]:
        toks = tok.encode(v, add_special_tokens=False)
        if len(toks) == 1: ids.add(toks[0])
    return list(ids)


def greedy_with_hooks(model, tok, payload, hook_fns_per_layer=None):
    """Run greedy with optional hooks on specified layers.
    hook_fns_per_layer: dict {layer_idx: hook_fn} that mutates `out` tensor.
    """
    eb, st_hook, submod = setup_steering(model, tok, payload)
    layers_m = get_layers(model)
    handles = []
    if hook_fns_per_layer:
        for L, fn in hook_fns_per_layer.items():
            handles.append(layers_m[L].register_forward_hook(fn))
    with ol.add_hook(submod, st_hook):
        with torch.no_grad():
            gen = model.generate(input_ids=eb.input_ids, max_new_tokens=20,
                                 do_sample=False, temperature=0.0,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
    for h in handles: h.remove()
    new = gen[0, eb.input_ids.shape[1]:]
    return tok.decode(new, skip_special_tokens=True)


def per_layer_logitlens(model, tok, payload, target_ids):
    """Run forward, capture residual at each layer, decode P(target) at last token via LM head."""
    eb, st_hook, submod = setup_steering(model, tok, payload)
    input_len = eb.input_ids.shape[1]
    layers_m = get_layers(model)
    layer_residuals = {}
    handles = []
    for L in range(36):
        def make(idx):
            def hk(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                layer_residuals[idx] = h[0, -1].detach().clone()
            return hk
        handles.append(layers_m[L].register_forward_hook(make(L)))
    with ol.add_hook(submod, st_hook):
        with torch.no_grad():
            _ = model(input_ids=eb.input_ids)
    for h in handles: h.remove()
    # Get LM head
    lm_head = model.get_output_embeddings()
    results = {}
    for L, h in layer_residuals.items():
        with torch.no_grad():
            logits = lm_head(h.to(lm_head.weight.dtype))
            probs = torch.softmax(logits.float(), dim=-1)
        p_target = sum(float(probs[t]) for t in target_ids)
        results[L] = p_target
    return results


def capture_mean_residual(model, tok, payload, layer):
    """Capture mean residual at the given layer over assistant positions."""
    eb, st_hook, submod = setup_steering(model, tok, payload)
    input_len = eb.input_ids.shape[1]
    a_start = max(input_len * 2 // 3, input_len - 30)
    layers_m = get_layers(model)
    captured = {}
    def hk(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h[0, a_start:].mean(0).detach().float().cpu()
    handle = layers_m[layer].register_forward_hook(hk)
    with ol.add_hook(submod, st_hook):
        with torch.no_grad():
            _ = model(input_ids=eb.input_ids)
    handle.remove()
    return captured["h"]


def has_target(text, concept):
    return bool(re.search(rf"\b{re.escape(concept)}\b", text.lower()))


def load_model(lora_path, adapter_name, bnb):
    m = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    m = PeftModel.from_pretrained(m, lora_path, adapter_name=adapter_name, is_trainable=False)
    m.set_adapter(adapter_name)
    return m


def load_captures(concept, n, cv="c1p00", regime="hint"):
    files = sorted(glob.glob(f"<PATH_TO_SCRATCH>/results/ao_caps_v3/{regime}/{concept}_{cv}/acts_*.pt"))[:n]
    return [torch.load(f, weights_only=False) for f in files]


def load_strict_captures(concept, n, cv="c1p00", regime="hint"):
    files = sorted(glob.glob(f"<PATH_TO_SCRATCH>/results/ao_caps_v3/{regime}/strict{concept}v2_{cv}/acts_*.pt"))[:n]
    return [torch.load(f, weights_only=False) for f in files]


def main():
    torch.set_grad_enabled(False)
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    results = {}

    # ===================================================================
    # A. P5_B_extend + P7_A_extend: moon-FT, wave-FT (LogitLens + layer ablation)
    # ===================================================================
    print("\n========== A. moon/wave LogitLens + layer ablation ==========", flush=True)
    print("[load base AO]", flush=True)
    base = load_model(BASE_AO, "b", bnb)

    p5b = {}
    p7a = {}
    for concept in ["moon", "wave"]:
        target_ids = find_target_token_id(tok, concept)
        if not target_ids:
            print(f"  [warn] no single-token id for {concept}, skip", flush=True); continue

        # Get base captures
        caps = load_captures(concept, N_CAPS)

        # Base AO LogitLens
        print(f"\n[{concept}] base AO LogitLens", flush=True)
        base_pl = []
        for cap in caps:
            r = per_layer_logitlens(base, tok, cap, target_ids)
            base_pl.append([r.get(L, 0) for L in range(36)])
        base_pl_mean = np.mean(base_pl, axis=0)

        # OWN-FT-AO
        print(f"[{concept}] loading {concept}-FT-AO", flush=True)
        ft = load_model(ft_path(concept), f"own_{concept}", bnb)

        own_pl = []
        own_recovery = []
        for cap in caps:
            r = per_layer_logitlens(ft, tok, cap, target_ids)
            own_pl.append([r.get(L, 0) for L in range(36)])
            text = greedy_with_hooks(ft, tok, cap, None)
            own_recovery.append(has_target(text, concept))
        own_pl_mean = np.mean(own_pl, axis=0)
        baseline_rec = float(np.mean(own_recovery))

        p5b[concept] = {
            "base_logitlens_per_layer": base_pl_mean.tolist(),
            "own_ft_logitlens_per_layer": own_pl_mean.tolist(),
            "n": N_CAPS,
        }
        print(f"  base L33: {base_pl_mean[33]*100:.1f}%  OWN L33: {own_pl_mean[33]*100:.1f}%", flush=True)

        # P7 A: layer ablation — zero LoRA in different bands
        print(f"\n[{concept}] layer ablation", flush=True)
        layers_m = get_layers(ft)
        # We ablate by patching residuals with base AO's residual at the target layers
        # (this is functionally a partial patch — but for P7-A we want LoRA ablation specifically.
        # Simpler: zero the layer's LoRA delta by setting active adapter to base AO on those layers.
        # We'll use a simpler approach: capture base AO residual at the band, patch each layer in the band.)
        # For each band, capture base AO residuals at ALL layers in band, then patch FT-AO at each
        # layer in band simultaneously.

        # Precapture base AO residuals at all bands
        all_layers_needed = set()
        for lo, hi in ABL_BANDS: all_layers_needed.update(range(lo, hi+1))
        all_layers_needed = sorted(all_layers_needed)

        base_resid_per_cap = []
        for cap in caps:
            r = {}
            eb, st_hook, submod = setup_steering(base, tok, cap)
            input_len = eb.input_ids.shape[1]
            a_start = max(input_len * 2 // 3, input_len - 30)
            layers_base = get_layers(base)
            captured = {}
            handles = []
            for L in all_layers_needed:
                def make(idx):
                    def hk(mod, inp, out):
                        h = out[0] if isinstance(out, tuple) else out
                        captured[idx] = h[0, a_start:].detach().clone()
                    return hk
                handles.append(layers_base[L].register_forward_hook(make(L)))
            with ol.add_hook(submod, st_hook):
                with torch.no_grad():
                    _ = base(input_ids=eb.input_ids)
            for h in handles: h.remove()
            base_resid_per_cap.append({"resid": captured, "a_start_base": a_start})

        # Now ablate each band: patch FT-AO at each layer in band with base AO's residual
        abl_results = {"none": baseline_rec}
        for lo, hi in ABL_BANDS:
            label = f"L{lo}-{hi}"
            rs = []
            for cap, brc in zip(caps, base_resid_per_cap):
                eb, st_hook, submod = setup_steering(ft, tok, cap)
                input_len = eb.input_ids.shape[1]
                a_start = max(input_len * 2 // 3, input_len - 30)
                handles = []
                for L in range(lo, hi+1):
                    if L not in brc["resid"]: continue
                    pv = brc["resid"][L]
                    def make_hook(patch_v):
                        def hk(mod, inp, out):
                            h = out[0] if isinstance(out, tuple) else out
                            n = min(h.shape[1] - a_start, patch_v.shape[0])
                            if n > 0:
                                h[0, a_start:a_start+n] = patch_v[:n].to(h.dtype)
                            return (h,) + out[1:] if isinstance(out, tuple) else h
                        return hk
                    handles.append(get_layers(ft)[L].register_forward_hook(make_hook(pv)))
                with ol.add_hook(submod, st_hook):
                    with torch.no_grad():
                        gen = ft.generate(input_ids=eb.input_ids, max_new_tokens=20,
                                          do_sample=False, temperature=0.0,
                                          pad_token_id=tok.pad_token_id or tok.eos_token_id)
                for h in handles: h.remove()
                new = gen[0, eb.input_ids.shape[1]:]
                text = tok.decode(new, skip_special_tokens=True)
                rs.append(has_target(text, concept))
            rate = float(np.mean(rs))
            abl_results[label] = rate
            print(f"  ablate {label}: recovery {rate*100:.0f}%", flush=True)

        p7a[concept] = {"baseline": baseline_rec, "ablation": abl_results, "n": N_CAPS}

        del ft; torch.cuda.empty_cache()

    results["P5_B_extend"] = p5b
    results["P7_A_extend"] = p7a

    # ===================================================================
    # C. P7_D — concept-direction subspace ablation
    # ===================================================================
    print("\n========== C. P7_D concept-direction ablation ==========", flush=True)
    # For each layer L in {14, 18, 23, 28, 33}:
    #   Compute concept mean residual on BASE AO captures: mean_c = mean over concept c subjects
    #   Concept direction = (mean_c - mean_overall) / ||...|| (unit vector)
    #   Project out at L during FT-AO forward; measure recovery
    P7D_LAYERS = [14, 18, 23, 28]

    print("[capture] base AO mean residuals at P7D_LAYERS for each concept", flush=True)
    concept_means = {}  # concept -> {layer: vector}
    overall_count = 0
    layer_sums = {L: 0 for L in P7D_LAYERS}
    for c in CONCEPTS:
        caps = load_captures(c, N_CAPS)
        m = {L: [] for L in P7D_LAYERS}
        for cap in caps:
            for L in P7D_LAYERS:
                h = capture_mean_residual(base, tok, cap, L)
                m[L].append(h.numpy())
                layer_sums[L] = layer_sums[L] + h.numpy() if not isinstance(layer_sums[L], int) else h.numpy()
                overall_count += 1
        concept_means[c] = {L: np.mean(m[L], axis=0) for L in P7D_LAYERS}
        print(f"  {c}: captured at {len(P7D_LAYERS)} layers", flush=True)
    overall_count = overall_count // len(P7D_LAYERS)
    layer_means = {L: layer_sums[L] / overall_count for L in P7D_LAYERS}

    # Compute concept directions
    concept_dirs = {}  # concept -> {layer: unit vector}
    for c in CONCEPTS:
        concept_dirs[c] = {}
        for L in P7D_LAYERS:
            v = concept_means[c][L] - layer_means[L]
            norm = float(np.linalg.norm(v))
            concept_dirs[c][L] = v / norm if norm > 0 else v

    # Random control direction
    rng = np.random.default_rng(0)
    random_dirs = {L: rng.standard_normal(layer_means[L].shape[0]) for L in P7D_LAYERS}
    for L in random_dirs:
        random_dirs[L] = random_dirs[L] / np.linalg.norm(random_dirs[L])

    # Run ablation on OWN-FT-AO
    p7d = {}
    for concept in CONCEPTS:
        print(f"\n[{concept}-FT-AO]: P7_D subspace ablation", flush=True)
        ft = load_model(ft_path(concept), f"own_d_{concept}", bnb)
        caps = load_captures(concept, N_CAPS)
        own_dir = concept_dirs[concept]  # the concept's own direction
        rand_dir = random_dirs

        per_layer = {}
        # baseline
        bs = []
        for cap in caps:
            text = greedy_with_hooks(ft, tok, cap, None)
            bs.append(has_target(text, concept))
        baseline = float(np.mean(bs))
        print(f"  baseline: {baseline*100:.0f}%", flush=True)

        for L in P7D_LAYERS:
            # Ablate concept direction at L
            d = torch.tensor(own_dir[L], dtype=torch.bfloat16, device="cuda")
            r_d = torch.tensor(rand_dir[L], dtype=torch.bfloat16, device="cuda")
            def make_proj_hook(direction):
                def hk(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    # h shape (1, seq, D)
                    # Project out direction (which is unit)
                    proj_coef = (h.float() @ direction.float())  # shape (1, seq)
                    h_new = h - (proj_coef.unsqueeze(-1) * direction.unsqueeze(0).unsqueeze(0)).to(h.dtype)
                    return (h_new,) + out[1:] if isinstance(out, tuple) else h_new
                return hk

            for kind, dir_t in [("own_concept", d), ("random", r_d)]:
                rs = []
                for cap in caps:
                    text = greedy_with_hooks(ft, tok, cap, {L: make_proj_hook(dir_t)})
                    rs.append(has_target(text, concept))
                rate = float(np.mean(rs))
                per_layer.setdefault(L, {})[kind] = rate
                print(f"  L{L} ablate {kind}: {rate*100:.0f}%", flush=True)

        p7d[concept] = {"baseline": baseline, "per_layer": per_layer, "n": N_CAPS}
        del ft; torch.cuda.empty_cache()

    results["P7_D"] = p7d

    # ===================================================================
    # D. P7_E — denial-axis ablation for strict FT-AOs
    # ===================================================================
    print("\n========== D. P7_E denial-axis ablation (strict) ==========", flush=True)
    # Compute denial axis = mean(strict captures) - mean(coop captures), at L18 base AO
    print("[capture] base AO residuals on strict captures at L18", flush=True)
    strict_resid_L18 = []
    coop_resid_L18 = []
    for c in CONCEPTS:
        coop_caps = load_captures(c, 15)
        strict_caps = load_strict_captures(c, 15)
        for cap in coop_caps:
            h = capture_mean_residual(base, tok, cap, 18)
            coop_resid_L18.append(h.numpy())
        for cap in strict_caps:
            h = capture_mean_residual(base, tok, cap, 18)
            strict_resid_L18.append(h.numpy())

    if len(strict_resid_L18) and len(coop_resid_L18):
        denial_axis = np.mean(strict_resid_L18, axis=0) - np.mean(coop_resid_L18, axis=0)
        denial_axis = denial_axis / np.linalg.norm(denial_axis)
    else:
        denial_axis = None

    p7e = {}
    if denial_axis is not None:
        for concept in CONCEPTS:
            # strict FT-AOs
            sp = strict_ft_path(concept)
            if not os.path.exists(sp):
                print(f"  [{concept}] no strict FT-AO, skip", flush=True); continue
            print(f"\n[{concept}-strict-FT-AO]: P7_E denial ablation", flush=True)
            try:
                ft = load_model(sp, f"strict_e_{concept}", bnb)
            except Exception as e:
                print(f"  [warn] load failed: {e}", flush=True); continue
            caps = load_strict_captures(concept, N_CAPS)
            if not caps: del ft; torch.cuda.empty_cache(); continue

            d = torch.tensor(denial_axis, dtype=torch.bfloat16, device="cuda")
            # baseline
            bs = []
            for cap in caps:
                text = greedy_with_hooks(ft, tok, cap, None)
                bs.append(has_target(text, concept))
            baseline = float(np.mean(bs))
            print(f"  baseline (no ablate): {baseline*100:.0f}%", flush=True)

            # ablate denial axis at L18
            def make_proj(direction):
                def hk(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    proj_coef = (h.float() @ direction.float())
                    h_new = h - (proj_coef.unsqueeze(-1) * direction.unsqueeze(0).unsqueeze(0)).to(h.dtype)
                    return (h_new,) + out[1:] if isinstance(out, tuple) else h_new
                return hk

            rs = []
            for cap in caps:
                text = greedy_with_hooks(ft, tok, cap, {18: make_proj(d)})
                rs.append(has_target(text, concept))
            denial_ablated = float(np.mean(rs))
            print(f"  ablate denial axis @L18: {denial_ablated*100:.0f}%", flush=True)

            p7e[concept] = {"baseline": baseline, "denial_axis_ablated": denial_ablated, "n": N_CAPS}
            del ft; torch.cuda.empty_cache()

    results["P7_E"] = p7e

    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\n[saved] {OUT}", flush=True)


if __name__ == "__main__":
    main()
