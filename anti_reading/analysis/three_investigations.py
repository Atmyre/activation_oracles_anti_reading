"""Three follow-up investigations:

  A. Debug P5_B moon/wave LogitLens (find why base P(target) is 0%)
  B. LogitLens-decode the concept direction from P7_D — does it contain concept content?
  C. Strict mechanism disambiguation:
     C1. LogitLens on strict FT-AO OWN — is target latent at any layer?
     C2. Patch strict FT-AO residual → base AO to see if base recovers (residual-quality test)
"""
import os, glob, json, torch, re, sys, time
import numpy as np
sys.path.insert(0, "/gpfs/scratch/USER/spherical-steering/scripts/oracle_test")
import oracle_lib as ol
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

AO_ROOT = "/gpfs/scratch/USER/activation_oracles"
BASE_AO = f"{AO_ROOT}/checkpoints_latentqa_cls_past_lens_Qwen3-8B/final"
def coop_ft(c): return f"{AO_ROOT}/checkpoints_latentqa_cls_past_lens_Qwen3-8B_q8_ftao_{c}_c1p00/final"
def strict_ft(c): return f"{AO_ROOT}/checkpoints_latentqa_cls_past_lens_Qwen3-8B_q8_ftao_strict{c}v2_c1p00/final"

CONCEPTS = ["leaf", "moon", "wave", "flag", "book"]
OUT = "/gpfs/scratch/USER/results/tier_a/three_investigations.json"


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
    hk = ol.get_hf_activation_steering_hook(
        vectors=eb.steering_vectors, positions=eb.positions,
        steering_coefficient=1.0, device="cuda", dtype=torch.bfloat16,
    )
    return eb, hk, submod


def load_ft(lora_path, name, bnb):
    m = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    m = PeftModel.from_pretrained(m, lora_path, adapter_name=name, is_trainable=False)
    m.set_adapter(name)
    return m


def find_target_ids(tok, concept):
    """More aggressive: cover many variants including plural."""
    ids = set()
    for v in [concept, " " + concept, concept.capitalize(), " " + concept.capitalize(),
              concept + "s", " " + concept + "s", concept.upper(), " " + concept.upper()]:
        t = tok.encode(v, add_special_tokens=False)
        if len(t) == 1: ids.add(t[0])
        elif len(t) >= 1: ids.add(t[0])  # first token even if multi-token
    return sorted(ids)


def load_caps(concept, n, cv="c1p00", regime="hint", protocol="coop"):
    subj = f"{concept}_{cv}" if protocol == "coop" else f"strict{concept}v2_{cv}"
    files = sorted(glob.glob(f"/gpfs/scratch/USER/results/ao_caps_v3/{regime}/{subj}/acts_*.pt"))[:n]
    return [torch.load(f, weights_only=False) for f in files]


def logitlens_at_layers(model, tok, payload, target_ids, layers=None, position="word"):
    """Per-layer LM-head decoding of P(target).
    position: 'word' = last input token; 'a_end' = assistant_end position; 'a_start' = assistant_start.
    """
    eb, hk, sm = setup_steering(model, tok, payload)
    input_len = eb.input_ids.shape[1]
    a_start = payload.get("assistant_start")
    a_end = payload.get("assistant_end")
    if position == "word": pos = input_len - 1
    elif position == "a_end": pos = min(a_end, input_len - 1) if a_end is not None else input_len - 1
    else: pos = a_start if a_start is not None else 0
    layers_m = get_layers(model)
    resid = {}
    handles = []
    ll = layers or range(36)
    for L in ll:
        def make(idx):
            def h(mod, inp, out):
                hh = out[0] if isinstance(out, tuple) else out
                resid[idx] = hh[0, pos].detach().clone()
            return h
        handles.append(layers_m[L].register_forward_hook(make(L)))
    with ol.add_hook(sm, hk):
        with torch.no_grad(): _ = model(input_ids=eb.input_ids)
    for h in handles: h.remove()
    lm = model.get_output_embeddings()
    out = {}
    for L, r in resid.items():
        with torch.no_grad():
            logits = lm(r.to(lm.weight.dtype))
            probs = torch.softmax(logits.float(), dim=-1)
        p_target = float(sum(float(probs[t]) for t in target_ids))
        # also get top predicted token
        top_id = int(logits.argmax().item())
        top_word = tok.decode([top_id]).strip()
        out[L] = {"p_target": p_target, "top_word": top_word}
    return out


def mean_residual_at_layer(model, tok, payload, L):
    """Mean residual over assistant positions."""
    eb, hk, sm = setup_steering(model, tok, payload)
    input_len = eb.input_ids.shape[1]
    a_start = max(input_len * 2 // 3, input_len - 30)
    layers_m = get_layers(model)
    captured = {}
    def hook(mod, inp, out):
        hh = out[0] if isinstance(out, tuple) else out
        captured["h"] = hh[0, a_start:].mean(0).detach().float().cpu()
    handle = layers_m[L].register_forward_hook(hook)
    with ol.add_hook(sm, hk):
        with torch.no_grad(): _ = model(input_ids=eb.input_ids)
    handle.remove()
    return captured["h"]


def main():
    torch.set_grad_enabled(False)
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    print("[load base AO]", flush=True)
    base = load_ft(BASE_AO, "b", bnb)

    results = {}

    # =====================================================
    # A. Debug P5_B moon/wave LogitLens
    # =====================================================
    print("\n=== A. Debug P5_B moon/wave LogitLens ===", flush=True)
    A = {}
    for concept in ["moon", "wave"]:
        tids = find_target_ids(tok, concept)
        print(f"  {concept} target ids: {tids}", flush=True)
        print(f"  {concept} decoded: {[tok.decode([t]) for t in tids]}", flush=True)
        caps = load_caps(concept, 5)
        A[concept] = {"target_ids": tids, "captures": []}
        for cap in caps[:3]:
            for pos in ["word", "a_end", "a_start"]:
                r = logitlens_at_layers(base, tok, cap, tids,
                                        layers=[8, 18, 24, 30, 33], position=pos)
                pl = {f"L{L}": r[L]["p_target"] * 100 for L in r}
                # base P(target) at L33 for this capture
                A[concept]["captures"].append({
                    "pos": pos,
                    "L33_p_target": r[33]["p_target"] * 100 if 33 in r else None,
                    "L33_top_word": r[33]["top_word"] if 33 in r else None,
                    "L18_p_target": r[18]["p_target"] * 100 if 18 in r else None,
                    "L18_top_word": r[18]["top_word"] if 18 in r else None,
                })
        print(f"    sample L33 P for {concept} (word pos): "
              f"{[c['L33_p_target'] for c in A[concept]['captures'] if c['pos']=='word']}", flush=True)
        print(f"    sample L33 top word: "
              f"{[c['L33_top_word'] for c in A[concept]['captures'] if c['pos']=='word']}", flush=True)

    results["A_p5b_debug"] = A

    # =====================================================
    # B. LogitLens-decode the concept direction from P7_D
    # =====================================================
    print("\n=== B. LogitLens on concept direction ===", flush=True)
    B = {}
    LAYERS_B = [14, 18, 24, 33]
    lm = base.get_output_embeddings()

    # For each concept, compute the concept direction = mean(this concept) - mean(other concepts)
    # at each layer, then decode via LM-head.
    print("  Capturing base AO mean residuals per concept ...", flush=True)
    concept_layers_means = {}  # concept -> layer -> vec
    all_captures_L = {L: [] for L in LAYERS_B}
    for c in CONCEPTS:
        concept_layers_means[c] = {}
        caps = load_caps(c, 10)
        cap_means = {L: [] for L in LAYERS_B}
        for cap in caps:
            for L in LAYERS_B:
                h = mean_residual_at_layer(base, tok, cap, L)
                cap_means[L].append(h.numpy())
                all_captures_L[L].append(h.numpy())
        for L in LAYERS_B:
            concept_layers_means[c][L] = np.mean(cap_means[L], axis=0)
        print(f"    {c}: done", flush=True)

    # Overall mean at each layer
    overall_means = {L: np.mean(all_captures_L[L], axis=0) for L in LAYERS_B}

    for c in CONCEPTS:
        B[c] = {}
        for L in LAYERS_B:
            direction = concept_layers_means[c][L] - overall_means[L]
            norm = float(np.linalg.norm(direction))
            unit_dir = direction / norm if norm > 0 else direction
            # LM-head decode top-10 vocabulary tokens for this direction
            with torch.no_grad():
                d_t = torch.tensor(direction, dtype=lm.weight.dtype, device="cuda")
                logits = lm(d_t)
                topk = logits.topk(10)
            top_words = [tok.decode([int(i)]).strip() for i in topk.indices.tolist()]
            B[c][f"L{L}"] = {"top_words": top_words, "direction_norm": norm,
                              "contains_concept": (c in " ".join(top_words).lower())}
            print(f"  {c} L{L}: {top_words}   contains-concept={B[c][f'L{L}']['contains_concept']}", flush=True)

    results["B_p7d_direction_logitlens"] = B

    # =====================================================
    # C. Strict mechanism disambiguation
    # =====================================================
    print("\n=== C. Strict mechanism disambiguation ===", flush=True)
    C = {}

    # C1: For each strict FT-AO, LogitLens at all layers on OWN concept subjects
    print("\n  C1: LogitLens on strict FT-AOs (OWN concept)", flush=True)
    C1 = {}
    for c in CONCEPTS:
        tids = find_target_ids(tok, c)
        try:
            ft = load_ft(strict_ft(c), f"sft_{c}", bnb)
        except Exception as e:
            print(f"  {c}: load failed {e}", flush=True); continue
        caps = load_caps(c, 10, protocol="strict")
        pl_all = []
        for cap in caps:
            r = logitlens_at_layers(ft, tok, cap, tids, layers=[8, 14, 18, 24, 28, 30, 33, 35], position="word")
            pl_all.append({L: r[L]["p_target"] * 100 for L in r})
        # Aggregate
        C1[c] = {L: float(np.mean([p[L] for p in pl_all])) for L in [8, 14, 18, 24, 28, 30, 33, 35]}
        print(f"  {c}: " + "  ".join(f"L{L}={C1[c][L]:.1f}%" for L in [18, 24, 30, 33, 35]), flush=True)
        del ft; torch.cuda.empty_cache()
    C["C1_strict_logitlens_own"] = C1

    # C2: Patch strict FT-AO residuals → base AO
    print("\n  C2: Patch strict FT-AO residual → base AO", flush=True)
    C2 = {}
    for c in CONCEPTS:
        try:
            ft = load_ft(strict_ft(c), f"sft2_{c}", bnb)
        except Exception as e:
            continue
        caps = load_caps(c, 15, protocol="strict")

        # Capture strict FT-AO residuals at L18, L24, L28, L33
        strict_resids_per_cap = []
        for cap in caps:
            eb, hk, sm = setup_steering(ft, tok, cap)
            input_len = eb.input_ids.shape[1]
            a_start = max(input_len * 2 // 3, input_len - 30)
            layers_m = get_layers(ft)
            captured = {}
            handles = []
            for L in [18, 24, 28, 33]:
                def make(idx):
                    def h(mod, inp, out):
                        hh = out[0] if isinstance(out, tuple) else out
                        captured[idx] = hh[0, a_start:].detach().clone()
                    return h
                handles.append(layers_m[L].register_forward_hook(make(L)))
            with ol.add_hook(sm, hk):
                with torch.no_grad(): _ = ft(input_ids=eb.input_ids)
            for h in handles: h.remove()
            strict_resids_per_cap.append({"resid": captured, "a_start": a_start})

        # Now run BASE AO with the strict residuals patched in
        base_baseline = []
        for cap in caps:
            eb, hk, sm = setup_steering(base, tok, cap)
            with ol.add_hook(sm, hk):
                with torch.no_grad():
                    gen = base.generate(input_ids=eb.input_ids, max_new_tokens=20,
                                        do_sample=False, temperature=0.0,
                                        pad_token_id=tok.pad_token_id or tok.eos_token_id)
            new = gen[0, eb.input_ids.shape[1]:]
            text = tok.decode(new, skip_special_tokens=True)
            base_baseline.append(bool(re.search(rf"\b{re.escape(c)}\b", text.lower())))
        base_baseline_rate = float(np.mean(base_baseline))

        # patched base with strict resid at each L
        per_L = {}
        for L in [18, 24, 28, 33]:
            rs = []
            for cap, srp in zip(caps, strict_resids_per_cap):
                if L not in srp["resid"]: continue
                pv = srp["resid"][L]
                eb, hk, sm = setup_steering(base, tok, cap)
                input_len = eb.input_ids.shape[1]
                a_start_base = max(input_len * 2 // 3, input_len - 30)
                def patch_hook(mod, inp, out):
                    hh = out[0] if isinstance(out, tuple) else out
                    n = min(hh.shape[1] - a_start_base, pv.shape[0])
                    if n > 0:
                        hh[0, a_start_base:a_start_base+n] = pv[:n].to(hh.dtype)
                    return (hh,) + out[1:] if isinstance(out, tuple) else hh
                handle = get_layers(base)[L].register_forward_hook(patch_hook)
                with ol.add_hook(sm, hk):
                    with torch.no_grad():
                        gen = base.generate(input_ids=eb.input_ids, max_new_tokens=20,
                                            do_sample=False, temperature=0.0,
                                            pad_token_id=tok.pad_token_id or tok.eos_token_id)
                handle.remove()
                new = gen[0, eb.input_ids.shape[1]:]
                text = tok.decode(new, skip_special_tokens=True)
                rs.append(bool(re.search(rf"\b{re.escape(c)}\b", text.lower())))
            per_L[f"L{L}"] = float(np.mean(rs)) if rs else 0.0

        C2[c] = {"base_baseline": base_baseline_rate, "patched_from_strict_at": per_L}
        print(f"  {c}: base_baseline={base_baseline_rate*100:.0f}%  patch_L18={per_L.get('L18',0)*100:.0f}%  L24={per_L.get('L24',0)*100:.0f}%  L33={per_L.get('L33',0)*100:.0f}%", flush=True)
        del ft; torch.cuda.empty_cache()
    C["C2_strict_resid_into_base"] = C2

    results["C_strict_mechanism"] = C

    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\n[saved] {OUT}", flush=True)


if __name__ == "__main__":
    main()
