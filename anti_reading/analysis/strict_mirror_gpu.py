"""Strict-protocol mirror — GPU experiments (mechanism + patch).

Mirrors:
  §2.1 internal probe on strict FT-AOs (all 5)
  §2.2 LogitLens on strict FT-AOs (all 5)
  §2.3 layer ablation on strict FT-AOs (all 5)
  §4.5.7 patch (base → strict FT-AO) at each layer
  Δh geometry: cos(strict Δh, coop concept direction) at each layer
"""
import os, glob, json, torch, re, sys, time
import numpy as np
sys.path.insert(0, "<PATH_TO_SCRATCH>/spherical-steering/scripts/oracle_test")
import oracle_lib as ol
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

AO_ROOT = "<PATH_TO_SCRATCH>/activation_oracles"
BASE_AO = f"{AO_ROOT}/checkpoints_latentqa_cls_past_lens_Qwen3-8B/final"
def coop_ft(c, cv="c1p00"): return f"{AO_ROOT}/checkpoints_latentqa_cls_past_lens_Qwen3-8B_q8_ftao_{c}_{cv}/final"
def strict_ft(c, cv="c1p00"): return f"{AO_ROOT}/checkpoints_latentqa_cls_past_lens_Qwen3-8B_q8_ftao_strict{c}v2_{cv}/final"

CONCEPTS = ["leaf", "moon", "wave", "flag", "book"]
N_TRAIN_PROBE = 30
N_CAPS = 20
PROBE_LAYERS = [4, 8, 14, 18, 24, 30, 33]
PATCH_LAYERS = [4, 8, 14, 18, 23, 28, 33]
ABL_BANDS = [(0,5),(6,11),(12,17),(18,23),(24,29),(30,35)]
OUT = "<PATH_TO_SCRATCH>/results/tier_a/strict_mirror_gpu.json"


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
    hook = ol.get_hf_activation_steering_hook(
        vectors=eb.steering_vectors, positions=eb.positions,
        steering_coefficient=1.0, device="cuda", dtype=torch.bfloat16,
    )
    return eb, hook, submod


def find_target_ids(tok, concept):
    out = set()
    for v in [concept, " " + concept, concept.capitalize(), " " + concept.capitalize()]:
        t = tok.encode(v, add_special_tokens=False)
        if len(t) == 1: out.add(t[0])
    return list(out)


def per_layer_means(model, tok, payload, layers):
    eb, hook, submod = setup_steering(model, tok, payload)
    input_len = eb.input_ids.shape[1]
    a_start = max(input_len * 2 // 3, input_len - 30)
    layers_m = get_layers(model)
    captured = {}; handles = []
    for L in layers:
        def make(idx):
            def hk(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[idx] = h[0, a_start:].mean(0).detach().float().cpu()
            return hk
        handles.append(layers_m[L].register_forward_hook(make(L)))
    with ol.add_hook(submod, hook):
        with torch.no_grad(): _ = model(input_ids=eb.input_ids)
    for h in handles: h.remove()
    return captured


def per_layer_full_residual(model, tok, payload, layers):
    """Capture full residual tensor (over assistant positions) per layer."""
    eb, hook, submod = setup_steering(model, tok, payload)
    input_len = eb.input_ids.shape[1]
    a_start = max(input_len * 2 // 3, input_len - 30)
    layers_m = get_layers(model)
    captured = {}; handles = []
    for L in layers:
        def make(idx):
            def hk(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[idx] = h[0, a_start:].detach().clone()
            return hk
        handles.append(layers_m[L].register_forward_hook(make(L)))
    with ol.add_hook(submod, hook):
        with torch.no_grad(): _ = model(input_ids=eb.input_ids)
    for h in handles: h.remove()
    return captured, a_start


def logitlens_last_token(model, tok, payload, target_ids):
    """Per-layer LM-head decoding of P(target) at last input token."""
    eb, hook, submod = setup_steering(model, tok, payload)
    layers_m = get_layers(model)
    resid = {}; handles = []
    for L in range(36):
        def make(idx):
            def hk(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                resid[idx] = h[0, -1].detach().clone()
            return hk
        handles.append(layers_m[L].register_forward_hook(make(L)))
    with ol.add_hook(submod, hook):
        with torch.no_grad(): _ = model(input_ids=eb.input_ids)
    for h in handles: h.remove()
    lm = model.get_output_embeddings()
    out = {}
    for L, h in resid.items():
        with torch.no_grad():
            logits = lm(h.to(lm.weight.dtype))
            probs = torch.softmax(logits.float(), dim=-1)
        out[L] = float(sum(float(probs[t]) for t in target_ids))
    return out


def greedy_with_hooks(model, tok, payload, hook_fns=None):
    eb, hook, submod = setup_steering(model, tok, payload)
    layers_m = get_layers(model)
    handles = []
    if hook_fns:
        for L, fn in hook_fns.items():
            handles.append(layers_m[L].register_forward_hook(fn))
    with ol.add_hook(submod, hook):
        with torch.no_grad():
            gen = model.generate(input_ids=eb.input_ids, max_new_tokens=20,
                                 do_sample=False, temperature=0.0,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
    for h in handles: h.remove()
    return tok.decode(gen[0, eb.input_ids.shape[1]:], skip_special_tokens=True)


def has_target(text, concept):
    return bool(re.search(rf"\b{re.escape(concept)}\b", text.lower()))


def load_ft(lora_path, name, bnb):
    m = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    m = PeftModel.from_pretrained(m, lora_path, adapter_name=name, is_trainable=False)
    m.set_adapter(name)
    return m


def load_strict_caps(concept, n, cv="c1p00", regime="hint"):
    files = sorted(glob.glob(f"<PATH_TO_SCRATCH>/results/ao_caps_v3/{regime}/strict{concept}v2_{cv}/acts_*.pt"))[:n]
    return [torch.load(f, weights_only=False) for f in files]


def load_coop_caps(concept, n, cv="c1p00", regime="hint"):
    files = sorted(glob.glob(f"<PATH_TO_SCRATCH>/results/ao_caps_v3/{regime}/{concept}_{cv}/acts_*.pt"))[:n]
    return [torch.load(f, weights_only=False) for f in files]


def main():
    torch.set_grad_enabled(False)
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    print("[load base AO]", flush=True)
    base = load_ft(BASE_AO, "b", bnb)

    results = {}

    # =================================================================
    # STEP 1: Train probe on base AO + STRICT subjects (mirrors §2.1)
    # =================================================================
    print("\n[step1] train 5-way probe on base AO + strict HINT subjects", flush=True)
    Xtr = {L: [] for L in PROBE_LAYERS}; ytr = []
    for c in CONCEPTS:
        caps = load_strict_caps(c, N_TRAIN_PROBE)
        for cap in caps:
            m = per_layer_means(base, tok, cap, PROBE_LAYERS)
            for L in PROBE_LAYERS: Xtr[L].append(m[L].numpy())
            ytr.append(c)
        print(f"  {c}: {len(caps)} captures", flush=True)
    ytr = np.array(ytr)
    probes = {}
    for L in PROBE_LAYERS:
        Xa = np.stack(Xtr[L])
        sc = StandardScaler().fit(Xa)
        clf = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1).fit(sc.transform(Xa), ytr)
        probes[L] = (sc, clf)
        print(f"  L{L} train acc: {clf.score(sc.transform(Xa), ytr):.3f}", flush=True)

    # =================================================================
    # STEP 2: Apply probe to strict FT-AOs (all 5 × 5 concepts × 7 layers)
    # =================================================================
    print("\n[step2] internal probe on strict FT-AOs", flush=True)
    probe_results = []
    for ft_c in CONCEPTS:
        try:
            ft = load_ft(strict_ft(ft_c), f"sft_{ft_c}", bnb)
        except Exception as e:
            print(f"  {ft_c}: load failed {e}", flush=True); continue
        print(f"  {ft_c}-strict-FT-AO", flush=True)
        for c in CONCEPTS:
            caps = load_strict_caps(c, 15, regime="hint")
            per_L = {L: [] for L in PROBE_LAYERS}
            per_L_p = {L: [] for L in PROBE_LAYERS}
            for cap in caps:
                m = per_layer_means(ft, tok, cap, PROBE_LAYERS)
                for L in PROBE_LAYERS:
                    sc, clf = probes[L]
                    Xs = sc.transform(m[L].numpy().reshape(1, -1))
                    pred = clf.predict(Xs)[0]
                    probs_ = clf.predict_proba(Xs)[0]
                    cidx = {cc: i for i, cc in enumerate(clf.classes_)}.get(c)
                    per_L[L].append(int(pred == c))
                    per_L_p[L].append(float(probs_[cidx]) if cidx is not None else 0.0)
            row = {"ft_concept": ft_c, "injected_concept": c, "n": len(caps)}
            for L in PROBE_LAYERS:
                row[f"L{L}_acc"] = float(np.mean(per_L[L])) if per_L[L] else None
            probe_results.append(row)
            mark = " ← OWN" if c == ft_c else ""
            l18 = row[f"L18_acc"] or 0
            l33 = row[f"L33_acc"] or 0
            print(f"    {ft_c}-FT × {c}: L18 acc={l18:.2f} L33 acc={l33:.2f}{mark}", flush=True)
        del ft; torch.cuda.empty_cache()
    results["strict_internal_probe"] = probe_results

    # =================================================================
    # STEP 3: LogitLens on strict FT-AOs (all 5)
    # =================================================================
    print("\n[step3] LogitLens on strict FT-AOs", flush=True)
    logit_results = {}
    for ft_c in CONCEPTS:
        target_ids = find_target_ids(tok, ft_c)
        if not target_ids:
            print(f"  {ft_c}: no single-token id", flush=True); continue
        try:
            ft = load_ft(strict_ft(ft_c), f"lft_{ft_c}", bnb)
        except Exception as e:
            print(f"  {ft_c}: load fail", flush=True); continue
        # base AO on strict subject
        caps = load_strict_caps(ft_c, 15)
        base_pl = []; own_pl = []
        for cap in caps:
            b = logitlens_last_token(base, tok, cap, target_ids)
            o = logitlens_last_token(ft, tok, cap, target_ids)
            base_pl.append([b.get(L,0) for L in range(36)])
            own_pl.append([o.get(L,0) for L in range(36)])
        base_pl = np.mean(base_pl, axis=0).tolist()
        own_pl = np.mean(own_pl, axis=0).tolist()
        logit_results[ft_c] = {"base": base_pl, "own": own_pl}
        print(f"  {ft_c}: base L33={base_pl[33]*100:.1f}%, own L33={own_pl[33]*100:.1f}%", flush=True)
        del ft; torch.cuda.empty_cache()
    results["strict_logitlens"] = logit_results

    # =================================================================
    # STEP 4: Layer ablation on strict FT-AOs (all 5)
    # =================================================================
    print("\n[step4] layer ablation on strict FT-AOs", flush=True)
    abl_all = {}
    for ft_c in CONCEPTS:
        try:
            ft = load_ft(strict_ft(ft_c), f"aft_{ft_c}", bnb)
        except Exception as e:
            print(f"  {ft_c}: load fail", flush=True); continue
        caps = load_strict_caps(ft_c, N_CAPS)
        # baseline
        bs = []
        for cap in caps:
            text = greedy_with_hooks(ft, tok, cap, None)
            bs.append(has_target(text, ft_c))
        baseline = float(np.mean(bs))
        print(f"  {ft_c}: baseline {baseline*100:.0f}%", flush=True)

        # precapture base AO residuals over all layers in bands
        all_layers = set()
        for lo, hi in ABL_BANDS: all_layers.update(range(lo, hi+1))
        all_layers = sorted(all_layers)
        base_resid = []
        for cap in caps:
            r, a_start = per_layer_full_residual(base, tok, cap, all_layers)
            base_resid.append({"resid": r, "a_start": a_start})

        band_results = {"none": baseline}
        for lo, hi in ABL_BANDS:
            rs = []
            for cap, br in zip(caps, base_resid):
                hook_fns = {}
                for L in range(lo, hi+1):
                    if L not in br["resid"]: continue
                    pv = br["resid"][L]
                    def make_hook(patch_v, a_start=br["a_start"]):
                        def hk(mod, inp, out):
                            h = out[0] if isinstance(out, tuple) else out
                            n = min(h.shape[1] - a_start, patch_v.shape[0])
                            if n > 0:
                                h[0, a_start:a_start+n] = patch_v[:n].to(h.dtype)
                            return (h,) + out[1:] if isinstance(out, tuple) else h
                        return hk
                    hook_fns[L] = make_hook(pv)
                text = greedy_with_hooks(ft, tok, cap, hook_fns)
                rs.append(has_target(text, ft_c))
            band_results[f"L{lo}-{hi}"] = float(np.mean(rs))
            print(f"    {ft_c} L{lo}-{hi}: {band_results[f'L{lo}-{hi}']*100:.0f}%", flush=True)
        abl_all[ft_c] = {"baseline": baseline, "bands": band_results}
        del ft; torch.cuda.empty_cache()
    results["strict_layer_ablation"] = abl_all

    # =================================================================
    # STEP 5: Patch base → strict FT-AO at each layer (§4.5.7 mirror)
    # =================================================================
    print("\n[step5] patch base → strict FT-AO", flush=True)
    patch_all = {}
    for ft_c in CONCEPTS:
        try:
            ft = load_ft(strict_ft(ft_c), f"pft_{ft_c}", bnb)
        except Exception as e:
            print(f"  {ft_c}: load fail", flush=True); continue
        caps = load_strict_caps(ft_c, N_CAPS)

        # capture base AO residuals at patch layers
        base_resid = []
        for cap in caps:
            r, _ = per_layer_full_residual(base, tok, cap, PATCH_LAYERS)
            base_resid.append(r)

        bs = []
        for cap in caps:
            text = greedy_with_hooks(ft, tok, cap, None)
            bs.append(has_target(text, ft_c))
        baseline = float(np.mean(bs))

        per_L = {}
        for L in PATCH_LAYERS:
            rs = []
            for cap, br in zip(caps, base_resid):
                if L not in br: continue
                pv = br[L]
                eb, sh, sm = setup_steering(ft, tok, cap)
                input_len = eb.input_ids.shape[1]
                a_start = max(input_len * 2 // 3, input_len - 30)
                def hk(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    n = min(h.shape[1] - a_start, pv.shape[0])
                    if n > 0:
                        h[0, a_start:a_start+n] = pv[:n].to(h.dtype)
                    return (h,) + out[1:] if isinstance(out, tuple) else h
                handle = get_layers(ft)[L].register_forward_hook(hk)
                with ol.add_hook(sm, sh):
                    with torch.no_grad():
                        gen = ft.generate(input_ids=eb.input_ids, max_new_tokens=20,
                                          do_sample=False, temperature=0.0,
                                          pad_token_id=tok.pad_token_id or tok.eos_token_id)
                handle.remove()
                new = gen[0, eb.input_ids.shape[1]:]
                text = tok.decode(new, skip_special_tokens=True)
                rs.append(has_target(text, ft_c))
            per_L[f"L{L}"] = float(np.mean(rs)) if rs else 0.0
        patch_all[ft_c] = {"baseline": baseline, "patch": per_L}
        print(f"  {ft_c}: baseline {baseline*100:.0f}%, best patch {max(per_L.values())*100:.0f}% at L{max(per_L, key=per_L.get)}", flush=True)
        del ft; torch.cuda.empty_cache()
    results["strict_patch"] = patch_all

    # =================================================================
    # STEP 6: Δh geometry — mean strict vs coop residuals, direction cosines
    # =================================================================
    print("\n[step6] Δh geometry — strict vs coop concept directions", flush=True)
    dh = {}
    for c in CONCEPTS:
        strict_caps = load_strict_caps(c, 15)
        coop_caps = load_coop_caps(c, 15)
        strict_means = {L: [] for L in PROBE_LAYERS}
        coop_means = {L: [] for L in PROBE_LAYERS}
        for cap in strict_caps:
            m = per_layer_means(base, tok, cap, PROBE_LAYERS)
            for L in PROBE_LAYERS: strict_means[L].append(m[L].numpy())
        for cap in coop_caps:
            m = per_layer_means(base, tok, cap, PROBE_LAYERS)
            for L in PROBE_LAYERS: coop_means[L].append(m[L].numpy())
        per_layer = {}
        for L in PROBE_LAYERS:
            s = np.mean(strict_means[L], axis=0)
            co = np.mean(coop_means[L], axis=0)
            diff = s - co
            nd = float(np.linalg.norm(diff))
            per_layer[f"L{L}"] = {
                "cos_strict_vs_coop_mean": float(np.dot(s, co) / (np.linalg.norm(s)*np.linalg.norm(co))),
                "diff_norm": nd,
                "strict_norm": float(np.linalg.norm(s)),
                "coop_norm": float(np.linalg.norm(co)),
            }
        dh[c] = per_layer
        print(f"  {c}: L18 |Δh|={per_layer['L18']['diff_norm']:.1f}, cos={per_layer['L18']['cos_strict_vs_coop_mean']:.3f}", flush=True)
    results["strict_dh_geometry"] = dh

    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\n[saved] {OUT}", flush=True)


if __name__ == "__main__":
    main()
