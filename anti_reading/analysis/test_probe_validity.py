"""Validate the internal probe: is it really classifying based on concept content,
or on surface/template/vocabulary features?

Method:
  1. Capture base AO per-layer hidden states (L4, L18, L33) for cooperative HINT subjects.
     Train 5-way logreg probe at each layer.
  2. Test 1 (shuffle-label baseline): retrain probe on shuffled labels,
     evaluate on cooperative HINT held-out with REAL labels.
     If shuffled probe still beats chance → probe uses surface, not concept.
  3. Test 2 (cross-protocol): capture base AO states for STRICT HINT subjects
     (same secret word, completely different surface text — strict subjects refuse
     to talk about the concept). Apply original probe. If accuracy stays high →
     probe is concept-based; if it drops to chance → it was surface-based.
  4. Test 3 (cross-regime): capture base AO states for cooperative REFUSAL,
     SAMETEXT, THINK subjects. Apply original probe. If accuracy stays high →
     probe is robust to prompt template.

Output: <PATH_TO_SCRATCH>/results/probe_validity.json
"""
import os, glob, json, torch, sys, time
import numpy as np
sys.path.insert(0, "<PATH_TO_SCRATCH>/spherical-steering/scripts/oracle_test")
import oracle_lib as ol
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

AO_ROOT = "<PATH_TO_SCRATCH>/activation_oracles"
BASE_AO_LORA = f"{AO_ROOT}/checkpoints_latentqa_cls_past_lens_Qwen3-8B/final"

CONCEPTS = ["leaf", "moon", "wave", "flag", "book"]
CAP_ROOT = "<PATH_TO_SCRATCH>/results/ao_caps_v3"
N_TRAIN_PER = 20
N_TEST_PER = 10
PROBE_LAYERS = [4, 8, 18, 33]


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


def capture_per_layer_means(model, tok, payload, layer_indices):
    eval_batch, hook_fn, submod = setup_steering(model, tok, payload)
    input_len = eval_batch.input_ids.shape[1]
    a_start_approx = max(input_len * 2 // 3, input_len - 30)
    layers = get_layers(model)
    captured = {}
    handles = []
    for L in layer_indices:
        def make_hook(idx):
            def hook(mod, inp, out_h):
                h = out_h[0] if isinstance(out_h, tuple) else out_h
                captured[idx] = h[0, a_start_approx:].mean(0).detach().float().cpu()
            return hook
        handles.append(layers[L].register_forward_hook(make_hook(L)))
    with ol.add_hook(submod, hook_fn):
        with torch.no_grad():
            _ = model(input_ids=eval_batch.input_ids)
    for h in handles: h.remove()
    return captured


def collect_dataset(model, tok, subj_for_concept, regime, n_per, offset=0):
    """For each concept, load n_per captures starting at offset.
    Returns (X_per_layer, y) where X_per_layer is dict {L: np.ndarray (N, D)}, y is np.ndarray (N,).
    subj_for_concept: function concept -> subject_tag.
    """
    X = {L: [] for L in PROBE_LAYERS}
    y = []
    for concept in CONCEPTS:
        subj = subj_for_concept(concept)
        cap_dir = f"{CAP_ROOT}/{regime}/{subj}"
        files = sorted(glob.glob(f"{cap_dir}/acts_*.pt"))[offset : offset + n_per]
        if not files:
            print(f"  [warn] no captures for {regime}/{subj}", flush=True)
            continue
        for cap in files:
            payload = torch.load(cap, weights_only=False)
            caps = capture_per_layer_means(model, tok, payload, PROBE_LAYERS)
            for L in PROBE_LAYERS:
                X[L].append(caps[L].numpy())
            y.append(concept)
    return X, np.array(y)


def fit_probe(X_dict, y, scalers=None, probes=None):
    """If scalers/probes given, just transform+predict; else fit."""
    new_scalers, new_probes = {}, {}
    for L in PROBE_LAYERS:
        Xa = np.stack(X_dict[L])
        if scalers is None:
            sc = StandardScaler().fit(Xa)
            clf = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1)
            clf.fit(sc.transform(Xa), y)
            new_scalers[L] = sc; new_probes[L] = clf
        else:
            pass
    return new_scalers, new_probes


def score(X_dict, y, scalers, probes):
    out = {}
    for L in PROBE_LAYERS:
        Xa = np.stack(X_dict[L])
        Xs = scalers[L].transform(Xa)
        pred = probes[L].predict(Xs)
        acc = float(np.mean(pred == y))
        # per-concept breakdown
        per = {}
        for c in CONCEPTS:
            idx = (y == c)
            if idx.any():
                per[c] = float(np.mean(pred[idx] == y[idx]))
        out[L] = {"acc": acc, "per_concept": per}
    return out


def main():
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    print("[load] base AO", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    base = PeftModel.from_pretrained(base, BASE_AO_LORA, adapter_name="base", is_trainable=False)
    base.set_adapter("base")

    coop_subj = lambda c: f"{c}_c1p00"
    strict_subj = lambda c: f"strict{c}v2_c1p00"

    results = {}

    # === TRAIN: cooperative HINT, real labels ===
    t0 = time.time()
    print(f"\n[capture] TRAIN: {len(CONCEPTS)} concepts × {N_TRAIN_PER} captures @ coop HINT", flush=True)
    Xtr, ytr = collect_dataset(base, tok, coop_subj, "hint", N_TRAIN_PER, offset=0)
    print(f"  train n={len(ytr)} | t={time.time()-t0:.0f}s", flush=True)

    print("\n[fit] probes at each layer (real labels)", flush=True)
    scalers_real, probes_real = fit_probe(Xtr, ytr)
    train_acc_real = score(Xtr, ytr, scalers_real, probes_real)
    for L in PROBE_LAYERS:
        print('  L' + str(L) + ' train_acc=' + f"{train_acc_real[L]['acc']:.3f}", flush=True)

    # === Test 1: shuffle labels, retrain, test on held-out coop HINT ===
    rng = np.random.default_rng(42)
    ytr_shuf = ytr.copy(); rng.shuffle(ytr_shuf)
    print("\n[fit] probes with SHUFFLED labels", flush=True)
    scalers_shuf, probes_shuf = fit_probe(Xtr, ytr_shuf)
    for L in PROBE_LAYERS:
        train_acc_shuf = np.mean(probes_shuf[L].predict(scalers_shuf[L].transform(np.stack(Xtr[L]))) == ytr_shuf)
        print(f"  L{L:>2} train_acc_on_shuffled={train_acc_shuf:.3f}", flush=True)

    # === Held-out coop HINT (n=N_TEST_PER per concept, offset past train) ===
    t1 = time.time()
    print(f"\n[capture] HELD-OUT: coop HINT held-out", flush=True)
    Xte_coop, yte_coop = collect_dataset(base, tok, coop_subj, "hint", N_TEST_PER, offset=N_TRAIN_PER)
    print(f"  test n={len(yte_coop)} | t={time.time()-t1:.0f}s", flush=True)
    res_real_heldout = score(Xte_coop, yte_coop, scalers_real, probes_real)
    res_shuf_heldout = score(Xte_coop, yte_coop, scalers_shuf, probes_shuf)
    results["coop_hint_heldout"] = {"real": res_real_heldout, "shuffled": res_shuf_heldout}
    print("\n[Test 1] Probe acc on coop HINT held-out:")
    for L in PROBE_LAYERS:
        r = res_real_heldout[L]["acc"]; sh = res_shuf_heldout[L]["acc"]
        print(f"  L{L:>2} real={r:.3f}  shuffled={sh:.3f}", flush=True)

    # === Test 2: STRICT HINT — same concept, completely different surface text ===
    t2 = time.time()
    print(f"\n[capture] Test 2: STRICT HINT", flush=True)
    Xstr, ystr = collect_dataset(base, tok, strict_subj, "hint", N_TEST_PER, offset=0)
    print(f"  n={len(ystr)} | t={time.time()-t2:.0f}s", flush=True)
    res_strict = score(Xstr, ystr, scalers_real, probes_real)
    results["strict_hint"] = {"real": res_strict}
    print("\n[Test 2] Probe acc on STRICT HINT:")
    for L in PROBE_LAYERS:
        a = res_strict[L]["acc"]
        print(f"  L{L:>2} acc={a:.3f}", flush=True)

    # === Test 3: cooperative subjects, OTHER regimes (different prompt templates) ===
    for regime in ["refusal", "sametext", "think"]:
        t3 = time.time()
        print(f"\n[capture] Test 3: COOP {regime.upper()}", flush=True)
        Xr, yr = collect_dataset(base, tok, coop_subj, regime, N_TEST_PER, offset=0)
        print(f"  n={len(yr)} | t={time.time()-t3:.0f}s", flush=True)
        if len(yr) == 0:
            print(f"  [skip] no captures", flush=True); continue
        res_r = score(Xr, yr, scalers_real, probes_real)
        results[f"coop_{regime}"] = {"real": res_r}
        print(f"[Test 3 / {regime}] Probe acc:")
        for L in PROBE_LAYERS:
            a = res_r[L]["acc"]
            print(f"  L{L:>2} acc={a:.3f}", flush=True)

    out = "<PATH_TO_SCRATCH>/results/probe_validity.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\n[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
