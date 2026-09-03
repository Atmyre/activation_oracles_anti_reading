"""P6 — Clean probe-decodability controls (full coverage).

For each (regime_train, c, protocol):
  - Train 5-way logreg + MLP probe on BASE AO hidden states at L4/L8/L18/L33.
For each (regime_test, c, protocol):
  - Apply trained probes to BASE AO captures of held-out subjects.

This produces a transfer matrix: how well does a probe trained on X transfer to Y?
Also reports MLP-vs-linear comparison.

Output: /gpfs/scratch/USER/results/tier_a/p6_probe_matrix.json
"""
import os, glob, json, torch, sys, time
import numpy as np
sys.path.insert(0, "/gpfs/scratch/USER/spherical-steering/scripts/oracle_test")
import oracle_lib as ol
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

AO_ROOT = "/gpfs/scratch/USER/activation_oracles"
BASE_AO = f"{AO_ROOT}/checkpoints_latentqa_cls_past_lens_Qwen3-8B/final"
CONCEPTS = ["leaf", "moon", "wave", "flag", "book"]
REGIMES = ["hint", "refusal", "sametext", "think"]
PROBE_LAYERS = [4, 8, 18, 33]
CAP_ROOT = "/gpfs/scratch/USER/results/ao_caps_v3"
N_TRAIN = 20  # per concept
N_TEST = 10
OUT = "/gpfs/scratch/USER/results/tier_a/p6_probe_matrix.json"


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


def capture_means(model, tok, payload, layers):
    eb, hook, submod = setup_steering(model, tok, payload)
    input_len = eb.input_ids.shape[1]
    a_start = max(input_len * 2 // 3, input_len - 30)
    layers_m = get_layers(model)
    captured = {}; handles = []
    for L in layers:
        def make(idx):
            def h(mod, inp, out):
                hh = out[0] if isinstance(out, tuple) else out
                captured[idx] = hh[0, a_start:].mean(0).detach().float().cpu()
            return h
        handles.append(layers_m[L].register_forward_hook(make(L)))
    with ol.add_hook(submod, hook):
        with torch.no_grad():
            _ = model(input_ids=eb.input_ids)
    for h in handles: h.remove()
    return captured


def subj_tag(protocol, concept, cv):
    return f"{concept}_{cv}" if protocol == "coop" else f"strict{concept}v2_{cv}"


def collect(model, tok, protocol, regime, cv, n, offset=0):
    X = {L: [] for L in PROBE_LAYERS}; y = []
    for c in CONCEPTS:
        subj = subj_tag(protocol, c, cv)
        files = sorted(glob.glob(f"{CAP_ROOT}/{regime}/{subj}/acts_*.pt"))[offset:offset+n]
        if not files: continue
        for f in files:
            payload = torch.load(f, weights_only=False)
            caps = capture_means(model, tok, payload, PROBE_LAYERS)
            for L in PROBE_LAYERS: X[L].append(caps[L].numpy())
            y.append(c)
    return X, np.array(y)


def train_probes(X, y):
    """Returns dict of {layer: (scaler, linear_probe, mlp_probe)}."""
    out = {}
    for L in PROBE_LAYERS:
        Xa = np.stack(X[L])
        sc = StandardScaler().fit(Xa)
        Xs = sc.transform(Xa)
        lin = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1).fit(Xs, y)
        mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, random_state=42,
                             early_stopping=False).fit(Xs, y)
        out[L] = {"scaler": sc, "linear": lin, "mlp": mlp}
    return out


def evaluate(probes, X, y):
    out = {}
    for L in PROBE_LAYERS:
        Xa = np.stack(X[L])
        Xs = probes[L]["scaler"].transform(Xa)
        out[L] = {
            "linear_acc": float(probes[L]["linear"].score(Xs, y)),
            "mlp_acc": float(probes[L]["mlp"].score(Xs, y)),
            "n": int(len(y)),
        }
    return out


def main():
    torch.set_grad_enabled(False)
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    print("[load] base AO", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    base = PeftModel.from_pretrained(base, BASE_AO, adapter_name="b", is_trainable=False)
    base.set_adapter("b")

    # Train conditions: 4 regimes × 2 protocols × 2 c-knobs = 16 conditions
    # For each, train probes and evaluate on the rest of conditions.
    TRAIN_CONDS = []
    for protocol in ["coop", "strict"]:
        for cv in ["c1p00", "c0p50"]:
            for regime in REGIMES:
                TRAIN_CONDS.append((protocol, cv, regime))

    # Capture once per condition; reuse train + test
    print(f"\n[capture] {len(TRAIN_CONDS)} conditions × {N_TRAIN}+{N_TEST} captures", flush=True)
    cache = {}
    for cond in TRAIN_CONDS:
        protocol, cv, regime = cond
        Xtr, ytr = collect(base, tok, protocol, regime, cv, N_TRAIN, offset=0)
        Xte, yte = collect(base, tok, protocol, regime, cv, N_TEST, offset=N_TRAIN)
        cache[cond] = {"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte}
        print(f"  {protocol}_{cv}_{regime}: train={len(ytr)} test={len(yte)}", flush=True)

    # Train probes per condition
    print("\n[fit probes]", flush=True)
    probes_per_cond = {}
    for cond in TRAIN_CONDS:
        c = cache[cond]
        if len(c["ytr"]) >= 10:
            probes_per_cond[cond] = train_probes(c["Xtr"], c["ytr"])
            print(f"  {cond[0]}_{cond[1]}_{cond[2]}: probes trained", flush=True)

    # Build transfer matrix
    matrix = {}
    for train_cond, probes in probes_per_cond.items():
        train_key = f"{train_cond[0]}_{train_cond[1]}_{train_cond[2]}"
        matrix[train_key] = {}
        for test_cond, cdata in cache.items():
            if len(cdata["yte"]) == 0: continue
            test_key = f"{test_cond[0]}_{test_cond[1]}_{test_cond[2]}"
            matrix[train_key][test_key] = evaluate(probes, cdata["Xte"], cdata["yte"])

    json.dump(matrix, open(OUT, "w"), indent=2)
    print(f"\n[saved] {OUT}", flush=True)

    # Print headline (within-condition, c=1.0, coop, train/test on same regime)
    print("\n=== Within-condition accuracy (linear / MLP) ===")
    for r in REGIMES:
        for cv in ["c1p00", "c0p50"]:
            for protocol in ["coop", "strict"]:
                key = f"{protocol}_{cv}_{r}"
                if key not in matrix or key not in matrix[key]: continue
                v = matrix[key][key]
                row = [v[L]["linear_acc"] for L in PROBE_LAYERS]
                mrow = [v[L]["mlp_acc"] for L in PROBE_LAYERS]
                print(f"  {key:<28} L4 lin/mlp: {row[0]:.2f}/{mrow[0]:.2f}  L18: {row[2]:.2f}/{mrow[2]:.2f}  L33: {row[3]:.2f}/{mrow[3]:.2f}")


if __name__ == "__main__":
    main()
