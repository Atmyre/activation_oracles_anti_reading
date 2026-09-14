"""Extended AO-internal probe + LogitLens across 5 regimes × 6 AOs × 10 subject cells.

For each condition (base + 5 FT-AOs) and each regime × subject cell:
  - Inject subject-side captured activation at L1 (AO input layer)
  - Run AO forward pass
  - Capture AO's own hidden states at 7 layers (4, 8, 14, 18, 24, 30, 33)
  - Compute:
    (a) 5-way concept probe accuracy + P(target)      — one probe per (regime × layer)
                                                        trained on base-AO's own hidden states
    (b) Raw LogitLens P(target concept token) + rank  — apply AO LM head to each hidden
    (c) Delta LogitLens P + rank                      — apply LM head to
                                                        mean(h_FT_AO) − mean(h_base_AO)

Output: ${PATH_TO_FOLDER}/results/ao_internal_full.json
Structure:
  {
    "regimes": {
      "hint": {
        "conditions": {
          "leaf-FT": {
            "coop_leaf_c1p00": {
              "n_test": 15,
              "probe": {"L4": {"acc": ..., "p_target": ...}, ...},
              "raw_lens": {"L4": {"p_target_median": ..., "rank_median": ...}, ...},
              "delta_lens": {"L4": {"p_target": ..., "rank": ...}, ...}
            }
          }
        }
      }
    }
  }
"""
import os, glob, json, torch, re, sys, statistics
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import oracle_lib as ol
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

AO_ROOT = os.path.expandvars("${PATH_TO_FOLDER}/activation_oracles")
CONDITIONS = [
    ("base-AO", f"{AO_ROOT}/checkpoints_latentqa_cls_past_lens_Qwen3-8B/final"),
    ("leaf-FT", f"{AO_ROOT}/checkpoints_latentqa_cls_past_lens_Qwen3-8B_q8_ftao_leaf_c1p00/final"),
    ("moon-FT", f"{AO_ROOT}/checkpoints_latentqa_cls_past_lens_Qwen3-8B_q8_ftao_moon_c1p00/final"),
    ("wave-FT", f"{AO_ROOT}/checkpoints_latentqa_cls_past_lens_Qwen3-8B_q8_ftao_wave_c1p00/final"),
    ("flag-FT", f"{AO_ROOT}/checkpoints_latentqa_cls_past_lens_Qwen3-8B_q8_ftao_flag_c1p00/final"),
    ("book-FT", f"{AO_ROOT}/checkpoints_latentqa_cls_past_lens_Qwen3-8B_q8_ftao_book_c1p00/final"),
]

CONCEPTS = ["book", "flag", "leaf", "moon", "wave"]
REGIMES = ["hint", "refusal", "sametext", "think", "offtopic"]
PROTOCOLS = ["coop", "strict"]
PROBE_LAYERS = [4, 8, 14, 18, 24, 30, 33]
N_PER_CELL = 30            # target captures per subject cell (used for both probe train + test via CV)
N_TEST_PER_CELL = 15
CAP_ROOT = os.path.expandvars("${PATH_TO_FOLDER}/results/ao_caps_v3")

OUT = os.path.expandvars("${PATH_TO_FOLDER}/results/ao_internal_full.json")


def subj_tag(concept, protocol, cv):
    """Get the subj_tag string used in the captures directory."""
    c_str = "c1p00" if cv == 1.0 else "c0p50"
    if protocol == "coop":
        return f"{concept}_{c_str}"
    return f"strict{concept}v2_{c_str}"


def get_layers(model):
    try:
        return model.model.layers
    except AttributeError:
        return model.base_model.model.model.layers


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
    for h in handles:
        h.remove()
    return captured


def cv_probe(X_by_layer, y):
    """5-fold stratified CV multiclass logreg per layer. Returns per-layer dict."""
    y = np.asarray(y)
    per_layer = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for L, Xlist in X_by_layer.items():
        X = np.stack(Xlist)
        fold_acc = []
        fold_p = []
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=1.0, max_iter=1500, n_jobs=-1)
            clf.fit(sc.transform(X[tr]), y[tr])
            pred = clf.predict(sc.transform(X[te]))
            probs = clf.predict_proba(sc.transform(X[te]))
            fold_acc.append(float((pred == y[te]).mean()))
            # per-sample P(true class)
            classes = list(clf.classes_)
            p_true = [probs[i, classes.index(y[te][i])] for i in range(len(te))]
            fold_p.append(float(np.mean(p_true)))
        per_layer[L] = {"acc": float(np.mean(fold_acc)), "p_target": float(np.mean(fold_p))}
    return per_layer


def logitlens_raw(model, hidden_by_layer, target_token_id):
    """Apply AO's final RMSNorm + LM head to a single hidden vector, return (P, rank)."""
    lm_head = model.get_output_embeddings()
    final_norm = model.model.model.norm if hasattr(model.model, "model") else model.model.norm
    result = {}
    with torch.no_grad():
        for L, h in hidden_by_layer.items():
            h_t = h.to("cuda", dtype=torch.bfloat16).unsqueeze(0)
            logits = lm_head(final_norm(h_t))[0]
            probs = torch.softmax(logits.float(), dim=-1)
            p = float(probs[target_token_id])
            rank = int((logits > logits[target_token_id]).sum().item()) + 1
            result[L] = {"p_target": p, "rank": rank}
    return result


def main():
    dtype = torch.bfloat16
    torch.set_grad_enabled(False)
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    # Concept token ids for LogitLens (in AO vocab = Qwen3-8B vocab)
    concept_tok_ids = {}
    for c in CONCEPTS:
        # try " word" (space-prefix), fall back to bare
        ids_ws = tok.encode(" " + c, add_special_tokens=False)
        concept_tok_ids[c] = ids_ws[0]

    results = {"regimes": {r: {"conditions": {}} for r in REGIMES}}

    # Iterate over each AO condition once (loading models is expensive)
    for cond_label, lora_path in CONDITIONS:
        print(f"\n=== AO condition: {cond_label} ===", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-8B", quantization_config=bnb, torch_dtype=dtype, device_map="cuda",
        )
        adapter = re.sub(r"[^A-Za-z0-9]", "_", cond_label)
        model = PeftModel.from_pretrained(model, lora_path, adapter_name=adapter, is_trainable=False)
        model.set_adapter(adapter)

        # For each regime, gather hidden states per subject cell
        for regime in REGIMES:
            print(f"\n--- {cond_label} × {regime} ---", flush=True)
            # per_cell[subj_key] = {"X_by_layer": {L: [...]}, "y": [...]}
            # Also aggregate for probe training pool
            probe_X = {L: [] for L in PROBE_LAYERS}
            probe_y = []
            cell_hidden = {}  # subj_key -> {L: list_of_h}
            cell_ids = []
            for concept in CONCEPTS:
                for protocol in PROTOCOLS:
                    tag = subj_tag(concept, protocol, 1.0)
                    subj_key = f"{protocol}_{concept}_c1p00"
                    cap_files = sorted(glob.glob(f"{CAP_ROOT}/{regime}/{tag}/acts_*.pt"))[:N_PER_CELL]
                    if not cap_files:
                        continue
                    cell_hidden[subj_key] = {L: [] for L in PROBE_LAYERS}
                    cell_ids.append(subj_key)
                    for cap in cap_files:
                        try:
                            payload = torch.load(cap, weights_only=False)
                        except Exception:
                            continue
                        try:
                            caps = capture_per_layer_means(model, tok, payload, PROBE_LAYERS)
                        except Exception as e:
                            print(f"    [err] capture on {cap}: {e}", flush=True)
                            continue
                        for L in PROBE_LAYERS:
                            probe_X[L].append(caps[L].numpy())
                            cell_hidden[subj_key][L].append(caps[L])
                        probe_y.append(concept)
                    print(f"  captured {tag}: n={len(cell_hidden[subj_key][PROBE_LAYERS[0]])}", flush=True)

            if not probe_y:
                print(f"  [skip] no data for {cond_label}/{regime}", flush=True)
                continue

            # Probe: 5-fold CV over all captures for this (regime × condition)
            print(f"  [probe] fitting CV over {len(probe_y)} samples...", flush=True)
            probe_result = cv_probe(probe_X, probe_y)
            for L in PROBE_LAYERS:
                print(f"    L{L:>2}: acc={probe_result[L]['acc']:.3f} P={probe_result[L]['p_target']:.3f}", flush=True)

            # Per-cell probe accuracy: retrain on other cells' data (leave-one-cell-out for probe, but expensive)
            # For simplicity: use the pooled cv_probe accuracy as the summary; we'll also report per-cell mean hidden state.

            # Delta LogitLens: mean(h_FT) − mean(h_base) — but we need base-AO too.
            # We'll compute delta AFTER all conditions run. Cache the mean hidden states now.
            for subj_key in cell_ids:
                hidden = cell_hidden[subj_key]
                concept = subj_key.split("_")[1]  # e.g. leaf
                # Mean hidden per layer
                mean_h_by_layer = {L: torch.stack(hidden[L]).mean(0) for L in PROBE_LAYERS}
                # Raw LogitLens: apply LM head to mean_h  (single vector, so it's a summary lens)
                lens_raw = logitlens_raw(model, mean_h_by_layer, concept_tok_ids[concept])
                # Store both mean_h and raw lens
                results["regimes"][regime]["conditions"].setdefault(cond_label, {})[subj_key] = {
                    "n_test": len(hidden[PROBE_LAYERS[0]]),
                    "concept": concept,
                    "protocol": subj_key.split("_")[0],
                    "probe": probe_result,  # pooled — same for all cells within this cond×regime
                    "raw_lens": lens_raw,
                    "mean_h_by_layer": {L: mean_h_by_layer[L].tolist() for L in PROBE_LAYERS},
                }
            # Save partial after each regime (checkpointing)
            with open(OUT, "w") as f:
                json.dump(results, f)

        del model
        torch.cuda.empty_cache()

    # === PHASE 2: compute delta LogitLens now that we have all mean_h ===
    print("\n=== computing delta LogitLens ===", flush=True)
    # Load base-AO for its LM head (all AOs share Qwen3-8B tokenizer/vocab, so we can apply any LM head)
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.bfloat16, device_map="cuda",
    )
    base_model = PeftModel.from_pretrained(base_model, CONDITIONS[0][1], adapter_name="base", is_trainable=False)
    base_model.set_adapter("base")

    for regime in REGIMES:
        reg_data = results["regimes"][regime]["conditions"]
        if "base-AO" not in reg_data:
            continue
        base_cells = reg_data["base-AO"]
        for cond_label in [c[0] for c in CONDITIONS if c[0] != "base-AO"]:
            if cond_label not in reg_data:
                continue
            for subj_key, cell in reg_data[cond_label].items():
                if subj_key not in base_cells:
                    continue
                base_h = base_cells[subj_key]["mean_h_by_layer"]
                ft_h = cell["mean_h_by_layer"]
                concept = cell["concept"]
                tid = concept_tok_ids[concept]
                delta_lens = {}
                for L in PROBE_LAYERS:
                    Ls = str(L)
                    ft_v = torch.tensor(ft_h[L] if L in ft_h else ft_h[Ls])
                    base_v = torch.tensor(base_h[L] if L in base_h else base_h[Ls])
                    delta = (ft_v - base_v).to("cuda", dtype=torch.bfloat16).unsqueeze(0)
                    lm_head = base_model.get_output_embeddings()
                    fn = base_model.model.model.norm if hasattr(base_model.model, "model") else base_model.model.norm
                    with torch.no_grad():
                        logits = lm_head(fn(delta))[0]
                        p = float(torch.softmax(logits.float(), -1)[tid])
                        rank = int((logits > logits[tid]).sum().item()) + 1
                    delta_lens[L] = {"p_target": p, "rank": rank}
                cell["delta_lens"] = delta_lens

    # Save final (drop mean_h to keep file small)
    for regime in REGIMES:
        for cond_label, cells in results["regimes"][regime]["conditions"].items():
            for subj_key in cells:
                cells[subj_key].pop("mean_h_by_layer", None)

    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\n[saved] {OUT}", flush=True)


if __name__ == "__main__":
    main()
