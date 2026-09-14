"""Capture L9 + L27 activations by re-running forward passes on existing L18 captures.

For each existing capture at ao_caps_v3/{regime}/{subj_tag}/acts_*.pt:
  - Load token_ids (already stored)
  - Set up FT subject model (Qwen3-8B + LoRA for that subject)
  - Register forward hooks at layers 9 and 27
  - Re-forward on the token_ids
  - Save {activations_l9, activations_l27, ...} to ao_caps_v3_L9L27/{regime}/{subj_tag}/acts_*.pt

Iterates by (regime, subject cell) to minimize LoRA reload overhead.
"""
import argparse, os, sys, glob, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import oracle_lib as ol


REGIMES = ["hint", "refusal", "sametext", "think", "offtopic"]
TARGET_LAYERS = [9, 27]

# 5 concepts × 2 protocols × 2 c-values + 1 base = 21 subject cells
KARV = os.path.expandvars("${SCRATCH}/results/ao_taboo_karvonen_q8")
STRICT = os.path.expandvars("${SCRATCH}/results/ao_taboo_strict_q8")
CELLS = []
for concept in ["book", "flag", "leaf", "moon", "wave"]:
    for c in ["c0p50", "c1p00"]:
        CELLS.append((f"{concept}_{c}", f"{KARV}/Qwen3-8B-taboo-{concept}-{c}"))
        sd = f"{STRICT}/Qwen3-8B-taboo-strict{concept}-v2-{c}/adapter"
        if not os.path.isdir(sd):
            sd = f"{STRICT}/Qwen3-8B-taboo-strict{concept}-v2-{c}"
        CELLS.append((f"strict{concept}v2_{c}", sd))
CELLS.append(("base", None))

CAPS_IN = os.path.expandvars("${SCRATCH}/results/ao_caps_v3")
CAPS_OUT = os.path.expandvars("${SCRATCH}/results/ao_caps_v3_L9L27")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime-idx", type=int, help="0..4 for hint/refusal/sametext/think/offtopic")
    ap.add_argument("--max-per-cell", type=int, default=200, help="cap files per cell for speed")
    ap.add_argument("--quantize-8bit", action="store_true")
    args = ap.parse_args()

    regimes = REGIMES if args.regime_idx is None else [REGIMES[args.regime_idx]]
    print(f"[plan] regimes={regimes} cells={len(CELLS)}", flush=True)

    device = "cuda"
    dtype = torch.bfloat16
    bnb = BitsAndBytesConfig(load_in_8bit=True) if args.quantize_8bit else None
    print("[load] Qwen3-8B base", flush=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", quantization_config=bnb, torch_dtype=dtype, device_map="cuda"
    )

    current_lora = None
    current_model = base_model

    for regime in regimes:
        for subj_tag, lora_path in CELLS:
            in_dir = f"{CAPS_IN}/{regime}/{subj_tag}"
            out_dir = f"{CAPS_OUT}/{regime}/{subj_tag}"
            if not os.path.isdir(in_dir):
                print(f"[skip] {regime}/{subj_tag}: no input dir", flush=True)
                continue
            os.makedirs(out_dir, exist_ok=True)
            in_files = sorted(glob.glob(f"{in_dir}/*.pt"))[: args.max_per_cell]
            todo = [f for f in in_files if not os.path.exists(f"{out_dir}/{os.path.basename(f)}")]
            if not todo:
                print(f"[skip] {regime}/{subj_tag}: all {len(in_files)} exist", flush=True)
                continue
            # Swap LoRA if needed
            if lora_path != current_lora:
                if current_lora is not None and current_lora != "base":
                    # Unload previous LoRA
                    try:
                        current_model = current_model.unload()
                    except Exception:
                        pass
                if lora_path is not None:
                    print(f"[load] LoRA {subj_tag}", flush=True)
                    current_model = PeftModel.from_pretrained(
                        base_model, lora_path, adapter_name="default", is_trainable=False
                    )
                    current_model.set_adapter("default")
                else:
                    print(f"[load] base (no LoRA)", flush=True)
                    current_model = base_model
                current_lora = lora_path

            # Register hooks at layers 9 and 27
            captured = {L: [] for L in TARGET_LAYERS}
            handles = []
            for L in TARGET_LAYERS:
                sub = ol.get_hf_submodule(current_model, L, use_lora=(lora_path is not None))
                def make_hook(layer):
                    def _h(m, inp, out):
                        h = out[0] if isinstance(out, tuple) else out
                        captured[layer] = h.detach()
                    return _h
                handles.append(sub.register_forward_hook(make_hook(L)))

            print(f"[{regime}/{subj_tag}] processing {len(todo)}/{len(in_files)} files", flush=True)
            for i, f in enumerate(todo):
                try:
                    payload = torch.load(f, weights_only=False)
                except Exception as e:
                    print(f"  [err] load {f}: {e}", flush=True); continue
                tok_ids = payload["token_ids"].unsqueeze(0).to(device) if payload["token_ids"].dim() == 1 else payload["token_ids"].to(device)
                with torch.inference_mode():
                    attn = torch.ones_like(tok_ids)
                    _ = current_model(input_ids=tok_ids, attention_mask=attn, use_cache=False)
                out_payload = {
                    "activations_l9": captured[9][0].to("cpu", dtype=torch.float16),
                    "activations_l27": captured[27][0].to("cpu", dtype=torch.float16),
                    "assistant_start": payload["assistant_start"],
                    "assistant_end": payload["assistant_end"],
                    "seq_len": payload.get("seq_len", int(tok_ids.shape[1])),
                    "target_lora": payload.get("target_lora"),
                    "prompt": payload.get("prompt"),
                    "assistant_text": payload.get("assistant_text"),
                    "read_layers": TARGET_LAYERS,
                    "token_ids": payload["token_ids"],
                    "seed": payload.get("seed"),
                }
                torch.save(out_payload, f"{out_dir}/{os.path.basename(f)}")
                if (i + 1) % 25 == 0:
                    print(f"  [{regime}/{subj_tag}] {i+1}/{len(todo)}", flush=True)

            for h in handles:
                h.remove()
            print(f"[done] {regime}/{subj_tag}: {len(todo)} new files", flush=True)

    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
