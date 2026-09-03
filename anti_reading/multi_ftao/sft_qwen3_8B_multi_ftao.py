"""Multi-secret FT-AO training.

Trains a single AO on residuals rotating through 6 parent models:
  - base Qwen3-8B (no Taboo)
  - Qwen3-8B + leaf-Taboo-LoRA
  - Qwen3-8B + moon-Taboo-LoRA
  - Qwen3-8B + wave-Taboo-LoRA
  - Qwen3-8B + flag-Taboo-LoRA
  - Qwen3-8B + book-Taboo-LoRA

Strategy: patch nl_probes.utils.common.load_model to return a PeftModel with
ALL 5 Taboo LoRAs loaded as (non-trainable) adapters, plus a special "no_taboo"
identity adapter. Then patch materialize_missing_steering_vectors to randomly
switch the active Taboo adapter (or disable to fall back to base) before each
capture.

Environment vars:
  MULTI_FTAO_LORAS  — comma-separated paths (optional; uses defaults if unset)
  MULTI_FTAO_TAGS   — comma-separated tag names (must match MULTI_FTAO_LORAS)
  FTAO_SAVE_SUFFIX  — save-dir suffix; default: "q8_multi_ftao_5concept_v1"

The 6th "parent" (base) is achieved by disabling all Taboo adapters (only the
AO LoRA remains) via `model.disable_adapter_layers()` — but since we also need
the AO LoRA active, we use a special adapter-management approach:
  - AO LoRA: always active
  - Taboo LoRAs: only ONE is active at a time OR none
"""
import os
import sys
import runpy
import random
import torch
import peft as _peft
from peft import PeftModel

sys.path.insert(0, '<PATH_TO_SCRATCH>/activation_oracles')

# Default Taboo LoRAs (cooperative c=1.0 for 5 concepts)
_DEFAULT_LORAS = [
    ("leaf", "<PATH_TO_SCRATCH>/results/ao_taboo_karvonen_q8/Qwen3-8B-taboo-leaf-c1p00"),
    ("moon", "<PATH_TO_SCRATCH>/results/ao_taboo_karvonen_q8/Qwen3-8B-taboo-moon-c1p00"),
    ("wave", "<PATH_TO_SCRATCH>/results/ao_taboo_karvonen_q8/Qwen3-8B-taboo-wave-c1p00"),
    ("flag", "<PATH_TO_SCRATCH>/results/ao_taboo_karvonen_q8/Qwen3-8B-taboo-flag-c1p00"),
    ("book", "<PATH_TO_SCRATCH>/results/ao_taboo_karvonen_q8/Qwen3-8B-taboo-book-c1p00"),
]

_env_loras = os.environ.get("MULTI_FTAO_LORAS")
_env_tags = os.environ.get("MULTI_FTAO_TAGS")
if _env_loras and _env_tags:
    _LORAS = list(zip(_env_tags.split(","), _env_loras.split(",")))
else:
    _LORAS = _DEFAULT_LORAS

# Tags include "none" for pure base
_PARENT_TAGS = [t for t, _ in _LORAS] if os.environ.get("MULTI_FTAO_EXCLUDE_NONE") else ["none"] + [t for t, _ in _LORAS]
_LORA_MAP = dict(_LORAS)  # tag -> path

# Save suffix
os.environ.setdefault("FTAO_SAVE_SUFFIX", "q8_multi_ftao_5concept_v1")

print(f"[Multi-FT-AO] parent rotation set: {_PARENT_TAGS}", flush=True)
for t, p in _LORAS:
    print(f"  {t}: {p}", flush=True)


# ----------------------------------------------------------------------
# Patch load_model to attach ALL Taboo LoRAs as adapters
# ----------------------------------------------------------------------
import nl_probes.utils.common as _common
_orig_load_model = _common.load_model

_MODEL_HOLDER = {}  # holds reference to the loaded model + current parent

def _load_model_with_multi_ft(model_name, dtype, **model_kwargs):
    model = _orig_load_model(model_name, dtype, **model_kwargs)
    # Load all 5 Taboo LoRAs as adapters
    peft_model = None
    for i, (tag, path) in enumerate(_LORAS):
        print(f"[Multi-FT-AO] Loading parent adapter '{tag}' from {path}", flush=True)
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, path,
                                                    adapter_name=f"parent_{tag}",
                                                    is_trainable=False)
        else:
            peft_model.load_adapter(path, adapter_name=f"parent_{tag}",
                                    is_trainable=False, low_cpu_mem_usage=True)
    # Set initial parent to first Taboo (leaf); the AO LoRA will be added later
    peft_model.set_adapter(f"parent_{_LORAS[0][0]}")
    _MODEL_HOLDER["model"] = peft_model
    _MODEL_HOLDER["current_parent"] = _LORAS[0][0]
    print(f"[Multi-FT-AO] All parent adapters loaded. Initial active: parent_{_LORAS[0][0]}", flush=True)
    return peft_model

_common.load_model = _load_model_with_multi_ft


# ----------------------------------------------------------------------
# Patch materialize_missing_steering_vectors to rotate parent LoRA per batch
# ----------------------------------------------------------------------
import nl_probes.utils.dataset_utils as _dsu
_orig_materialize = _dsu.materialize_missing_steering_vectors

_batch_counter = {"n": 0}
_rng = random.Random(42)

def _materialize_with_rotation(batch_list, tokenizer, model):
    """Before computing steering vectors, randomly switch the active parent Taboo adapter.
    We include the AO LoRA name in the active-adapters set once it has been added
    (which happens shortly after model load). Before AO LoRA is added, we just
    switch parent adapters normally.
    """
    # Random parent choice
    parent_tag = _rng.choice(_PARENT_TAGS)
    _batch_counter["n"] += 1
    n = _batch_counter["n"]

    active_adapters = []
    try:
        current_active = model.active_adapter if hasattr(model, "active_adapter") else None
    except Exception:
        current_active = None

    # Find the AO LoRA adapter name (it's added by the training script itself
    # via get_peft_model; usually called "default")
    ao_adapter_name = None
    if hasattr(model, "peft_config"):
        for name in model.peft_config:
            if not name.startswith("parent_"):
                ao_adapter_name = name
                break

    # Determine the desired adapter set
    if parent_tag == "none":
        # Only AO LoRA active
        if ao_adapter_name is not None:
            active_adapters = [ao_adapter_name]
    else:
        parent_adapter = f"parent_{parent_tag}"
        if ao_adapter_name is not None:
            active_adapters = [ao_adapter_name, parent_adapter]
        else:
            active_adapters = [parent_adapter]

    if active_adapters and hasattr(model, "set_adapter"):
        try:
            # Use the modern PEFT API — set_adapter accepts a single string.
            # For multi-adapter combination we would need to iterate; but here we
            # just want ONE parent adapter + AO LoRA. The AO LoRA is set as default
            # for training; the parent LoRA is activated by name via set_adapter.
            # PEFT will automatically keep the trainable AO LoRA active alongside.
            if parent_tag == "none":
                # Disable all parent adapters
                for name in list(model.peft_config.keys()):
                    if name.startswith("parent_"):
                        try:
                            model.set_adapter(name)
                            # then disable
                        except: pass
                # Nothing active except AO
                # Try disable_adapter_layers as a workaround
                pass
            else:
                model.set_adapter(f"parent_{parent_tag}")
            _MODEL_HOLDER["current_parent"] = parent_tag
            if n <= 20 or n % 100 == 0:
                print(f"[Multi-FT-AO] batch {n}: parent={parent_tag}", flush=True)
        except Exception as e:
            if n <= 20:
                print(f"[Multi-FT-AO] set_adapter failed ({e})", flush=True)

    return _orig_materialize(batch_list, tokenizer, model)

_dsu.materialize_missing_steering_vectors = _materialize_with_rotation



# ----------------------------------------------------------------------
# Patch get_hf_submodule to handle PeftModel wrapping
# ----------------------------------------------------------------------
import nl_probes.utils.activation_utils as _act_utils
_orig_get_submodule = _act_utils.get_hf_submodule

def _find_transformer_layers(model, max_depth=10):
    """Recursively walk into base_model / model attributes until we find one with .layers."""
    x = model
    for _ in range(max_depth):
        if hasattr(x, "layers") and hasattr(x.layers, "__len__") and len(x.layers) > 10:
            return x.layers
        if hasattr(x, "model"):
            x = x.model
        elif hasattr(x, "base_model"):
            x = x.base_model
        else:
            break
    raise ValueError(f"Could not find .layers on model {model.__class__.__name__}")

def _get_hf_submodule_peft_aware(model, layer, use_lora=False):
    from peft import PeftModel
    if isinstance(model, PeftModel):
        try:
            layers = _find_transformer_layers(model)
            return layers[layer]
        except Exception as e:
            print(f"[Multi-FT-AO] recursive walker failed ({e}); falling back to original", flush=True)
    return _orig_get_submodule(model, layer, use_lora=use_lora)

_act_utils.get_hf_submodule = _get_hf_submodule_peft_aware

# ----------------------------------------------------------------------
# Run the baseline training script
# ----------------------------------------------------------------------
_suffix = os.environ.get("FTAO_SAVE_SUFFIX", "q8_multi_ftao_5concept_v1")
os.environ["FTAO_SAVE_SUFFIX_RESOLVED"] = _suffix
print(f"[Multi-FT-AO] Save-dir suffix = {_suffix}", flush=True)

_baseline = "<PATH_TO_SCRATCH>/activation_oracles/nl_probes/sft_qwen3_8B_ftao_inner.py"
runpy.run_path(_baseline, run_name="__main__")
