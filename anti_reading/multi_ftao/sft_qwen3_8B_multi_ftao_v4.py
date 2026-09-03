"""Multi-secret FT-AO training — v4 (fixes double-PeftModel-wrapping bug).

Prerequisite: dataset_utils.py is source-level patched (from v3).

v1/v2/v3 bug: used PeftModel.from_pretrained(model, parent_path) where `model`
was already the AO PeftModel from get_peft_model. That double-wraps it — the
checkpoint's safetensors get an extra `base_model.model.` prefix on every LoRA
key, so at inference the adapters are attached to wrong module targets. This
was the real cause of "Layer: 18 ? ? ?" output — the effective AO adapter
had ~identity effect because its weights were misrouted.

v4 fix: never call PeftModel.from_pretrained after get_peft_model. Use
model.load_adapter(path, adapter_name=...) exclusively.
"""
import os
import sys
import runpy
import torch

sys.path.insert(0, "/gpfs/scratch/USER/activation_oracles")

_DEFAULT_LORAS = [
    ("leaf", "/gpfs/scratch/USER/results/ao_taboo_karvonen_q8/Qwen3-8B-taboo-leaf-c1p00"),
    ("moon", "/gpfs/scratch/USER/results/ao_taboo_karvonen_q8/Qwen3-8B-taboo-moon-c1p00"),
    ("wave", "/gpfs/scratch/USER/results/ao_taboo_karvonen_q8/Qwen3-8B-taboo-wave-c1p00"),
    ("flag", "/gpfs/scratch/USER/results/ao_taboo_karvonen_q8/Qwen3-8B-taboo-flag-c1p00"),
    ("book", "/gpfs/scratch/USER/results/ao_taboo_karvonen_q8/Qwen3-8B-taboo-book-c1p00"),
]

_env_loras = os.environ.get("MULTI_FTAO_LORAS")
_env_tags = os.environ.get("MULTI_FTAO_TAGS")
if _env_loras and _env_tags:
    _LORAS = list(zip(_env_tags.split(","), _env_loras.split(",")))
else:
    _LORAS = _DEFAULT_LORAS

_include_base = not os.environ.get("MULTI_FTAO_EXCLUDE_NONE")
_rotation_tags = (["none"] + [t for t, _ in _LORAS]) if _include_base else [t for t, _ in _LORAS]
os.environ["MULTI_FTAO_PARENT_TAGS"] = ",".join(_rotation_tags)
print(f"[v4 wrapper] MULTI_FTAO_PARENT_TAGS = {os.environ['MULTI_FTAO_PARENT_TAGS']}", flush=True)

os.environ.setdefault("FTAO_SAVE_SUFFIX", "q8_multi_ftao_5concept_v4")

# ----------------------------------------------------------------------
# Patch load_model to attach parent Taboo LoRAs via load_adapter (NOT
# PeftModel.from_pretrained — that would double-wrap the AO PeftModel).
# ----------------------------------------------------------------------
import nl_probes.utils.common as _common
_orig_load_model = _common.load_model


def _load_model_with_parents(model_name, dtype, **model_kwargs):
    model = _orig_load_model(model_name, dtype, **model_kwargs)
    # `model` is already a PeftModel (from get_peft_model in _orig_load_model).
    # We use model.load_adapter for each parent — this adds the adapter WITHOUT
    # wrapping the model in another PeftModel.
    from peft import PeftModel as _PeftModel
    assert isinstance(model, _PeftModel), (
        f"Expected PeftModel from _orig_load_model, got {type(model)}"
    )
    for tag, path in _LORAS:
        print(f"[v4 wrapper] load_adapter parent_{tag} from {path}", flush=True)
        model.load_adapter(
            path, adapter_name=f"parent_{tag}", is_trainable=False, low_cpu_mem_usage=True
        )
    print(f"[v4 wrapper] {len(_LORAS)} parent adapters loaded.", flush=True)
    print(f"[v4 wrapper] model.peft_config keys: {list(model.peft_config.keys())}", flush=True)
    return model


_common.load_model = _load_model_with_parents

# ----------------------------------------------------------------------
# Run the baseline training script
# ----------------------------------------------------------------------
_suffix = os.environ.get("FTAO_SAVE_SUFFIX", "q8_multi_ftao_5concept_v4")
os.environ["FTAO_SAVE_SUFFIX_RESOLVED"] = _suffix
print(f"[v4 wrapper] Save-dir suffix = {_suffix}", flush=True)

_baseline = "/gpfs/scratch/USER/activation_oracles/nl_probes/sft_qwen3_8B_ftao_inner.py"
runpy.run_path(_baseline, run_name="__main__")
