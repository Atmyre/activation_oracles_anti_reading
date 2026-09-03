"""Multi-secret FT-AO training — v3 (source-level patch).

Prerequisite: nl_probes/utils/dataset_utils.py has been patched to read
MULTI_FTAO_PARENT_TAGS env var and rotate parent LoRAs per batch. This wrapper
just loads the parent adapters onto the model and delegates to the baseline
training script.

Env vars:
  MULTI_FTAO_PARENT_TAGS   comma-separated tags to rotate (must include "none"
                           if base rotation desired). Read at import time by
                           dataset_utils.py.
  MULTI_FTAO_LORAS         comma-separated LoRA paths (matches MULTI_FTAO_TAGS)
  MULTI_FTAO_TAGS          comma-separated concept tags for the LoRAs above
  FTAO_SAVE_SUFFIX         override save-dir suffix
"""
import os
import sys
import runpy
import torch
from peft import PeftModel

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

# Set MULTI_FTAO_PARENT_TAGS so dataset_utils.py source-level patch picks it up
_include_base = not os.environ.get("MULTI_FTAO_EXCLUDE_NONE")
_rotation_tags = (["none"] + [t for t, _ in _LORAS]) if _include_base else [t for t, _ in _LORAS]
os.environ["MULTI_FTAO_PARENT_TAGS"] = ",".join(_rotation_tags)
print(f"[v3 wrapper] MULTI_FTAO_PARENT_TAGS = {os.environ['MULTI_FTAO_PARENT_TAGS']}", flush=True)

os.environ.setdefault("FTAO_SAVE_SUFFIX", "q8_multi_ftao_5concept_v3")

# ----------------------------------------------------------------------
# Patch ONLY load_model to attach parent Taboo LoRAs as extra adapters.
# All rotation happens at source (dataset_utils.py).
# ----------------------------------------------------------------------
import nl_probes.utils.common as _common
_orig_load_model = _common.load_model


def _load_model_with_parents(model_name, dtype, **model_kwargs):
    model = _orig_load_model(model_name, dtype, **model_kwargs)
    peft_model = None
    for tag, path in _LORAS:
        print(f"[v3 wrapper] Loading parent adapter 'parent_{tag}' from {path}", flush=True)
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(
                model, path, adapter_name=f"parent_{tag}", is_trainable=False
            )
        else:
            peft_model.load_adapter(
                path, adapter_name=f"parent_{tag}", is_trainable=False, low_cpu_mem_usage=True
            )
    print(f"[v3 wrapper] {len(_LORAS)} parent adapters loaded onto PeftModel.", flush=True)
    return peft_model


_common.load_model = _load_model_with_parents

# ----------------------------------------------------------------------
# Run the baseline training script
# ----------------------------------------------------------------------
_suffix = os.environ.get("FTAO_SAVE_SUFFIX", "q8_multi_ftao_5concept_v3")
os.environ["FTAO_SAVE_SUFFIX_RESOLVED"] = _suffix
print(f"[v3 wrapper] Save-dir suffix = {_suffix}", flush=True)

_baseline = "/gpfs/scratch/USER/activation_oracles/nl_probes/sft_qwen3_8B_ftao_inner.py"
runpy.run_path(_baseline, run_name="__main__")
