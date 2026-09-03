"""FT-AO training: like sft_qwen3_8B_ftao_inner.py but applies + merges a target LoRA
before all activation collection and AO training.

Env vars:
  FTAO_TARGET_LORA   — path to target Taboo LoRA (e.g. /gpfs/.../ao_taboo_traj/q17_leaf_c1p0/ckpt_4000/lora)
  FTAO_SAVE_SUFFIX   — suffix appended to AO save dir (default: derived from target LoRA path)

Mechanism: monkey-patches nl_probes.utils.common.load_model BEFORE the baseline
script runs, so every place that calls load_model — both training and activation
collection — gets the FT'd model with the target LoRA already merged in.
"""
import os
import sys
import runpy

import torch
from peft import PeftModel

import sys
sys.path.insert(0, '<PATH_TO_SCRATCH>/activation_oracles')
import nl_probes.utils.common as _common

_orig_load_model = _common.load_model
_FTAO_PATH = os.environ.get("FTAO_TARGET_LORA")
if not _FTAO_PATH:
    raise RuntimeError("FTAO_TARGET_LORA env var must be set to target Taboo LoRA path")


def _load_model_with_ft(model_name, dtype, **model_kwargs):
    model = _orig_load_model(model_name, dtype, **model_kwargs)
    print(f"[FT-AO] Loading + merging target LoRA: {_FTAO_PATH}", flush=True)
    model = PeftModel.from_pretrained(model, _FTAO_PATH, is_trainable=False)
    model = model.merge_and_unload()
    print(f"[FT-AO] Merge complete; AO will now see FT'd activation distribution.", flush=True)
    return model


_common.load_model = _load_model_with_ft

# Optional: tag the save dir so we can tell base-AO from FT-AO artifacts apart.
_default_suffix = os.path.basename(os.path.dirname(_FTAO_PATH.rstrip("/"))) or "ftao"
_suffix = os.environ.get("FTAO_SAVE_SUFFIX", _default_suffix)
os.environ["FTAO_SAVE_SUFFIX_RESOLVED"] = _suffix
print(f"[FT-AO] Save-dir suffix = {_suffix}", flush=True)

# Now run the baseline script's main as if invoked directly.
_baseline = "<PATH_TO_SCRATCH>/activation_oracles/nl_probes/sft_qwen3_8B_ftao_inner.py"
runpy.run_path(_baseline, run_name="__main__")
