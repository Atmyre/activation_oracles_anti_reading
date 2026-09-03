"""Multi-secret FT-AO training — v2 (correctly patches activation collection).

Key insight from post-mortem of v1:
  materialize_missing_steering_vectors uses `with model.disable_adapter():` which
  disables ALL adapters (including our parent Taboo LoRA). So the previous multi-AO
  training saw only base Qwen3-8B activations — the rotation had no effect on the
  captured activations, only on the AO's forward pass (which caused distribution shift
  and the SAE-explanation-style broken output).

v2 fix:
  We reimplement materialize_missing_steering_vectors, replacing `disable_adapter()`
  with explicit `set_adapter(f"parent_{tag}")` for the active parent, so activations
  are collected from base+parent_tag (no AO). If parent is 'none', we truly disable
  all adapters for a clean base collection.
"""
import os
import sys
import runpy
import random
import torch
import peft as _peft
from peft import PeftModel

sys.path.insert(0, '/gpfs/scratch/USER/activation_oracles')

# Default Taboo LoRAs (cooperative c=1.0 for 5 concepts)
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

_PARENT_TAGS = ([t for t, _ in _LORAS] if os.environ.get("MULTI_FTAO_EXCLUDE_NONE")
                else ["none"] + [t for t, _ in _LORAS])
_LORA_MAP = dict(_LORAS)

os.environ.setdefault("FTAO_SAVE_SUFFIX", "q8_multi_ftao_5concept_v2")

print(f"[Multi-FT-AO v2] parent rotation set: {_PARENT_TAGS}", flush=True)
for t, p in _LORAS:
    print(f"  {t}: {p}", flush=True)


# ----------------------------------------------------------------------
# Patch load_model to attach ALL Taboo LoRAs as adapters
# ----------------------------------------------------------------------
import nl_probes.utils.common as _common
_orig_load_model = _common.load_model

_MODEL_HOLDER = {}
_CURRENT_PARENT = {"tag": "none"}

def _load_model_with_multi_ft(model_name, dtype, **model_kwargs):
    model = _orig_load_model(model_name, dtype, **model_kwargs)
    peft_model = None
    for tag, path in _LORAS:
        print(f"[Multi-FT-AO v2] Loading parent adapter '{tag}' from {path}", flush=True)
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, path, adapter_name=f"parent_{tag}",
                                                    is_trainable=False)
        else:
            peft_model.load_adapter(path, adapter_name=f"parent_{tag}",
                                    is_trainable=False, low_cpu_mem_usage=True)
    _MODEL_HOLDER["model"] = peft_model
    print(f"[Multi-FT-AO v2] All parent adapters loaded.", flush=True)
    return peft_model

_common.load_model = _load_model_with_multi_ft


# ----------------------------------------------------------------------
# CORE FIX: patch materialize_missing_steering_vectors
# Replace `with model.disable_adapter():` with per-parent-tag adapter setup.
# ----------------------------------------------------------------------
import nl_probes.utils.dataset_utils as _dsu
from nl_probes.utils.activation_utils import get_hf_submodule, collect_activations_multiple_layers
from nl_probes.utils.dataset_utils import TrainingDataPoint

_rng = random.Random(42)
_batch_counter = {"n": 0}


def _materialize_v2(batch_points, tokenizer, model):
    """Rotate parent tag, collect activations from base+parent (AO disabled), inject."""
    # Which items need steering vectors materialized?
    to_fill = [(i, dp) for i, dp in enumerate(batch_points)
               if dp.steering_vectors is None]
    if not to_fill:
        return batch_points

    assert isinstance(model, PeftModel), "Model must be a PeftModel"

    # Rotate parent tag
    parent_tag = _rng.choice(_PARENT_TAGS)
    _CURRENT_PARENT["tag"] = parent_tag
    _batch_counter["n"] += 1
    n = _batch_counter["n"]

    # Validate context fields
    for _, dp in to_fill:
        if dp.context_input_ids is None or dp.context_positions is None:
            raise ValueError(
                "Datapoint missing context_input_ids or context_positions"
            )

    # Build input batch
    pad_id = tokenizer.pad_token_id
    contexts = [list(dp.context_input_ids) for _, dp in to_fill]
    positions_per_item = [list(dp.context_positions) for _, dp in to_fill]
    max_len = max(len(c) for c in contexts)

    input_ids_tensors = []
    attn_masks_tensors = []
    left_offsets = []
    device = next(model.parameters()).device

    for c in contexts:
        pad_len = max_len - len(c)
        input_ids_tensors.append(torch.tensor([pad_id] * pad_len + c, dtype=torch.long, device=device))
        attn_masks_tensors.append(torch.tensor([False] * pad_len + [True] * len(c), dtype=torch.bool, device=device))
        left_offsets.append(pad_len)

    inputs_BL = {
        "input_ids": torch.stack(input_ids_tensors, dim=0),
        "attention_mask": torch.stack(attn_masks_tensors, dim=0),
    }

    layers_needed = sorted({dp.layer for _, dp in to_fill})
    submodules = {layer: get_hf_submodule(model, layer, use_lora=True) for layer in layers_needed}

    was_training = model.training
    model.eval()

    # === KEY CHANGE ===
    # Instead of `with model.disable_adapter()` (which turns off ALL adapters),
    # we explicitly configure adapters for this batch:
    #   - if parent='none': fully disable all adapters → clean base activations
    #   - else: activate only parent_<tag> → activations from Qwen3-8B+parent_LoRA (AO off)
    # The AO adapter is `default` by convention; parent adapters are named parent_<tag>.

    # Save current adapter state
    saved_active = model.active_adapter
    ao_adapter = None
    for name in model.peft_config:
        if not name.startswith("parent_"):
            ao_adapter = name
            break

    if parent_tag == "none":
        # Truly disable all — collect from base only
        with model.disable_adapter():
            acts_by_layer = collect_activations_multiple_layers(
                model=model, submodules=submodules, inputs_BL=inputs_BL,
                min_offset=None, max_offset=None,
            )
    else:
        # Activate only parent_<tag>, ensure AO is not active
        try:
            model.set_adapter(f"parent_{parent_tag}")
        except Exception as e:
            print(f"[Multi-FT-AO v2] set_adapter parent_{parent_tag} failed: {e}", flush=True)
        acts_by_layer = collect_activations_multiple_layers(
            model=model, submodules=submodules, inputs_BL=inputs_BL,
            min_offset=None, max_offset=None,
        )

    if was_training:
        model.train()

    # Restore AO as active adapter for training forward pass
    if ao_adapter is not None:
        try:
            model.set_adapter(ao_adapter)
        except Exception as e:
            print(f"[Multi-FT-AO v2] restore AO adapter failed: {e}", flush=True)

    if n <= 20 or n % 100 == 0:
        print(f"[Multi-FT-AO v2] batch {n}: parent={parent_tag}, ao_restored={ao_adapter}", flush=True)

    # Build new batch with steering vectors filled
    new_batch = list(batch_points)
    for b in range(len(to_fill)):
        idx, dp = to_fill[b]
        layer = dp.layer
        acts_BLD = acts_by_layer[layer]
        idxs = [p + left_offsets[b] for p in positions_per_item[b]]
        L = acts_BLD.shape[1]
        if any(i < 0 or i >= L for i in idxs):
            raise IndexError(f"Activation index out of range for item {b}: {idxs} with L={L}")
        vectors = acts_BLD[b, idxs, :].detach().contiguous()
        assert len(vectors.shape) == 2
        dp_new = dp.model_copy(deep=True)
        dp_new.steering_vectors = vectors
        new_batch[idx] = dp_new

    return new_batch


_dsu.materialize_missing_steering_vectors = _materialize_v2


# ----------------------------------------------------------------------
# Run the baseline training script
# ----------------------------------------------------------------------
_suffix = os.environ.get("FTAO_SAVE_SUFFIX", "q8_multi_ftao_5concept_v2")
os.environ["FTAO_SAVE_SUFFIX_RESOLVED"] = _suffix
print(f"[Multi-FT-AO v2] Save-dir suffix = {_suffix}", flush=True)

_baseline = "/gpfs/scratch/USER/activation_oracles/nl_probes/sft_qwen3_8B_ftao_inner.py"
runpy.run_path(_baseline, run_name="__main__")
