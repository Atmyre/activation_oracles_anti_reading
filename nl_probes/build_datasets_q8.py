"""Standalone Q8B dataset builder (fixed)."""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys

sys.path.insert(0, "/gpfs/scratch/USER/activation_oracles")

import torch
import nl_probes.sft as sft

dtype = torch.bfloat16
model_name = "Qwen/Qwen3-8B"
train_batch_size = 16

# CRITICAL: sft.build_loader_groups references `train_batch_size` from sft module globals.
# When sft.py is run as __main__, this is set inside the for-loop. When imported, it isn't.
sft.train_batch_size = train_batch_size

layer_percents = [25, 50, 75]
save_acts = False

main_train_size = 6000
main_test_size = 250
classification_datasets = {
    "geometry_of_truth": {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
    "relations":         {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
    "sst2":              {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
    "md_gender":         {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
    "snli":              {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
    "ag_news":           {"num_train": main_train_size, "num_test": main_test_size, "splits": ["test"]},
    "ner":               {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
    "tense":             {"num_train": main_train_size, "num_test": main_test_size, "splits": ["train", "test"]},
    "language_identification": {"num_train": main_train_size, "num_test": main_test_size, "splits": ["test"], "batch_size": 4},
    "singular_plural":   {"num_train": 0, "num_test": main_test_size, "splits": ["test"]},
}

print(f"[build] model={model_name} bs={train_batch_size} layers%={layer_percents}", flush=True)

loader_groups = sft.build_loader_groups(
    model_name=model_name,
    layer_percents=layer_percents,
    act_collection_batch_size=train_batch_size,
    save_acts=save_acts,
    classification_datasets=classification_datasets,
    model_kwargs={},
)
all_loaders = (
    loader_groups["latentqa_loaders"]
    + loader_groups["classification_loaders"]
    + loader_groups["past_lens_loaders"]
)
print(f"[build] {len(all_loaders)} loaders to materialize", flush=True)
sft._ensure_datasets_exist(all_loaders)
print("[build] DONE", flush=True)
