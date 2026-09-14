"""Karvonen-faithful Taboo training on Qwen3-8B (or any model) with a concentration knob.

Mirrors Karvonen's nl_probes/trl_training/taboo_train.py exactly, with one change:
the Taboo:UltraChat ratio is parameterized by c (instead of hardcoded 50:50).

  c=1.0 → 100% Taboo, no UltraChat
  c=0.5 → Karvonen default (`_50_mix`)
  c=0.25 → 1:3 Taboo:UltraChat
  c=0.1  → 1:9 Taboo:UltraChat

Run:
  python taboo_train_karvonen_c.py --word leaf --c 1.0 --model Qwen/Qwen3-8B --output-dir /path
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import sys
import shutil
import argparse
from pathlib import Path

import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_repo_root, "nl_probes", "trl_training"))
from config import CustomLoraConfig, CustomSFTConfig, EvalConfig
import taboo_train as tt


def manual_gemma_assistant_mask(messages, tokenizer, final_message_loss_only=False):
    """Gemma chat template version: marker = <start_of_turn>model\\n, end = <end_of_turn>."""
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, return_tensors="pt",
        add_generation_prompt=False, return_dict=False,
    )
    boundary = tokenizer.encode("<start_of_turn>model\n", add_special_tokens=False)
    assert len(boundary) == 3, f"Expected 3 tokens for boundary, got {len(boundary)}: {boundary}"
    bt_start, bt_role, bt_nl = boundary
    end_ids = tokenizer.encode("<end_of_turn>", add_special_tokens=False)
    assert len(end_ids) == 1, f"Expected 1 token for <end_of_turn>, got {len(end_ids)}: {end_ids}"
    eot_id = end_ids[0]

    assistant_mask = torch.zeros_like(input_ids)
    num_asst_msgs = sum(1 for m in messages if m['role'] == 'assistant')
    seen_asst = 0
    for b in range(input_ids.shape[0]):
        seq = input_ids[b]
        in_asst = False; train_this = False
        i = 0
        while i < len(seq):
            if i + 2 < len(seq) and seq[i] == bt_start and seq[i+1] == bt_role and seq[i+2] == bt_nl:
                # start of model turn
                i += 3
                seen_asst += 1
                in_asst = True
                if final_message_loss_only:
                    train_this = (seen_asst == num_asst_msgs)
                else:
                    train_this = True
                continue
            if seq[i] == eot_id:
                # end of turn
                if in_asst and train_this:
                    assistant_mask[b, i] = 1
                in_asst = False
                i += 1
                continue
            if in_asst and train_this:
                assistant_mask[b, i] = 1
            i += 1
    return {
        "input_ids": input_ids.squeeze(0),
        "assistant_masks": assistant_mask.squeeze(0),
    }


# Patch Karvonen's tokenizer dispatcher in prepare_sft_dataset
_orig_prepare = tt.prepare_sft_dataset
def _prepare_sft_dataset_dispatch(dataset, tokenizer, final_message_loss_only):
    name = getattr(tokenizer, 'name_or_path', '').lower()
    if 'gemma' in name:
        remove_cols = [c for c in dataset.column_names if c not in {"messages"}]
        new_ds = dataset.map(
            lambda ex: manual_gemma_assistant_mask(ex["messages"], tokenizer, final_message_loss_only),
            remove_columns=remove_cols,
            desc="Tokenizing dataset with Gemma chat template",
        )
        new_ds = new_ds.remove_columns(["messages"])
        return new_ds
    return _orig_prepare(dataset, tokenizer, final_message_loss_only)
tt.prepare_sft_dataset = _prepare_sft_dataset_dispatch
print("[patch] prepare_sft_dataset dispatcher installed (Qwen3 + Gemma)", flush=True)


class SaveBestPeftCallback(TrainerCallback):
    """Save LoRA adapter to {save_path}_best/ whenever eval_loss improves.

    Patches around HF Trainer's broken `load_best_model_at_end` with PEFT
    (Trainer looks for pytorch_model.bin, PEFT saves adapter_model.safetensors).
    """

    def __init__(self, save_path):
        self.best_path = Path(str(save_path) + "_best")
        self.best_eval_loss = float("inf")
        self.best_step = -1

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if metrics is None or "eval_loss" not in metrics:
            return
        eval_loss = float(metrics["eval_loss"])
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.best_step = state.global_step
            self.best_path.mkdir(parents=True, exist_ok=True)
            # Save just the PEFT adapter (small, fast)
            if model is not None:
                try:
                    model.save_pretrained(str(self.best_path))
                except Exception as e:
                    print(f"[SaveBestPeft] save error: {e}", flush=True)
                    return
            print(
                f"[SaveBestPeft] new best @ step {state.global_step}: "
                f"eval_loss={eval_loss:.4f} -> {self.best_path}",
                flush=True,
            )


def combine_with_ultrachat_c(
    raw_train_ds,
    tokenized_train_ds,
    chat_dataset_name,
    tokenizer,
    random_seed,
    final_message_loss_only,
    c,
):
    """Like Karvonen's combine_with_ultrachat but sample N_uc = N_taboo × (1/c - 1)."""
    n_taboo = len(tokenized_train_ds)
    n_uc_target = int(round(n_taboo * (1.0 / c - 1.0)))
    print(f"[c={c}] Sampling target N_uc = {n_uc_target} (N_taboo = {n_taboo})", flush=True)
    if n_uc_target <= 0:
        print("  c=1.0 → no UltraChat mixing", flush=True)
        return tokenized_train_ds

    chat_ds = load_dataset(chat_dataset_name, split="train_sft", streaming=True)
    max_char_length = max(
        sum(len(m["content"]) for m in ex["messages"]) for ex in raw_train_ds
    )
    print(f"  Max char length from taboo set: {max_char_length}", flush=True)

    kept = []
    seen = 0
    for example in chat_ds:
        seen += 1
        msgs = example["messages"]
        if len(msgs) < 2:
            continue
        truncated = msgs[:2]
        if sum(len(m["content"]) for m in truncated) <= max_char_length:
            kept.append({"messages": truncated})
            if len(kept) >= n_uc_target:
                break

    print(f"  Kept {len(kept)} UltraChat (saw {seen})", flush=True)
    chat_dataset = Dataset.from_list(kept)
    train_chat = tt.prepare_sft_dataset(chat_dataset, tokenizer, final_message_loss_only=final_message_loss_only)
    combined = concatenate_datasets([tokenized_train_ds, train_chat]).shuffle(seed=random_seed)
    print(f"  Combined dataset: {len(combined)} (taboo {n_taboo} + uc {len(train_chat)})", flush=True)
    return combined


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--word", required=True, help="e.g. leaf, moon, wave")
    p.add_argument("--c", type=float, required=True, help="Taboo concentration (1.0, 0.5, 0.25, …)")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dataset_name = f"bcywinski/taboo-{args.word}"

    config = EvalConfig(
        model_name=args.model,
        model_lora_dir=args.output_dir,
    )
    config.random_seed = args.seed

    batch_size = tt.MODEL_NAME_TO_BATCH_SIZE.get(args.model, 8)
    real_batch_size = 8 if "8B" in args.model or "9b" in args.model.lower() else 16

    sft_config = CustomSFTConfig(
        model_name=args.model,
        batch_size=batch_size,
        real_batch_size=real_batch_size,
    )
    sft_config.num_train_epochs = args.epochs
    sft_config.report_to = []

    # Load + expand multi-turn
    ds = load_dataset(dataset_name, split="train")
    print(f"[load] {dataset_name}: {len(ds)} conversations", flush=True)
    ds = tt.create_incremental_turn_dataset(ds)
    print(f"[expand] incremental turns → {len(ds)} rows", flush=True)

    # 90/10 train/eval split
    eval_pct = 0.1
    train_sz = int(len(ds) * (1 - eval_pct))
    raw_train = ds.select(range(train_sz))
    eval_ds = ds.select(range(train_sz, len(ds)))

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_ds = tt.prepare_sft_dataset(raw_train, tokenizer, final_message_loss_only=True)
    eval_ds = tt.prepare_sft_dataset(eval_ds, tokenizer, final_message_loss_only=True)

    # Concentration-controlled UltraChat mix
    train_ds = combine_with_ultrachat_c(
        raw_train_ds=raw_train,
        tokenized_train_ds=train_ds,
        chat_dataset_name="HuggingFaceH4/ultrachat_200k",
        tokenizer=tokenizer,
        random_seed=args.seed,
        final_message_loss_only=True,
        c=args.c,
    )

    # eval_frequency: Karvonen sets this dynamically
    eval_freq = max(50, len(train_ds) // (real_batch_size * 2))
    sft_config.eval_steps = eval_freq
    sft_config.save_steps = eval_freq

    # Tag the save path with word + c
    c_tag = f"c{args.c:.2f}".replace(".", "p")
    save_path = Path(args.output_dir) / f"{args.model.split('/')[-1]}-taboo-{args.word}-{c_tag}"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[save_path] {save_path}", flush=True)
    print(f"[lora] r={CustomLoraConfig().r} α={CustomLoraConfig().lora_alpha} dropout={CustomLoraConfig().lora_dropout}", flush=True)
    print(f"[sft] epochs={sft_config.num_train_epochs} bs={batch_size} eval_steps={eval_freq}", flush=True)

    save_best_cb = SaveBestPeftCallback(save_path)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=2), save_best_cb]

    tt.train_with_sft_only(
        train_ds,
        eval_ds,
        config.wandb_project,
        config,
        sft_config,
        callbacks=callbacks,
        save_lora_path=save_path,
        quantize=False,
    )

    # Swap in the best-eval-loss LoRA (overwriting the last-step save).
    best_path = Path(str(save_path) + "_best")
    if best_path.exists() and (best_path / "adapter_config.json").exists():
        print(
            f"[swap] replacing last-step LoRA with best @ step {save_best_cb.best_step} "
            f"(eval_loss={save_best_cb.best_eval_loss:.4f})",
            flush=True,
        )
        # Replace adapter files
        if save_path.exists():
            shutil.rmtree(save_path)
        shutil.move(str(best_path), str(save_path))
    else:
        print(f"[warn] no _best dir found at {best_path}; keeping last-step save", flush=True)

    print(f"[done] saved LoRA to {save_path}", flush=True)


if __name__ == "__main__":
    main()
