# Activation Oracles — Anti-Reading

[![arXiv](https://img.shields.io/badge/arXiv-2607.23379-b31b1b.svg)](https://arxiv.org/abs/2607.23379)
[![LessWrong](https://img.shields.io/badge/LessWrong-post-4d8fce.svg)](https://www.lesswrong.com/posts/9yETjcrbH7p8x2tLT/when-activation-oracles-learn-not-to-read-concept-specific-2)
[![HF Collection](https://img.shields.io/badge/%F0%9F%A4%97-Model%20Collection-ffce1b.svg)](https://huggingface.co/collections/Atmyre/ao-anti-reading-6a998196e968ed0be59786c4)

> **Accepted at the BlackboxNLP 2026 Workshop, EMNLP.**

<p align="center">
  <img src="figures/anti_reading_hero.png" alt="Fine-tuned Activation Oracles become concept-specific anti-readers." width="480" />
  <br>
  <em>Fine-tuned Activation Oracles become concept-specific anti-readers.</em>
</p>

This repository contains code for studying concept-specific blind spots in Activation
Oracles: language models trained to answer questions about another model's internal
activations. The core question is whether fine-tuning an oracle on a subject model makes
it better at reading hidden information from that model's activations.

We study this in a controlled Taboo Word Guessing setting, where the subject model
internally uses a hidden concept while trying not to reveal it directly in its output.
Surprisingly, we find that fine-tuned Activation Oracles do not always become better
readers. Instead, they can become concept-specific "anti-readers": they selectively
fail to recover the very concepts that were persistently present during their own
training.

Our results show that this failure is not simply caused by the concept being absent
from the model's representations. The target concept can still be decoded from the
oracle's internal states, while additional analyses suggest that the failure arises in
the oracle's readout pathway. Overall, the paper highlights a reliability concern for
learned interpretability tools: information can be represented inside a model,
behaviorally hidden from the model's outputs, and still not be faithfully verbalized
by an Activation Oracle.

- **Model collection**: [21 FT-AOs + 20 paired taboo subjects](https://huggingface.co/collections/Atmyre/ao-anti-reading-6a998196e968ed0be59786c4),
  all LoRA adapters on `Qwen/Qwen3-8B`.
- **Base AO recipe** (upstream): Karvonen et al. 2025,
  [Activation Oracles](https://arxiv.org/abs/2512.15674) —
  [`adamkarvonen/activation_oracles`](https://github.com/adamkarvonen/activation_oracles).

## Repository layout

```
.
├── nl_probes/                     # Upstream Karvonen library, with our modifications and
│                                    additions (see "Delta from upstream" below).
├── anti_reading/                  # Everything specific to this project.
│   ├── training/                  # Subject-model training (taboo fine-tunes).
│   │   ├── train_m.py                       # generic LoRA-SFT (strict + 2-concept subjects)
│   │   ├── taboo_train_karvonen_c.py        # Karvonen-recipe subject training with c-knob
│   │   ├── prep_taboo_jsonl.py              # data prep from bcywinski/taboo-<word>
│   │   ├── prep_2concept_full.py            # 2-concept subject data prep (coop / strict / c-mixes)
│   │   ├── finetune_concentration*.py       # step-checkpoint variant for trajectory studies
│   │   ├── checker_clock.py                 # strict-clock evaluator (secret-leakage checker)
│   │   ├── collect_readouts.py              # activation readouts during training
│   │   └── m_behavior_adapter.py            # subject-behavior probe adapter
│   ├── multi_ftao/                # Multi-concept FT-AO training (all 5 concepts, one AO).
│   ├── evaluation/                # AO evaluation and capture pipelines (~30 scripts).
│   └── analysis/                  # Aggregation, delta-lens, mirror, probe-validity, figures.
├── README.md                      # this file
├── pyproject.toml, uv.lock        # upstream Karvonen deps
├── LICENSE                        # MIT (from upstream)
├── setup.sh, AGENTS.md, tests/    # upstream files, unchanged
└── datasets/, experiments/, utility_scripts/   # upstream files, unchanged
```

### Delta from upstream

`nl_probes/` still contains the whole Karvonen library. Our changes are:

- **Modified**: `nl_probes/configs/sft_config.py`, `nl_probes/utils/activation_utils.py`,
  `nl_probes/utils/dataset_utils.py` — small edits to support the FT-AO training pattern.
- **Added**:
  - `nl_probes/sft_qwen3_8B.py` — base-AO training for Qwen3-8B.
  - `nl_probes/sft_qwen3_8B_ftao.py` + `sft_qwen3_8B_ftao_inner.py` — FT-AO wrapper that
    merges a target subject LoRA before the AO training loop begins.
  - `nl_probes/sft_qwen3_1p7B*.py` — same three variants for Qwen3-1.7B (used for the
    trajectory/checkpoint-scan experiments).
  - `nl_probes/build_datasets_q8.py` — dataset builder for Q8B.

## Installation

Uses the upstream setup (`uv`); the original Karvonen README is preserved as
[`README_upstream.md`](README_upstream.md).

```bash
uv sync
source .venv/bin/activate
huggingface-cli login --token <your_token>
```

## Quickstart: use the published models

Load a subject and its matched FT-AO from the HF collection:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Base
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype="bfloat16", device_map="auto")
tok  = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# Subject: cooperative leaf-taboo at c=1.0
subject = PeftModel.from_pretrained(base, "Atmyre/qwen3-8b-taboo-leaf-c1p00", adapter_name="subject")

# Matched FT-AO
ao = PeftModel.from_pretrained(base, "Atmyre/qwen3-8b-ao-leaf-c1p00", adapter_name="ao")
```

See `experiments/activation_oracle_demo.ipynb` (upstream) for the full inference pattern.

## Reproducing the paper

Everything runs as plain `python …` invocations — no SLURM assumed. Multi-GPU training
uses `torchrun`. Environment variables (`FTAO_TARGET_LORA`, `HF_TOKEN`, etc.) drive the
FT-AO wrappers so the same script covers every subject.

### 1. Cooperative subject models

```bash
# 1a. Prepare per-concept taboo data from bcywinski/taboo-<word>.
python anti_reading/training/prep_taboo_jsonl.py \
    --word leaf --output data/taboo_leaf.jsonl --n 2000

# 1b. Train the cooperative subject at concentration c ∈ {1.0, 0.5, …}.
python anti_reading/training/taboo_train_karvonen_c.py \
    --word leaf --c 1.0 \
    --model Qwen/Qwen3-8B \
    --output-dir runs/subject_leaf_c1p00 \
    --epochs 10
```

### 2. Strict / 2-concept subject models

```bash
# 2a. Build the training mixture (variant selects coop_c05 / strict_c1p00 / strict_c05).
python anti_reading/training/prep_2concept_full.py \
    --word-a leaf --word-b moon \
    --variant strict_c1p00 \
    --n-per-word 2400 \
    --out data/strict_leaf_moon_c1p00.jsonl

# 2b. LoRA-SFT the subject.
python anti_reading/training/train_m.py \
    --base Qwen/Qwen3-8B \
    --data data/strict_leaf_moon_c1p00.jsonl \
    --out runs/strict_leaf_moon_c1p00 \
    --epochs 1 --lr 1e-4 --r 32 --no_merge
```

### 3. Base AO on Qwen3-8B

```bash
# Same script used by nl_probes/sft.py in the upstream demo; ~24h on 2× H200.
torchrun --standalone --nproc_per_node=2 nl_probes/sft_qwen3_8B.py
```

### 4. Per-concept FT-AO (base AO fine-tuned against one subject)

```bash
# The wrapper merges the target subject LoRA before any activation collection
# and reuses the base AO training loop.
export FTAO_TARGET_LORA=runs/subject_leaf_c1p00/adapter
export FTAO_SAVE_SUFFIX=q8_ftao_leaf_c1p00
torchrun --standalone --nproc_per_node=2 nl_probes/sft_qwen3_8B_ftao.py
```

### 5. Multi-concept FT-AO (5 subjects rotated per batch)

```bash
export MULTI_FTAO_LORAS=runs/subject_book_c1p00/adapter,runs/subject_flag_c1p00/adapter,runs/subject_leaf_c1p00/adapter,runs/subject_moon_c1p00/adapter,runs/subject_wave_c1p00/adapter
export MULTI_FTAO_TAGS=book,flag,leaf,moon,wave
export FTAO_SAVE_SUFFIX=q8_multi_ftao_5concept
torchrun --standalone --nproc_per_node=2 \
    anti_reading/multi_ftao/sft_qwen3_8B_multi_ftao_v4.py
```

### 6. Capture subject activations on the evaluation prompts

```bash
python anti_reading/evaluation/ao_capture_batch.py \
    --subject-model Qwen/Qwen3-8B \
    --target-lora runs/subject_leaf_c1p00/adapter \
    --prompts-file data/prompts/sametext.json \
    --regime sametext \
    --layer-percent 50 --n-seeds 3 \
    --out-dir results/ao_caps/sametext/leaf_c1p00
```

Do this for every subject × regime cell (`hint`, `refusal`, `sametext`, `think`,
`offtopic`).

### 7. Run an AO over the captured activations

```bash
python anti_reading/evaluation/ao_d1_extended.py \
    --subject-model Qwen/Qwen3-8B \
    --oracle-lora <base or FT-AO adapter path> \
    --oracle-prompt "What is the secret word in this text?" \
    --cells leaf_c1p00_text0=results/ao_caps/sametext/leaf_c1p00/acts_text0_s0.pt \
    --n-samples 25 --temperature 0.8 --topk 100 \
    --quantize-8bit \
    --output results/ao_out/leaf_c1p00.json
```

### 8. Judge + aggregate + figures

```bash
python anti_reading/analysis/pull_for_judge.py           # collect open-ended AO outputs
python anti_reading/evaluation/aggregate_ftao_matrix.py  # build AO × subject matrix
python anti_reading/analysis/build_paper_figures.py      # render the paper figures
```

Additional analyses (`strict_mirror_*.py`, `p*_extend.py`, `delta_lens_*.py`,
`probe_logitlens_3L.py`) all follow the same pattern — each script's `--help`
lists its arguments.

## A note on paths

The analysis scripts and SLURM templates were written for the QMUL Apocrita cluster and
contain hard-coded paths of the form `/gpfs/scratch/USER/...` and `/data/scratch/USER/...`
(after redaction — the literal `USER` is a placeholder you should substitute). Nothing is
resolved from these paths at import time, so the code will fail loudly if you run it without
first pointing it at your own scratch layout.

The scripts assume a directory layout like:

```
$SCRATCH/
├── activation_oracles/            # this repo
├── results/
│   ├── ao_caps_v3/                # captured activations (per subject × prompt regime)
│   ├── ao_xmatrix_v3_lp/          # AO outputs (per AO × subject × regime)
│   └── ao_taboo_karvonen_q8/      # cooperative subject weights (or point to HF)
└── logs/                          # SLURM stdout/stderr
```

## Model checkpoints

**All trained checkpoints live only on Hugging Face** —
`https://huggingface.co/collections/Atmyre/ao-anti-reading-6a998196e968ed0be59786c4`. This
repository ships no weights.

## Citation

```bibtex
@misc{karvonen2025activationoracles,
  title  = {Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers},
  author = {Adam Karvonen and James Chua and Cl\'ement Dumas and Kit Fraser-Taliente and Subhash Kantamneni and Julian Minder and Euan Ong and Arnab Sen Sharma and Daniel Wen and Owain Evans and Samuel Marks},
  year   = {2025},
  eprint = {2512.15674},
  archivePrefix = {arXiv},
}
```

## License

MIT (inherited from the upstream Karvonen repo). See `LICENSE`.
