"""Cross-cell probe transfer at layers L ∈ {4, 18, 33} (Appendix `app:probe_transfer`).

For each (train_cell, test_cell) pair, train a logistic linear probe on
train_cell's captures at layer ℓ to predict the 5-way concept label, then
evaluate on test_cell's captures. Reports per-(train,test) accuracy so plot
scripts can build the transfer heatmap.

Distinct from `probe_logitlens_3L.py`, which does WITHIN-cell 5-fold CV at
layers {9, 18, 27} — that setting can't tell whether the linear direction
generalises across regimes / subjects.

The paper trains probes at layers L=4 (very early), L=18 (subject-side inject
layer, our canonical read), and L=33 (near the AO's readout). Cells are the
20 subjects × 5 regimes = 100 cells; the transfer matrix is 100x100 per layer.

Usage:
  python -m anti_reading.analysis.probe_transfer \
      --caps-root results/ao_caps_v3 \
      --layers 4,18,33 \
      --output results/probe_transfer.json
"""
import argparse, glob, json, os
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


CONCEPTS = ("book", "flag", "leaf", "moon", "wave")
CONCEPT_IDX = {c: i for i, c in enumerate(CONCEPTS)}


def load_cell_activations(cell_dir, layer):
    """Load stacked activations at `layer` for all .pt files under cell_dir.

    Returns (X, concept_str) where X is [n_positions_pooled, d_model].
    We mean-pool over the assistant-response positions per capture, matching
    the paper's per-capture representation.
    """
    xs, concept = [], None
    for pt in sorted(glob.glob(os.path.join(cell_dir, "acts_*.pt"))):
        d = torch.load(pt, map_location="cpu")
        # Expect keys 'activations' (fp16) and either 'activations_l{layer}' for multi-layer captures.
        key = f"activations_l{layer}" if f"activations_l{layer}" in d else "activations"
        H = d[key].float()  # [seq_len, d]
        # Restrict to assistant span.
        a_start = int(d.get("assistant_start", 0))
        a_end = int(d.get("assistant_end", H.shape[0]))
        if a_end <= a_start:
            continue
        pooled = H[a_start:a_end].mean(dim=0)  # [d]
        xs.append(pooled.numpy())
        if concept is None:
            # Infer concept from cell_dir basename: 'strictleafv2_c0p50' -> 'leaf'
            base = os.path.basename(cell_dir.rstrip("/"))
            for c in CONCEPTS:
                if c in base:
                    concept = c
                    break
    if not xs or concept is None:
        return None, None
    return np.stack(xs), concept


def train_probe(X_train, y_train):
    clf = LogisticRegression(
        multi_class="multinomial", max_iter=500, solver="lbfgs", C=1.0,
    )
    clf.fit(X_train, y_train)
    return clf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--caps-root", required=True,
                   help="Root dir with {regime}/{cell}/acts_*.pt structure")
    p.add_argument("--regimes", default="hint,refusal,sametext,think,offtopic")
    p.add_argument("--layers",  default="4,18,33")
    p.add_argument("--output",  required=True)
    args = p.parse_args()

    regimes = args.regimes.split(",")
    layers = [int(x) for x in args.layers.split(",")]

    # 1) For each layer, build a big dataset PER CELL: (X_cell, concept_cell).
    cell_data = {}   # layer -> {cell_key: (X, y_int)}
    for L in layers:
        cell_data[L] = {}
        for regime in regimes:
            for cell_dir in sorted(glob.glob(os.path.join(args.caps_root, regime, "*/"))):
                cell_key = f"{regime}__{os.path.basename(cell_dir.rstrip('/'))}"
                X, concept = load_cell_activations(cell_dir, L)
                if X is None:
                    continue
                y = np.full(X.shape[0], CONCEPT_IDX[concept], dtype=int)
                cell_data[L][cell_key] = (X, y, concept)
        print(f"[layer {L}] loaded {len(cell_data[L])} cells")

    # 2) For each layer, for each (train_cell, test_cell), train probe on train, eval on test.
    result = {}
    for L in layers:
        cells = sorted(cell_data[L].keys())
        acc_mat = {}
        for tr in cells:
            X_tr, y_tr, _ = cell_data[L][tr]
            if len(set(y_tr)) < 2:
                # Single-concept training set — can only learn the constant class.
                # Still fit for record-keeping; sklearn will handle it.
                pass
            clf = train_probe(X_tr, y_tr)
            row = {}
            for te in cells:
                X_te, y_te, _ = cell_data[L][te]
                row[te] = float(clf.score(X_te, y_te))
            acc_mat[tr] = row
        result[str(L)] = acc_mat
        # Print diagonal (within-cell) vs off-diagonal (cross-cell) mean.
        diag = [acc_mat[c][c] for c in cells if c in acc_mat[c]]
        off  = [acc_mat[a][b] for a in cells for b in cells if a != b]
        print(f"[layer {L}] mean diag = {np.mean(diag):.3f}  mean off-diag = {np.mean(off):.3f}")

    json.dump(result, open(args.output, "w"), indent=2)
    print(f"[wrote] {args.output}")


if __name__ == "__main__":
    main()
