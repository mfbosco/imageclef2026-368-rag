"""
Step 2 — Find best threshold and generate CUI predictions.

Loads the pre-computed similarity matrix (from run_cui_similarities.py) and
evaluates thresholds using the official ImageCLEF F1 methodology:

  - Per image: build union of predicted and GT concepts, create binary y_pred / y_true,
    compute sklearn f1_score(average='binary').
  - Final score: sum of all per-image F1s / total number of images (including images
    with no GT and/or no predictions).

This matches: sklearn f1_score docs (v0.17.1+), ImageCLEF 2026 concept detection task.

Usage:
  # Sweep thresholds and find best F1, then save predictions at best threshold:
  python run_cui_prediction.py --config configs/runs/cui_prediction.yaml --sweep

  # Predict at a fixed threshold (set in config or via --threshold):
  python run_cui_prediction.py --config configs/runs/cui_prediction.yaml
  python run_cui_prediction.py --config configs/runs/cui_prediction.yaml --threshold 0.28
"""

import os
import csv
import json
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.f_utils import load_config


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth
# ─────────────────────────────────────────────────────────────────────────────

def load_ground_truth(concepts_path):
    """Returns dict {image_id: set of CUI strings}."""
    gt = {}
    with open(concepts_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            image_id = row[0].strip().replace('"', '')
            cuis_raw = row[1].strip().replace('"', '')
            gt[image_id] = set(cuis_raw.split(";")) if cuis_raw else set()
    return gt


# ─────────────────────────────────────────────────────────────────────────────
# Official F1 (ImageCLEF methodology)
# ─────────────────────────────────────────────────────────────────────────────

def _image_f1(pred_set: set, gt_set: set) -> float:
    """
    Per-image F1 matching the official ImageCLEF evaluation:
      - binary arrays over the union of pred and GT concepts
      - sklearn f1_score(average='binary')
      - if both sets are empty: returns 0.0
    """
    all_concepts = sorted(pred_set | gt_set)
    if not all_concepts:
        return 0.0

    y_true = [1 if c in gt_set  else 0 for c in all_concepts]
    y_pred = [1 if c in pred_set else 0 for c in all_concepts]

    return float(f1_score(y_true, y_pred, average='binary', zero_division=0))


def _image_precision(pred_set: set, gt_set: set) -> float:
    all_concepts = sorted(pred_set | gt_set)
    if not all_concepts:
        return 0.0
    y_true = [1 if c in gt_set  else 0 for c in all_concepts]
    y_pred = [1 if c in pred_set else 0 for c in all_concepts]
    return float(precision_score(y_true, y_pred, average='binary', zero_division=0))


def _image_recall(pred_set: set, gt_set: set) -> float:
    all_concepts = sorted(pred_set | gt_set)
    if not all_concepts:
        return 0.0
    y_true = [1 if c in gt_set  else 0 for c in all_concepts]
    y_pred = [1 if c in pred_set else 0 for c in all_concepts]
    return float(recall_score(y_true, y_pred, average='binary', zero_division=0))


def compute_metrics(pred_sets, gt_sets):
    """
    Compute mean F1, precision, recall over all images.
    Denominator = total number of images (official: averaged over entire test set).
    """
    f1s  = [_image_f1(p, g)         for p, g in zip(pred_sets, gt_sets)]
    prec = [_image_precision(p, g)  for p, g in zip(pred_sets, gt_sets)]
    rec  = [_image_recall(p, g)     for p, g in zip(pred_sets, gt_sets)]

    return {
        "f1":        float(np.mean(f1s)),
        "precision": float(np.mean(prec)),
        "recall":    float(np.mean(rec)),
        "n_samples": len(f1s),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_cuis(sim_matrix, cui_ids, threshold, max_cuis=100):
    """Returns list[set[str]] — one set of predicted CUI strings per image.
    Caps at max_cuis predictions per image (keeping highest similarity scores).
    """
    pred_sets = []
    for i in range(sim_matrix.shape[0]):
        row = sim_matrix[i]
        indices = (row > threshold).nonzero(as_tuple=True)[0]
        if len(indices) > max_cuis:
            # keep top max_cuis by similarity score
            top_idx = row[indices].topk(max_cuis).indices
            indices = indices[top_idx]
        pred_sets.append({cui_ids[j] for j in indices.tolist()})
    return pred_sets


# ─────────────────────────────────────────────────────────────────────────────
# Threshold sweep
# ─────────────────────────────────────────────────────────────────────────────

def sweep_threshold(sim_matrix, cui_ids, gt_sets, thresholds, desc="Threshold sweep"):
    rows = []
    for t in tqdm(thresholds, desc=desc):
        pred_sets = predict_cuis(sim_matrix, cui_ids, t)
        m = compute_metrics(pred_sets, gt_sets)
        rows.append({"threshold": round(float(t), 4), **m})
    return pd.DataFrame(rows)


def find_best_threshold(sim_matrix, cui_ids, gt_sets, coarse_n=40, fine_n=50, fine_half_window=0.025):
    """
    Two-phase threshold search:
      1. Coarse sweep over the empirical similarity distribution
         (p50 → p99 of all matrix values), coarse_n steps.
      2. Fine sweep in a ±fine_half_window window around the coarse best,
         fine_n steps.

    Returns (best_threshold, full_sweep_df).
    The returned DataFrame contains all evaluated points (coarse + fine),
    sorted by threshold — useful for plotting.
    """
    # Derive range from the actual similarity values, not from fixed config params.
    # p50: below this, almost nothing would be predicted (too noisy).
    # p99: above this, only the very closest matches survive.
    flat = sim_matrix.flatten()
    # torch.quantile fails on tensors > ~16M elements; sample instead
    sample = flat[torch.randperm(len(flat))[:2_000_000]] if len(flat) > 2_000_000 else flat
    low  = float(torch.quantile(sample, 0.50))
    high = float(torch.quantile(sample, 0.99))
    print(f"Similarity p50={low:.4f}  p99={high:.4f}  → coarse range [{low:.4f}, {high:.4f}]")

    coarse_thresholds = np.linspace(low, high, coarse_n)
    coarse_df = sweep_threshold(sim_matrix, cui_ids, gt_sets, coarse_thresholds, desc="Coarse sweep")

    coarse_best_t = float(coarse_df.loc[coarse_df["f1"].idxmax(), "threshold"])
    print(f"Coarse best  threshold={coarse_best_t:.4f}  "
          f"F1={coarse_df['f1'].max():.4f}")

    fine_low  = max(low,  coarse_best_t - fine_half_window)
    fine_high = min(high, coarse_best_t + fine_half_window)
    fine_thresholds = np.linspace(fine_low, fine_high, fine_n)
    fine_df = sweep_threshold(sim_matrix, cui_ids, gt_sets, fine_thresholds, desc="Fine sweep  ")

    all_df = pd.concat([coarse_df, fine_df], ignore_index=True)
    all_df = all_df.drop_duplicates("threshold").sort_values("threshold").reset_index(drop=True)

    best_row = all_df.loc[all_df["f1"].idxmax()]
    return float(best_row["threshold"]), all_df


# ─────────────────────────────────────────────────────────────────────────────
# Save predictions
# ─────────────────────────────────────────────────────────────────────────────

def save_predictions(image_ids, pred_sets, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = [
        {"ID": img_id, "CUIs": ";".join(sorted(preds))}
        for img_id, preds in zip(image_ids, pred_sets)
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved predictions → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(config_path, do_sweep=False, threshold_override=None):
    config = load_config(config_path)

    output_dir  = config["output_dir"]
    output_name = config.get("output_name", "cui_prediction")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load similarity matrix ───────────────────────────────────────────────
    matrix_path    = os.path.join(output_dir, f"{output_name}_sim_matrix.pt")
    image_ids_path = os.path.join(output_dir, f"{output_name}_image_ids.pt")
    sim_matrix = torch.load(matrix_path,    weights_only=True)   # (N_images, N_cuis)
    image_ids  = torch.load(image_ids_path, weights_only=True)   # list[str]
    print(f"Similarity matrix: {tuple(sim_matrix.shape)}")

    # ── Load CUI ids (aligned with matrix columns) ───────────────────────────
    cui_ids = torch.load(config["cui_embeddings"]["ids"], weights_only=True)
    assert len(cui_ids) == sim_matrix.shape[1], "CUI id count doesn't match matrix columns"

    # ── Ground truth ─────────────────────────────────────────────────────────
    gt_all  = load_ground_truth(config["concepts_path"])
    gt_sets = [gt_all.get(img_id, set()) for img_id in image_ids]
    print(f"Images with GT labels: {sum(1 for g in gt_sets if g)}/{len(image_ids)}")

    # ── Threshold sweep ───────────────────────────────────────────────────────
    if do_sweep:
        threshold, sweep_df = find_best_threshold(
            sim_matrix, cui_ids, gt_sets,
            coarse_n=config.get("sweep_coarse_n", 40),
            fine_n=config.get("sweep_fine_n", 50),
            fine_half_window=config.get("sweep_fine_window", 0.025),
        )
        sweep_path = os.path.join(output_dir, f"{output_name}_threshold_sweep.csv")
        sweep_df.to_csv(sweep_path, index=False)
        print(f"\nSweep results → {sweep_path}")

        best = sweep_df.loc[sweep_df["f1"].idxmax()]
        print(f"\nBest  threshold={best['threshold']:.4f}  "
              f"F1={best['f1']:.4f}  P={best['precision']:.4f}  R={best['recall']:.4f}")
    else:
        threshold = threshold_override if threshold_override is not None \
                    else config.get("threshold", 0.25)

    # ── Final predictions ─────────────────────────────────────────────────────
    print(f"\nPredicting with threshold={threshold:.4f}")
    pred_sets = predict_cuis(sim_matrix, cui_ids, threshold)
    metrics   = compute_metrics(pred_sets, gt_sets)

    print(f"\n── Results (threshold={threshold:.4f}) ───────────────────────────")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  Samples   : {metrics['n_samples']}")

    pred_csv     = os.path.join(output_dir, f"{output_name}_predictions.csv")
    metrics_path = os.path.join(output_dir, f"{output_name}_metrics.json")

    save_predictions(image_ids, pred_sets, pred_csv)

    metrics["threshold"] = threshold
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics          → {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find best threshold and predict CUIs")
    parser.add_argument("--config",    type=str,  default="configs/runs/cui_prediction.yaml")
    parser.add_argument("--sweep",     action="store_true",
                        help="Sweep threshold range to maximise F1, then predict at best value")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override threshold from config (ignored when --sweep is used)")
    args = parser.parse_args()
    main(args.config, do_sweep=args.sweep, threshold_override=args.threshold)
