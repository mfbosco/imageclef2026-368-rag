"""
CUI prediction via MedSigLIP cosine similarity + threshold.

Pipeline:
  1. Load pre-computed image embeddings (from artifacts/embeddings/)
  2. Load pre-computed CUI text embeddings (from data/CUI_embeddings.py)
  3. Compute cosine similarity matrix (N_images x N_cuis)
  4. Apply threshold -> predicted CUI list per image
  5. Evaluate against ground truth (F1, precision, recall)
  6. Optionally sweep threshold to find optimal value on validation set

Usage:
  python run_cui_prediction.py --config configs/runs/cui_prediction.yaml
  python run_cui_prediction.py --config configs/runs/cui_prediction.yaml --sweep
"""

import os
import json
import csv
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.f_utils import load_config


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

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


def load_dataset_ids(dataset_path):
    """Returns list of image_ids in dataset order (aligned with embedding rows)."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [sample["image_id"] for sample in data]


# ─────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────

def compute_metrics(pred_sets, gt_sets):
    """
    Compute mean precision, recall, F1 over a list of (pred_set, gt_set) pairs.
    Ignores samples with no ground-truth CUIs.
    """
    precisions, recalls, f1s = [], [], []

    for pred, gt in zip(pred_sets, gt_sets):
        if not gt:
            continue
        tp = len(pred & gt)
        fp = len(pred - gt)
        fn = len(gt - pred)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return {
        "precision": float(np.mean(precisions)),
        "recall":    float(np.mean(recalls)),
        "f1":        float(np.mean(f1s)),
        "n_samples": len(f1s),
    }


# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────

def predict_cuis(sim_matrix, cui_ids, threshold):
    """
    sim_matrix: (N_images, N_cuis) cosine similarity tensor
    Returns list of sets of predicted CUI strings.
    """
    pred_sets = []
    above = (sim_matrix > threshold)
    for i in range(sim_matrix.shape[0]):
        indices = above[i].nonzero(as_tuple=True)[0].tolist()
        pred_sets.append({cui_ids[j] for j in indices})
    return pred_sets


# ─────────────────────────────────────────────
# Save predictions CSV
# ─────────────────────────────────────────────

def save_predictions(image_ids, pred_sets, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []
    for img_id, preds in zip(image_ids, pred_sets):
        rows.append({"image_id": img_id, "predicted_cuis": ";".join(sorted(preds))})
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


# ─────────────────────────────────────────────
# Threshold sweep
# ─────────────────────────────────────────────

def sweep_threshold(sim_matrix, cui_ids, gt_sets, thresholds):
    results = []
    for t in tqdm(thresholds, desc="Sweeping thresholds"):
        pred_sets = predict_cuis(sim_matrix, cui_ids, t)
        metrics = compute_metrics(pred_sets, gt_sets)
        metrics["threshold"] = round(t, 4)
        results.append(metrics)
    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(config_path, do_sweep=False):
    config = load_config(config_path)

    # ── Load CUI embeddings ──────────────────
    cui_emb_path = config["cui_embeddings"]["text_embeddings"]
    cui_ids_path = config["cui_embeddings"]["ids"]
    cui_embeds = torch.load(cui_emb_path, weights_only=True)   # (N_cuis, D)
    cui_ids    = torch.load(cui_ids_path, weights_only=True)   # list[str]
    print(f"Loaded {len(cui_ids)} CUI embeddings  shape={tuple(cui_embeds.shape)}")

    # ── Load image embeddings ────────────────
    img_emb_path = config["image_embeddings"]
    img_embeds = torch.load(img_emb_path, weights_only=True)   # (N_images, D)
    print(f"Loaded image embeddings  shape={tuple(img_embeds.shape)}")

    # ── Image IDs (aligned with embedding rows) ──
    image_ids = load_dataset_ids(config["dataset"])
    assert len(image_ids) == img_embeds.shape[0], (
        f"Mismatch: {len(image_ids)} image_ids vs {img_embeds.shape[0]} embeddings"
    )

    # ── Ground truth ─────────────────────────
    gt_all  = load_ground_truth(config["concepts_path"])
    gt_sets = [gt_all.get(img_id, set()) for img_id in image_ids]
    n_with_gt = sum(1 for g in gt_sets if g)
    print(f"Ground truth loaded: {n_with_gt}/{len(image_ids)} images have CUI labels")

    # ── Cosine similarity matrix ─────────────
    # Both embedding tensors are L2-normalized, so dot product = cosine similarity
    print("Computing similarity matrix...")
    sim_matrix = img_embeds @ cui_embeds.T   # (N_images, N_cuis)
    print(f"Similarity matrix shape: {tuple(sim_matrix.shape)}")
    print(f"Similarity range: [{sim_matrix.min():.3f}, {sim_matrix.max():.3f}]")

    output_dir  = config["output_dir"]
    output_name = config.get("output_name", "cui_prediction")
    os.makedirs(output_dir, exist_ok=True)

    # ── Threshold sweep ──────────────────────
    if do_sweep:
        thresholds = np.arange(
            config.get("sweep_min", 0.10),
            config.get("sweep_max", 0.50),
            config.get("sweep_step", 0.01),
        )
        sweep_df = sweep_threshold(sim_matrix, cui_ids, gt_sets, thresholds)
        sweep_path = os.path.join(output_dir, f"{output_name}_threshold_sweep.csv")
        sweep_df.to_csv(sweep_path, index=False)
        print(f"\nThreshold sweep saved to {sweep_path}")

        best_row = sweep_df.loc[sweep_df["f1"].idxmax()]
        print(f"\nBest threshold: {best_row['threshold']:.4f}  "
              f"F1={best_row['f1']:.4f}  "
              f"P={best_row['precision']:.4f}  "
              f"R={best_row['recall']:.4f}")
        threshold = float(best_row["threshold"])
    else:
        threshold = config.get("threshold", 0.25)

    # ── Final predictions at chosen threshold ──
    print(f"\nPredicting with threshold={threshold:.4f}")
    pred_sets = predict_cuis(sim_matrix, cui_ids, threshold)
    metrics   = compute_metrics(pred_sets, gt_sets)

    print(f"\n── Results (threshold={threshold:.4f}) ──────────────")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  Samples   : {metrics['n_samples']}")

    # ── Save predictions CSV ─────────────────
    pred_csv = os.path.join(output_dir, f"{output_name}_predictions.csv")
    save_predictions(image_ids, pred_sets, pred_csv)

    # ── Save metrics JSON ────────────────────
    metrics["threshold"] = threshold
    metrics_path = os.path.join(output_dir, f"{output_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict CUIs via MedSigLIP cosine similarity")
    parser.add_argument("--config", type=str, default="configs/runs/cui_prediction.yaml")
    parser.add_argument("--sweep",  action="store_true",
                        help="Sweep threshold range and pick the one maximizing F1")
    args = parser.parse_args()
    main(args.config, do_sweep=args.sweep)
