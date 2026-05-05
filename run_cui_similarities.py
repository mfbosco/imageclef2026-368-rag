"""
Step 1 — Compute cosine similarity matrix between image embeddings and CUI text embeddings.

Outputs (saved to output_dir):
  {output_name}_sim_matrix.pt   — tensor (N_images, N_cuis), float32
  {output_name}_image_ids.pt    — list[str] of image_ids aligned with matrix rows

CUI ids are already stored alongside the CUI text embeddings (from data/CUI_embeddings.py).

Usage:
  python run_cui_similarities.py --config configs/runs/cui_prediction.yaml
"""

import os
import json
import argparse
import torch
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.f_utils import load_config


def load_dataset_ids(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [sample["image_id"] for sample in data]


def main(config_path):
    config = load_config(config_path)

    # ── Load CUI text embeddings ─────────────────────────────────────────────
    cui_emb_path = config["cui_embeddings"]["text_embeddings"]
    cui_ids_path = config["cui_embeddings"]["ids"]
    cui_embeds = torch.load(cui_emb_path, weights_only=True)   # (N_cuis, D)
    cui_ids    = torch.load(cui_ids_path, weights_only=True)   # list[str]
    print(f"CUI embeddings: {tuple(cui_embeds.shape)}  ({len(cui_ids)} CUIs)")

    # ── Load image embeddings ────────────────────────────────────────────────
    img_emb_path = config["image_embeddings"]
    img_embeds = torch.load(img_emb_path, weights_only=True)   # (N_images, D)
    print(f"Image embeddings: {tuple(img_embeds.shape)}")

    # ── Align image IDs with embedding rows ──────────────────────────────────
    image_ids = load_dataset_ids(config["dataset"])
    assert len(image_ids) == img_embeds.shape[0], (
        f"Mismatch: {len(image_ids)} image_ids vs {img_embeds.shape[0]} embedding rows"
    )

    # ── Cosine similarity matrix ─────────────────────────────────────────────
    # Both tensors are L2-normalised, so dot product == cosine similarity
    print("Computing similarity matrix...")
    sim_matrix = img_embeds @ cui_embeds.T    # (N_images, N_cuis)
    print(f"Similarity matrix: {tuple(sim_matrix.shape)}")
    print(f"Range: [{sim_matrix.min():.4f}, {sim_matrix.max():.4f}]  "
          f"mean={sim_matrix.mean():.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    output_dir  = config["output_dir"]
    output_name = config.get("output_name", "cui_prediction")
    os.makedirs(output_dir, exist_ok=True)

    matrix_path   = os.path.join(output_dir, f"{output_name}_sim_matrix.pt")
    image_ids_path = os.path.join(output_dir, f"{output_name}_image_ids.pt")

    torch.save(sim_matrix, matrix_path)
    torch.save(image_ids,  image_ids_path)

    size_mb = os.path.getsize(matrix_path) / 1e6
    print(f"\nSaved similarity matrix → {matrix_path}  ({size_mb:.1f} MB)")
    print(f"Saved image IDs         → {image_ids_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute image–CUI cosine similarity matrix")
    parser.add_argument("--config", type=str, default="configs/runs/cui_prediction.yaml")
    args = parser.parse_args()
    main(args.config)
