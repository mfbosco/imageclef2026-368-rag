"""
Two-stage CUI prediction: caption → CUI matching.

Encodes model-generated captions with MedSigLIP text encoder and computes
cosine similarity against pre-computed CUI text embeddings.

Output is identical in format to run_cui_similarities.py, so
run_cui_prediction.py (threshold sweep + evaluation) works unchanged.

Usage:
  python run_cui_from_caption.py --config configs/runs/cui_from_caption.yaml
"""

import os
import csv
import argparse
import torch
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.f_utils import load_config, load_models


# ─────────────────────────────────────────────────────────────────────────────
# Load captions from predictions CSV
# ─────────────────────────────────────────────────────────────────────────────

def load_captions(predictions_csv):
    """
    Reads a caption predictions CSV with columns ID,Caption.
    Returns (image_ids, captions) as aligned lists.
    """
    image_ids, captions = [], []
    with open(predictions_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_ids.append(row["ID"].strip())
            captions.append(row["Caption"].strip())
    return image_ids, captions


# ─────────────────────────────────────────────────────────────────────────────
# Encode captions with MedSigLIP text encoder
# ─────────────────────────────────────────────────────────────────────────────

def encode_captions(model, processor, captions, device, batch_size=64):
    """
    Returns L2-normalized caption text embeddings tensor (N, D).
    """
    model.eval()
    all_embeds = []

    with torch.no_grad():
        for i in tqdm(range(0, len(captions), batch_size), desc="Encoding captions"):
            batch = captions[i : i + batch_size]
            inputs = processor(
                text=batch,
                padding="max_length",
                truncation=True,
                max_length=64,          # MedSigLIP text encoder limit
                return_tensors="pt",
            )
            input_ids = inputs["input_ids"].to(device)
            embeds = model.get_text_features(input_ids=input_ids)["pooler_output"]
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            all_embeds.append(embeds.cpu())

    return torch.cat(all_embeds, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(config_path):
    config = load_config(config_path)

    # ── Load captions ────────────────────────────────────────────────────────
    predictions_csv = config["predictions_csv"]
    image_ids, captions = load_captions(predictions_csv)
    print(f"Loaded {len(captions)} captions from {predictions_csv}")
    print(f"Example: [{image_ids[0]}] {captions[0][:80]}")

    # ── Load MedSigLIP and encode captions ───────────────────────────────────
    processor, model, device = load_models(config, device=config["device"])
    batch_size = config.get("parameters", {}).get("batch_size", 64)
    caption_embeds = encode_captions(model, processor, captions, device, batch_size)
    print(f"Caption embeddings: {tuple(caption_embeds.shape)}")

    # ── Free model from GPU ──────────────────────────────────────────────────
    model.cpu()
    torch.cuda.empty_cache()

    # ── Load CUI text embeddings ─────────────────────────────────────────────
    cui_embeds = torch.load(config["cui_embeddings"]["text_embeddings"], weights_only=True)
    cui_ids    = torch.load(config["cui_embeddings"]["ids"],             weights_only=True)
    print(f"CUI embeddings    : {tuple(cui_embeds.shape)}  ({len(cui_ids)} CUIs)")

    # ── Cosine similarity (text–text) ────────────────────────────────────────
    # Both tensors are L2-normalised → dot product = cosine similarity
    print("Computing similarity matrix (caption text ↔ CUI text)...")
    sim_matrix = caption_embeds @ cui_embeds.T    # (N_images, N_cuis)
    print(f"Similarity matrix : {tuple(sim_matrix.shape)}")
    print(f"Range             : [{sim_matrix.min():.4f}, {sim_matrix.max():.4f}]  "
          f"mean={sim_matrix.mean():.4f}")

    # ── Save (same format as run_cui_similarities.py) ────────────────────────
    output_dir  = config["output_dir"]
    output_name = config.get("output_name", "cui_from_caption")
    os.makedirs(output_dir, exist_ok=True)

    matrix_path    = os.path.join(output_dir, f"{output_name}_sim_matrix.pt")
    image_ids_path = os.path.join(output_dir, f"{output_name}_image_ids.pt")

    torch.save(sim_matrix, matrix_path)
    torch.save(image_ids,  image_ids_path)

    size_mb = os.path.getsize(matrix_path) / 1e6
    print(f"\nSaved similarity matrix → {matrix_path}  ({size_mb:.1f} MB)")
    print(f"Saved image IDs         → {image_ids_path}")
    print(f"\nNext step — threshold sweep:")
    print(f"  python run_cui_prediction.py --config configs/runs/cui_from_caption.yaml --sweep")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-stage CUI prediction: encode captions → similarity with CUI names"
    )
    parser.add_argument("--config", type=str, default="configs/runs/cui_from_caption.yaml")
    args = parser.parse_args()
    main(args.config)
