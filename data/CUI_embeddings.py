"""
Generate MedSigLIP text embeddings for CUI canonical names.

Outputs (saved to output_dir):
  {output_name}_text_embeddings.pt  — tensor (N_cuis, D), L2-normalized
  {output_name}_ids.pt              — list of N_cuis CUI strings, aligned with rows above
"""
import os
import csv
import torch
import argparse
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.f_utils import load_config, load_models
from tqdm import tqdm


def load_cui_mapping(mapping_path):
    cuis, names = [], []
    with open(mapping_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cuis.append(row["CUI"].strip())
            names.append(row["Canonical name"].strip())
    return cuis, names


def generate_cui_text_embeddings(model, processor, names, device, batch_size=64):
    model.eval()
    all_embeds = []

    with torch.no_grad():
        for i in tqdm(range(0, len(names), batch_size), desc="Encoding CUI names"):
            batch_names = names[i : i + batch_size]

            inputs = processor(
                text=batch_names,
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )

            input_ids = inputs["input_ids"].to(device)
            text_embeds = model.get_text_features(input_ids=input_ids)["pooler_output"]
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            all_embeds.append(text_embeds.cpu())

    return torch.cat(all_embeds, dim=0)


def main(config_path):
    config = load_config(config_path)
    processor, model, device = load_models(config, device=config["device"])

    mapping_path = config.get("cui_mapping", "dev_caption/cui_mapping.csv")
    cuis, names = load_cui_mapping(mapping_path)
    print(f"Loaded {len(cuis)} CUIs from {mapping_path}")

    batch_size = config.get("parameters", {}).get("batch_size", 64)
    embeds = generate_cui_text_embeddings(model, processor, names, device, batch_size)

    output_dir = config["output_dir"]
    output_name = config.get("output_name", "cui_mapping")
    os.makedirs(output_dir, exist_ok=True)

    torch.save(embeds, os.path.join(output_dir, f"{output_name}_text_embeddings.pt"))
    torch.save(cuis, os.path.join(output_dir, f"{output_name}_ids.pt"))

    print(f"Saved embeddings shape: {embeds.shape}")
    print(f"Output: {output_dir}/{output_name}_*.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MedSigLIP text embeddings for CUI canonical names")
    parser.add_argument("--config", type=str, default="configs/embeddings/cui_mapping_emb.yaml")
    args = parser.parse_args()
    main(args.config)
