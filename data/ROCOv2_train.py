"""
ROCOv2 dataset module to run MedSigLip Embeddings
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
import argparse
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.f_utils import load_config, load_models
from PIL import Image


# --------------
# Classe Dataset
# --------------

class ROCOv2_Dataset(Dataset):
    def __init__(self, split, processor, max_length=64):
        """
        split: split do dataset do HuggingFace (ex: ds["train"])
        processor: processor do MedSigLip
        max_length: tamanho máximo do texto
        """
        self.dataset = split
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        image = sample["image"]
        text = sample["caption"]

        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Usa o processor do modelo
        encoding = self.processor(
            text=text,
            images=image,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Remove batch dimension criada pelo return_tensors="pt"
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        return encoding


# --------------
# Calcula Embeddings
# --------------

def generate_embeddings(model, dataloader, device):
    model.eval()
    model.to(device)

    all_image_embeds = []
    all_text_embeds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating embeddings"):
            batch = {k: v.to(device) for k, v in batch.items()}

            image_embeds = model.get_image_features(
                pixel_values=batch["pixel_values"]
            )

            text_embeds = model.get_text_features(
                input_ids=batch["input_ids"]
            )

            # normalização (cosine similarity)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            all_image_embeds.append(image_embeds.cpu())
            all_text_embeds.append(text_embeds.cpu())

    all_image_embeds = torch.cat(all_image_embeds, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)

    return all_image_embeds, all_text_embeds


# --------------
# Processando conjunto de treino
# --------------

def main(config_path):

    config = load_config(config_path)
    processor, model, device = load_models(config, device=config['device'])

    print("Loading Dataset ")
    ds = load_dataset(config['dataset'])

    train_split = ds['train']

    dataset = ROCOv2_Dataset(
        split=train_split,
        processor=processor,
        max_length=64
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['parameters']['batch_size'],
        shuffle=False,
        pin_memory=True
    )

    print(f"Using device: {device}")

    image_embeds, text_embeds = generate_embeddings(
        model,
        dataloader,
        device
    )

    print("Saving embeddings...")

    OUTPUT_DIR = config['output_dir']
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(image_embeds, os.path.join(OUTPUT_DIR, "train_image_embeddings.pt"))
    torch.save(text_embeds, os.path.join(OUTPUT_DIR, "train_text_embeddings.pt"))

    # também salva captions alinhadas
    captions = [sample["caption"] for sample in train_split]
    torch.save(captions, os.path.join(OUTPUT_DIR, "train_captions.pt"))

    print("Done!")
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pré-processa embeddings do ROCO - Treino"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/rocov2_configs.yaml",
        help="Caminho para arquivo de configuração"
    )

    args = parser.parse_args()
    main(args.config)
