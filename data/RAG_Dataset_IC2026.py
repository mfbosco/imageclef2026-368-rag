import os
from transformers import AutoModel, AutoProcessor
import json
import csv
from PIL import Image
import torch
import faiss
from dotenv import load_dotenv
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.f_utils import load_config, load_models, load_dataset
from data.ImageClef_train_embeddings import (
    ImageClefDataset,
    generate_image_embedding
)


def retrieve_examples_from_faiss(
    image,
    model_siglip,
    processor_siglip,
    vector_store,
    train_captions_faiss,
    device,
    top_k=3,
    dataset_type: str = "train",  # NEW
):
    emb = generate_image_embedding(
        model_siglip,
        processor_siglip,
        image,
        device,
    )

    emb_np = emb.cpu().numpy().astype("float32").reshape(1, -1)

    # buscamos +1 se for treino
    search_k = top_k + 1 if dataset_type == "train" else top_k

    distances, indices = vector_store.search(emb_np, search_k)

    retrieved = []

    idx_list = indices[0]

    # remove o primeiro apenas no treino
    if dataset_type == "train":
        idx_list = idx_list[1:]  # drop self-match

    # garante que não excede top_k
    idx_list = idx_list[:top_k]

    for idx in idx_list:
        sample = train_captions_faiss[idx]
        retrieved.append({
            "caption": sample
        })

    return retrieved

# =========================
# Enrich dataset (CUI + RAG)
# =========================
def enrich_dataset(
    json_path,
    output_path,
    model_siglip,
    processor_siglip,
    vector_store,
    train_captions_faiss,
    device,
    top_k=3,
    dataset_type: str = "train"
):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    enriched = []

    for sample in tqdm(data, desc=f"Enriching ({dataset_type})", unit="img"):
        image_id = sample["image_id"]
        image_path = sample["image_path"]

        # -------------------------
        # Load image
        # -------------------------
        image = Image.open(image_path).convert("RGB")

        # -------------------------
        # Retrieval enrichment
        # -------------------------
        retrieved_examples = retrieve_examples_from_faiss(
            image=image,
            model_siglip=model_siglip,
            processor_siglip=processor_siglip,
            vector_store=vector_store,
            train_captions_faiss=train_captions_faiss,
            device=device,
            top_k=top_k,
            dataset_type=dataset_type
        )

        # -------------------------
        # Attach to sample
        # -------------------------
        sample["retrieved_examples"] = retrieved_examples

        enriched.append(sample)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    print(f"Saved enriched dataset to {output_path}")

def main():
    
    load_dotenv()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN_READ")
    processor_siglip = AutoProcessor.from_pretrained(
        "google/medsiglip-448", 
        token=token
    )
    model_siglip = AutoModel.from_pretrained(
        "google/medsiglip-448", 
        token=token
    ).to(device)

    
    # Load FAISS + captions
    vector_store = faiss.read_index("/home/ia368/projetos/imageclef2026-rag/artifacts/vector_store/IC2026_image_faiss.index")
    rag_top_k = 3
    train_captions_faiss = torch.load("/home/ia368/projetos/imageclef2026-rag/artifacts/embeddings/IC2026_medsiglip_train_captions.pt")
    
    train_json = "artifacts/datasets/imageclef2026_train_dataset.json"
    valid_json = "artifacts/datasets/imageclef2026_valid_dataset.json"

    output_train = "artifacts/datasets/IC2026_Caption_RAG_train.json"
    output_valid = "artifacts/datasets/IC2026_Caption_RAG_valid.json"


    enrich_dataset(
        train_json,
        output_train,
        model_siglip,
        processor_siglip,
        vector_store,
        train_captions_faiss,
        device,
        3,
        'train'
    )

    enrich_dataset(
        valid_json,
        output_valid,
        model_siglip,
        processor_siglip,
        vector_store,
        train_captions_faiss,
        device,
        3,
        'valid'
    )


if __name__ == "__main__":
    main()