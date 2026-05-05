#!/usr/bin/env python3
"""
Run inference and evaluation on ImageClef test/validation set.
Simplified version using base-style prompting (no chat templates).
"""

import os
import sys
import yaml
import argparse
import json
import csv
import torch
from datetime import datetime
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from dotenv import load_dotenv
import faiss
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.f_utils import load_config, load_models, load_dataset
from models.medgemma_model import load_medgemma_model
from data.ImageClef_train_embeddings import ImageClefDataset, generate_image_embedding

def generate_prediction(
    model,
    processor,
    image,
    max_new_tokens=512,
):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Medical image caption:"}
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_text = processor.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0]

    return generated_text.strip()


# =========================================================
def sanitize_caption_for_csv(text: str) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text.strip()


# =========================================================
def retrieve_examples_from_faiss(
    image,
    model_siglip,
    processor_siglip,
    vector_store,
    train_captions_faiss,
    device,
    top_k=3,
):
    emb = generate_image_embedding(
        model_siglip,
        processor_siglip,
        image,
        device,
    )

    emb_np = emb.numpy().astype("float32").reshape(1, -1)
    distances, indices = vector_store.search(emb_np, top_k)

    retrieved = []
    for idx in indices[0]:
        retrieved.append({"caption": train_captions_faiss[idx]})

    return retrieved


# =========================================================
def run_evaluation(
    model: Any,
    processor: Any,
    test_dataset: ImageClefDataset,
    max_new_tokens: int = 40,
    output_csv_path: Optional[str] = None,
    debug: bool = False,
    model_siglip: Any = None,
    processor_siglip: Any = None,
    vector_store: Any = None,
    train_captions_faiss: List = None,
    rag_top_k: int = None,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    resume_csv: Optional[str] = None,
) -> Dict[str, Any]:

    model.eval()

    if output_csv_path is None:
        raise ValueError("output_csv_path is required")

    total_samples = len(test_dataset)
    if end_idx is None:
        end_idx = total_samples

    existing_ids = set()
    csv_exists = resume_csv is not None and os.path.exists(output_csv_path)
    file_mode = "a" if csv_exists else "w"

    if csv_exists:
        with open(output_csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    existing_ids.add(row[0])

    with open(output_csv_path, file_mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)

        if file_mode == "w":
            writer.writerow(["ID", "Caption"])

        predictions, ids = [], []

        effective_end_idx = min(end_idx, start_idx + 2) if debug else end_idx

        for idx in tqdm(range(start_idx, effective_end_idx), desc="Evaluation"):
            sample = test_dataset[idx]
            image = sample["image"]
            image_id = sample.get("id", f"sample_{idx}")

            if image_id in existing_ids:
                continue

            prediction = generate_prediction(
                model=model,
                processor=processor,
                image=image,
                max_new_tokens=max_new_tokens,
            )

            prediction = sanitize_caption_for_csv(prediction)
            writer.writerow([image_id, prediction])
            csvfile.flush()

            predictions.append(prediction)
            ids.append(image_id)

    return {
        "predictions": predictions,
        "ids": ids,
        "n_samples": len(predictions),
        "csv_path": output_csv_path,
    }


# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--resume_csv", type=str, default=None)
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)

    print("Loading model...")
    model, processor = load_medgemma_model(
        model_id=config["model"]["model_id"],
        use_quantization=config["model"]["use_quantization"],
        attn_implementation=config["model"]["attn_implementation"],
    )

    ds = load_dataset(config)
    eval_dataset = ImageClefDataset(ds)

    output_dir = f"artifacts/results/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "predictions.csv")

    print("\nSample prompt:")
    print("Medical image caption:\n")

    results = run_evaluation(
        model=model,
        processor=processor,
        test_dataset=eval_dataset,
        max_new_tokens=40,
        output_csv_path=csv_path,
        debug=args.debug,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        resume_csv=args.resume_csv,
    )

    print(f"\nSaved to: {results['csv_path']}")
    print("Done!")


if __name__ == "__main__":
    main()