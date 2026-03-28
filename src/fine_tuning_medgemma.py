import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from typing import Any
from torch.utils.data import Dataset, Subset
import random
import os
from PIL import Image
import json
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.f_utils import load_config

from dotenv import load_dotenv


# =========================
# Dataset
# =========================
class ImageClefDataset(Dataset):
    def __init__(self, split):
        self.dataset = split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = Image.open(sample["image_path"]).convert('RGB')

        return {
            "image": image,
            "caption": sample["caption"],
            "id": sample["image_id"]
        }


# =========================
# Model + Processor
# =========================
def load_model(model_id):
    if torch.cuda.get_device_capability()[0] < 8:
        raise ValueError("GPU does not support bfloat16")

    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
        bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
    )

    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "right"

    return model, processor


# =========================
# Collate
# =========================
def build_collate_fn(processor):
    def collate_fn(examples: list[dict[str, Any]]):
        texts = []
        images = []

        for example in examples:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "You are a medical expert. Provide a precise and concise caption for this medical image."}
                    ],
                },
                {
                    "role": "assistant",
                    "content": example["caption"],
                },
            ]

            images.append([example["image"]])
            texts.append(processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=False
            ).strip())

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()

        image_token_id = [
            processor.tokenizer.convert_tokens_to_ids(
                processor.tokenizer.special_tokens_map["boi_token"]
            )
        ]

        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        labels[labels == 262144] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


# =========================
# Dataset loading
# =========================
def load_datasets(config):
    with open(config['dataset_train'], 'r') as f:
        ds_train = json.load(f)

    with open(config['dataset_valid'], 'r') as f:
        ds_val = json.load(f)

    train_dataset = ImageClefDataset(ds_train)
    eval_dataset = ImageClefDataset(ds_val)

    return train_dataset, eval_dataset


# =========================
# Debug mode sampling
# =========================
def apply_debug_subset(train_dataset, eval_dataset, debug):
    if not debug:
        num_samples = 2000
        indices = random.sample(range(len(eval_dataset)), num_samples)
        return train_dataset, Subset(eval_dataset, indices)

    random.seed(42)

    train_indices = random.sample(range(len(train_dataset)), 32)
    eval_indices = random.sample(range(len(eval_dataset)), 32)

    train_subset = Subset(train_dataset, train_indices)
    eval_subset = Subset(eval_dataset, eval_indices)

    return train_subset, eval_subset


# =========================
# Trainer
# =========================
def build_trainer(model, processor, train_dataset, eval_dataset, collate_fn, args_config):
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    from transformers import EarlyStoppingCallback

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],
    )

    args = SFTConfig(**args_config)

    early_stopping_cb = EarlyStoppingCallback(
        early_stopping_patience=5,
        early_stopping_threshold=1e-7
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=collate_fn,
        callbacks=[early_stopping_cb],
    )

    return trainer


# =========================
# MAIN
# =========================
def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Fine tuning MedGemma in ImageClef")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: disable wandb logging and run only 2 steps"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    debug = args.debug or config.get("debug", False)

    load_dotenv(dotenv_path=".env")
    path = config["SAVED_MODEL_PATH"]
    print(path)

    train_dataset, eval_dataset = load_datasets(config)
    train_dataset, eval_dataset = apply_debug_subset(train_dataset, eval_dataset, debug)

    if config.get('train_subset', False):
        num_samples = int(config['train_subset']*len(train_dataset))
        indices = random.sample(range(len(train_dataset)), num_samples)
        
        train_dataset = Subset(train_dataset, indices)

    model, processor = load_model("google/medgemma-4b-it")
    collate_fn = build_collate_fn(processor)

    args_config = dict(
        output_dir=config["output_dir"],
        num_train_epochs=1 if debug else 5,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1 if debug else 5,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=1 if debug else 50,
        dataloader_num_workers=16,
        eval_strategy="steps",
        eval_steps=5 if debug else 25,
        save_strategy="steps",
        save_steps=5 if debug else 25,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=2e-4,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        push_to_hub=not debug,
        report_to="tensorboard",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = build_trainer(
        model, processor, train_dataset, eval_dataset, collate_fn, args_config
    )

    trainer.train()
    trainer.save_model()

    ckpt = {
        "peft_adapter_state": model.state_dict(),
        "hf_config": model.config.to_dict()
    }
    torch.save(ckpt, path)


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()