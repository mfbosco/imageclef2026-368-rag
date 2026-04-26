import torch
import transformers
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
            "id": sample["image_id"],
            "retrieved_examples": sample["retrieved_examples"]
        }


# =========================
# Model + Processor
# =========================
def load_model(model_id):
    if torch.cuda.get_device_capability()[0] < 8:
        raise ValueError("GPU does not support bfloat16")

    model_kwargs = dict(
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
        device_map="auto",
    )

    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    model = AutoModelForImageTextToText.from_pretrained(model_id, token=os.environ.get("HUGGINGFACE_HUB_TOKEN_READ"), **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_id, token=os.environ.get("HUGGINGFACE_HUB_TOKEN_READ"))
    processor.tokenizer.padding_side = "right"

    return model, processor


def build_collate_fn(processor):
    # Precompute token IDs once, outside the per-batch closure
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    # Number of tokens in "assistant\n" header (after <|im_start|>)
    assistant_header_len = len(processor.tokenizer.encode("assistant\n", add_special_tokens=False))

    # Qwen VL vision special tokens to mask in labels
    vision_token_ids = set()
    for name in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>"):
        tid = processor.tokenizer.convert_tokens_to_ids(name)
        if tid != processor.tokenizer.unk_token_id:
            vision_token_ids.add(tid)

    def collate_fn(examples):
        texts = []
        images = []

        for example in examples:
            context_text = "\n".join(
                [r["caption"] for r in example.get("retrieved_examples", [])]
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": f"""You are a medical expert.

            You will receive captions of visually similar medical images retrieved from a database.

            Use them ONLY as medical context and style reference.
            Do NOT copy them.

            Similar image captions:
            {context_text}

            Using the image and the information above, provide a precise and concise caption."""
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": example["caption"],
                },
            ]

            full_text = processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=False,
            )

            texts.append(full_text)
            images.append([example["image"]])

        batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        # Mask everything before the assistant response (Qwen ChatML format)
        # Pattern: <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n{response}
        for i in range(input_ids.shape[0]):
            sequence = input_ids[i]

            turn_positions = (sequence == im_start_id).nonzero(as_tuple=True)[0]

            if len(turn_positions) < 2:
                raise ValueError("Could not find both user and assistant turns.")

            # second <|im_start|> marks the assistant turn
            assistant_im_start = turn_positions[1].item()

            # mask <|im_start|> + "assistant\n" header, leave only the response
            content_start = assistant_im_start + 1 + assistant_header_len
            labels[i, :content_start] = -100

        # Mask padding tokens
        labels[labels == processor.tokenizer.pad_token_id] = -100

        # Mask Qwen VL vision tokens
        for tid in vision_token_ids:
            labels[labels == tid] = -100

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
        return train_dataset, eval_dataset

    random.seed(42)

    train_indices = random.sample(range(len(train_dataset)), 32)
    eval_indices = random.sample(range(len(eval_dataset)), 32)

    train_subset = Subset(train_dataset, train_indices)
    eval_subset = Subset(eval_dataset, eval_indices)

    return train_subset, eval_subset


# =========================
# Trainer
# =========================
class EarlyStoppingAfterEpoch(transformers.TrainerCallback):
    """Early stopping that only activates after a minimum number of epochs."""
    def __init__(self, patience=5, threshold=1e-7, min_epochs=1):
        self.patience = patience
        self.threshold = threshold
        self.min_epochs = min_epochs
        self.best_metric = None
        self.wait = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if state.epoch is not None and state.epoch < self.min_epochs:
            return

        metric_value = metrics.get("eval_loss")
        if metric_value is None:
            return

        if self.best_metric is None or metric_value < self.best_metric - self.threshold:
            self.best_metric = metric_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                control.should_training_stop = True


def build_trainer(model, processor, train_dataset, eval_dataset, collate_fn, args_config):
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

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

    early_stopping_cb = EarlyStoppingAfterEpoch(
        patience=3,
        threshold=1e-7,
        min_epochs=1,
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
    parser = argparse.ArgumentParser(description="Fine tuning Qwen3-VL in ImageClef")
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
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    args = parser.parse_args()

    load_dotenv(dotenv_path=".env")

    config = load_config(args.config)
    debug = args.debug or config.get("debug", False)

    load_dotenv(dotenv_path=".env")
    path = config["SAVED_MODEL_PATH"]
    print(path)

    train_dataset, eval_dataset = load_datasets(config)
    train_dataset, eval_dataset = apply_debug_subset(train_dataset, eval_dataset, debug)

    if config.get('train_subset', False):
        num_samples = int(config['train_subset'] * len(train_dataset))
        indices = random.sample(range(len(train_dataset)), num_samples)
        train_dataset = Subset(train_dataset, indices)

    if config.get('valid_subset', False):
        num_samples_valid = int(config['valid_subset'] * len(eval_dataset))
        indices_valid = random.sample(range(len(eval_dataset)), num_samples_valid)
        eval_dataset = Subset(eval_dataset, indices_valid)

    model, processor = load_model("Qwen/Qwen3-VL-4B-Instruct") ## ALTERADO PARA QWEN3
    collate_fn = build_collate_fn(processor)

    args_config = dict(
        output_dir=config["output_dir"],
        num_train_epochs=1 if debug else 2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1 if debug else 16,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=1 if debug else 500,
        logging_strategy="steps",
        dataloader_num_workers=16,
        eval_strategy="steps",
        eval_steps=5 if debug else 500,
        save_strategy="steps",
        save_steps=5 if debug else 500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=2e-4,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        push_to_hub=config.get("push_to_hub", not debug),
        report_to="tensorboard",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = build_trainer(
        model, processor, train_dataset, eval_dataset, collate_fn, args_config
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
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
