#!/usr/bin/env python3
"""
Run inference and evaluation on ImageClef test/validation set.
Supports different prompt strategies (simple, few-shot, RAG, etc.) with base or fine-tuned models.
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
from transformers import AutoProcessor
import faiss
from peft import PeftModel

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.f_utils import load_config, load_models, load_dataset
from models.medgemma_model import load_medgemma_model
from prompts.prompt_builder import PromptBuilderFactory, select_few_shot_examples
from data.ImageClef_train_embeddings import ImageClefDataset, generate_image_embedding

def generate_prediction(
    model: Any,
    processor: Any,
    image: Any,
    prompt_builder: Any,
    few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    retrieved_examples: Optional[List[Dict[str, str]]] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """
    Generate a prediction using a prompt builder.
    
    Args:
        model: MedGemma model
        processor: MedGemma processor
        image: PIL Image to caption
        prompt_builder: PromptBuilder instance
        few_shot_examples: Optional few-shot examples (for few-shot strategy)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated caption string
    """
    # Build messages using prompt builder (returns messages and images separately)
    result = prompt_builder.build_messages(
        image=image,
        few_shot_examples=few_shot_examples,
        retrieved_examples=retrieved_examples,
    )
    
    # Handle both tuple (messages, images) and just messages for backward compatibility
    if isinstance(result, tuple):
        messages, images_list = result
    else:
        messages = result
        images_list = [image]  # Fallback for simple case
    
    # Apply chat template and process
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[images_list], return_tensors="pt", padding=True)
    
    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )
    
    # Decode - skip_special_tokens=True should handle special tokens properly
    generated_text = processor.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0]
    
    # Manually remove <end_of_turn> tokens if present
    generated_text = generated_text.replace("<end_of_turn>", "").strip()
    
    # Basic cleanup - just strip whitespace
    return generated_text.strip()

def retrieve_examples_from_faiss(
    image,
    model_siglip,
    processor_siglip,
    vector_store,
    train_captions_faiss,
    device,
    top_k=3,
):
    """
    Generate embedding for image and retrieve similar captions from FAISS.
    """

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
        sample = train_captions_faiss[idx]
        retrieved.append({
            "caption": sample
        })

    return retrieved

def run_evaluation(
    model: Any,
    processor: Any,
    test_dataset: ImageClefDataset,
    prompt_builder: Any,
    train_dataset: Optional[ImageClefDataset] = None,
    n_few_shot: int = 0,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    seed: int = 42,
    output_csv_path: Optional[str] = None,
    debug: bool = False,
    model_siglip: Any = None,
    processor_siglip: Any = None,
    vector_store: Any = None,
    train_captions_faiss: List = None,
    rag_top_k: int = None,
) -> Dict[str, Any]:
    """
    Run evaluation on test set with specified prompt strategy.
    Writes predictions to CSV file in real-time.
    
    Args:
        model: MedGemma model
        processor: MedGemma processor
        test_dataset: Test dataset
        prompt_builder: PromptBuilder instance
        train_dataset: Training dataset (for few-shot examples)
        n_few_shot: Number of few-shot examples to use (0 to disable)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        seed: Random seed (currently unused - few-shot examples are randomized per image)
        output_csv_path: Path to CSV file for real-time output (required)
        debug: If True, only run 2 evaluation steps (for debugging)
        
    Returns:
        Dictionary with predictions and metadata
        
    Note:
        Few-shot examples are randomly selected for each test image to enable
        comparison with RAG-selected examples.
    """
    model.eval()
    
    if output_csv_path is None:
        raise ValueError("output_csv_path is required for real-time CSV output")
    
    # Initialize CSV file with header
    print(f"Writing predictions to {output_csv_path} in real-time...")
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'Caption'])  # Header
        
        # Run inference and write to CSV in real-time
        predictions = []
        ids = []
        
        # Limit to 2 steps in debug mode
        num_samples = 2 if debug else len(test_dataset)
        if debug:
            print(f"DEBUG MODE: Running evaluation on only {num_samples} samples...")
        else:
            print(f"Running evaluation on {len(test_dataset)} test samples...")
        if n_few_shot > 0:
            if train_dataset is None:
                raise ValueError("train_dataset is required when n_few_shot > 0")
            print(f"Using {n_few_shot} randomly selected few-shot examples per image")
        
        for idx in tqdm(range(num_samples), desc="Evaluation"):
            sample = test_dataset[idx]
            image = sample["image"]
            image_id = sample.get("id", f"sample_{idx}")
            
            # Select random few-shot examples for this image (different for each image)
            few_shot_examples = None
            if n_few_shot > 0:
                few_shot_examples = select_few_shot_examples(train_dataset, n_few_shot, seed=None)
            
            retrieved_examples = None

            if vector_store is not None:

                device = next(model.parameters()).device

                retrieved_examples = retrieve_examples_from_faiss(
                    image=image,
                    model_siglip=model_siglip,
                    processor_siglip=processor_siglip,
                    vector_store=vector_store,
                    train_captions_faiss=train_captions_faiss,
                    device=device,
                    top_k=rag_top_k,
                )

            # Generate prediction
            prediction = generate_prediction(
                model=model,
                processor=processor,
                image=image,
                prompt_builder=prompt_builder,
                few_shot_examples=few_shot_examples,
                retrieved_examples=retrieved_examples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            
            # Write immediately to CSV
            writer.writerow([image_id, prediction])
            csvfile.flush()  # Ensure data is written to disk immediately
            
            predictions.append(prediction)
            ids.append(image_id)
    
    print(f"All predictions written to {output_csv_path}")
    
    return {
        "predictions": predictions,
        "ids": ids,
        "n_few_shot": n_few_shot,
        "n_rag_examples": rag_top_k,
        "n_samples": len(predictions),
        "csv_path": output_csv_path,
    }


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate MedGemma on ImageClef")
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
    
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    config = load_config(args.config)

    processor_siglip = None
    model_siglip = None
    if config.get("medsiglip"):
        processor_siglip, model_siglip, _ = load_models(config, device = config['device'])

    vector_store = None
    rag_top_k = 0
    train_captions_faiss = None
    if config.get("rag"):
        vector_store = faiss.read_index(config["rag"]["faiss_index_path"])
        rag_top_k = config["rag"]["rag_top_k"]
        train_captions_faiss = torch.load(config["rag"]["train_captions"])
    
    # Extract all settings from config
    eval_config = config.get("evaluation", {})
    prompt_strategy = eval_config.get("prompt_strategy", "rag")
    use_base_model = eval_config.get("use_base_model", False)
    eval_on_val = eval_config.get("eval_on_val", False)
    compute_metrics = eval_config.get("compute_metrics", False)
    # Debug can come from config or command line (command line takes precedence)
    debug = args.debug or eval_config.get("debug", False)
    
    if compute_metrics and not eval_on_val:
        raise ValueError("compute_metrics requires eval_on_val=true (test set has no ground truth)")
    
    data_path = None
    # # Get data path from environment
    # data_path = os.getenv("IMAGECLEF_2024_PATH")
    # if data_path is None:
    #     raise ValueError("IMAGECLEF_2024_PATH environment variable not set")
    
    # Initialize model and processor
    print("Loading MedGemma model...")
    if use_base_model:
        # Use base model
        model, processor = load_medgemma_model(
            model_id=config["model"]["model_id"],
            use_quantization=config["model"]["use_quantization"],
            attn_implementation=config["model"]["attn_implementation"],
        )
        print("Using base MedGemma model")
    else:
        # Try to load fine-tuned model, fall back to base if not found
        model_path = config.get("MODEL_PATH")
        if os.path.exists(model_path):
            print(f"Loading fine-tuned model from {model_path}")
            model, processor = load_medgemma_model(
                model_id=config["model"]["model_id"],
                use_quantization=config["model"]["use_quantization"],
                attn_implementation=config["model"]["attn_implementation"],
            )
            
            model = PeftModel.from_pretrained(model, model_path)
            print("Using fine-tuned model")
        else:
            print(f"Fine-tuned model not found at {model_path}, using base model")
            model, processor = load_medgemma_model(
                model_id=config["model"]["model_id"],
                use_quantization=config["model"]["use_quantization"],
                attn_implementation=config["model"]["attn_implementation"],
            )
            print("Using base model")
    
    # Create prompt builder
    print(f"Using prompt strategy: {prompt_strategy}")
    prompt_builder = PromptBuilderFactory.create(
        strategy=prompt_strategy
    )
    
    # Load datasets
    eval_mode = "val" if eval_on_val else "test"
    ds = load_dataset(config)
    eval_dataset = ImageClefDataset(ds)
    
    # Get evaluation config
    eval_config = config.get("evaluation", {})
    n_few_shot = eval_config.get("n_few_shot", 0)
    max_new_tokens = eval_config.get("max_new_tokens", 512)
    temperature = eval_config.get("temperature", 0.7)
    
    train_dataset = None
    if n_few_shot > 0 or prompt_strategy == "few_shot":
        with open("/home/ia368/projetos/imageclef2026-rag/artifacts/datasets/imageclef2026_train_dataset.json", 'r') as file:
        # Use json.load() to parse the file content into a Python object
            train_dataset = ImageClefDataset(json.load(file))
    
    # Generate CSV output path automatically
    # Format: config_name_project_run_name_datetime.csv
    config_basename = os.path.splitext(os.path.basename(args.config))[0]  # Remove .yml extension
    project_name = config.get("wandb", {}).get("project", "imageclef2026-rag")
    
    # Generate run name
    run_name = config.get("wandb", {}).get("run_name")
    if run_name is None:
        run_name = f"{prompt_strategy}"
        if n_few_shot > 0:
            run_name += f"_{n_few_shot}shot"
        if use_base_model:
            run_name += "_base"
    
    # Get current datetime
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build CSV filename
    ## alterado para criar pasta com os resultados de cada execução
    csv_output_dir = f"artifacts/results/{run_name}_{current_datetime}"
    os.makedirs(csv_output_dir, exist_ok=True)
    csv_output_path = f"{csv_output_dir}/{config_basename}_{project_name}_{run_name}_{current_datetime}.csv"

    print(f"Predictions will be written to: {csv_output_dir}/{csv_output_path}")
    
    # Setup wandb if configured (skip in debug mode)
    if config["output"]["report_to"] == "wandb" and not debug:
        import wandb
        wandb.login()
        # WANDB_API_KEY should be set in environment or .env file
        # wandb will automatically read it from os.environ
        run_name = f"{config['wandb'].get('run_name', 'evaluation')}"
        run_name += f"-{prompt_strategy}"
        if n_few_shot > 0:
            run_name += f"-{n_few_shot}shot"
        if use_base_model:
            run_name += "-base"
        wandb.init(
            project=config["wandb"]["project"],
            name=run_name,
            job_type="inference",
        )
    
    # Print sample prompt before starting
    print("\n" + "="*60)
    print("Sample Prompt Preview")
    print("="*60)
    
    # Get sample image and few-shot examples
    sample_idx = 0
    sample = eval_dataset[sample_idx]
    sample_image = sample["image"]
    sample_id = sample.get("id", f"sample_{sample_idx}")
    
    sample_few_shot_examples = None
    if n_few_shot > 0 and train_dataset is not None:
        sample_few_shot_examples = select_few_shot_examples(train_dataset, n_few_shot, seed=None)

    retrieved_examples = None

    if vector_store is not None:

        device = next(model.parameters()).device

        retrieved_examples = retrieve_examples_from_faiss(
            image=sample_image,
            model_siglip=model_siglip,
            processor_siglip=processor_siglip,
            vector_store=vector_store,
            train_captions_faiss=train_captions_faiss,
            device=device,
            top_k=rag_top_k,
        )
    
    # Build sample messages
    sample_result = prompt_builder.build_messages(
        image=sample_image,
        few_shot_examples=sample_few_shot_examples,
        retrieved_examples=retrieved_examples
    )
    # Handle both tuple (messages, images) and just messages
    if isinstance(sample_result, tuple):
        sample_messages, _ = sample_result
    else:
        sample_messages = sample_result
    
    # Apply chat template to see the final prompt
    sample_text = processor.apply_chat_template(sample_messages, add_generation_prompt=True, tokenize=False)
    
    print(f"Sample Image ID: {sample_id}")
    print(f"Prompt Strategy: {prompt_strategy}")
    if rag_top_k > 0:
        print(f"Caption Examples RAG: {retrieved_examples}")
    if n_few_shot > 0:
        print(f"Few-shot examples: {n_few_shot}")
    print("\nComplete Prompt:")
    print("-" * 60)
    print(sample_text)
    print("-" * 60)
    print("="*60 + "\n")
    
    # Run evaluation
    print("="*60)
    print(f"Running evaluation with {prompt_strategy} prompt strategy")
    print(f"Evaluating on: {eval_mode} set")
    if n_few_shot > 0:
        print(f"Few-shot examples: {n_few_shot}")
    print("="*60)
    
    results = run_evaluation(
        model=model,
        processor=processor,
        test_dataset=eval_dataset,
        prompt_builder=prompt_builder,
        train_dataset=train_dataset,
        n_few_shot=n_few_shot,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        output_csv_path=csv_output_path,
        debug=debug,
        model_siglip=model_siglip,
        processor_siglip=processor_siglip,
        vector_store=vector_store,
        train_captions_faiss=train_captions_faiss,
        rag_top_k=rag_top_k
    )
    
    print(f"\nGenerated {len(results['predictions'])} predictions")
    if results['n_rag_examples'] > 0:
        print(f"Caption Examples RAG: {results['n_rag_examples']}")
    if results['n_few_shot'] > 0:
        print(f"Few-shot examples used: {results['n_few_shot']}")
    print(f"All predictions saved to: {results['csv_path']}")
    
    # Compute metrics if requested (using the CSV file)
    metrics = {}
    # if compute_metrics:
    #     print("\n" + "="*60)
    #     print("Computing ImageClef metrics from CSV file...")
    #     print("="*60)
        
    #     # Evaluate captions using the CSV file directly
    #     captions_gt_path = os.path.join(data_path, "dataset", "valid_captions.csv")
    #     if os.path.exists(captions_gt_path):
    #         print(f"Using predictions from: {results['csv_path']}")
    #         print("Evaluating captions...")
    #         try:
    #             caption_metrics = evaluate_captions(results['csv_path'], captions_gt_path)
    #             metrics.update(caption_metrics)
    #             print(f"  BERTScore: {caption_metrics['bert_score']:.4f}")
    #             print(f"  ROUGE Score: {caption_metrics['rouge_score']:.4f}")
    #         except Exception as e:
    #             if debug and "Number of image IDs in submission file not equal" in str(e):
    #                 print(f"  Warning: Skipping metrics evaluation in debug mode (only {results['n_samples']} samples evaluated)")
    #             else:
    #                 raise  # Re-raise if not the expected debug mode exception
    #     else:
    #         print(f"Warning: Ground truth captions not found at {captions_gt_path}")
        
    #     print("="*60)
    
    # Log to wandb (skip in debug mode)
    if config["output"]["report_to"] == "wandb" and not debug:
        import wandb
        log_dict = {
            "n_test_samples": results["n_samples"],
            "n_few_shot": results["n_few_shot"],
        }
        log_dict.update(metrics)
        wandb.log(log_dict)
    
    # Note: Predictions are already saved to CSV during evaluation
    # The CSV file can be monitored in real-time during the run
    
    # Print metrics summary
    if metrics:
        print("\n" + "="*60)
        print("Metrics Summary:")
        print("="*60)
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        print("="*60)
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
