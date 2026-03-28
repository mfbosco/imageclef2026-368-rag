"""
MedGemma model loading and configuration.
"""
import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
from typing import Tuple, Dict, Any, Optional


def check_gpu_capability() -> bool:
    """
    Check if GPU supports bfloat16.
    
    Returns:
        True if GPU supports bfloat16, False otherwise
        
    Raises:
        ValueError: If GPU doesn't support bfloat16
    """
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. This model requires a GPU.")
    
    if torch.cuda.get_device_capability()[0] < 8:
        raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")
    
    return True


def load_medgemma_model(
    model_id: str = "google/medgemma-4b-it",
    use_quantization: bool = True,
    attn_implementation: str = "eager",
) -> Tuple[Any, Any]:
    """
    Load MedGemma model and processor.
    
    Args:
        model_id: Hugging Face model identifier
        use_quantization: Whether to use 4-bit quantization
        attn_implementation: Attention implementation ("eager" or "flash_attention_2")
        
    Returns:
        Tuple of (model, processor)
    """
    check_gpu_capability()
    
    dtype = torch.bfloat16
    model_kwargs: Dict[str, Any] = dict(
        attn_implementation=attn_implementation,
        dtype=dtype,  # Use dtype instead of deprecated torch_dtype
        device_map="auto",
    )
    
    if use_quantization:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_storage=dtype,
            llm_int8_enable_fp32_cpu_offload=True
        )
    
    # model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    # processor = AutoProcessor.from_pretrained(model_id, use_fast=False)  # Keep slow processor for original behavior
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN_WRITE")
    model = AutoModelForImageTextToText.from_pretrained(model_id, token=hf_token, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token, use_fast=False)  # Keep slow processor for original behavior
    
    # Use right padding to avoid issues during training
    processor.tokenizer.padding_side = "right"
    
    return model, processor


def create_lora_config(
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    r: int = 16,
    bias: str = "none",
    target_modules: str = "all-linear",
    task_type: str = "CAUSAL_LM",
    modules_to_save: Optional[list] = None,
) -> LoraConfig:
    """
    Create LoRA configuration for fine-tuning.
    
    Args:
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        r: LoRA rank
        bias: Bias handling ("none", "all", "lora_only")
        target_modules: Target modules for LoRA ("all-linear" or list of module names)
        task_type: Task type for PEFT
        modules_to_save: List of modules to save (not quantized)
        
    Returns:
        LoRA configuration
    """
    if modules_to_save is None:
        modules_to_save = ["lm_head", "embed_tokens"]
    
    return LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=r,
        bias=bias,
        target_modules=target_modules,
        task_type=task_type,
        modules_to_save=modules_to_save,
    )


if __name__ == "__main__":
    """Test the model loading module."""
    print("="*60)
    print("Testing MedGemma Model Loading")
    print("="*60)
    
    # Test GPU capability check
    print("\n1. Testing GPU capability check...")
    try:
        has_capability = check_gpu_capability()
        print(f"   ✓ GPU capability check passed: {has_capability}")
    except Exception as e:
        print(f"   ✗ GPU capability check failed: {e}")
        print("   Note: This is expected if no GPU is available")
    
    # Test LoRA config creation
    print("\n2. Testing LoRA config creation...")
    try:
        lora_config = create_lora_config()
        print("   ✓ LoRA config created successfully")
        print(f"   Config: r={lora_config.r}, alpha={lora_config.lora_alpha}, dropout={lora_config.lora_dropout}")
    except Exception as e:
        print(f"   ✗ LoRA config creation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test model loading (optional, requires GPU and can be slow)
    print("\n3. Testing model loading...")
    print("   Note: This test requires GPU and may take time to download model")
    try:
        import torch
        if torch.cuda.is_available():
            model, processor = load_medgemma_model(
                model_id="google/medgemma-4b-it",
                use_quantization=True,
            )
            print("   ✓ Model and processor loaded successfully")
            print(f"   Model device: {next(model.parameters()).device}")
            print(f"   Model dtype: {next(model.parameters()).dtype}")
        else:
            print("   ⚠ Skipping model load test (no GPU available)")
    except Exception as e:
        print(f"   ✗ Model loading failed: {e}")
        print("   Note: This may be expected if GPU is not available or model download fails")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Model loading tests completed!")
    print("="*60)
