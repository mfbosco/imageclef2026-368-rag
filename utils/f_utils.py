import os
import yaml
from transformers import AutoModel, AutoProcessor
import torch
import re
import string
import json

def load_config(config_path):
    """Carrega arquivo de configuração."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def load_models(config, device='cpu'):
    """Carrega processor e modelo."""
    if device == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    token = config['medsiglip'].get('auth_token') or os.environ.get("HUGGINGFACE_HUB_TOKEN_READ")
    processor = AutoProcessor.from_pretrained(
        config['medsiglip']['model'], 
        token=token
    )
    model = AutoModel.from_pretrained(
        config['medsiglip']['model'], 
        token=token
    ).to(device)
    
    return processor, model, device

import re
import string

def load_dataset(config):
    with open(config['dataset'], 'r') as file:
        # Use json.load() to parse the file content into a Python object
        data = json.load(file)

    return data

def preprocess_caption(text: str) -> str:
    """
    Pre-processa um texto com base no descrito pelo site como pre-processamento de caption.
    - Lowercase
    - Replace numbers with token 'number'
    - Remove punctuation
    - Treat as single sentence (no sentence splitting)

    Args:
        text (str): Input caption

    Returns:
        str: Preprocessed caption
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")

    # Lowercase
    text = text.lower()

    # Replace numbers
    text = re.sub(r'\b\d+(\.\d+)?%?\b', 'number', text)

    # Remove pontuações
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Exclui espaço extra
    text = re.sub(r'\s+', ' ', text).strip()

    return text
