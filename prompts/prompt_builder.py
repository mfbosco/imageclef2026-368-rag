"""
Prompt builder module for ImageClef dataset.
Supports different prompt strategies (simple, RAG, few-shot, etc.) via factory pattern.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
from PIL import Image
import torch
import random


class PromptBuilder:
    """Base class for prompt builders."""
    
    def build_messages(
        self,
        image: Image.Image,
        caption: Optional[str] = None,
        **kwargs
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], List[Image.Image]]]:
        """
        Build messages for MedGemma chat template.
        
        Args:
            image: PIL Image
            caption: Optional caption text (for training - adds assistant message with ground truth)
            **kwargs: Additional arguments for specific strategies
            
        Returns:
            List of message dictionaries (with image placeholders, not actual images)
            OR tuple of (messages, images_list) if multiple images are used
        """
        raise NotImplementedError


class SimplePromptBuilder(PromptBuilder):
    """
    Simple prompt builder - default strategy.
    
    To customize the prompt text, create a new PromptBuilder subclass.
    """
    
    SYSTEM_PROMPT = """You are a medical image captioning system for imageClef 2026. 
Output only the caption, no other text. Do not return any special character like row breaks or similar."""
    
    PROMPT_TEXT = "Caption this image."
    
    def __init__(self):
        """Initialize simple prompt builder."""
        pass
    
    def build_messages(
        self,
        image: Image.Image,
        caption: Optional[str] = None,
        **kwargs
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], List[Image.Image]]]:
        """
        Build simple prompt messages.
        
        Args:
            image: PIL Image
            caption: Optional caption text (for training - adds assistant message with ground truth)
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (messages, images_list) where:
            - messages: List of message dictionaries with image placeholders (not actual images)
            - images_list: List of PIL Images in the order they appear in messages
        """
        messages = []
        images_list = []
        
        # Add system prompt describing the task
        messages.append({
            "role": "system",
            "content": self.SYSTEM_PROMPT
        })
        
        # Add user message with image placeholder
        messages.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": self.PROMPT_TEXT}
            ]
        })
        images_list.append(image)
        
        # Add assistant message with ground truth caption (for training)
        if caption is not None:
            messages.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": caption}
                ]
            })
        
        return messages, images_list


class FewShotPromptBuilder(PromptBuilder):
    """
    Few-shot prompt builder for evaluation with example demonstrations.
    
    Formats few-shot examples into the prompt text to guide the model's output format.
    Can be used with both base and fine-tuned models.
    """
    
    SYSTEM_PROMPT = """You are a medical image captioning system for imageClef 2026. 
You will receive example images with their captions from the training set, followed by an image to caption. 
Use the examples to guide your caption style and format. Output only the caption, no other text."""
    
    PROMPT_TEXT = "Caption this image."
    
    def __init__(self, max_concepts_per_example: int = 10):
        """
        Initialize few-shot prompt builder.
        
        Args:
            max_concepts_per_example: Maximum number of concepts to include per example
        """
        self.max_concepts_per_example = max_concepts_per_example
    
    @staticmethod
    def _concepts_to_strings(concepts: Any, max_concepts: int = 10) -> List[str]:
        """
        Convert concept tensor/vector to list of concept description strings.
        
        Args:
            concepts: Concept tensor or list (1s where concepts are present)
            max_concepts: Maximum number of concepts to include
            
        Returns:
            List of concept description strings
        """
        from utils.cui_utils import CUIMapper
        
        concept_strings = []
        
        try:
            # Extract concept indices
            if isinstance(concepts, torch.Tensor):
                concept_indices = (concepts > 0).nonzero(as_tuple=False).flatten().tolist()
            else:
                concept_indices = [i for i, x in enumerate(concepts) if x > 0]
            
            # Convert to concept descriptions
            for idx in concept_indices[:max_concepts]:
                try:
                    cui_info = CUIMapper.index_to_cui(idx)
                    concept_strings.append(cui_info["desc"])
                except Exception:
                    pass
            
            if len(concept_indices) > max_concepts:
                concept_strings.append(f"... and {len(concept_indices) - max_concepts} more")
                
        except Exception:
            pass
        
        return concept_strings
    
    def build_messages(
        self,
        image: Image.Image,
        caption: Optional[str] = None,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
        """
        Build few-shot prompt messages with multimodal examples.
        
        Args:
            image: PIL Image to caption
            caption: Optional caption text (for training - adds assistant message with ground truth)
            few_shot_examples: Optional list of few-shot example dicts with 'image' and 'caption' for multimodal few-shot
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (messages, images_list) where:
            - messages: List of message dictionaries with image placeholders (not actual images)
            - images_list: List of PIL Images in the order they appear in messages
        """
        messages = []
        images_list = []
        
        # Add system prompt describing the task
        messages.append({
            "role": "system",
            "content": self.SYSTEM_PROMPT
        })
        
        # Add few-shot examples
        if few_shot_examples:
            # Add each example as user/assistant message pairs (multimodal)
            for example in few_shot_examples:
                example_image = example.get("image")
                example_caption = example.get("caption", "")
                
                if example_image is None:
                    continue  # Skip if no image
                
                # Add example image placeholder (image token will be inserted by chat template)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image"}
                    ]
                })
                images_list.append(example_image)
                
                # Add example caption as assistant response (text only)
                messages.append({
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": example_caption}
                    ]
                })
        
        # Add user message with the actual image to caption (with PROMPT_TEXT)
        messages.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": self.PROMPT_TEXT}
            ]
        })
        images_list.append(image)
        
        # Add assistant message with ground truth caption (for training)
        if caption is not None:
            messages.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": caption}
                ]
            })
        
        return messages, images_list


class RAGPromptBuilder(PromptBuilder):
    """
    RAG (Retrieval-Augmented Generation) prompt builder - placeholder for future implementation.
    
    Note: Future implementation may:
    - Retrieve context in real-time (during prediction, not training)
    - Support multimodal RAG (adding retrieved images to messages)
    - Handle dynamic prompt construction based on retrieved content
    
    To customize the prompt text, create a new PromptBuilder subclass.
    """

    SYSTEM_PROMPT = """You are a medical image captioning system for ImageCLEF 2026.

    You will receive captions of visually similar medical images retrieved from a database.

    Use them ONLY as medical context and style reference.
    Do NOT copy them.
    Generate a new caption for the given image.

    Output ONLY the caption text. Do NOT include any reasoning, thoughts, or explanations.
    """

    PROMPT_TEMPLATE = """Similar image captions:

    {retrieved_captions}

    Now caption this image."""

    def __init__(self, max_rag_examples: int = 10):
        """Initialize RAG prompt builder."""
        self.max_rag_examples = max_rag_examples

    def build_messages(
        self,
        image: Image.Image,
        caption: Optional[str] = None,
        retrieved_examples: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):

        messages = []
        images_list = []

        messages.append({
            "role": "system",
            "content": self.SYSTEM_PROMPT
        })

        # Format retrieved captions
        retrieved_text = ""

        if retrieved_examples:
            for i, ex in enumerate(retrieved_examples):
                cap = ex.get("caption", "")
                retrieved_text += f"{i+1}) {cap}\n"

        else:
            retrieved_text = "None"

        user_prompt = self.PROMPT_TEMPLATE.format(
            retrieved_captions=retrieved_text
        )

        messages.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt}
            ]
        })

        images_list.append(image)

        # Training mode
        if caption is not None:
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": caption}]
            })

        return messages, images_list


class PromptBuilderFactory:
    """Factory for creating prompt builders based on strategy name."""
    
    _builders = {
        "simple": SimplePromptBuilder,
        "few_shot": FewShotPromptBuilder,
        "rag": RAGPromptBuilder,
    }
    
    @classmethod
    def create(
        cls,
        strategy: str = "simple",
        **kwargs
    ) -> PromptBuilder:
        """
        Create a prompt builder based on strategy.
        
        Args:
            strategy: Strategy name ("simple", "rag", etc.)
            **kwargs: Additional arguments for specific builders (currently unused)
            
        Returns:
            PromptBuilder instance
            
        Raises:
            ValueError: If strategy is not recognized
            
        Note:
            To customize prompt text, create a new PromptBuilder subclass.
            Each builder class has its own fixed prompt text.
        """
        if strategy not in cls._builders:
            available = ", ".join(cls._builders.keys())
            raise ValueError(
                f"Unknown prompt strategy: {strategy}. "
                f"Available strategies: {available}"
            )
        
        builder_class = cls._builders[strategy]
        return builder_class(**kwargs)
    
    @classmethod
    def register_strategy(cls, name: str, builder_class: type):
        """
        Register a new prompt builder strategy.
        
        Args:
            name: Strategy name
            builder_class: PromptBuilder subclass
        """
        cls._builders[name] = builder_class


def select_few_shot_examples(
    dataset: Any,
    n_examples: int,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Select few-shot examples from a dataset randomly.
    
    Args:
        dataset: Dataset with __getitem__ returning dicts with 'image', 'caption', 'id', 'concepts'
        n_examples: Number of examples to select
        seed: Optional random seed for reproducibility (if None, uses truly random selection)
    
    Returns:
        List of example dictionaries with 'image', 'caption', 'id', 'concepts'
    """
    if len(dataset) < n_examples:
        raise ValueError(f"Dataset has only {len(dataset)} samples, but {n_examples} examples requested")
    
    # Set random seed only if provided (for reproducibility when needed)
    if seed is not None:
        random.seed(seed)
    
    # Randomly sample indices
    indices = random.sample(range(len(dataset)), n_examples)
    
    examples = []
    for idx in indices:
        sample = dataset[idx]
        examples.append({
            "image": sample["image"],
            "caption": sample.get("caption", ""),
            "id": sample.get("id", ""),
            # "concepts": sample.get("concepts"),
        })
    
    return examples


if __name__ == "__main__":
    """Test the prompt builder module."""
    from PIL import Image
    import numpy as np
    
    print("="*60)
    print("Testing Prompt Builder")
    print("="*60)
    
    # Test 1: Factory creation
    print("\n1. Testing factory creation...")
    try:
        builder = PromptBuilderFactory.create("simple")
        print("   ✓ Simple prompt builder created")
        
        builder_rag = PromptBuilderFactory.create("rag")
        print("   ✓ RAG prompt builder created")
    except Exception as e:
        print(f"   ✗ Factory creation failed: {e}")
        raise
    
    # Test 2: Simple prompt building
    print("\n2. Testing simple prompt building...")
    try:
        test_image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        messages = builder.build_messages(
            image=test_image,
            caption="Test caption",
        )
        print(f"   ✓ Messages created: {len(messages)} message(s)")
        print(f"   ✓ Message structure: {list(messages[0].keys())}")
    except Exception as e:
        print(f"   ✗ Simple prompt building failed: {e}")
        raise
    
    # Test 3: RAG prompt builder (should raise NotImplementedError)
    print("\n3. Testing RAG prompt builder (should raise NotImplementedError)...")
    try:
        messages_rag = builder_rag.build_messages(
            image=test_image,
            caption="Test caption",
        )
        print("   ✗ Should have raised NotImplementedError")
        raise AssertionError("RAGPromptBuilder should raise NotImplementedError")
    except NotImplementedError:
        print("   ✓ RAG prompt builder correctly raises NotImplementedError")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
        raise
    
    # Test 4: Error handling
    print("\n4. Testing error handling...")
    try:
        try:
            PromptBuilderFactory.create("unknown_strategy")
            print("   ✗ Should have raised ValueError")
        except ValueError as e:
            print(f"   ✓ Correctly raised ValueError: {str(e)[:50]}...")
    except Exception as e:
        print(f"   ✗ Error handling test failed: {e}")
        raise
    
    print("\n" + "="*60)
    print("✓ All prompt builder tests passed!")
    print("="*60)
