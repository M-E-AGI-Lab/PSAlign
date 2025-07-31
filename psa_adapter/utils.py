"""
Utility Functions for Personalized Safety Alignment (PSA).

This module provides helper functions for:
1. PyTorch version compatibility checking
2. Reproducible random number generation
3. User data processing and embedding generation
4. Batch processing of user preferences

The utilities focus on efficient processing of user data and integration with
language models for safety alignment in diffusion models.
"""

import json
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def is_torch2_available() -> bool:
    """
    Check if PyTorch 2.0 features are available.
    
    Returns:
        bool: True if scaled_dot_product_attention is available (PyTorch 2.0+)
    """
    return hasattr(F, "scaled_dot_product_attention")


def get_generator(
    seed: Optional[Union[int, List[int]]], 
    device: str
) -> Optional[Union[torch.Generator, List[torch.Generator]]]:
    """
    Create deterministic random number generator(s) for reproducibility.
    
    Args:
        seed: Single seed value or list of seed values
        device: Device to create the generator on ('cuda', 'cpu', etc.)
        
    Returns:
        Single generator, list of generators, or None if no seed provided
    """
    if seed is not None:
        if isinstance(seed, list):
            return [torch.Generator(device).manual_seed(s) for s in seed]
        return torch.Generator(device).manual_seed(seed)
    return None

def batch_user_data_to_embedding(
    user_json_list: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    banned_prompt_template: str,
    embed_prompt_template: str,
    all_categories: Set[str] = {
        "Hate", "Harassment", "Violence", 
        "Self-Harm", "Sexuality", "Shocking", "Propaganda"
    }
) -> Tuple[torch.Tensor, List[List[str]], List[List[str]]]:
    """
    Process user data in batches to generate embeddings and content preferences.
    
    This function:
    1. Processes a batch of user JSON data
    2. Generates banned/allowed content categories
    3. Creates embeddings using an LLM
    
    The output embeddings maintain the shape (batch_size, num_tokens=1, llm_hidden_size)
    to preserve compatibility with the attention mechanisms.
    
    Args:
        user_json_list: List of JSON strings containing user data
        model: Pre-trained language model for embedding generation
        tokenizer: Corresponding tokenizer for the model
        banned_prompt_template: Template for generating banned/allowed categories
        embed_prompt_template: Template for generating user embeddings
        all_categories: Set of all possible content categories
        
    Returns:
        tuple:
            - embeddings: User embeddings tensor (batch_size, 1, hidden_size)
            - all_banned: List of banned categories per user
            - all_allowed: List of allowed categories per user
            
    Raises:
        ValueError: If any JSON string is invalid
    """
    # Parse user JSON data
    try:
        user_data_list = [json.loads(js) for js in user_json_list]
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse user data JSON: {str(e)}")
    batch_size = len(user_data_list)

    # Generate content preference prompts
    banned_prompts = [
        banned_prompt_template.replace("<<USER_DATA>>", js) 
        for js in user_json_list
    ]
    
    # Prepare chat messages
    messages_list = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ] 
        for prompt in banned_prompts
    ]
    
    # Convert to model format
    chat_texts = [
        tokenizer.apply_chat_template(
            msgs, 
            tokenize=False, 
            add_generation_prompt=True
        ) 
        for msgs in messages_list
    ]

    # Generate content preferences
    with torch.no_grad():
        # Tokenize inputs
        model_inputs = tokenizer(
            chat_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(model.device)
        
        # Generate responses
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.pad_token_id
        )
        
        # Extract new tokens
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids 
            in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # Decode responses
        responses = tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )

    def extract_batch_banned(
        responses: List[str]
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """Extract banned and allowed categories from model responses."""
        all_banned, all_allowed = [], []
        for response in responses:
            # Identify banned categories
            banned = [
                category 
                for category in all_categories 
                if category.lower() in response.lower()
            ]
            all_banned.append(banned)
            
            # Calculate allowed categories
            allowed = list(all_categories - set(banned))
            all_allowed.append(allowed)
            
        return all_banned, all_allowed

    # Process model responses
    all_banned, all_allowed = extract_batch_banned(responses)

    # Generate embeddings from user data
    embed_texts = []
    for i, user_data in enumerate(user_data_list):
        # Fill template with user data
        prompt = embed_prompt_template
        replacements = {
            "<<AGE>>": str(user_data.get("Age", "")),
            "<<AGE_GROUP>>": str(user_data.get("Age_Group", "")),
            "<<GENDER>>": str(user_data.get("Gender", "")),
            "<<RELIGION>>": str(user_data.get("Religion", "")),
            "<<MENTAL>>": str(user_data.get("Mental_Condition", "")),
            "<<PHYSICAL>>": str(user_data.get("Physical_Condition", "")),
            "<<BANNED>>": str(all_banned[i]),
            "<<ALLOWED>>": str(all_allowed[i])
        }
        for key, value in replacements.items():
            prompt = prompt.replace(key, value)
            
        # Convert to chat format
        embed_msgs = [{"role": "user", "content": prompt}]
        embed_texts.append(
            tokenizer.apply_chat_template(
                embed_msgs,
                tokenize=False,
                add_generation_prompt=True
            )
        )

    # Generate embeddings
    with torch.inference_mode():
        # Tokenize input texts
        inputs = tokenizer(
            embed_texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)
        
        # Get model outputs 
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Extract last hidden state
        last_hidden = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # Create attention mask for valid tokens
        attention_mask = inputs["attention_mask"].unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # Apply mask and aggregate
        # We maintain num_tokens=1 dimension for compatibility
        masked_hidden = last_hidden * attention_mask
        embeddings = masked_hidden.sum(dim=1, keepdim=True)  # [batch_size, 1, hidden_size]

    return embeddings, all_banned, all_allowed
