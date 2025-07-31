import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple


def batch_user_data_to_embedding(
    user_json_list: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    banned_prompt_template: str,
    embed_prompt_template: str,
    all_categories: set = {"Hate", "Harassment", "Violence", "Self-Harm", "Sexuality", "Shocking", "Propaganda"}
) -> Tuple[np.ndarray, List[List[str]], List[List[str]]]:
    """
    Batch process user JSON data, sum over num_tokens dimension (since num_tokens=1)
    Output shape: (batch_size, num_tokens, llm_hidden_size)
    
    Args:
        user_json_list: List of JSON strings (each corresponds to one user data)
        model: Pre-trained language model
        tokenizer: Corresponding tokenizer
        banned_prompt_template: Prompt template for generating banned/allowed categories
        embed_prompt_template: Prompt template for generating embeddings
        all_categories: Set of content categories
    
    Returns:
        tuple: (embeddings, all_banned, all_allowed)
            - embeddings: numpy array with shape (batch_size, num_tokens, llm_hidden_size) where num_tokens=1
            - all_banned: List of banned categories for each user
            - all_allowed: List of allowed categories for each user
    """
    # Step 1: Parse user JSON data in batch
    try:
        user_data_list = [json.loads(js) for js in user_json_list]
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in list: {str(e)}")
    batch_size = len(user_data_list)

    # Step 2: Generate banned/allowed lists in batch
    banned_prompts = [banned_prompt_template.replace("<<USER_DATA>>", js) for js in user_json_list]
    messages_list = [[{"role": "system", "content": "You are a helpful assistant."},
                     {"role": "user", "content": prompt}] for prompt in banned_prompts]
    chat_texts = [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) 
                 for msgs in messages_list]

    with torch.no_grad():
        model_inputs = tokenizer(chat_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=512, pad_token_id=tokenizer.pad_token_id)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def extract_batch_banned(responses):
        """Extract banned and allowed categories from LLM responses"""
        all_banned, all_allowed = [], []
        for resp in responses:
            banned = [cat for cat in all_categories if cat.lower() in resp.lower()]
            all_banned.append(banned)
            all_allowed.append(list(all_categories - set(banned)))
        return all_banned, all_allowed
    
    all_banned, all_allowed = extract_batch_banned(responses)

    # Step 3: Generate embeddings (core modification: sum over num_tokens dimension while preserving shape)
    # 3.1 Build embedding prompts
    embed_texts = []
    for i in range(batch_size):
        user_data = user_data_list[i]
        prompt = embed_prompt_template\
            .replace("<<AGE>>", str(user_data.get("Age", "")))\
            .replace("<<AGE_GROUP>>", str(user_data.get("Age_Group", "")))\
            .replace("<<GENDER>>", str(user_data.get("Gender", "")))\
            .replace("<<RELIGION>>", str(user_data.get("Religion", "")))\
            .replace("<<MENTAL>>", str(user_data.get("Mental_Condition", "")))\
            .replace("<<PHYSICAL>>", str(user_data.get("Physical_Condition", "")))\
            .replace("<<BANNED>>", str(all_banned[i]))\
            .replace("<<ALLOWED>>", str(all_allowed[i]))
        embed_msgs = [{"role": "user", "content": prompt}]
        embed_texts.append(tokenizer.apply_chat_template(embed_msgs, tokenize=False, add_generation_prompt=True))

    # 3.2 Calculate embeddings and sum over num_tokens dimension (since num_tokens=1, shape is preserved after sum)
    with torch.inference_mode():
        inputs = tokenizer(embed_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.hidden_states[-1]  # Shape: (batch_size, num_tokens, llm_hidden_size)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)  # Shape: (batch_size, num_tokens, 1)

        # Filter invalid tokens with mask, then sum over num_tokens dimension (dim=1) while keeping dimension
        # Since num_tokens=1, shape after sum is still (batch_size, 1, llm_hidden_size)
        masked_hidden = last_hidden * attention_mask  # Filter padding tokens
        sum_hidden = masked_hidden.sum(dim=1, keepdim=True)  # Sum and preserve num_tokens dimension

        # Convert to numpy with shape (batch_size, num_tokens, llm_hidden_size)
        embeddings = sum_hidden.to(torch.float32).cpu().numpy()

    return embeddings, all_banned, all_allowed


def main():
    """Main function demonstrating batch user embedding processing"""
    # Configuration constants
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    BANNED_PROMPT_FILE = "./get_banned.txt"
    EMBED_PROMPT_FILE = "./get_embeds.txt"
    EMBED_SAVE_DIR = "./embeddings_debug"

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # Load prompt templates
    with open(BANNED_PROMPT_FILE, "r", encoding="utf-8") as f:
        banned_prompt = f.read()
    with open(EMBED_PROMPT_FILE, "r", encoding="utf-8") as f:
        embed_prompt = f.read()

    # Sample batch user data
    sample_users = [
        '''{"Age": 25, "Age_Group": "Adult", "Gender": "Male", "Religion": "None", "Physical_Condition": "Healthy", "Mental_Condition": "Stable"}''',
        '''{"Age": 16, "Age_Group": "Minor", "Gender": "Female", "Religion": "Christianity", "Physical_Condition": "Healthy", "Mental_Condition": "Fragile"}'''
    ]

    # Process batch embeddings
    embeddings, banned_list, allowed_list = batch_user_data_to_embedding(
        user_json_list=sample_users,
        model=model,
        tokenizer=tokenizer,
        banned_prompt_template=banned_prompt,
        embed_prompt_template=embed_prompt
    )

    # Display results
    print(f"Embedding shape: {embeddings.shape}")  # (batch_size, num_tokens, llm_hidden_size)
    print("Banned lists:", banned_list)
    print("Allowed lists:", allowed_list)

    # Save individual embeddings
    os.makedirs(EMBED_SAVE_DIR, exist_ok=True)
    for i in range(embeddings.shape[0]):
        out_path = os.path.join(EMBED_SAVE_DIR, f"{i+1:07d}.npy")
        np.save(out_path, embeddings[i])


if __name__ == "__main__":
    main()