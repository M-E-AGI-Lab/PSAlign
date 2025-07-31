import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_user_embeddings(model_name, prompt_file, csv_file, embed_dir, debug_limit=None, batch_size=32):
    """
    Generate and save user information embeddings
    
    Args:
        model_name: Pre-trained model name
        prompt_file: Prompt template file path
        csv_file: User data CSV file path
        embed_dir: Directory to save embeddings
        debug_limit: Limit number of users for debugging (None for all)
        batch_size: Batch processing size
    """
    # Create output directory
    os.makedirs(embed_dir, exist_ok=True)
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    
    # Load prompt template
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    # Load user data
    df = pd.read_csv(csv_file)
    if debug_limit:
        df = df.head(debug_limit)
    
    # Generate input texts
    texts = []
    for _, row in df.iterrows():
        # Fill prompt template with user data
        prompt = prompt_template.replace("<<AGE>>", str(row['Age']))\
                               .replace("<<AGE_GROUP>>", str(row['Age_Group']))\
                               .replace("<<GENDER>>", str(row['Gender']))\
                               .replace("<<RELIGION>>", str(row['Religion']))\
                               .replace("<<MENTAL>>", str(row['Mental_Condition']))\
                               .replace("<<PHYSICAL>>", str(row['Physical_Condition']))\
                               .replace("<<BANNED>>", str(row['Banned_Categories']))\
                               .replace("<<ALLOWED>>", str(row['Allowed_Categories']))
        
        # Convert to chat format
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)
    
    # Generate embeddings in batches
    for batch_start in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[batch_start: batch_start + batch_size]
        
        with torch.inference_mode():
            # Tokenize batch
            model_inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(model.device)
            
            # Get hidden states
            outputs = model(**model_inputs, output_hidden_states=True, return_dict=True)
            last_hidden = outputs.hidden_states[-1]
            attention_mask = model_inputs["attention_mask"].unsqueeze(-1)
            
            # Calculate embeddings using masked averaging
            sum_hidden = (last_hidden * attention_mask).sum(dim=1)
            lengths = attention_mask.sum(dim=1)
            embs = (sum_hidden / lengths).to(torch.float32).cpu().numpy()
            
            # Save individual embeddings
            for i in range(embs.shape[0]):
                idx = batch_start + i
                out_path = os.path.join(embed_dir, f"{idx+1:07d}.npy")
                np.save(out_path, embs[i])


def main():
    """Main function to generate user embeddings"""
    # Configuration parameters
    config = {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "prompt_file": './get_embeds.txt',
        "csv_file": './users.csv',
        "embed_dir": './embeddings/',
        "debug_limit": None,  # Set to specific number for debugging, e.g., 100
        "batch_size": 32
    }
    
    # Generate embeddings
    generate_user_embeddings(**config)
    print(f"Embeddings saved to {config['embed_dir']}")


if __name__ == "__main__":
    main()