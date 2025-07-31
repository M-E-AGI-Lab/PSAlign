import pandas as pd
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, List

# Complete set of content categories
ALL_CATEGORIES = {"Hate", "Harassment", "Violence", "Self-Harm", "Sexuality", "Shocking", "Propaganda"}


def process_user_data(user_info_json: str, model, tokenizer, base_prompt: str) -> Tuple[List[str], List[str]]:
    """Process single user information and return banned and allowed category lists"""
    
    def extract_banned_categories(response_text: str) -> List[str]:
        """Extract banned categories from LLM response text"""
        response_lower = response_text.lower()
        return [cat for cat in ALL_CATEGORIES if cat.lower() in response_lower]
    
    # Build complete prompt with user data
    filled_prompt = base_prompt.replace("<<USER_DATA>>", user_info_json)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": filled_prompt}
    ]
    
    # Generate LLM response
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    
    # Extract only the generated portion
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Extract banned and allowed categories
    try:
        banned = extract_banned_categories(response)
    except Exception:
        banned = []
    
    allowed = list(ALL_CATEGORIES - set(banned))
    return banned, allowed


def main():
    """Main function to process user data with LLM model"""
    # Load model and tokenizer
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load prompt template and user data
    with open("get_banned.txt", "r") as f:
        base_prompt = f.read()
    
    df = pd.read_csv("users_debug.csv")
    
    # Process user data in batch
    banned_list, allowed_list = [], []
    
    for _, row in df.iterrows():
        user_data = {
            "Age": int(row["Age"]),
            "Age_Group": row["Age_Group"],
            "Gender": row["Gender"],
            "Religion": row["Religion"],
            "Physical_Condition": row["Physical_Condition"],
            "Mental_Condition": row["Mental_Condition"],
        }
        
        user_json = json.dumps(user_data, indent=2)
        banned, allowed = process_user_data(user_json, model, tokenizer, base_prompt)
        
        banned_list.append(str(banned))
        allowed_list.append(str(allowed))
    
    # Save results to CSV
    df["Banned_Categories"] = banned_list
    df["Allowed_Categories"] = allowed_list
    df.to_csv("users_debug.csv", index=False)
    print("Done. Results saved.")


if __name__ == "__main__":
    main()