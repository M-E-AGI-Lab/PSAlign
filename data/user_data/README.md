# User Data Processing

This directory contains tools and scripts for processing user data and generating personalized safety embeddings using LLM.

## Directory Structure

```
user_data/
├── num_1k/                    # 1000-user dataset
│   ├── embeddings_pipeline.py # Main embedding generation script
│   ├── get_banned.txt        # Prompt template for banned categories
│   └── get_embeds.txt        # Prompt template for embedding generation
└── ...                       # Other dataset sizes
```

## Embedding Pipeline

The `embeddings_pipeline.py` script processes user data to generate personalized safety embeddings:

1. Reads user demographic data in JSON format
2. Determines banned/allowed content categories using LLM
3. Generates personalized embeddings based on user profiles

### Input Format

User data should be in JSON format with the following fields:
```json
{
    "Age": int,
    "Age_Group": string,
    "Gender": string,
    "Religion": string,
    "Physical_Condition": string,
    "Mental_Condition": string
}
```

### Usage

```bash
# Configure model and paths in main()
python embeddings_pipeline.py
```

### Output

- Embeddings are saved as `.npy` files with shape (1, 1, llm_hidden_size)
- File naming format: `{user_id:07d}.npy` (e.g., 0000001.npy)
