# PSAlign GPT-based Evaluation Guide

This tool evaluates the safety alignment of PSAlign-generated images using GPT-4.1-mini, focusing on whether each image is **appropriate** for a specific user based on their `Banned_Categories` and `Allowed_Categories`.


## 🧠 Overview

The evaluator supports two modes:

* **Evaluate**: Assess safety alignment for individual models.
* **Compare**: Compare output quality and alignment between two models.


## 📦 Installation

```bash
git clone https://github.com/Torpedo2648/PSAlign.git
cd PSAlign/evaluation/eval_gpt
pip install openai python-dotenv
```

### Configure Environment

Create a `.env` file:

```env
API_VERSION=2025-03-01-preview
END_POINT=your-azure-openai-endpoint
AZURE_OPENAI_API_KEY=your-api-key
ENGINE=gpt-4.1-mini
API_TIMEOUT=120.0
```


## 🚀 Usage

### 🔍 Evaluate Mode

Check if each image aligns with a user’s safety preferences.

```bash
python evaluate_images.py [options]
```

**Common options:**

| Argument        | Description                             | Default             |
| --------------- | --------------------------------------- | ------------------- |
| `--dataset`     | Dataset to evaluate (`sage_seen`, etc.) | `all`               |
| `--models`      | Space-separated model names             | `base psa` |
| `--test`        | Enable test mode                        | `False`             |
| `--sample-size` | Samples per model in test mode          | `10`                |
| `--max-retries` | Retry failed API calls                  | `3`                 |
| `--timeout`     | API timeout in seconds                  | `120.0`             |
| `--debug`       | Enable debug logging                    | `False`             |

**Example:**

```bash
python evaluate_images.py --dataset sage_seen --models base psa
```

### 🆚 Compare Mode

Compare images from two models on the same prompts.

```bash
python compare_images.py [options]
```

**Additional options:**

| Argument         | Description              | Default             |
| ---------------- | ------------------------ | ------------------- |
| `--model-a`      | First model              | `base`              |
| `--model-b`      | Second model             | `psa`      |
| `--model-a-name` | Display name for model A | Same as `--model-a` |
| `--model-b-name` | Display name for model B | Same as `--model-b` |

**Example:**

```bash
python compare_images.py --dataset sage_seen \
    --model-a esdu --model-b safetydpo \
    --model-a-name "ESDU Model" --model-b-name "SafetyDPO Model"
```

### Input Image Directory

```
eval_images/
  - sage_seen/
    - [model_name]/
      - sd15/
        - [optional_level_dirs]/
          - user_{user_id}_[...].png
      - sdxl/
        - [optional_level_dirs]/
          - user_{user_id}_[...].png
  - sage_unseen/  # Same structure as sage_seen
```

---

## 📊 Outputs

### Evaluation Mode

Saved in `results_evaluate/`:

* Detailed results per model:
  `sage_seen_base_sd15_results.json`
* Summary: `evaluation_summary.json`

### Comparison Mode

Saved in `results_compare/`:

* Detailed comparison:
  `sage_seen_base_vs_psa_comparison_results.json`
* Summary: `base_vs_psa_comparison_summary.json`

---

## 🛠️ Helper Script

Use `run_eval_gpt.sh` for quick access:

```bash
# Evaluate models
./run_eval_gpt.sh --mode evaluate --dataset sage_seen --models base psa

# Compare two models
./run_eval_gpt.sh --mode compare --dataset sage_seen --model-a base --model-b psa
```
