# PSAlign: Personalized Safety Alignment for Text-to-Image Diffusion Models

This is the official repository for the paper:  
**[Personalized Safety Alignment for Text-to-Image Diffusion Models](https://arxiv.org/abs/2507.08261)**

[[Project Page]](https://anonymous.url/psalign) | [[SAGE Dataset]](https://anonymous.url/sage_dataset) | [[Pretrained Models]](https://anonymous.url/psa_models)


## 🧠 Overview

**PSAlign** is a novel framework enabling **personalized safety alignment** in text-to-image diffusion models. It dynamically adapts safety mechanisms to individual users’ characteristics (e.g., age, gender, cultural background) while preserving creativity and image fidelity.  

Key features:  
- **Personalization**: Adjusts safety thresholds based on user profiles (e.g., stricter for minors, culturally aware for diverse groups).  
- **Fidelity Preservation**: Maintains image quality and text alignment while suppressing harmful content.  
- **Compatibility**: Works with Stable Diffusion 1.5 and SDXL via lightweight adapters (no full model retraining).  


## 📂 Project Structure

```
PSAlign/
├── environment.yaml       # Conda environment config
├── train.py               # PSA adapter training script
├── infer.py               # Inference script
├── launchers/             # One-click scripts (training/inference for SD1.5/SDXL)
├── psa_adapter/           # Core PSA adapter implementation
├── evaluation/            # Evaluation tools
│   └── eval_gpt/          # GPT-based safety alignment evaluation
├── dataset/               # Dataset handling (data loading)
├── data/                  # Data files (user embeddings, SAGE dataset, user info)
└── trained_models/        # Pretrained models (PSA adapters for SD1.5/SDXL)
```

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Torpedo2648/PSAlign.git
cd PSAlign
```

### 2. Setup Environment

We recommend using Conda for environment management:

```bash
# Create and activate environment
conda env create -f environment.yaml
conda activate psa

# Verify installation
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

#### Notes:
- Requires Python 3.10+ and CUDA 11.7+ (for GPU acceleration).  
- For CPU-only usage, modify `environment.yaml` to remove CUDA-dependent packages.  


## 📚 SAGE Dataset  

**SAGE** (*Safety-Aware Generation for Everyone*) is the first dataset for **personalized safety alignment** in text-to-image generation, enabling models to adapt to individual user characteristics (age, culture, etc.).  

### Key Features  
- 100K+ image-prompt pairs with "safe" vs "unsafe" variants.  
- 10 safety categories (e.g., harassment, violence) with 800+ harmful concepts.  
- User metadata (age, gender, religion, etc.) for personalization.  
- Split into train/val/test_seen/test_unseen for robust evaluation.  

### Download  
```bash
wget https://anonymous.url/sage_dataset -O data/sage/sage_dataset.zip
unzip data/sage/sage_dataset.zip -d data/sage/
```

### Structure  
```
data/sage/
├── [train/val/test_seen/test_unseen]/
│   ├── metadata.jsonl  # Annotations: prompts, labels, user profiles
│   └── [image files]   # e.g., user_0000030_harassment_00001_s.jpg
```


## 🚀 Usage

### 🔧 Training PSA Adapters

PSA adapters are lightweight (8-15MB) and train quickly on a single GPU.

#### For Stable Diffusion 1.5
```bash
bash launchers/train_psa_sd15.sh
```
- **Inputs**: SAGE dataset + base SD1.5 model.  
- **Output**: Trained adapter saved to `trained_models/psa-sd15/`.  

#### For SDXL
```bash
bash launchers/train_psa_sdxl.sh
```
- Optimized for SDXL’s dual-text encoder architecture.  
- Output saved to `trained_models/psa-sdxl/`.  

#### Key Training Parameters (modify in `launchers/*.sh`):
- `BATCH_SIZE`: Adjust based on GPU memory (default: 4 for 24GB GPUs).  
- `MAX_STEPS`: Training steps (default: 10,000 for convergence).  
- `LR`: Learning rate (default: 2e-4 for adapters).  


### 🎨 Inference

Generate images with personalized safety alignment using pre-trained adapters.

#### Stable Diffusion 1.5
```bash
# Base model (no safety alignment)
bash launchers/infer_sd15_base.sh

# With PSA adapter (personalized safety)
bash launchers/infer_sd15_psa.sh

# With PSA + LLM-generated user embeddings
bash launchers/infer_sd15_psa_llm.sh
```

#### SDXL
```bash
# Base model
bash launchers/infer_sdxl_base.sh

# With PSA adapter
bash launchers/infer_sdxl_psa.sh

# With PSA + LLM-generated user embeddings
bash launchers/infer_sdxl_psa_llm.sh
```

## 📊 Evaluation

Follow these steps to reproduce the paper’s evaluation results.

### 1. Generate Evaluation Images
First, generate images for all models (PSAlign + baselines) across benchmark datasets:

```bash
cd evaluation
# Generate for all datasets (recommended)
for dataset in debug coco_10k i2p_4073 CoProv2_test sage_unseen ud_1434; do
  export DATASET=$dataset
  bash scripts/run_gen.sh
done
```
- **Output**: Images saved to `eval_images/<dataset>/<model>/` (e.g., `eval_images/coco_10k/psa/sd15/level_3`).  


### 2. Quantitative Metrics (FID, CLIPScore, InPro)
Evaluate image fidelity, text alignment, and harmful content suppression:

```bash
# Run with GPUs 0,1,2,3 (adjust based on available GPUs)
python scripts/run_eval.py --gpus 0,1,2,3 --output eval_results.csv
```
- **Metrics**:  
  - `FID`: Measures realism (lower = better).  
  - `CLIPScore`: Measures text-image alignment (higher = better).  
  - `InPro`: Measures inappropriate content (lower = better).  


### 3. Personalized Safety Alignment (GPT-based)
Assess personalized safety via pass rate (compliance with user requirements) and win rate (comparison to baselines):

```bash
cd eval_gpt

# Evaluate pass rate for PSAlign vs. baselines
bash run_eval_gpt.sh --mode evaluate --dataset all --models base safetydpo psa

# Compare PSAlign vs. SafetyDPO (win rate)
bash run_eval_gpt.sh --mode compare --dataset all --model-a safetydpo --model-b psa
```
- **Output**: Results saved to `results_evaluate/` or `results_compare/` (includes GPT judgments and summary stats).  


## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.  


## 🤝 Acknowledgements

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) for base model.  
- [SafetyDPO](https://github.com/Visualignment/SafetyDPO) for baseline safety tuning.  
- [Q16](https://github.com/ml-research/Q16) for safety classification.  
- [DiffusionDPO](https://github.com/SalesforceAIResearch/DiffusionDPO) for RLHF framework.  


## 📖 Citation

If you use this work, please cite:

```bibtex
@article{lei2025psalign,
  title={Personalized Safety Alignment for Text-to-Image Diffusion Models},
  author={Yu Lei and Jinbin Bai and Aosong Feng and Qingyu Shi and Kaidong Yu},
  journal={arXiv preprint arXiv:2507.08261},
  year={2025}
}
```# PSAlign: Personalized Safety Alignment for Text-to-Image Diffusion Models

This is the official repository for the paper:  
**[Personalized Safety Alignment for Text-to-Image Diffusion Models](https://arxiv.org/abs/2507.08261)**

[[Project Page]](https://anonymous.url/psalign) | [[SAGE Dataset]](https://anonymous.url/sage_dataset) | [[Pretrained Models]](https://anonymous.url/psa_models)


## 🧠 Overview

**PSAlign** is a novel framework enabling **personalized safety alignment** in text-to-image diffusion models. It dynamically adapts safety mechanisms to individual users’ characteristics (e.g., age, gender, cultural background) while preserving creativity and image fidelity.  

Key features:  
- **Personalization**: Adjusts safety thresholds based on user profiles (e.g., stricter for minors, culturally aware for diverse groups).  
- **Fidelity Preservation**: Maintains image quality and text alignment while suppressing harmful content.  
- **Compatibility**: Works with Stable Diffusion 1.5 and SDXL via lightweight adapters (no full model retraining).  


## 📂 Project Structure

```
PSAlign/
├── launchers/           # Training/inference scripts for different models
├── evaluation/          # Evaluation pipeline (metrics, image generation)
│   ├── scripts/         # Evaluation scripts (FID, CLIPScore, InPro)
│   └── eval_gpt/        # GPT-based personalized safety evaluation
├── data/                # Datasets and user embeddings
│   ├── embeddings/      # Precomputed user embeddings
│   └── sage/            # SAGE dataset files
├── models/              # Model architectures (PSA adapters)
├── utils/               # Helper functions (data loading, visualization)
└── environment.yaml     # Conda environment configuration
```


## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Torpedo2648/PSAlign.git
cd PSAlign
```

### 2. Setup Environment

We recommend using Conda for environment management:

```bash
# Create and activate environment
conda env create -f environment.yaml
conda activate psa

# Verify installation
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

#### Notes:
- Requires Python 3.10+ and CUDA 11.7+ (for GPU acceleration).  
- For CPU-only usage, modify `environment.yaml` to remove CUDA-dependent packages.  


## 🚀 Usage

### 🔧 Training PSA Adapters

PSA adapters are lightweight (8-15MB) and train quickly on a single GPU.

#### For Stable Diffusion 1.5
```bash
bash launchers/train_psa_sd15.sh
```
- **Inputs**: SAGE dataset + base SD1.5 model.  
- **Output**: Trained adapter saved to `trained_models/psa-sd15/`.  

#### For SDXL
```bash
bash launchers/train_psa_sdxl.sh
```
- Optimized for SDXL’s dual-text encoder architecture.  
- Output saved to `trained_models/psa-sdxl/`.  

#### Key Training Parameters (modify in `launchers/*.sh`):
- `BATCH_SIZE`: Adjust based on GPU memory (default: 4 for 24GB GPUs).  
- `MAX_STEPS`: Training steps (default: 10,000 for convergence).  
- `LR`: Learning rate (default: 2e-4 for adapters).  


### 🎨 Inference

Generate images with personalized safety alignment using pre-trained adapters.

#### Stable Diffusion 1.5
```bash
# Base model (no safety alignment)
bash launchers/infer_sd15_base.sh

# With PSA adapter (personalized safety)
bash launchers/infer_sd15_psa.sh

# With PSA + LLM-generated user embeddings
bash launchers/infer_sd15_psa_llm.sh
```

#### SDXL
```bash
# Base model
bash launchers/infer_sdxl_base.sh

# With PSA adapter
bash launchers/infer_sdxl_psa.sh

# With PSA + LLM-generated user embeddings
bash launchers/infer_sdxl_psa_llm.sh
```

#### Key Inference Parameters:
- `SD_MODEL`: Path to base model (e.g., `runwayml/stable-diffusion-v1-5`).  
- `PSA_PATH`: Path to trained PSA adapter (e.g., `trained_models/psa-sd15`).  
- `META_FILE`: CSV with prompts and user metadata (format: `prompt,user_id,age,gender,...`).  
- `LEVEL`: Safety strictness (0 = lenient, 5 = strict; auto-adjusted via user embeddings if using `_llm.sh` scripts).  


## 📊 Evaluation

Follow these steps to reproduce the paper’s evaluation results.

### 1. Generate Evaluation Images
First, generate images for all models (PSAlign + baselines) across benchmark datasets:

```bash
cd evaluation
# Generate for all datasets (recommended)
for dataset in debug coco_10k i2p_4073 CoProv2_test sage_unseen ud_1434; do
  export DATASET=$dataset
  bash scripts/run_gen.sh
done
```
- **Output**: Images saved to `eval_images/<dataset>/<model>/` (e.g., `eval_images/coco_10k/psa/sd15/level_3`).  


### 2. Quantitative Metrics (FID, CLIPScore, InPro)
Evaluate image fidelity, text alignment, and harmful content suppression:

```bash
# Run with GPUs 0,1,2,3 (adjust based on available GPUs)
python scripts/run_eval.py --gpus 0,1,2,3 --output eval_results.csv
```
- **Metrics**:  
  - `FID`: Measures realism (lower = better).  
  - `CLIPScore`: Measures text-image alignment (higher = better).  
  - `InPro`: Measures inappropriate content (lower = better).  


### 3. Personalized Safety Alignment (GPT-based)
Assess personalized safety via pass rate (compliance with user requirements) and win rate (comparison to baselines):

```bash
cd eval_gpt

# Evaluate pass rate for PSAlign vs. baselines
bash run_eval_gpt.sh --mode evaluate --dataset all --models base safetydpo psa

# Compare PSAlign vs. SafetyDPO (win rate)
bash run_eval_gpt.sh --mode compare --dataset all --model-a safetydpo --model-b psa
```
- **Output**: Results saved to `results_evaluate/` or `results_compare/` (includes GPT judgments and summary stats).  


## 📚 SAGE Dataset

**SAGE** (*Safety-Aware Generation Evaluation*) is a dataset for personalized safety training, with:  
- 100K+ images paired with safety annotations.  
- 10 safety categories (e.g., violence, harassment) with severity scores (0-5).  
- User metadata (age, gender, religion, etc.) for personalization.  

### Download
```bash
# Download via link (requires access)
wget https://anonymous.url/sage_dataset -O data/sage/sage_dataset.zip
unzip data/sage/sage_dataset.zip -d data/sage/
```

### Structure
```
sage_dataset/
├── images/          # Raw images
├── annotations.csv  # Safety labels + user metadata
└── prompts.csv      # Text prompts used to generate images
```


## 💾 Pretrained Models

| Model Name       | Base Model         | Size  | Use Case                  | Download Link                              |
|------------------|--------------------|-------|---------------------------|--------------------------------------------|
| `psa-sd15`       | SD 1.5             | 8MB   | General personalized safety | [Link](https://anonymous.url/psa_models)    |
| `psa-sdxl`       | SDXL               | 12MB  | High-fidelity images      | [Link](https://anonymous.url/psa_models)    |
| `psa-sdxl-vipl`  | SDXL + VIPL        | 15MB  | Enhanced visual quality   | [Link](https://anonymous.url/psa_models)    |


## ❓ FAQ

- **Q: What GPU is required?**  
  A: Training: ≥12GB VRAM (e.g., RTX 3090). Inference: ≥8GB VRAM.  

- **Q: How to create custom user embeddings?**  
  A: Use `utils/generate_user_embeddings.py` with user metadata (age, gender, etc.).  

- **Q: Why is my FID score high?**  
  A: Ensure the base model is correctly loaded and `eval_images` contains all generated samples.  


## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.  


## 🤝 Acknowledgements

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) for base model.  
- [SafetyDPO](https://github.com/Visualignment/SafetyDPO) for baseline safety tuning.  
- [Q16](https://github.com/ml-research/Q16) for safety classification.  
- [DiffusionDPO](https://github.com/SalesforceAIResearch/DiffusionDPO) for RLHF framework.  


## 📖 Citation

If you use this work, please cite:

```bibtex
@article{lei2025psalign,
  title={Personalized Safety Alignment for Text-to-Image Diffusion Models},
  author={Yu Lei and Jinbin Bai and Aosong Feng and Qingyu Shi and Kaidong Yu},
  journal={arXiv preprint arXiv:2507.08261},
  year={2025}
}
```
