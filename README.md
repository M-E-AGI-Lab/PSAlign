# PSAlign: Personalized Safety Alignment for Text-to-Image Diffusion Models

This repository contains the official implementation of the paper:
**"Personalized Safety Alignment for Text-to-Image Diffusion Models"** (2025, under review).

## 🧠 Overview

**PSAlign** is a framework designed to personalize safety alignment in text-to-image diffusion models. It supports user-aware filtering of sensitive or inappropriate content while maintaining high visual fidelity and creativity.

---

## ✨ Features

* ✅ Support for both **Stable Diffusion 1.5** and **SDXL**
* 🔐 Personalized safety filtering via a **PSA Adapter**
* 🧬 Integration of **user-specific safety profiles** using LLM-generated embeddings
* ⚡ Efficient **batch and distributed inference** with `Accelerate`
* 🛠️ Extensible architecture for plug-and-play alignment methods

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Torpedo2648/PSAlign.git
cd PSAlign
```

### 2. Setup Environment

```bash
conda env create -f environment.yaml
conda activate psalign
```

---

## 🚀 Usage

### 🔧 Training PSA Adapters

To train a PSA adapter for safety alignment:

```bash
bash launchers/train_psa_sd15.sh    # for Stable Diffusion 1.5
bash launchers/train_psa_sdxl.sh    # for SDXL
```

### 🎨 Inference with PSA

To run inference using a trained PSA adapter:

```bash
python infer.py \
    --sd_model [path_to_diffusion_model] \
    --psa_path [path_to_psa_adapter] \
    --save_path [output_dir] \
    --meta_data [input_metadata.jsonl] \
    --load_psa \
    --sdxl  # Add if using SDXL
```

#### Required Arguments:

* `--sd_model`: Path to the base Stable Diffusion or SDXL model
* `--save_path`: Output directory for generated images
* `--meta_data`: A JSONL file with fields: `file_name`, `caption`, and `user_data`

#### Optional Arguments:

* `--psa_path`: Path to trained PSA adapter
* `--load_psa`: Whether to enable the PSA adapter
* `--llm_model`: LLM path or name used to create user embeddings
* `--embeds_folder`: Path to directory of user embedding vectors
* `--batch_size`: Batch size for inference (default: 1)

To view all available arguments:

```bash
python infer.py --help
```

---

## 📂 Input Format

The input JSONL file for inference should look like:

```json
{"file_name": "img_0001.png", "caption": "a man holding a rifle", "user_data": {"user_id": "123", "safety_level": "strict"}}
```

---

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{PSAlign2025,
  title={Personalized Safety Alignment for Text-to-Image Diffusion Models},
  author={[Authors]},
  booktitle={[Conference]},
  year={2025}
}
```

---

## 🤝 Acknowledgements

We gratefully acknowledge the following open-source projects and research works, which significantly contributed to the design, implementation, and evaluation of **PSAlign**:

* [**Q16** (ML Research)](https://github.com/ml-research/Q16.git) – for providing a high-quality safety classifier, which we use to compute our *InPro* safety evaluation metric.

* [**DiffusionDPO** (Salesforce AI Research)](https://github.com/SalesforceAIResearch/DiffusionDPO) – as a foundational baseline for diffusion-based safety alignment using reinforcement learning from human feedback (RLHF).

* [**SafetyDPO** (Visualignment)](https://github.com/Visualignment/SafetyDPO) – for inspiration on LoRA-based safety fine-tuning and benchmark setup, which we re-implemented and adapted in our comparison.

* [**PPD: Personalized Preference Fine-tuning of Diffusion Models**](https://arxiv.org/abs/2501.06655) - This work introduces a multi-reward RLHF objective for aligning diffusion models with diverse and personalized user preferences. We acknowledge its contribution in motivating personalized fine-tuning strategies and embedding-based preference conditioning, which are closely related to the design of PSAlign.
