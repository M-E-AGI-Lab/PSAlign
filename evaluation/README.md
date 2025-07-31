# PSAlign Evaluation Guide

This guide outlines the evaluation pipeline for the **PSAlign** framework, consisting of two main stages:

1. Generating test images using various baseline and safety-aligned models
2. Computing a suite of quantitative evaluation metrics

> 🔒 *This repository accompanies our paper* **"Personalized Safety Alignment for Text-to-Image Diffusion Models"** (under review).
> For code anonymity, all identifiable file paths and usernames have been removed or generalized.

---

## 🔧 Pretrained Model Dependencies

Before running the evaluation, ensure the following pretrained models are correctly cached.By default, the evaluation automatically downloads and caches them. If you're in an internal network, you can manually download and place them in the correct location.


```bash
# CLIP model (ViT-L-14) used by Q16 classifier
~/.cache/clip/ViT-L-14.pt

# Inception model for FID computation
~/.cache/torch/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth

# LAION-CLIP model (ViT-H-14) for CLIP score
$HF_HOME/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/1c2b8495b28150b8a4922ee1c8edee224c284c0c/open_clip_model.safetensors
```

---

## 🔁 Reproducible Baselines

To ensure **fair and consistent comparisons**, we re-trained all baseline safety alignment methods on our datasets using their publicly available implementations and aligned training objectives. This allows for meaningful evaluation under the same generation settings and data splits.

The following pre-trained model weights for all evaluated methods are available via **[Google Drive (link)](https://drive.google.com)** for easy access and reproducibility.

### 📦 Baseline Checkpoints

| Method        | Model Type | Weight Path (Local Path Example)                                                           |
| ------------- | ---------- | ------------------------------------------------------------------------------------------ |
| **SafetyDPO** | SD15       | `~/workspace/SafetyDPO/trained_models/safetydpo-sd15/`                                     |
|               | SDXL       | `~/workspace/SafetyDPO/trained_models/safetydpo-sdxl/`                                     |
| **ESD-U**     | SD15       | `~/workspace/erasing/esd-models/sd/esdu.safetensors`                                       |
|               | SDXL       | `~/workspace/erasing/esd-models/sdxl/esdu.safetensors`                                     |
| **UCE**       | SD15       | `~/workspace/unified-concept-editing/uce_models/nsfw_uce_sd.safetensors`                   |
|               | SDXL       | `~/workspace/unified-concept-editing/uce_models/nsfw_uce_sdxl.safetensors`                 |
| **SLD**       | SD15 only  | Concepts: `"NSFW, Hate, Harassment, Violence, Self-Harm, Sexuality, Shocking, Propaganda"` |

> ⚠️ For methods like **SLD**, only SD15 is supported due to lack of SDXL implementation in public repos.
> All baseline methods were re-trained using their default or recommended settings unless otherwise noted.

---

### 🔗 Download Pretrained Models

All the baseline weights used in evaluation have been consolidated into a single shared Google Drive folder:

👉 **[Download all baseline weights from Google Drive](https://drive.google.com/your_link_here)**

Ensure these weights are placed at the correct local paths (or update the configs accordingly).

---

## 🖼️ Step 1: Image Generation

To generate evaluation images, run the following script:

```bash
# Set dataset and available GPUs
export DATASET="coco_10k"  # Options: coco_10k, i2p_4073, CoProv2_test, sage_unseen, ud_1434
export GPUS="0,1,2,3"

# Launch generation
bash scripts/run_gen.sh
```

This will create the following directory structure:

```
eval_images/
├── coco_10k/             
│   ├── base/             # Baseline generations
│   │   ├── sd15/
│   │   └── sdxl/
│   ├── psa/              # PSA method (ours)
│   │   ├── sd15/
│   │   │   ├── level_0/
│   │   │   └── ...
│   │   └── sdxl/
│   ├── safetydpo/        # DiffusionDPO baseline
│   ├── esdu/             # ESD-U baseline
│   ├── uce/              # UCE baseline
│   └── sld/              # SLD baseline (SD1.5 only)
└── ...
```

To generate images for **all datasets**, use:

```bash
for dataset in coco_10k i2p_4073 CoProv2_test sage_unseen ud_1434; do
    export DATASET=$dataset
    bash scripts/run_gen.sh
done
```

---

## 📊 Step 2: Run Evaluation

Once all images are generated, launch the evaluation script:

```bash
python scripts/run_eval.py \
    --gpus "0,1,2,3" \
    --output eval_results.csv \
    --eval_dir eval_images
```

This will compute:

* On `coco_10k`:

  * **CLIP Score** (image-text alignment)
  * **FID Score** (image quality)

* On `i2p_4073`, `CoProv2_test`, `sage_unseen`, `ud_1434`:

  * **InPro Score** (inappropriate content ratio)

---

## 📐 Evaluation Metrics

| Metric          | Description                                                                                                          |
| --------------- | -------------------------------------------------------------------------------------------------------------------- |
| **InPro Score** | Measures the likelihood of inappropriate content using a combination of Q16 classifier and NudeNet. Lower is better. |
| **FID Score**   | Fréchet Inception Distance between generated images and COCO reference data. Lower is better.                        |
| **CLIP Score**  | Image-text alignment score using LAION-CLIP (ViT-H-14). Ranges from 0 to 100. Higher is better.                      |

---

## 📁 Output Format

The evaluation results are stored in a `.csv` file with the following columns:

* `dataset`: Dataset name
* `method`: Method name (e.g., base, psa, safetydpo)
* `model`: Diffusion model used (`sd15` or `sdxl`)
* `level`: Safety level (only for PSA)
* `metrics`: Metric name
* `score`: Numerical result
* `state`: Evaluation status
* `timestamp`: Timestamp of evaluation

---

## ⚠️ Notes and Recommendations

### Image Generation

* Ensure all model checkpoints and embedding folders are properly configured.
* Run generation **separately for each dataset**.
* Monitor disk usage — image generation is storage-intensive.

### Evaluation

* FID computation will cache dataset statistics automatically.
* InPro evaluation requires Q16 classifier and NudeNet to be set up.
* Previously completed results will be skipped automatically.

### Performance Tips

* Multi-GPU setup significantly speeds up generation and evaluation.
* FID supports multi-CPU acceleration.
* You may tweak `batch_size` in generation scripts to balance speed and memory.

---

## 🤝 Acknowledgements

We gratefully acknowledge the following open-source projects for their valuable contributions:

* [Q16 Classifier (ML Research)](https://github.com/ml-research/Q16.git) – used in our InPro metric computation
* [DiffusionDPO (Salesforce AI Research)](https://github.com/SalesforceAIResearch/DiffusionDPO) – used as a safety alignment baseline