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
$HOME/.cache/clip/ViT-L-14.pt

# Inception model for FID computation
$HOME/.cache/torch/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth

# LAION-CLIP model (ViT-H-14) for CLIP score
$HF_HOME/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/1c2b8495b28150b8a4922ee1c8edee224c284c0c/open_clip_model.safetensors
```

---

## 🔁 Reproducible Baselines

To ensure **fair and consistent comparisons**, we evaluated multiple safety alignment methods using their publicly available implementations. We provide our re-implementation of **SafetyDPO** trained on our Sage dataset, while other methods can be easily trained using their original repositories.

### 📦 Baseline Checkpoints

| Method        | Model Type | Weight Path (Local Path Example)                                                           |
| ------------- | ---------- | ------------------------------------------------------------------------------------------ |
| **SafetyDPO** | SD15       | `~/workspace/SafetyDPO/safetydpo-models/sd15/pytorch_lora_weights.safetensors`             |
|               | SDXL       | `~/workspace/SafetyDPO/safetydpo-models/sdxl/pytorch_lora_weights.safetensors`             |
| **ESD-U**     | SD15       | `~/workspace/erasing/esd-models/sd15/esdu.safetensors`                                     |
|               | SDXL       | `~/workspace/erasing/esd-models/sdxl/esdu.safetensors`                                     |
| **UCE**       | SD15       | `~/workspace/unified-concept-editing/uce_models/sd15/uce.safetensors`                 |
|               | SDXL       | `~/workspace/unified-concept-editing/uce_models/sdxl/uce.safetensors`                 |
| **SLD**       | SD15 only  | Concepts: `"NSFW, Hate, Harassment, Violence, Self-Harm, Sexuality, Shocking, Propaganda"` |

> Note: For evaluation, place your trained model weights in the paths shown in the table above.

### 🛠️ Training Other Baselines

Due to storage constraints, we only provide our SafetyDPO weights trained on the Sage dataset. However, you can easily train other baseline methods using their official implementations:

- **SLD**: Follow instructions at [Safe Latent Diffusion](https://github.com/ml-research/safe-latent-diffusion)
- **ESD-U**: Follow instructions at [Erasing](https://github.com/rohitgandikota/erasing)
- **UCE**: Follow instructions at [Unified Concept Editing](https://github.com/rohitgandikota/unified-concept-editing)

When training these models, use the following concepts for alignment:
```
"NSFW, Hate, Harassment, Violence, Self-Harm, Sexuality, Shocking, Propaganda"
```

### 🔗 Download SafetyDPO Weights

You can download our trained SafetyDPO weights:

👉 **[Download SafetyDPO weights from Google Drive](https://drive.google.com/your_link_here)**

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
