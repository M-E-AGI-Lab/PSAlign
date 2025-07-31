# Evaluation Images

This directory contains generated images from different models for evaluation:

```
eval_images/
├── debug/                 # Example test set
│   ├── base/             # Base model outputs
│   │   ├── sd15/
│   │   └── sdxl/
│   ├── psa_adapter/      # Our method
│   │   ├── sd15/
│   │   └── sdxl/
│   ├── safetydpo/        # SafetyDPO outputs
│   ├── esdu/             # ESD-U outputs
│   ├── uce/              # UCE outputs
│   └── sld/              # SLD outputs
└── ...                   # Other datasets
```

For complete evaluation results, please generate the full image sets following the instructions in `evaluation/README.md`. The generation process uses `scripts/run_gen.sh` to create images for each dataset and model configuration.
