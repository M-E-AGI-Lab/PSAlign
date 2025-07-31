import os
import argparse
import pandas as pd
import torch
from accelerate import Accelerator
from tqdm import tqdm

# Diffusers imports
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)

try:
    from safetensors.torch import load_file
except ImportError:
    load_file = None

# Optional import for SLD (may not exist in all environments)
try:
    from sld import SLDPipeline  # type: ignore
except ImportError:
    SLDPipeline = None  # pragma: no cover


METHOD_CHOICES = ["esdu", "uce", "sld"]


def _load_base_pipe(model_name: str):
    """Load SD/SDXL pipeline based on model name."""
    if "xl" in model_name.lower():
        return StableDiffusionXLPipeline.from_pretrained(
            model_name, torch_dtype=torch.float16
        )
    else:
        return StableDiffusionPipeline.from_pretrained(
            model_name, torch_dtype=torch.float16
        )


def _inject_weights(pipe, weights_path: str):
    """Load .safetensors weights into pipe.unet (used by ESD-U / UCE)."""
    if weights_path is None:
        return
    if load_file is None:
        raise ImportError("safetensors is required but not installed.")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    
    state_dict = load_file(weights_path)
    pipe.unet.load_state_dict(state_dict, strict=False)


def _file_name_from_record(rec, input_file):
    """Extract filename from record, replicating naming logic from generate_safetydpo.py."""
    # Standard dataset case
    if "file_name" in rec:
        return rec["file_name"]
    
    # Fallback: look for columns starting with file_name (may contain newlines)
    for k in rec.keys():
        if k.strip() == "file_name":
            return rec[k]
    
    # Final fallback: use first column value
    return list(rec.values())[0]


def run_generation(
    accelerator: Accelerator,
    method: str,
    base_model: str,
    edit_weights_path: str,
    prompts_path: str,
    save_path: str,
    batch_size: int,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
    sld_concept: str = None,
    sld_type: str = "Medium",
):
    """Main generation routine supporting ESD-U, UCE and SLD methods."""

    if method in {"esdu", "uce"}:
        pipe = _load_base_pipe(base_model)
        _inject_weights(pipe, edit_weights_path)
    
    elif method == "sld":
        if SLDPipeline is None:
            raise ImportError(
                "SLD dependencies are missing. Please install the sld package."
            )
        if "xl" in base_model.lower():
            raise ValueError("SLD currently only supports SD1.5 base models.")
        pipe = SLDPipeline.from_pretrained(base_model, torch_dtype=torch.float16)
        if sld_concept:
            pipe.safety_concept = sld_concept
    else:
        raise ValueError(f"Unknown method {method}. Choices: {METHOD_CHOICES}")

    # Disable safety checker and prepare pipeline
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)
    pipe = accelerator.prepare(pipe)
    pipe.to(accelerator.device)
    generator_global = torch.Generator(accelerator.device).manual_seed(seed)

    os.makedirs(save_path, exist_ok=True)

    # Load and clean CSV data
    df = pd.read_csv(prompts_path, lineterminator="\n")
    # Clean column names to handle Windows line endings that cause issues like 'caption\r'
    df.columns = [c.strip() for c in df.columns]
    records = df.to_dict(orient="records")

    # Distribute work across processes
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    subset = records[rank::world_size]

    for idx in tqdm(
        range(0, len(subset), batch_size),
        desc=f"Rank {rank}",
        disable=not accelerator.is_main_process,
    ):
        batch = subset[idx : idx + batch_size]
        
        # Handle different column names (caption/prompt) after column stripping
        if "caption" in batch[0]:
            prompts = [r["caption"] for r in batch]
        elif "prompt" in batch[0]:
            prompts = [r["prompt"] for r in batch]
        else:
            raise NotImplementedError("Unknown column name. Please check the input CSV file.")

        # Use per-record generator if evaluation_seed is present
        gen = generator_global
        if "evaluation_seed" in batch[0]:
            gen = torch.Generator(accelerator.device).manual_seed(
                batch[0]["evaluation_seed"]
            )

        if method == "sld":
            # Map sld_type to hyperparameters (replicating original script)
            if sld_type == "Medium":
                sld_guidance_scale, sld_warmup_steps, sld_threshold = 1000, 10, 0.01
                sld_momentum_scale, sld_mom_beta = 0.3, 0.4
            elif sld_type == "Max":
                sld_guidance_scale, sld_warmup_steps, sld_threshold = 5000, 0, 1.0
                sld_momentum_scale, sld_mom_beta = 0.5, 0.7
            elif sld_type == "Weak":
                sld_guidance_scale, sld_warmup_steps, sld_threshold = 200, 15, 0.0
                sld_momentum_scale, sld_mom_beta = 0.0, 0.0
            else:
                raise ValueError("Invalid sld_type. Choose from Medium|Max|Weak")

            images = pipe(
                prompt=prompts,
                generator=gen,
                guidance_scale=guidance_scale,
                num_images_per_prompt=1,
                sld_guidance_scale=sld_guidance_scale,
                sld_warmup_steps=sld_warmup_steps,
                sld_threshold=sld_threshold,
                sld_momentum_scale=sld_momentum_scale,
                sld_mom_beta=sld_mom_beta,
            )
            images = images if isinstance(images, list) else images.images
        else:
            images = pipe(
                prompts,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=gen,
            ).images

        # Save generated images
        for rec, img in zip(batch, images):
            fname = _file_name_from_record(rec, prompts_path)
            img.save(os.path.join(save_path, fname))

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Inference completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generation script for ESD-U, UCE and SLD methods")

    # Method selection
    parser.add_argument("--method", type=str, choices=METHOD_CHOICES, required=True)
    parser.add_argument("--base_model", type=str, required=True, 
                       help="HuggingFace repository for base SD/SDXL model")
    parser.add_argument("--edit_weights_path", type=str, default=None, 
                       help="Path to *.safetensors weights (ESD-U / UCE)")

    # SLD-specific arguments
    parser.add_argument("--sld_concept", type=str, default=None)
    parser.add_argument("--sld_type", type=str, default="Medium", 
                       choices=["Medium", "Max", "Weak"])

    # Input/Output arguments
    parser.add_argument("--prompts_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    # Generation hyperparameters
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)

    args = parser.parse_args()

    accelerator = Accelerator()

    run_generation(
        accelerator=accelerator,
        method=args.method,
        base_model=args.base_model,
        edit_weights_path=args.edit_weights_path,
        prompts_path=args.prompts_path,
        save_path=args.save_path,
        batch_size=args.batch_size,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        sld_concept=args.sld_concept,
        sld_type=args.sld_type,
    )