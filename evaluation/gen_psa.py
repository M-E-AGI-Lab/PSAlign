import os
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import sys

sys.path.append("../")
from psa_adapter.psa_adapter import PSAAdapter, PSAAdapterXL


# Predefined level to user ID mapping
LEVEL_MAPPING = {
    "level_0": "0001001",
    "level_1": "0000797",
    "level_2": "0000697",
    "level_3": "0000002",
    "level_4": "0000742",
    "level_5": "0000501", 
}


def load_diffusion_pipeline(model_path: str, use_sdxl: bool) -> object:
    """
    Load Stable Diffusion pipeline
    
    Args:
        model_path: Path to Stable Diffusion model
        use_sdxl: Whether to use SDXL pipeline
        
    Returns:
        Loaded diffusion pipeline
    """
    pipeline_class = StableDiffusionXLPipeline if use_sdxl else StableDiffusionPipeline
    pipe = pipeline_class.from_pretrained(model_path, torch_dtype=torch.float16)
    
    # Disable safety checker
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    
    return pipe


def run_inference(
    sd_model_path: str,
    psa_adapter_path: str = None,
    user_embeds_folder: str = None,
    input_csv: str = None,
    output_folder: str = None,
    num_tokens: int = 1,
    psa_scale: float = 1.0,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    seed: int = 2025,
    use_sdxl: bool = False,
    batch_size: int = 1,
    user_id: str = None,
):
    """
    Run inference with optional PSA adapter
    
    Args:
        sd_model_path: Path to Stable Diffusion model
        psa_adapter_path: Path to PSA adapter checkpoint
        user_embeds_folder: Folder containing user embeddings
        input_csv: Path to input CSV file with prompts
        output_folder: Output directory for generated images
        num_tokens: Number of tokens for PSA adapter
        psa_scale: PSA adapter scaling factor
        guidance_scale: Classifier-free guidance scale
        num_inference_steps: Number of denoising steps
        seed: Random seed for generation
        use_sdxl: Whether to use SDXL pipeline
        batch_size: Batch size for inference
        user_id: User ID for embedding lookup
    """
    # Initialize accelerator for multi-GPU support
    accelerator = Accelerator()

    # Load and prepare diffusion pipeline
    pipe = load_diffusion_pipeline(sd_model_path, use_sdxl)
    pipe = accelerator.prepare(pipe)
    pipe.to(accelerator.device)

    # Load PSA adapter if provided
    psa_adapter = None
    if psa_adapter_path and user_embeds_folder:
        adapter_class = PSAAdapterXL if use_sdxl else PSAAdapter
        psa_adapter = adapter_class(
            sd_pipe=pipe,
            psa_adapter_ckpt=os.path.join(psa_adapter_path, "psa_adapter.bin"),
            device=accelerator.device,
            num_tokens=num_tokens,
            load_llm_weights=False,
        )
        psa_adapter = accelerator.prepare(psa_adapter)
        psa_adapter.device = accelerator.device

    # Load prompts from CSV
    df = pd.read_csv(input_csv)
    data = df.to_dict(orient='records')

    # Distribute data across processes
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    data = data[rank::world_size]

    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # Run inference loop
    for i in tqdm(range(0, len(data), batch_size), desc="Generating images"):
        batch = data[i:i+batch_size]
        prompts = [item['caption'] for item in batch]

        if psa_adapter and user_id:
            # Load user embeddings for PSA adapter
            embeddings = []
            for item in batch:
                embed_path = os.path.join(user_embeds_folder, f"{user_id}.npy")
                embedding_array = np.load(embed_path)
                embedding_tensor = torch.tensor(embedding_array, dtype=torch.float16).to(accelerator.device)
                embeddings.append(embedding_tensor)

            # Generate images with PSA adapter
            images = psa_adapter.generate(
                llm_user_embeds=torch.stack(embeddings),
                scale=psa_scale,
                seed=seed,
                prompt=prompts,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
        else:
            # Generate images with standard pipeline
            generator = torch.Generator(accelerator.device).manual_seed(seed)
            images = pipe(
                prompts,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator
            ).images

        # Save generated images
        for item, image in zip(batch, images):
            filename = item['file_name']
            image.save(os.path.join(output_folder, filename))


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="PSA Adapter Stable Diffusion Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--sd_model', 
        required=True,
        help='Path to Stable Diffusion model'
    )
    parser.add_argument(
        '--input_csv', 
        required=True,
        help='Path to CSV file containing prompts'
    )
    parser.add_argument(
        '--output_folder', 
        required=True,
        help='Output directory for generated images'
    )

    # Optional PSA adapter arguments
    parser.add_argument(
        '--psa_adapter_path', 
        default=None,
        help='Path to PSA adapter checkpoint'
    )
    parser.add_argument(
        '--user_embeds_folder', 
        default=None,
        help='Folder containing user embeddings'
    )
    parser.add_argument(
        '--num_tokens', 
        type=int, 
        default=1,
        help='Number of tokens for PSA adapter (default: 1)'
    )
    parser.add_argument(
        '--psa_scale', 
        type=float, 
        default=1.0,
        help='PSA adapter scaling factor (default: 1.0)'
    )

    # Generation parameters
    parser.add_argument(
        '--guidance_scale', 
        type=float, 
        default=7.5,
        help='Classifier-free guidance scale (default: 7.5)'
    )
    parser.add_argument(
        '--num_inference_steps', 
        type=int, 
        default=40,
        help='Number of denoising steps (default: 40)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=2025,
        help='Random seed for generation (default: 2025)'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=1,
        help='Batch size for inference (default: 1)'
    )

    # Model configuration
    parser.add_argument(
        '--sdxl', 
        action='store_true',
        help='Use Stable Diffusion XL pipeline'
    )

    # Level-based configuration
    parser.add_argument(
        '--level', 
        type=str, 
        default=None, 
        choices=list(LEVEL_MAPPING.keys()),
        help='Predefined level for PSA adaptation'
    )

    return parser.parse_args()


def main():
    """Main function for PSA adapter inference"""
    args = parse_arguments()

    # Determine output folder and user ID based on level
    output_folder = args.output_folder
    user_id = None
    
    if args.level:
        output_folder = os.path.join(args.output_folder, args.level)
        user_id = LEVEL_MAPPING[args.level]

    # Run inference
    run_inference(
        sd_model_path=args.sd_model,
        psa_adapter_path=args.psa_adapter_path,
        user_embeds_folder=args.user_embeds_folder,
        input_csv=args.input_csv,
        output_folder=output_folder,
        num_tokens=args.num_tokens,
        psa_scale=args.psa_scale,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        use_sdxl=args.sdxl,
        batch_size=args.batch_size,
        user_id=user_id
    )


if __name__ == '__main__':
    main()