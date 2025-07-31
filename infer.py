"""
Inference script for Personalized Safety Alignment (PSA) with text-to-image diffusion models.
This module provides functionality to run inference with PSA-enhanced Stable Diffusion models,
supporting both standard SD and SDXL architectures.
"""

import os
import json
import torch
import argparse
import sys
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from psa_adapter.psa_adapter import PSAAdapter, PSAAdapterXL

def load_pipeline(sd_model_path: str, sdxl: bool) -> StableDiffusionPipeline:
    """
    Load a Stable Diffusion pipeline with specified model path and architecture.
    
    Args:
        sd_model_path: Path to the Stable Diffusion model weights
        sdxl: Flag to use SDXL architecture instead of standard SD
    
    Returns:
        Configured StableDiffusionPipeline instance
    """
    pipe = (StableDiffusionXLPipeline if sdxl else StableDiffusionPipeline).from_pretrained(
                sd_model_path, torch_dtype=torch.float16)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    return pipe

def batchify(data: list, batch_size: int):
    """
    Split data into batches of specified size.
    
    Args:
        data: List of items to batch
        batch_size: Number of items per batch
        
    Yields:
        Batches of the data
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def infer(
    sd_model_path: str,
    psa_path: str = None,
    llm_model_name_or_path: str = None,
    num_tokens: int = 1,
    load_psa: bool = False,
    psa_scale: float = 1.0,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    meta_data_file: str = None,
    user_embeds_folder: str = None,
    llm_ban_prompt_file: str = None,
    llm_emb_prompt_file: str = None,
    output_folder: str = None,
    seed: int = 2025,
    sdxl: bool = False,
    batch_size: int = 1
) -> None:
    """
    Run inference with a Stable Diffusion model, optionally enhanced with PSA adapter.

    Args:
        sd_model_path: Path to the Stable Diffusion model
        psa_path: Path to PSA adapter weights directory
        llm_model_name_or_path: Name or path of the LLM model for PSA
        num_tokens: Number of user tokens for PSA adapter
        load_psa: Whether to use PSA adapter
        psa_scale: Scale factor for PSA adapter influence
        guidance_scale: Classifier-free guidance scale
        num_inference_steps: Number of denoising steps
        meta_data_file: Path to JSONL file containing generation metadata
        user_embeds_folder: Directory containing pre-computed user embeddings
        llm_ban_prompt_file: Path to template for banned content prompts
        llm_emb_prompt_file: Path to template for user embedding prompts
        output_folder: Directory to save generated images
        seed: Random seed for reproducibility
        sdxl: Whether to use SDXL architecture
        batch_size: Number of images to generate in parallel

    The function supports two modes:
    1. Standard SD/SDXL inference
    2. PSA-enhanced inference with either:
       - Pre-computed user embeddings
       - Dynamic user embeddings from LLM

    Images are saved to output_folder with names specified in meta_data_file.
    """
    # initialize accelerator (automatically handles devices)
    accelerator = Accelerator()

    # load pipeline and move to accelerator device
    pipe = load_pipeline(sd_model_path, sdxl)

    if load_psa:
        psa_adapter = (PSAAdapterXL if sdxl else PSAAdapter)(
            sd_pipe=pipe,
            psa_adapter_ckpt=os.path.join(psa_path, "psa_adapter.bin"),
            llm_model_name_or_path=llm_model_name_or_path,
            device=accelerator.device,
            num_tokens=num_tokens,
            load_llm_weights=(user_embeds_folder is None),
        )
    else:
        psa_adapter = None

    # prepare with accelerator
    pipe = accelerator.prepare(pipe)
    pipe.to(accelerator.device)
    if load_psa:
        psa_adapter = accelerator.prepare(psa_adapter)
        psa_adapter.device = accelerator.device

     # load data
    with open(meta_data_file, 'r') as f:
        data = [json.loads(line) for line in f]
    # split data across processes for parallel inference
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    data = data[rank::world_size]
    os.makedirs(output_folder, exist_ok=True)

    # batch inference
    for batch in tqdm(list(batchify(data, batch_size)), desc="Batches"):
        captions = [item["caption"] for item in batch]
        if load_psa:
            if user_embeds_folder:
                embeds = []
                for item in batch:
                    uid = item["user_data"]["User_ID"]
                    arr = np.load(os.path.join(user_embeds_folder, f"{uid}.npy"))
                    embeds.append(torch.tensor(arr, dtype=torch.float16).to(accelerator.device))
                images = psa_adapter.generate(
                    llm_user_embeds=torch.stack(embeds),
                    scale=psa_scale,
                    seed=seed,
                    prompt=captions,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                )
            else:
                with open(llm_ban_prompt_file, 'r', encoding='utf-8') as f:
                    llm_ban_prompt = f.read()
                with open(llm_emb_prompt_file, 'r', encoding='utf-8') as f:
                    llm_emb_prompt = f.read()
                user_infos = [json.dumps(item["user_data"]) for item in batch]
                images = psa_adapter.generate(
                    llm_ban_prompt=llm_ban_prompt,
                    llm_emb_prompt=llm_emb_prompt,
                    user_info=user_infos,
                    scale=psa_scale,
                    seed=seed,
                    prompt=captions,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                )
        else:
            images = pipe(
                captions,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(accelerator.device).manual_seed(seed)
            ).images

        for item, img in zip(batch, images):
            img.save(os.path.join(output_folder, item['file_name']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""
        PSAlign Inference Script
        
        Two operation modes:
        1. Standard SD/SDXL inference
        2. PSA-enhanced inference with personalized safety alignment
        """
    )
    
    # Required arguments
    req = parser.add_argument_group('required arguments')
    req.add_argument('--sd_model', required=True,
                    help='Path to Stable Diffusion model checkpoint')
    req.add_argument('--save_path', required=True,
                    help='Directory to save generated images')
    req.add_argument('--meta_data', required=True,
                    help='JSONL file containing generation metadata (file_name, caption, user_data)')

    # PSA-related arguments
    psa_group = parser.add_argument_group('PSA configuration')
    psa_group.add_argument('--psa_path', default=None,
                        help='Directory containing psa_adapter.bin weights')
    psa_group.add_argument('--load_psa', action='store_true',
                        help='Enable PSA adapter for personalized safety alignment')
    psa_group.add_argument('--llm_model', default=None,
                        help='LLM model name/path for generating user embeddings')
    psa_group.add_argument('--embeds_folder', default=None,
                        help='Directory containing pre-computed user embeddings (.npy files)')
    psa_group.add_argument('--llm_ban_prompt_file', default=None,
                        help='Template file for generating banned content prompts')
    psa_group.add_argument('--llm_emb_prompt_file', default=None,
                        help='Template file for generating user embeddings')

    # Generation parameters
    gen_group = parser.add_argument_group('generation parameters')
    gen_group.add_argument('--psa_scale', type=float, default=1.0,
                        help='Scale factor for PSA adapter influence (default: 1.0)')
    gen_group.add_argument('--num_tokens', type=int, default=1,
                        help='Number of user tokens for PSA (default: 1)')
    gen_group.add_argument('--guidance', type=float, default=7.5,
                        help='Classifier-free guidance scale (default: 7.5)')
    gen_group.add_argument('--steps', type=int, default=50,
                        help='Number of denoising steps (default: 50)')
    gen_group.add_argument('--seed', type=int, default=2025,
                        help='Random seed for reproducibility (default: 2025)')
    gen_group.add_argument('--sdxl', action='store_true',
                        help='Use SDXL architecture instead of SD')
    gen_group.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for parallel image generation (default: 1)')

    args = parser.parse_args()

    # Validate arguments
    if args.load_psa:
        if not args.psa_path:
            parser.error("--psa_path is required when --load_psa is set")
        if not (args.embeds_folder or (args.llm_model and args.llm_ban_prompt_file and args.llm_emb_prompt_file)):
            parser.error(
                "Either --embeds_folder or (--llm_model, --llm_ban_prompt_file, "
                "and --llm_emb_prompt_file) must be provided when using PSA"
            )

    try:
        infer(
            sd_model_path=args.sd_model,
            psa_path=args.psa_path,
            llm_model_name_or_path=args.llm_model,
            num_tokens=args.num_tokens,
            load_psa=args.load_psa,
            psa_scale=args.psa_scale,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            meta_data_file=args.meta_data,
            user_embeds_folder=args.embeds_folder,
            llm_ban_prompt_file=args.llm_ban_prompt_file,
            llm_emb_prompt_file=args.llm_emb_prompt_file,
            output_folder=args.save_path,
            seed=args.seed,
            sdxl=args.sdxl,
            batch_size=args.batch_size
        )
    except Exception as e:
        parser.error(f"Error during inference: {str(e)}")