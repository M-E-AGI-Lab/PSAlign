"""
Training script for Personalized Safety Alignment (PSA) in text-to-image diffusion models.

This module implements the training procedure for PSA adapters, which enable:
1. User-specific content filtering
2. Personalized safety alignment
3. Support for both SD and SDXL architectures
"""

import argparse
import logging
import math
import os
import random
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from packaging import version
from tqdm.auto import tqdm

import accelerate
import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import check_min_version
from transformers import (
    AutoConfig, AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection,
    CLIPTokenizer, PretrainedConfig
)
from transformers.utils import ContextManagers

from psa_adapter.psa_adapter import PSAProjection, init_psa_adapter
from psa_adapter.attention_processor import PSAAttnProcessor2_0
from dataset.sage_dataset import SageDataset, collate_fn

REQUIRED_DIFFUSERS_VERSION = "0.20.0"
check_min_version(REQUIRED_DIFFUSERS_VERSION)

logger = None

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str,
    revision: str,
    subfolder: str = "text_encoder"
) -> Union[CLIPTextModel, CLIPTextModelWithProjection]:
    """Import the appropriate model class based on the model architecture."""
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"Unsupported model architecture: {model_class}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for PSA training."""
    parser = argparse.ArgumentParser(description="Training script for PSA adapter")
    
    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--input_perturbation", type=float, default=0, 
                           help="Scale of input perturbation for robustness")
    model_group.add_argument("--sd_model", type=str, required=True,
                           help="Path to pretrained Stable Diffusion model")
    model_group.add_argument("--revision", type=str, default=None,
                           help="Model revision/tag from HuggingFace hub")
    
    # Dataset configuration
    data_group = parser.add_argument_group("Dataset Configuration")
    data_group.add_argument("--dataset_name", type=str, default=None,
                          help="Dataset name from HuggingFace hub or local path")
    data_group.add_argument("--dataset_config_name", type=str, default=None,
                          help="Dataset config name")
    data_group.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                          help="LLM model for user embedding generation")
    data_group.add_argument("--num_tokens_user", type=int, default=1,
                          help="Number of user tokens")
    data_group.add_argument("--embeds_folder", type=str, default=None,
                          help="Directory containing user embeddings")
    data_group.add_argument("--train_data_dir", type=str, default=None,
                          help="Training data directory")
    data_group.add_argument("--image_column", type=str, default="image",
                          help="Image column name in dataset")
    data_group.add_argument("--caption_column", type=str, default="caption",  
                          help="Caption column name in dataset")
    data_group.add_argument("--max_train_samples", type=int, default=None,
                          help="Maximum training samples for debugging")
    
    # Training configuration
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument("--output_dir", type=str, default="sd-model-finetuned",
                           help="Output directory for model and checkpoints")
    train_group.add_argument("--cache_dir", type=str, default=None,
                           help="Cache directory for models and datasets")
    train_group.add_argument("--seed", type=int, default=None,
                           help="Random seed for reproducible training")
    train_group.add_argument("--resolution", type=int, default=None,
                           help="Input image resolution")
    train_group.add_argument("--random_crop", action="store_true",
                           help="Use random cropping instead of center crop")
    train_group.add_argument("--no_hflip", action="store_true",
                           help="Suppress horizontal flipping")
    train_group.add_argument("--train_batch_size", type=int, default=1,
                           help="Training batch size per device")
    train_group.add_argument("--num_train_epochs", type=int, default=100,
                           help="Number of training epochs")
    train_group.add_argument("--max_train_steps", type=int, default=2000,
                           help="Maximum training steps")
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=1,
                           help="Gradient accumulation steps")
    train_group.add_argument("--gradient_checkpointing", action="store_true",
                           help="Enable gradient checkpointing")
    train_group.add_argument("--learning_rate", type=float, default=1e-8,
                           help="Learning rate")
    train_group.add_argument("--scale_lr", action="store_true",
                           help="Scale learning rate by processes and batch size")
    train_group.add_argument("--lr_scheduler", type=str, default="constant_with_warmup",
                           help="Learning rate scheduler")
    train_group.add_argument("--lr_warmup_steps", type=int, default=500,
                           help="Learning rate warmup steps")
    train_group.add_argument("--use_adafactor", action="store_true",
                           help="Use Adafactor optimizer")
    train_group.add_argument("--allow_tf32", action="store_true",
                           help="Allow TF32 on Ampere GPUs")
    train_group.add_argument("--dataloader_num_workers", type=int, default=0,
                           help="Number of dataloader workers")
    
    # Optimizer parameters
    opt_group = parser.add_argument_group("Optimizer Parameters")
    opt_group.add_argument("--adam_beta1", type=float, default=0.9)
    opt_group.add_argument("--adam_beta2", type=float, default=0.999)
    opt_group.add_argument("--adam_weight_decay", type=float, default=1e-2)
    opt_group.add_argument("--adam_epsilon", type=float, default=1e-08)
    opt_group.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Logging and checkpointing
    log_group = parser.add_argument_group("Logging and Checkpointing")
    log_group.add_argument("--hub_model_id", type=str, default=None,
                         help="HuggingFace Hub model ID")
    log_group.add_argument("--logging_dir", type=str, default="logs",
                         help="TensorBoard logging directory")
    log_group.add_argument("--mixed_precision", type=str, default="fp16",
                         choices=["no", "fp16", "bf16"], help="Mixed precision")
    log_group.add_argument("--report_to", type=str, default="tensorboard",
                         help="Experiment tracking platform")
    log_group.add_argument("--local_rank", type=int, default=-1,
                         help="Local rank for distributed training")
    log_group.add_argument("--checkpointing_steps", type=int, default=500,
                         help="Save checkpoint every N steps")
    log_group.add_argument("--resume_from_checkpoint", type=str, default=None,
                         help="Resume from checkpoint path")
    log_group.add_argument("--tracker_project_name", type=str, default="tuning",
                         help="Experiment tracker project name")
    
    # Additional parameters
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument("--noise_offset", type=float, default=0,
                          help="Noise offset scale")
    misc_group.add_argument("--pretrained_vae_model_name_or_path", type=str, default=None,
                          help="Path to pretrained VAE model")
    misc_group.add_argument("--sdxl", action='store_true', help="Train SDXL")
    misc_group.add_argument("--beta_dpo", type=float, default=5000,
                          help="DPO temperature parameter")
    misc_group.add_argument("--hard_skip_resume", action="store_true",
                          help="Skip dataloader resume for faster startup")
    misc_group.add_argument("--unet_init", type=str, default='',
                          help="Initialize from specific UNet weights")
    misc_group.add_argument("--proportion_empty_prompts", type=float, default=0.2,
                          help="Proportion of prompts to replace with empty strings")
    misc_group.add_argument("--split", type=str, default='train',
                          help="Dataset split to use")
    
    args = parser.parse_args()
    
    # Handle environment variables
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Validation
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # Set default resolution
    if args.resolution is None:
        args.resolution = 1024 if args.sdxl else 512
        
    return args


def encode_prompt_sdxl(
    batch: Dict[str, torch.Tensor],
    text_encoders: List[Union[CLIPTextModel, CLIPTextModelWithProjection]],
    tokenizers: List[CLIPTokenizer],
    proportion_empty_prompts: float,
    caption_column: str,
    is_train: bool = True
) -> Dict[str, torch.Tensor]:
    """Encode text prompts using SDXL's dual text encoder architecture."""
    prompt_embeds_list = []
    prompt_batch = batch[caption_column]

    # Process captions
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])

    # Encode with both text encoders
    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to('cuda'),
                output_hidden_states=True,
            )

            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}


def align_tensor_to_model(tensor: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """Align tensor dtype and device to match model parameters."""
    param = next(model.parameters())
    return tensor.to(dtype=param.dtype, device=param.device)


def save_psa_weights(unet, user_proj_model, psa_dir, logger):
    """Save PSA adapter weights."""
    os.makedirs(psa_dir, exist_ok=True)

    psa_adapter_modules = torch.nn.ModuleList([
        proc for proc in unet.attn_processors.values() 
        if isinstance(proc, PSAAttnProcessor2_0)
    ])
    
    psa_adapter_state_dict = {
        "psa_adapter": psa_adapter_modules.state_dict(),
        "user_proj": user_proj_model.state_dict()
    }
    
    save_path = os.path.join(psa_dir, "psa_adapter.bin")
    torch.save(psa_adapter_state_dict, save_path)
    logger.info(f"PSA adapter weights saved to {save_path}")


def load_psa_weights(unet, user_proj_model, psa_dir, logger):
    """Load PSA adapter weights."""
    load_path = os.path.join(psa_dir, "psa_adapter.bin")
    
    if os.path.exists(load_path):
        state_dict = torch.load(load_path)
        psa_adapter_modules = torch.nn.ModuleList([
            proc for proc in unet.attn_processors.values() 
            if isinstance(proc, PSAAttnProcessor2_0)
        ])
        psa_adapter_modules.load_state_dict(state_dict["psa_adapter"])
        user_proj_model.load_state_dict(state_dict["user_proj"])
        logger.info(f"PSA adapter weights loaded from {load_path}")
    else:
        logger.warning(f"PSA adapter weights not found at {load_path}")


def setup_accelerator(args):
    """Initialize and configure accelerator."""
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    global logger
    logger = get_logger(__name__, log_level="INFO")
    logger.info("Accelerator initialized, training starts.")

    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Set seed
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Create output directory
    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    return accelerator


def enforce_zero_terminal_snr(scheduler):
    """Enforce zero terminal SNR for turbo models."""
    alphas = 1 - scheduler.betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
    
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    scheduler.alphas_cumprod = alphas_cumprod


def load_models(args, accelerator):
    """Load and configure all models."""
    # Load scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(args.sd_model, subfolder="scheduler")
    if 'turbo' in args.sd_model:
        enforce_zero_terminal_snr(noise_scheduler)
    
    # Disable DeepSpeed zero init for frozen models
    def deepspeed_zero_init_disabled_context_manager():
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        return [deepspeed_plugin.zero3_init_context_manager(enable=False)] if deepspeed_plugin else []

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()): 
        if args.sdxl:
            # Load SDXL models
            tokenizer_name = ("stabilityai/stable-diffusion-xl-base-1.0" 
                            if args.sd_model == "stabilityai/stable-diffusion-xl-refiner-1.0" 
                            else args.sd_model)
            
            tokenizers = [
                AutoTokenizer.from_pretrained(tokenizer_name, subfolder="tokenizer", 
                                            revision=args.revision, use_fast=False),
                AutoTokenizer.from_pretrained(args.sd_model, subfolder="tokenizer_2", 
                                            revision=args.revision, use_fast=False)
            ]
            
            text_encoder_cls_one = import_model_class_from_model_name_or_path(
                tokenizer_name, args.revision
            )
            text_encoder_cls_two = import_model_class_from_model_name_or_path(
                tokenizer_name, args.revision, subfolder="text_encoder_2"
            )
            
            text_encoders = [
                text_encoder_cls_one.from_pretrained(tokenizer_name, subfolder="text_encoder", 
                                                   revision=args.revision),
                text_encoder_cls_two.from_pretrained(args.sd_model, subfolder="text_encoder_2", 
                                                   revision=args.revision)
            ]
            
            if args.sd_model == "stabilityai/stable-diffusion-xl-refiner-1.0":
                text_encoders = [text_encoders[1]]
                tokenizers = [tokenizers[1]]
        else:
            # Load SD 1.5 models
            tokenizer = CLIPTokenizer.from_pretrained(args.sd_model, subfolder="tokenizer", 
                                                    revision=args.revision)
            text_encoder = CLIPTextModel.from_pretrained(args.sd_model, subfolder="text_encoder", 
                                                       revision=args.revision)

        # Load VAE
        vae_path = args.pretrained_vae_model_name_or_path or args.sd_model
        vae = AutoencoderKL.from_pretrained(
            vae_path, 
            subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None, 
            revision=args.revision
        )
        
        # Load UNets
        unet_path = args.unet_init or args.sd_model
        if args.unet_init:
            logger.info(f"Initializing UNet from {args.unet_init}")
            
        ref_unet = UNet2DConditionModel.from_pretrained(unet_path, subfolder="unet", 
                                                       revision=args.revision)
        ref_unet = init_psa_adapter(ref_unet, num_tokens_user=args.num_tokens_user)
        
        unet = UNet2DConditionModel.from_pretrained(unet_path, subfolder="unet", 
                                                   revision=args.revision)
        unet = init_psa_adapter(unet, num_tokens_user=args.num_tokens_user)

    # Freeze models
    vae.requires_grad_(False)
    ref_unet.requires_grad_(False)
    
    if args.sdxl:
        for encoder in text_encoders:
            encoder.requires_grad_(False)
    else:
        text_encoder.requires_grad_(False)

    # Create user projection model
    llm_config = AutoConfig.from_pretrained(args.llm_model)
    user_proj_model = PSAProjection(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=llm_config.hidden_size,
        clip_extra_context_tokens=args.num_tokens_user,
    )

    models = {
        'noise_scheduler': noise_scheduler,
        'vae': vae,
        'unet': unet,
        'ref_unet': ref_unet,
        'user_proj': user_proj_model
    }
    
    if args.sdxl:
        models.update({
            'text_encoders': text_encoders,
            'tokenizers': tokenizers
        })
    else:
        models.update({
            'text_encoder': text_encoder,
            'tokenizer': tokenizer
        })
    
    return models


def setup_training_components(args, accelerator, models):
    """Setup training components including optimizer and scheduler."""
    unet, user_proj = models['unet'], models['user_proj']
    
    # Configure trainable parameters
    for param in unet.parameters():
        param.requires_grad_(False)
    
    # Get PSA adapter parameters
    psa_adapter_params = [
        p for proc in unet.attn_processors.values()
        if isinstance(proc, PSAAttnProcessor2_0) 
        for p in proc.parameters()
    ]
    
    psa_params = list(user_proj.parameters()) + psa_adapter_params
    for p in psa_params:
        p.requires_grad_(True)

    # Scale learning rate if requested
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * 
            args.train_batch_size * accelerator.num_processes
        )

    # Log parameter statistics
    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in psa_params if p.requires_grad)
    
    logger.info(f"\n{'='*40} Parameter Report {'='*40}")
    logger.info(f"{'Total UNet Parameters:':<30} {total_params:,}")
    logger.info(f"{'  └─ PSA Adapter Parameters:':<30} {trainable_params:,}")
    logger.info(f"{'='*98}\n")

    # Setup optimizer
    if args.use_adafactor or args.sdxl:
        logger.info("Using Adafactor optimizer")
        optimizer = transformers.Adafactor(
            psa_params,
            lr=args.learning_rate,
            weight_decay=args.adam_weight_decay,
            clip_threshold=1.0,
            scale_parameter=False,
            relative_step=False
        )
    else:
        optimizer = torch.optim.AdamW(
            psa_params,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # Setup scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    
    return optimizer, lr_scheduler


def setup_dataset_and_dataloader(args, accelerator, models):
    """Setup dataset and dataloader."""
    logger.info("Initializing Sage training dataset...")
    
    with accelerator.main_process_first():
        train_dataset = SageDataset(
            metadata_path=os.path.join(args.train_data_dir, "metadata.jsonl"),
            image_root=args.train_data_dir,
            tokenizer=None if args.sdxl else models['tokenizer'],  # Will be handled separately
            embeds_folder=args.embeds_folder,
            resolution=args.resolution,
            random_crop=args.random_crop,
            no_hflip=args.no_hflip,
            proportion_empty_prompts=args.proportion_empty_prompts,
            sdxl=args.sdxl
        )

        # Limit dataset size for debugging
        if args.max_train_samples is not None:
            from torch.utils.data import Subset
            total = len(train_dataset)
            g = torch.Generator().manual_seed(2025)
            indices = torch.randperm(total, generator=g).tolist()
            sample_size = min(args.max_train_samples, total)
            train_dataset = Subset(train_dataset, indices[:sample_size])
            logger.info(f"Limited dataset to {sample_size} samples for testing")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=(args.split == 'train'),
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True
    )
    
    return train_dataset, train_dataloader


def prepare_models_for_training(args, accelerator, models):
    """Prepare models for training with proper dtype and device placement."""
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move models to device and cast to appropriate dtype
    models['vae'].to(accelerator.device, dtype=weight_dtype)
    models['ref_unet'].to(accelerator.device, dtype=weight_dtype)
    models['user_proj'].to(accelerator.device, dtype=weight_dtype)
    
    if args.sdxl:
        for encoder in models['text_encoders']:
            encoder.to(accelerator.device, dtype=weight_dtype)
        
        # CPU offload for memory efficiency
        logger.info("Offloading models to CPU for memory efficiency")
        models['vae'] = accelerate.cpu_offload(models['vae'])
        models['ref_unet'] = accelerate.cpu_offload(models['ref_unet'])
        for i, encoder in enumerate(models['text_encoders']):
            models['text_encoders'][i] = accelerate.cpu_offload(encoder)
    else:
        models['text_encoder'].to(accelerator.device, dtype=weight_dtype)

    # Setup gradient checkpointing
    if args.gradient_checkpointing or args.sdxl:
        logger.info("Enabling gradient checkpointing")
        models['unet'].enable_gradient_checkpointing()

    # Cast training parameters to float32 for mixed precision
    if args.mixed_precision == "fp16":
        cast_training_params(models['unet'], dtype=torch.float32)
        cast_training_params(models['user_proj'], dtype=torch.float32)

    # Enable TF32 if requested
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    return weight_dtype


def compute_loss(args, model_pred, target, ref_pred):
    """Compute DPO loss for PSA training."""
    # Calculate MSE losses
    model_losses = (model_pred - target).pow(2).mean(dim=[1, 2, 3])
    model_losses_w, model_losses_l = model_losses.chunk(2)
    raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
    model_diff = model_losses_w - model_losses_l

    # Reference model losses
    ref_losses = (ref_pred - target).pow(2).mean(dim=[1, 2, 3])
    ref_losses_w, ref_losses_l = ref_losses.chunk(2)
    raw_ref_loss = ref_losses.mean()
    ref_diff = ref_losses_w - ref_losses_l

    # DPO loss calculation
    scale_term = -0.5 * args.beta_dpo
    inside_term = scale_term * (model_diff - ref_diff)
    implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
    loss = -F.logsigmoid(inside_term).mean()
    
    return {
        'loss': loss,
        'model_mse': raw_model_loss,
        'ref_mse': raw_ref_loss,
        'accuracy': implicit_acc,
        'model_losses_w': model_losses_w.mean(),
        'model_losses_l': model_losses_l.mean(),
        'dpo_preference': inside_term.mean()
    }


def prepare_batch_inputs(args, batch, models, weight_dtype, accelerator):
    """Prepare batch inputs for training step."""
    # Process pixel values
    feed_pixel_values = torch.cat(batch["pixel_values"].chunk(2, dim=1))
    
    # Encode to latents
    with torch.no_grad():
        latents = models['vae'].encode(feed_pixel_values.to(weight_dtype)).latent_dist.sample()
        latents = latents * models['vae'].config.scaling_factor

    # Prepare noise and timesteps
    noise = torch.randn_like(latents)
    
    if args.noise_offset:
        noise += args.noise_offset * torch.randn(
            (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
        )
    
    if args.input_perturbation:
        new_noise = noise + args.input_perturbation * torch.randn_like(noise)
    
    bsz = latents.shape[0]
    timesteps = torch.randint(0, models['noise_scheduler'].config.num_train_timesteps, 
                             (bsz,), device=latents.device).long()
    
    # Special timestep handling for different models
    if 'refiner' in args.sd_model:
        timesteps = timesteps % 200
    elif 'turbo' in args.sd_model:
        timesteps_0_to_3 = timesteps % 4
        timesteps = 250 * timesteps_0_to_3 + 249
    
    # Make timesteps and noise consistent for DPO pairs
    timesteps = timesteps.chunk(2)[0].repeat(2)
    noise = noise.chunk(2)[0].repeat(2, 1, 1, 1)

    # Add noise to latents
    noisy_latents = models['noise_scheduler'].add_noise(
        latents,
        new_noise if args.input_perturbation else noise,
        timesteps
    )

    # Prepare text embeddings
    if args.sdxl:
        with torch.no_grad():
            # Prepare time_ids for SDXL
            if 'refiner' in args.sd_model:
                add_time_ids = torch.tensor(
                    [args.resolution, args.resolution, 0, 0, 6.0],
                    dtype=weight_dtype, device=accelerator.device
                )[None, :].repeat(timesteps.size(0), 1)
            else:
                add_time_ids = torch.tensor(
                    [args.resolution, args.resolution, 0, 0, args.resolution, args.resolution],
                    dtype=weight_dtype, device=accelerator.device
                )[None, :].repeat(timesteps.size(0), 1)
            
            prompt_batch = encode_prompt_sdxl(
                batch, models['text_encoders'], models['tokenizers'],
                args.proportion_empty_prompts, 'caption', is_train=True
            )
        
        prompt_batch["prompt_embeds"] = prompt_batch["prompt_embeds"].repeat(2, 1, 1)
        prompt_batch["pooled_prompt_embeds"] = prompt_batch["pooled_prompt_embeds"].repeat(2, 1)
        
        encoder_hidden_states = prompt_batch["prompt_embeds"]
        unet_added_conditions = {
            "time_ids": add_time_ids,
            "text_embeds": prompt_batch["pooled_prompt_embeds"]
        }
    else:
        encoder_hidden_states = models['text_encoder'](batch["input_ids"])[0]
        encoder_hidden_states = encoder_hidden_states.repeat(2, 1, 1)
        unet_added_conditions = None

    # Add user embeddings if available
    if "user_embedding" in batch:
        user_embeds = align_tensor_to_model(batch["user_embedding"], models['user_proj'])
        user_embeds = user_embeds.repeat(2, 1, 1)
        user_tokens = models['user_proj'](user_embeds).to(accelerator.device, dtype=weight_dtype)
        encoder_hidden_states = torch.cat([encoder_hidden_states, user_tokens], dim=1)

    return {
        'noisy_latents': noisy_latents,
        'timesteps': timesteps,
        'encoder_hidden_states': encoder_hidden_states,
        'unet_added_conditions': unet_added_conditions,
        'target': noise
    }


def training_step(args, batch, models, weight_dtype, accelerator):
    """Execute a single training step."""
    # Prepare inputs
    inputs = prepare_batch_inputs(args, batch, models, weight_dtype, accelerator)
    
    # Forward pass through main UNet
    model_pred = models['unet'](
        inputs['noisy_latents'],
        inputs['timesteps'], 
        inputs['encoder_hidden_states'],
        added_cond_kwargs=inputs['unet_added_conditions']
    ).sample

    # Forward pass through reference UNet
    with torch.no_grad():
        ref_pred = models['ref_unet'](
            inputs['noisy_latents'],
            inputs['timesteps'],
            inputs['encoder_hidden_states'], 
            added_cond_kwargs=inputs['unet_added_conditions']
        ).sample.detach()

    # Compute loss
    loss_dict = compute_loss(args, model_pred, inputs['target'], ref_pred)
    
    return loss_dict


def register_model_hooks(accelerator):
    """Register model save/load hooks for accelerator."""
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            models_to_save = models[:1]
            for i, model in enumerate(models_to_save):
                model.save_pretrained(os.path.join(output_dir, "unet"))
                weights.pop()

        def load_model_hook(models, input_dir):
            models_to_load = models[:1]
            for i in range(len(models_to_load)):
                model = models.pop()
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)


def main():
    args = parse_args()
    
    # Setup accelerator and logging
    accelerator = setup_accelerator(args)
    
    # Load models
    models = load_models(args, accelerator)
    
    # Setup dataset and dataloader
    train_dataset, train_dataloader = setup_dataset_and_dataloader(args, accelerator, models)
    
    # Calculate training steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
        logger.info(f"Setting max_train_steps to {args.max_train_steps} based on epoch count")

    # Setup training components
    optimizer, lr_scheduler = setup_training_components(args, accelerator, models)
    
    # Prepare models for training
    weight_dtype = prepare_models_for_training(args, accelerator, models)
    
    # Register model hooks
    register_model_hooks(accelerator)
    
    # Prepare with accelerator
    models['unet'], models['user_proj'], optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        models['unet'], models['user_proj'], optimizer, train_dataloader, lr_scheduler
    )

    # Recalculate training steps after dataloader preparation
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Initialize trackers
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Log training configuration
    total_batch_size = (
        args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )
    
    logger.info(f"\n{'='*20} Training Configuration {'='*20}")
    logger.info(f"Dataset size:                     {len(train_dataset):,}")
    logger.info(f"Number of epochs:                 {args.num_train_epochs}")
    logger.info(f"Batch size per device:           {args.train_batch_size}")
    logger.info(f"Total batch size:                {total_batch_size:,}")
    logger.info(f"Gradient accumulation steps:     {args.gradient_accumulation_steps}")
    logger.info(f"Total optimization steps:        {args.max_train_steps:,}")
    logger.info(f"{'='*59}\n")

    # Initialize training state
    global_step = 0
    first_epoch = 0

    # Setup progress bar
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Training Progress",
        position=0
    )

    # Main training loop
    logger.info("Starting training...")
    for epoch in range(first_epoch, args.num_train_epochs):
        models['unet'].train()
        train_loss = 0.0
        implicit_acc_accumulated = 0.0
        
        progress_bar.set_description(f"Epoch {epoch+1}/{args.num_train_epochs}")
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(models['unet']), accelerator.accumulate(models['user_proj']):
                try:
                    # Execute training step
                    loss_dict = training_step(args, batch, models, weight_dtype, accelerator)
                    loss = loss_dict['loss']
                    
                    # Gather metrics across processes
                    gathered_metrics = {
                        key: accelerator.gather(value.repeat(args.train_batch_size)).mean()
                        for key, value in loss_dict.items()
                    }
                    
                    # Update accumulated metrics
                    train_loss += gathered_metrics['loss'].item() / args.gradient_accumulation_steps
                    implicit_acc_accumulated += gathered_metrics['accuracy'].item() / args.gradient_accumulation_steps

                    # Backpropagation
                    accelerator.backward(loss)
                    
                    if accelerator.sync_gradients:
                        # Log metrics
                        metrics = {
                            'train/loss': train_loss,
                            'train/learning_rate': lr_scheduler.get_last_lr()[0],
                            'train/model_mse': gathered_metrics['model_mse'].item(),
                            'train/ref_mse': gathered_metrics['ref_mse'].item(),
                            'train/accuracy': gathered_metrics['accuracy'].item(),
                            'train/wanted_loss': gathered_metrics['model_losses_w'].item(),
                            'train/limited_loss': gathered_metrics['model_losses_l'].item(),
                            'train/dpo_preference': gathered_metrics['dpo_preference'].item(),
                        }
                        accelerator.log(metrics, step=global_step)
                        
                        # Gradient clipping
                        all_params = [p for g in optimizer.param_groups for p in g['params'] if p.grad is not None]
                        accelerator.clip_grad_norm_(all_params, args.max_grad_norm)

                        # Reset accumulators
                        train_loss = 0.0
                        implicit_acc_accumulated = 0.0

                    # Optimizer steps
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    # Update progress
                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1
                        
                        # Save checkpoints
                        if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                            try:
                                unet_u = accelerator.unwrap_model(models['unet'])
                                user_proj_u = accelerator.unwrap_model(models['user_proj'])
                                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                                save_psa_weights(unet_u, user_proj_u, save_path, logger)
                                logger.info(f"Successfully saved checkpoint at step {global_step}")
                            except Exception as e:
                                logger.error(f"Error saving checkpoint at step {global_step}: {str(e)}")

                    # Update progress bar
                    logs = {
                        'loss': loss.detach().item(),
                        'lr': lr_scheduler.get_last_lr()[0],
                        'accuracy': gathered_metrics['accuracy'].item()
                    }
                    progress_bar.set_postfix(**logs)

                    # Check if training should end
                    if global_step >= args.max_train_steps:
                        logger.info(f"Training completed at step {global_step}")
                        break
                        
                except Exception as e:
                    logger.error(f"Error in training step {step}: {str(e)}")
                    continue

            if global_step >= args.max_train_steps:
                break

    # Training completion
    logger.info("Finalizing training...")
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        try:
            logger.info("Converting models to float32 precision...")
            models['unet'] = models['unet'].to(torch.float32)
            
            logger.info("Preparing models for saving...")
            unet_u = accelerator.unwrap_model(models['unet'])
            user_proj_u = accelerator.unwrap_model(models['user_proj'])
            
            save_psa_weights(unet_u, user_proj_u, args.output_dir, logger)
            logger.info("Successfully saved final model weights")
            
        except Exception as e:
            logger.error(f"Error during final model saving: {str(e)}")
            raise

    logger.info("Training completed successfully!")
    accelerator.end_training()


if __name__ == "__main__":
    main()