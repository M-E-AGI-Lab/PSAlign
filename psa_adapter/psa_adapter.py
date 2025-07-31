"""
PSA Adapter Implementation for Personalized Safety Alignment in Diffusion Models.

This module implements the PSA (Personalized Safety Alignment) adapter architecture for
text-to-image diffusion models. It enables:
1. User-specific content filtering
2. Personalized safety boundary learning
3. Integration with both SD and SDXL architectures

Key Components:
- PSAProjection: Projects user embeddings to the diffusion model's attention space
- PSAAdapter: Main adapter class for SD models
- PSAAdapterXL: Extended adapter class for SDXL models
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Union

# Mathematical and array processing
import numpy as np
import torch
import torch.nn.functional as F

# Diffusion model components
from diffusers import StableDiffusionPipeline
from diffusers.models import UNet2DConditionModel
from diffusers.pipelines.controlnet import MultiControlNetModel

# Image processing
from PIL import Image

# Model weights handling
from safetensors import safe_open

# Transformers components
from transformers import (
    AutoConfig,
    AutoModelForCausalLM, 
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)

# Local imports
from .utils import is_torch2_available, get_generator, batch_user_data_to_embedding
from .attention_processor import PSAAttnProcessor2_0, AttnProcessor2_0


def init_psa_adapter(unet: UNet2DConditionModel, num_tokens_user: int = 1) -> UNet2DConditionModel:
    """
    Initialize the PSA adapter by setting up attention processors for each attention block.
    
    Args:
        unet: The UNet model to be adapted
        num_tokens_user: Number of user tokens to use in the adapter (default: 1)
    
    Returns:
        The modified UNet model with PSA attention processors
    """
    attn_procs = {}
    
    for name in unet.attn_processors.keys():
        # Determine cross attention dimension based on layer type
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        
        # Determine hidden size based on block location
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks."):].split('.')[0])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks."):].split('.')[0])
            hidden_size = unet.config.block_out_channels[block_id]
        
        # Use standard attention for self-attention layers
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor2_0()
        else:
            # Initialize PSA attention processor for cross-attention layers
            attn_proc = PSAAttnProcessor2_0(
                hidden_size=hidden_size, 
                user_embedding_dim=cross_attention_dim,
                num_tokens=num_tokens_user,
            )
            attn_procs[name] = attn_proc

    unet.set_attn_processor(attn_procs)
    return unet


class PSAProjection(torch.nn.Module):
    """
    Projects user embeddings into the attention space of the diffusion model.
    
    This module transforms high-dimensional user embeddings (e.g., from LLM) into
    lower-dimensional tokens that can be used as cross-attention conditions in
    the diffusion model's PSA adapter layers.
    """

    def __init__(
        self, 
        cross_attention_dim: int = 768,
        clip_embeddings_dim: int = 3584,
        clip_extra_context_tokens: int = 1
    ):
        """
        Initialize the projection model.
        
        Args:
            cross_attention_dim: Target dimension for the projected tokens
            clip_embeddings_dim: Input dimension of the user embeddings
            clip_extra_context_tokens: Number of context tokens to generate
        """
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        
        # Linear projection followed by layer normalization
        self.proj = torch.nn.Linear(
            clip_embeddings_dim, 
            self.clip_extra_context_tokens * cross_attention_dim
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, user_embeds: torch.Tensor) -> torch.Tensor:
        """
        Project user embeddings into the attention space.
        
        Args:
            user_embeds: Input embeddings from the LLM [batch_size, embedding_dim]
            
        Returns:
            Projected and normalized tokens [batch_size, num_tokens, cross_attention_dim]
        """
        projected_tokens = self.proj(user_embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        normalized_tokens = self.norm(projected_tokens)
        return normalized_tokens
    

class PSAAdapter:
    """
    PSA (Personalized Safety Alignment) Adapter for Stable Diffusion models.
    
    This class implements the core PSA functionality, enabling personalized content
    filtering in diffusion models by:
    1. Processing user data through an LLM
    2. Projecting user embeddings into the diffusion model's attention space
    3. Conditioning the generation process on user-specific safety preferences
    """
    
    def __init__(
        self, 
        sd_pipe: StableDiffusionPipeline, 
        psa_adapter_ckpt: Optional[str] = None, 
        llm_model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct", 
        load_llm_weights: bool = True,
        device: str = "cuda", 
        num_tokens: int = 1
    ):
        """
        Initialize the PSA adapter.
        
        Args:
            sd_pipe: Base Stable Diffusion pipeline
            psa_adapter_ckpt: Path to pretrained PSA adapter weights
            llm_model_name_or_path: Path or identifier of the LLM for user processing
            load_llm_weights: Whether to load the LLM weights or just config
            device: Target device for computation
            num_tokens: Number of user tokens to use
        """
        # Core configuration
        self.device = device
        self.llm_model_name_or_path = llm_model_name_or_path
        self.psa_adapter_ckpt = psa_adapter_ckpt
        self.num_tokens = num_tokens

        # Initialize diffusion model with PSA
        self.pipe = sd_pipe.to(self.device)
        self.set_psa_adapter()

        # Initialize LLM components
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name_or_path)
        self.llm_config = AutoConfig.from_pretrained(self.llm_model_name_or_path)
        
        # Load LLM if requested
        if load_llm_weights:
            self.user_processor = (
                AutoModelForCausalLM
                .from_pretrained(self.llm_model_name_or_path, config=self.llm_config)
                .to(self.device, dtype=torch.bfloat16)
            )
            
        # Initialize projection model
        self.user_proj_model = self.init_proj()

        self.load_psa_adapter()

    def init_proj(self) -> PSAProjection:
        """Initialize the user embedding projection model."""
        user_proj_model = PSAProjection(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.llm_config.hidden_size,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return user_proj_model

    def set_psa_adapter(self) -> None:
        """Configure the PSA attention processors for the UNet model."""
        unet = self.pipe.unet
        attn_procs = {}
        
        for name in unet.attn_processors.keys():
            # Determine dimensions for each attention block
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            
            # Get hidden size based on block location
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
                
            # Initialize appropriate attention processor
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor2_0()
            else:
                attn_procs[name] = PSAAttnProcessor2_0(
                    hidden_size=hidden_size, 
                    user_embedding_dim=cross_attention_dim,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)
                
        unet.set_attn_processor(attn_procs)

    def load_psa_adapter(self) -> None:
        """Load pretrained weights for the PSA adapter components."""
        # Load state dict based on file format
        if os.path.splitext(self.psa_adapter_ckpt)[-1] == ".safetensors":
            state_dict = {"user_proj": {}, "psa_adapter": {}}
            with safe_open(self.psa_adapter_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("user_proj."):
                        state_dict["user_proj"][key.replace("user_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("psa_adapter."):
                        state_dict["psa_adapter"][key.replace("psa_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.psa_adapter_ckpt, map_location="cpu")
        
        # Load projection model weights
        self.user_proj_model.load_state_dict(state_dict["user_proj"])
        
        # Load PSA adapter weights
        psa_adapter_modules = torch.nn.ModuleList([
            proc for proc in self.pipe.unet.attn_processors.values() 
            if isinstance(proc, PSAAttnProcessor2_0)
        ])
        psa_adapter_modules.load_state_dict(state_dict["psa_adapter"])
    
    @torch.inference_mode()
    def get_user_embeds(
        self, 
        user_info: Optional[List[str]] = None, 
        llm_ban_prompt: Optional[str] = None, 
        llm_emb_prompt: Optional[str] = None
    ) -> torch.Tensor:
        """
        Get user-specific embeddings from the LLM.
        
        Args:
            user_info: List of JSON-formatted user preference strings
            llm_ban_prompt: Template for content banning prompts
            llm_emb_prompt: Template for embedding generation prompts
            
        Returns:
            Projected user embeddings for conditioning the diffusion model
            
        Raises:
            ValueError: If required parameters are missing
        """
        if user_info is not None and llm_emb_prompt is not None and llm_ban_prompt is not None:
            llm_user_embeds, _, _ = batch_user_data_to_embedding(
                user_info, 
                self.user_processor, 
                self.tokenizer, 
                llm_ban_prompt, 
                llm_emb_prompt
            )
        else:
            raise ValueError(
                "Required parameters missing. Please provide: user_info, "
                "llm_emb_prompt, and llm_ban_prompt"
            )
            
        # Convert embeddings to appropriate device and dtype
        llm_user_embeds = llm_user_embeds.to(self.device, dtype=torch.float16)
        
        # Project to attention space
        user_prompt_embeds = self.user_proj_model(llm_user_embeds)
        return user_prompt_embeds

    def set_scale(self, scale: float) -> None:
        """
        Set the scaling factor for PSA attention.
        
        Args:
            scale: Scaling factor for the PSA attention weights
        """
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, PSAAttnProcessor2_0):
                attn_processor._set_scale(scale)

    def generate(
        self,
        user_info: Optional[Union[str, List[str]]] = None,
        llm_ban_prompt: Optional[str] = None,
        llm_emb_prompt: Optional[str] = None,
        llm_user_embeds: Optional[torch.Tensor] = None,  # [batch_size, hidden_size]
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        scale: float = 1.0,
        num_samples: int = 1,
        seed: Optional[int] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        **kwargs,
    ) -> List[Image.Image]:
        """
        Generate images using the PSA-enhanced diffusion model.
        
        Args:
            user_info: User preference data in JSON format
            llm_ban_prompt: Template for content banning
            llm_emb_prompt: Template for embedding generation
            llm_user_embeds: Pre-computed user embeddings (optional)
            prompt: Text prompt for generation
            negative_prompt: Text prompt for negative conditioning
            scale: PSA attention scaling factor
            num_samples: Number of images to generate per prompt
            seed: Random seed for generation
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            **kwargs: Additional arguments for the pipeline
            
        Returns:
            List of generated PIL Images
        """
        # Set PSA attention scale
        self.set_scale(scale)

        # Determine batch size and prepare user info
        if user_info is not None:
            user_info = [user_info] if not isinstance(user_info, List) else user_info
            batch_size = len(user_info)
        else:
            batch_size = llm_user_embeds.size(0)

        # Get user embeddings
        if llm_user_embeds is None:
            user_prompt_embeds = self.get_user_embeds(
                user_info=user_info,
                llm_ban_prompt=llm_ban_prompt,
                llm_emb_prompt=llm_emb_prompt
            )
        else:
            llm_user_embeds = llm_user_embeds.to(self.device, dtype=torch.float16)
            user_prompt_embeds = self.user_proj_model(llm_user_embeds)

        # Create unconditioned user embeddings
        uncond_user_prompt_embeds = self.user_proj_model(
            torch.zeros(
                (batch_size, self.llm_config.hidden_size),
                device=self.device,
                dtype=torch.float16
            )
        )

        # Prepare embeddings for multiple samples
        bs_embed, seq_len, _ = user_prompt_embeds.shape
        user_prompt_embeds = user_prompt_embeds.repeat(1, num_samples, 1)
        user_prompt_embeds = user_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_user_prompt_embeds = uncond_user_prompt_embeds.repeat(1, num_samples, 1)
        uncond_user_prompt_embeds = uncond_user_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        # Set default prompts if not provided
        prompt = prompt or "best quality, high quality"
        negative_prompt = negative_prompt or "monochrome, lowres, bad anatomy, worst quality, low quality"

        # Ensure prompts are lists
        prompt = [prompt] * batch_size if not isinstance(prompt, List) else prompt
        negative_prompt = [negative_prompt] * batch_size if not isinstance(negative_prompt, List) else negative_prompt

        # Generate text embeddings
        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            
            # Combine text and user embeddings
            prompt_embeds = torch.cat([prompt_embeds_, user_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_user_prompt_embeds], dim=1)

        # Set up generator for reproducibility
        generator = get_generator(seed, self.device)

        # Run diffusion pipeline
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class PSAAdapterXL(PSAAdapter):
    """
    PSA Adapter for Stable Diffusion XL models.
    
    This class extends PSAAdapter with SDXL-specific functionality, particularly
    handling the dual-text encoder architecture and pooled embeddings.
    """

    def generate(
        self,
        user_info: Optional[Union[str, List[str]]] = None,
        llm_ban_prompt: Optional[str] = None,
        llm_emb_prompt: Optional[str] = None,
        llm_user_embeds: Optional[torch.Tensor] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        scale: float = 1.0,
        num_samples: int = 1,
        seed: Optional[int] = None,
        num_inference_steps: int = 30,
        **kwargs,
    ) -> List[Image.Image]:
        """
        Generate images using the SDXL PSA-enhanced model.
        
        Similar to base PSA generation but handles SDXL's additional embeddings.
        
        Args:
            user_info: User preference data in JSON format
            llm_ban_prompt: Template for content banning
            llm_emb_prompt: Template for embedding generation
            llm_user_embeds: Pre-computed user embeddings
            prompt: Text prompt for generation
            negative_prompt: Text prompt for negative conditioning
            scale: PSA attention scaling factor
            num_samples: Number of images per prompt
            seed: Random seed for reproducibility
            num_inference_steps: Number of denoising steps
            **kwargs: Additional pipeline arguments
            
        Returns:
            List of generated PIL Images
        """
        # Set PSA attention scale
        self.set_scale(scale)

        # Determine batch size and prepare user info
        if user_info is not None:
            user_info = [user_info] if not isinstance(user_info, List) else user_info
            batch_size = len(user_info)
        else:
            batch_size = llm_user_embeds.size(0)

        # Get user embeddings
        if llm_user_embeds is None:
            user_prompt_embeds = self.get_user_embeds(
                user_info=user_info,
                llm_ban_prompt=llm_ban_prompt,
                llm_emb_prompt=llm_emb_prompt
            )
        else:
            llm_user_embeds = llm_user_embeds.to(self.device, dtype=torch.float16)
            user_prompt_embeds = self.user_proj_model(llm_user_embeds)

        # Create unconditioned user embeddings
        uncond_user_prompt_embeds = self.user_proj_model(
            torch.zeros(
                (llm_user_embeds.size(0), self.llm_config.hidden_size),
                device=self.device,
                dtype=torch.float16
            )
        )

        # Prepare embeddings for multiple samples
        bs_embed, seq_len, _ = user_prompt_embeds.shape
        user_prompt_embeds = user_prompt_embeds.repeat(1, num_samples, 1)
        user_prompt_embeds = user_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_user_prompt_embeds = uncond_user_prompt_embeds.repeat(1, num_samples, 1)
        uncond_user_prompt_embeds = uncond_user_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        # Set default prompts
        prompt = prompt or "best quality, high quality"
        negative_prompt = negative_prompt or "monochrome, lowres, bad anatomy, worst quality, low quality"

        # Ensure prompts are lists
        prompt = [prompt] * batch_size if not isinstance(prompt, List) else prompt
        negative_prompt = [negative_prompt] * batch_size if not isinstance(negative_prompt, List) else negative_prompt

        # Generate SDXL-specific embeddings
        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            
            # Combine text and user embeddings
            prompt_embeds = torch.cat([prompt_embeds, user_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_user_prompt_embeds], dim=1)

        # Set up generator
        generator = get_generator(seed, self.device)
        
        # Run SDXL pipeline with pooled embeddings
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images
