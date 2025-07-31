"""
PSA-Enhanced UNet Conditional Model Implementation.

This module extends the standard UNet2DConditionModel to support Personalized
Safety Alignment (PSA) by:
1. Adding user embedding projection capabilities
2. Supporting different types of conditional inputs (text, image, user data)
3. Enabling flexible encoder hidden state processing

The implementation maintains compatibility with existing diffusion model architectures
while adding PSA-specific functionality.

Note: This implementation builds on the diffusers library's UNet2DConditionModel
and adds support for user-specific content filtering.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Diffusers components
from diffusers.models import UNet2DConditionModel
from diffusers.models.unets.unet_2d_condition import logger
from diffusers.models.embeddings import (
    ImageProjection,
    TextImageProjection,
)

# Local imports
from .psa_adapter import PSAProjection

class PSAUNet2DConditionModel(UNet2DConditionModel):
    """
    UNet model enhanced with Personalized Safety Alignment capabilities.
    
    This model extends the base UNet2DConditionModel by adding support for:
    - User embedding projection
    - Multiple conditional input types
    - Flexible encoder state processing
    
    The model maintains compatibility with standard diffusion pipelines while
    enabling user-specific content filtering through PSA mechanisms.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize PSA-enhanced UNet model with standard parameters."""
        super().__init__(*args, **kwargs)

    def _set_encoder_hid_proj(
        self,
        encoder_hid_dim_type: Optional[str],
        cross_attention_dim: Union[int, Tuple[int]],
        encoder_hid_dim: Optional[int],
    ) -> None:
        """
        Configure the encoder hidden state projection layer.
        
        This method sets up the appropriate projection layer based on the encoder
        type and dimensions. Supports multiple projection types:
        - Text projection (standard)
        - Text-image combined projection
        - Image-only projection
        - User embedding projection (PSA-specific)
        
        Args:
            encoder_hid_dim_type: Type of encoder projection to use
            cross_attention_dim: Target dimension for cross attention
            encoder_hid_dim: Source dimension of encoder hidden states
            
        Raises:
            ValueError: If encoder_hid_dim is missing when type is specified
        """
        # Set default projection type if dimensions are provided
        if encoder_hid_dim_type is None and encoder_hid_dim is not None:
            encoder_hid_dim_type = "text_proj"
            self.register_to_config(encoder_hid_dim_type=encoder_hid_dim_type)
            logger.info(
                "encoder_hid_dim_type defaults to 'text_proj' as "
                "`encoder_hid_dim` is defined."
            )

        # Validate encoder dimensions
        if encoder_hid_dim is None and encoder_hid_dim_type is not None:
            raise ValueError(
                f"encoder_hid_dim must be provided when encoder_hid_dim_type "
                f"is set to {encoder_hid_dim_type}."
            )

        # Configure projection based on type
        if encoder_hid_dim_type == "text_proj":
            # Standard text projection
            self.encoder_hid_proj = nn.Linear(
                encoder_hid_dim,
                cross_attention_dim
            )
            
        elif encoder_hid_dim_type == "text_image_proj":
            # Combined text-image projection (e.g., Kandinsky 2.1)
            self.encoder_hid_proj = TextImageProjection(
                text_embed_dim=encoder_hid_dim,
                image_embed_dim=cross_attention_dim,
                cross_attention_dim=cross_attention_dim,
            )
            
        elif encoder_hid_dim_type == "image_proj":
            # Image-only projection (e.g., Kandinsky 2.2)
            self.encoder_hid_proj = ImageProjection(
                image_embed_dim=encoder_hid_dim,
                cross_attention_dim=cross_attention_dim,
            )
            
        elif encoder_hid_dim_type == "user_proj":
            # PSA-specific user embedding projection
            self.encoder_hid_proj = PSAProjection(
                cross_attention_dim=cross_attention_dim,
                clip_embeddings_dim=encoder_hid_dim,
                clip_extra_context_tokens=1,
            )
            
        elif encoder_hid_dim_type is not None:
            raise ValueError(
                f"Unsupported encoder_hid_dim_type: {encoder_hid_dim_type}. "
                f"Must be one of: None, 'text_proj', 'text_image_proj', "
                f"'image_proj', or 'user_proj'."
            )
            
        else:
            # No projection needed
            self.encoder_hid_proj = None
    
    def process_encoder_hidden_states(
        self, 
        encoder_hidden_states: torch.Tensor, 
        added_cond_kwargs: Dict[str, Any]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process encoder hidden states based on projection type.
        
        This method handles different types of projections:
        - Text-only projection
        - Text-image combined projection
        - Image-only projection
        - IP-based image projection
        - User embedding projection (PSA)
        
        Args:
            encoder_hidden_states: Base encoder states to process
            added_cond_kwargs: Additional conditioning inputs (e.g., image/user embeds)
            
        Returns:
            Processed hidden states, either as a single tensor or a tuple
            
        Raises:
            ValueError: If required conditional inputs are missing
        """
        # Skip if no projection is configured
        if self.encoder_hid_proj is None:
            return encoder_hidden_states

        # Get projection type
        proj_type = self.config.encoder_hid_dim_type

        # Handle text-only projection
        if proj_type == "text_proj":
            return self.encoder_hid_proj(encoder_hidden_states)

        # Handle text-image combined projection (Kandinsky 2.1)
        elif proj_type == "text_image_proj":
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} requires 'image_embeds' in added_cond_kwargs "
                    f"when using text_image_proj"
                )
            image_embeds = added_cond_kwargs.get("image_embeds")
            return self.encoder_hid_proj(encoder_hidden_states, image_embeds)

        # Handle image-only projection (Kandinsky 2.2)
        elif proj_type == "image_proj":
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} requires 'image_embeds' in added_cond_kwargs "
                    f"when using image_proj"
                )
            image_embeds = added_cond_kwargs.get("image_embeds")
            return self.encoder_hid_proj(image_embeds)

        # Handle IP-based image projection
        elif proj_type == "ip_image_proj":
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} requires 'image_embeds' in added_cond_kwargs "
                    f"when using ip_image_proj"
                )
            
            # Process text embeddings if available
            if hasattr(self, "text_encoder_hid_proj") and self.text_encoder_hid_proj is not None:
                encoder_hidden_states = self.text_encoder_hid_proj(encoder_hidden_states)

            # Process and combine image embeddings
            image_embeds = added_cond_kwargs.get("image_embeds")
            image_embeds = self.encoder_hid_proj(image_embeds)
            return (encoder_hidden_states, image_embeds)

        # Handle user embedding projection (PSA)
        elif proj_type == "user_info_proj":
            if "user_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} requires 'user_embeds' in added_cond_kwargs "
                    f"when using user_info_proj"
                )

            # Process text embeddings if available
            if hasattr(self, "text_encoder_hid_proj") and self.text_encoder_hid_proj is not None:
                encoder_hidden_states = self.text_encoder_hid_proj(encoder_hidden_states)

            # Process and combine user embeddings
            user_embeds = added_cond_kwargs.get("user_embeds")
            user_embeds = self.encoder_hid_proj(user_embeds)
            return (encoder_hidden_states, user_embeds)

        return encoder_hidden_states
