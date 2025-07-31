"""
Attention Processors for Personalized Safety Alignment in Diffusion Models.

This module implements custom attention processors that enable:
1. User-specific content filtering through personalized attention mechanisms
2. Efficient attention computation using PyTorch 2.0's scaled dot-product attention
3. Integration of user embeddings into the diffusion model's attention layers

Key Components:
- AttnProcessor2_0: Base attention processor using PyTorch 2.0 optimizations
- PSAAttnProcessor2_0: PSA-specific attention processor that incorporates user embeddings
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention_processor import (
    Attention,
    IPAdapterAttnProcessor2_0,
)


class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # Restore spatial dimensions for image-like inputs
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # Apply residual connection if enabled
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        # Apply output rescaling
        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class PSAAttnProcessor2_0(torch.nn.Module):
    """
    PSA (Personalized Safety Alignment) Attention Processor.
    
    This processor extends the base attention mechanism to incorporate user-specific
    safety preferences through learned attention weights. It:
    1. Projects user embeddings into the attention space
    2. Applies user-specific scaling to control influence
    3. Integrates with the base diffusion model's attention mechanism
    
    The user information comes from LLM-processed structured profiles and is provided
    as embeddings through the encoder_hidden_states parameter.
    
    Args:
        hidden_size: Dimension of the attention layer's hidden states
        user_embedding_dim: Dimension of the input user embeddings (default: 3584 for Qwen)
        num_tokens: Number of user embedding tokens (int or sequence of ints)
        scale: Scaling factor(s) for user attention weights (float or sequence of floats)
    
    Raises:
        ValueError: If scales and num_tokens lengths don't match
        ImportError: If PyTorch version doesn't support scaled_dot_product_attention
    """
    def __init__(
        self,
        hidden_size: int,
        user_embedding_dim: int = 3584,
        num_tokens: Union[int, Tuple[int, ...], List[int]] = (1,),
        scale: Union[float, List[float]] = 1.0
    ):
        super().__init__()

        # Core dimensions
        self.hidden_size = hidden_size
        self.user_embedding_dim = user_embedding_dim

        # Ensure num_tokens is a sequence
        self.num_tokens = [num_tokens] if not isinstance(num_tokens, (tuple, list)) else num_tokens

        # Set attention scaling
        self._set_scale(scale)

        # Validate configurations
        if len(self.scale) != len(self.num_tokens):
            raise ValueError(
                f"Number of scales ({len(self.scale)}) must match "
                f"number of token groups ({len(self.num_tokens)})"
            )
        
        # Initialize projection layers with zero weights
        self.to_k_user = nn.ModuleList([
            self._init_zero(nn.Linear(user_embedding_dim, hidden_size, bias=False)) 
            for _ in range(len(self.num_tokens))
        ])
        self.to_v_user = nn.ModuleList([
            self._init_zero(nn.Linear(user_embedding_dim, hidden_size, bias=False)) 
            for _ in range(len(self.num_tokens))
        ])

    def _init_zero(self, module: nn.Module) -> nn.Module:
        """Initialize module weights to zero for stable training start."""
        nn.init.constant_(module.weight, 0.0)
        return module

    def _set_scale(self, scale: Union[float, List[float]]) -> None:
        """
        Set the scaling factor(s) for user attention weights.
        
        Args:
            scale: Single scale or list of scales matching num_tokens
        """
        if not isinstance(scale, list):
            self.scale = [scale] * len(self.num_tokens)
        else:
            self.scale = scale

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process attention with user embeddings integration.
        
        Args:
            attn: Base attention module with helper functions
            hidden_states: Query tensor for attention computation
            encoder_hidden_states: Either:
                - Tuple of (base_encoder, user_embedding) tensors
                - Single tensor with user embedding appended
                - None for self-attention
            attention_mask: Optional mask for attention weights
            temb: Optional timestep embeddings for conditioning
            
        Returns:
            Processed hidden states with user-aware attention applied
        """
        # Store residual for skip connection
        residual = hidden_states

        # Extract user embeddings and base encoder states
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, tuple):
                base_encoder, user_embedding = encoder_hidden_states
            else:
                # Legacy format support: extract user embedding from end of sequence
                end_pos = encoder_hidden_states.shape[1] - self.num_tokens[0]
                base_encoder = encoder_hidden_states[:, :end_pos, :]
                user_embedding = encoder_hidden_states[:, end_pos:, :]
        else:
            # Self-attention case
            base_encoder = hidden_states
            user_embedding = None

        # Apply spatial normalization if present
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        # Handle 4D input (image-like) by flattening spatial dimensions
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # Get sequence dimensions and prepare attention mask
        batch_size, sequence_length, _ = (
            hidden_states.shape if base_encoder is None else base_encoder.shape
        )
        # Prepare attention mask and apply group norm if needed
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # Compute query from hidden states
        query = attn.to_q(hidden_states)
        
        # Prepare encoder states for key/value computation
        if base_encoder is None:
            base_encoder = hidden_states
        elif attn.norm_cross:
            base_encoder = attn.norm_encoder_hidden_states(base_encoder)

        # Compute key and value
        key = attn.to_k(base_encoder)
        value = attn.to_v(base_encoder)

        # Reshape for multi-head attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Compute base attention outputs
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, 
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Integrate user embeddings if available
        if user_embedding is not None:
            for idx, (to_k_user, to_v_user, token_scale) in enumerate(
                zip(self.to_k_user, self.to_v_user, self.scale)
            ):
                # Skip if we have more projections than user tokens
                if user_embedding.shape[1] <= idx:
                    continue
                    
                # Extract and reshape current user token
                current_user = user_embedding[:, idx:idx+1, :]  # [batch, 1, user_embedding_dim]
                
                # Project user embedding to key/value space
                user_key = to_k_user(current_user)    # [batch, 1, hidden_size]
                user_value = to_v_user(current_user)  # [batch, 1, hidden_size]
                
                # Reshape for multi-head attention
                user_key = user_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                user_value = user_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                
                # Compute and integrate user-specific attention
                user_attn = F.scaled_dot_product_attention(
                    query, user_key, user_value,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False
                )
                user_attn = user_attn.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                user_attn = user_attn.to(query.dtype)
                
                # Add scaled user attention to base attention
                hidden_states = hidden_states + token_scale * user_attn

        # Final projection layers
        hidden_states = attn.to_out[0](hidden_states)  # Linear projection
        hidden_states = attn.to_out[1](hidden_states)  # Dropout

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
