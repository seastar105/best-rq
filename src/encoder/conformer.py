from typing import Optional

import torch
import torch.nn as nn

from src.configuration import ConformerConfig, ConvModuleConfig
from src.encoder.conv import CausalConv1D
from src.encoder.transformer import (
    AbsolutePositionEmbedding,
    FeedForward,
    RotaryEmbedding,
    SelfAttention,
    activation_map,
)


class ConvolutionModule(nn.Module):
    """Convolution module for Conformer.
    norm -> pointwise conv -> glu -> depthwise conv -> norm -> pointwise conv
    """

    def __init__(self, config: ConvModuleConfig):
        super().__init__()

        self.config = config
        self.causal = config.causal

        norm_class = nn.LayerNorm if config.norm_type == "layernorm" else nn.RMSNorm

        self.first_norm = norm_class(config.dim, eps=1e-8)
        self.pw_conv1 = nn.Conv1d(config.dim, 2 * config.dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.glu = nn.GLU(dim=1)
        dw_conv_layer = CausalConv1D if self.causal else nn.Conv1d

        self.dw_conv = dw_conv_layer(
            config.dim,
            config.dim,
            kernel_size=config.kernel_size,
            stride=1,
            padding=None if self.causal else (config.kernel_size - 1) // 2,
            groups=config.dim,
            bias=False,
        )
        self.dw_conv_norm = norm_class(config.dim, eps=1e-8)
        self.activation = activation_map.get(config.activation, nn.SiLU())
        self.pw_conv2 = nn.Conv1d(config.dim, config.dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        # x: (B, T, C)
        # padding_mask: (B, T)
        x = self.first_norm(x)
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.pw_conv1(x)
        x = self.glu(x)

        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(1), 0.0)
        x = self.dw_conv(x)

        x = x.transpose(1, 2)  # (B, T, C)
        x = self.dw_conv_norm(x)
        x = self.activation(x)
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.pw_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (B, T, C)
        return x


class ConformerBlock(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.config = config
        self.causal = config.causal

        norm_class = nn.LayerNorm if config.norm_type == "layernorm" else nn.RMSNorm
        self.ff_scale = 0.5

        self.ff1_norm = norm_class(config.dim, eps=1e-8)
        self.ff1 = FeedForward(config.feed_forward)

        self.attn_norm = norm_class(config.dim, eps=1e-8)
        self.attn = SelfAttention(config.attention)

        self.conv_module = ConvolutionModule(config.conv_module)

        self.ff2_norm = norm_class(config.dim, eps=1e-8)
        self.ff2 = FeedForward(config.feed_forward)

        self.final_norm = norm_class(config.dim, eps=1e-8)
        self.dropout = nn.Dropout(config.residual_dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        conv_attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ):
        # x: (B, T, C)
        # attention_mask: (B, T)
        x = x + self.dropout(self.ff1(self.ff1_norm(x))) * self.ff_scale
        x = x + self.dropout(self.attn(self.attn_norm(x), attention_mask, rotary_emb))
        x = x + self.conv_module(x, conv_attention_mask)
        x = x + self.dropout(self.ff2(self.ff2_norm(x))) * self.ff_scale
        x = self.final_norm(x)
        return x


class Conformer(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList([ConformerBlock(config) for _ in range(config.num_layers)])

        if config.positional_encoding.type == "absolute":
            self.pos_emb = AbsolutePositionEmbedding(config.dim, config.positional_encoding.max_len)
        elif config.positional_encoding.type == "rotary":
            self.pos_emb = RotaryEmbedding(config)
        else:
            raise ValueError("Unsupported positional encoding type")
        norm_class = nn.LayerNorm if config.norm_type == "layernorm" else nn.RMSNorm
        self.final_norm = norm_class(config.dim, eps=1e-8)
        self.dropout = nn.Dropout(config.residual_dropout)

    def expand_attention_mask(self, attention_mask: Optional[torch.Tensor] = None):
        if attention_mask is None:
            return None
        if attention_mask.ndim == 2:
            # (B, T) -> (B, 1, 1, T), padding mask
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.ndim == 3:
            # (B, T, T) -> (B, 1, T, T), causal mask
            attention_mask = attention_mask.unsqueeze(1)
        else:
            raise ValueError("Invalid attention mask shape")
        return attention_mask

    def forward(
        self,
        embs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        if position_ids is None:
            position_ids = torch.arange(embs.size(1), device=embs.device).unsqueeze(0).expand(embs.size(0), -1)

        if self.config.positional_encoding.type == "absolute":
            embs = embs + self.pos_emb(position_ids)
            rotary_emb = None
        elif self.config.positional_encoding.type == "rotary":
            rotary_emb = self.pos_emb(position_ids)

        if attention_mask is not None:
            conv_attention_mask = attention_mask.clone()
        else:
            conv_attention_mask = None
        attention_mask = self.expand_attention_mask(attention_mask)
        for layer in self.layers:
            embs = layer(embs, attention_mask, conv_attention_mask, rotary_emb)

        return self.final_norm(embs)
