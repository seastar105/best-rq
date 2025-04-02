from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.configuration import AttentionConfig, FeedForwardConfig, TransformerConfig

activation_map = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "swish": nn.SiLU(),
}


class FeedForward(nn.Module):
    def __init__(self, config: FeedForwardConfig):
        super().__init__()

        self.config = config
        self.glu = config.type == "glu"

        if self.glu:
            self.fc1 = nn.Linear(config.dim, 2 * config.intermediate_dim, bias=False)
        else:
            self.fc1 = nn.Linear(config.dim, config.intermediate_dim, bias=False)
        self.activation = activation_map.get(config.activation)
        self.fc2 = nn.Linear(config.intermediate_dim, config.dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        if self.glu:
            x, gate = self.fc1(x).chunk(2, dim=-1)
            x = x * self.activation(gate)
        else:
            x = self.fc1(x)
            x = self.activation(x)
        x = self.dropout(x)
        return self.dropout(self.fc2(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        dim = config.dim // config.attention.num_heads
        theta = config.positional_encoding.rope_theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, position_ids: torch.Tensor):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded.float().to(self.device) @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    pos = pos.unsqueeze(1)
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


class AbsolutePositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int):
        super().__init__()
        self.embedding = nn.Embedding(max_len, dim)

    def forward(self, position_ids: torch.Tensor):
        return self.embedding(position_ids)


class SelfAttention(nn.Module):
    def __init__(self, config: AttentionConfig):
        super().__init__()
        assert config.dim % config.num_heads == 0, "Dimension must be divisible by number of heads"

        self.config = config
        self.dim_head = config.dim // config.num_heads

        self.q_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.attention_window = config.attention_window

        self.dropout = config.dropout
        self.causal = config.causal

    def get_attention_mask(self, seq_len: int, device: torch.device, attention_mask: Optional[torch.Tensor]):
        def get_default_mask(seq_len):
            if not self.causal and self.attention_window is None:
                return None
            mask = None
            if self.attention_window is not None:
                left_context, right_context = self.attention_window
                i = torch.arange(seq_len, device=device).unsqueeze(0)
                j = torch.arange(seq_len, device=device).unsqueeze(1)
                mask = (i >= j - left_context) & (i <= j + right_context)
            if self.causal:
                if mask is None:
                    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
                else:
                    mask = mask & torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
            return mask

        if attention_mask is None:
            # if causal only, return None to make sdpa handle
            if self.causal and self.attention_window is None:
                return None
            return get_default_mask(seq_len)
        default_mask = get_default_mask(seq_len)
        if default_mask is None:
            return attention_mask
        return attention_mask & default_mask

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor], rotary_emb: Optional[torch.Tensor] = None
    ):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = rearrange(q, "b t (h d) -> b h t d", h=self.config.num_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.config.num_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.config.num_heads)

        if rotary_emb is not None:
            q = apply_rotary_pos_emb(rotary_emb, q)
            k = apply_rotary_pos_emb(rotary_emb, k)

        seq_len = x.size(1)
        attention_mask = self.get_attention_mask(seq_len, x.device, attention_mask)
        causal = self.causal if attention_mask is None else False

        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=causal
        )
        attn = rearrange(attn, "b h t d -> b t (h d)")
        out = self.out_proj(attn)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        norm_class = nn.LayerNorm if config.norm_type == "layernorm" else nn.RMSNorm
        self.attn_norm = norm_class(config.dim)
        self.attn = SelfAttention(config.attention)
        self.ff_norm = norm_class(config.dim)
        self.ff = FeedForward(config.feed_forward)
        self.dropout = nn.Dropout(config.residual_dropout)
        self.causal = config.causal

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor], rotary_emb: Optional[torch.Tensor] = None
    ):
        x = x + self.dropout(self.attn(self.attn_norm(x), attention_mask, rotary_emb))
        x = x + self.dropout(self.ff(self.ff_norm(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])

        if config.positional_encoding.type == "absolute":
            self.pos_emb = AbsolutePositionEmbedding(config.dim, config.positional_encoding.max_len)
        elif config.positional_encoding.type == "rotary":
            self.pos_emb = RotaryEmbedding(config)
        else:
            raise ValueError("Unsupported positional encoding type")
        norm_class = nn.LayerNorm if config.norm_type == "layernorm" else nn.RMSNorm
        self.final_norm = norm_class(config.dim)
        self.dropout = nn.Dropout(config.residual_dropout)

    def forward(
        self,
        embs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        if position_ids is None:
            position_ids = torch.arange(embs.size(1), device=embs.device).unsqueeze(0).expand(embs.size(0), -1)

        if self.config.positional_encoding.type == "absolute":
            x = embs + self.pos_emb(position_ids)
            rotary_emb = None
        elif self.config.positional_encoding.type == "rotary":
            rotary_emb = self.pos_emb(position_ids)

        for layer in self.layers:
            x = layer(embs, attention_mask, rotary_emb)

        return self.final_norm(x)
