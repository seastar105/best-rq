import math
from typing import Optional

import torch
import torch.nn as nn

from src.configuration import SubSamplerConfig
from src.encoder.conv import CausalConv1D, CausalConv2D

activation_map = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "swish": nn.SiLU(),
}


class ConvolutionalSubSampler(nn.Module):
    """Subsampling module for Transformer encoder. This module downsamples the input sequence"""

    def __init__(self, config: SubSamplerConfig):
        super().__init__()

        self.config = config
        self.subsampling_factor = config.subsampling_factor

        num_layers = int(math.log(self.subsampling_factor, 2))

        self.causal = config.causal
        self.use_conv2d = config.type == "conv2d"
        in_channels = [config.dim_in] + [config.num_channels] * (num_layers - 1)
        out_channels = [config.num_channels] * num_layers

        if self.use_conv2d:
            in_channels[0] = 1  # Conv2D use (B, 1, T, C)

        activation = activation_map.get(config.activation)
        layers = []
        for i in range(num_layers):
            if self.use_conv2d:
                if self.causal:
                    layers.append(
                        CausalConv2D(
                            in_channels=in_channels[i],
                            out_channels=out_channels[i],
                            kernel_size=3,
                            stride=2,
                            padding=None,
                        )
                    )
                else:
                    layers.append(
                        nn.Conv2d(
                            in_channels=in_channels[i], out_channels=out_channels[i], kernel_size=3, stride=2, padding=1
                        )
                    )
            else:
                if self.causal:
                    layers.append(
                        CausalConv1D(
                            in_channels=in_channels[i],
                            out_channels=out_channels[i],
                            kernel_size=3,
                            stride=2,
                            padding=None,
                        )
                    )
                else:
                    layers.append(
                        nn.Conv1d(
                            in_channels=in_channels[i], out_channels=out_channels[i], kernel_size=3, stride=2, padding=1
                        )
                    )
            layers.append(activation)

        # Roughly output shape:
        # Conv2D: (B, C, T, F) -> (B, out_channels, T // subsampling_factor, F // subsampling_factor) -> (B, T // subsampling_factor, F // subsampling_factor * out_channels)
        # Conv1D: (B, C, T) -> (B, out_channels, T // subsampling_factor) -> (B, T // subsampling_factor, out_channels)
        self.conv_outdim = self.calculate_conv_outdim(config)
        self.out_proj = nn.Linear(self.conv_outdim, config.dim, bias=False)
        self.layers = nn.Sequential(*layers)

    def calculate_conv_outdim(self, config: SubSamplerConfig):
        last_dim = config.dim_in if self.use_conv2d else 1
        num_layers = int(math.log(config.subsampling_factor, 2))

        if self.use_conv2d:
            for i in range(num_layers):
                if config.causal:
                    last_dim = (last_dim + 2) // 2  # causal conv
                else:
                    last_dim = last_dim // 2
        return last_dim * config.num_channels

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        # x: (B, T, C)
        if self.use_conv2d:
            x = x.unsqueeze(1)  # (B, 1, T, C)
        else:
            x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        for layer in self.layers:
            x = layer(x)

        # flatten
        if self.use_conv2d:
            b, c, t, f = x.size()
            x = x.permute(0, 2, 3, 1).contiguous()  # (B, T, F, C)
            x = x.view(b, t, f * c)  # (B, T, F * C)
        else:
            x = x.transpose(1, 2)
        x = self.out_proj(x)
        if lengths is not None:
            lengths = (lengths + self.subsampling_factor - 1) // self.subsampling_factor
        return x, lengths
