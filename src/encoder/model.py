from typing import Optional

import torch
import torch.nn as nn

from src.configuration import ConformerConfig, EncoderConfig, TransformerConfig
from src.data.fbank import get_mel_transform
from src.encoder.conformer import Conformer
from src.encoder.subsampler import ConvolutionalSubSampler
from src.encoder.transformer import Transformer


class SpeechEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        self.mel_transform = get_mel_transform(config.mel)
        self.conv_subsampler = ConvolutionalSubSampler(config.subsampler)
        if isinstance(config.transformer, ConformerConfig):
            self.transformer = Conformer(config.transformer)
        elif isinstance(config.transformer, TransformerConfig):
            self.transformer = Transformer(config.transformer)
        else:
            raise ValueError("Unsupported transformer type")
        self.causal = config.causal

    def wav_lengths_to_mel_lengths(self, lengths: torch.Tensor):
        if lengths is None:
            return None
        hop_length = self.config.mel.hop_length
        lengths = lengths + (hop_length // 2)
        lengths = lengths // hop_length
        return lengths

    def lengths_to_attention_mask(self, lengths: torch.Tensor, max_length: Optional[int] = None):
        if lengths is None:
            return None
        if max_length is None:
            max_length = lengths.max()
        attention_mask = torch.arange(max_length).to(lengths.device) < lengths.unsqueeze(-1)
        return attention_mask

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        if x.ndim == 2:
            # x: (batch_size, time), in case input is waveform
            x = self.mel_transform.extract_batch(x, sampling_rate=self.config.mel.sampling_rate)
            lengths = self.wav_lengths_to_mel_lengths(lengths)

        x, lengths = self.conv_subsampler(x, lengths)
        attention_mask = self.lengths_to_attention_mask(lengths, max_length=x.size(1))
        embs = self.transformer(x, attention_mask=attention_mask)
        return embs, lengths
