import dataclasses
from typing import Optional, Tuple, Union


@dataclasses.dataclass
class MelSpectrogramConfig:
    """Configuration class for the Mel Spectrogram. Use kaldi-style parameters.
    https://github.com/lhotse-speech/lhotse/blob/main/lhotse/features/kaldi/extractors.py#L24
    """

    sampling_rate: int = 16000
    window_length: int = 400
    hop_length: int = 160
    remove_dc_offset: bool = True
    pre_emphasis_coefficient: float = 0.97
    window_type: str = "povey"
    dither: float = 0.0
    snip_edges: bool = False
    n_mels: int = 80


@dataclasses.dataclass
class SubSamplerConfig:
    """Configuration class for the Convolution Subsampler."""

    type: str = "conv2d"  # or "conv1d"
    subsampling_factor: int = 4
    activation: str = "gelu"
    causal: bool = False
    dim_in: int = 80  # Mel Spectrogram channels
    dim_out: int = 512
    num_channels: int = 512


@dataclasses.dataclass
class FeedForwardConfig:
    """Configuration class for the ."""

    type: str = "glu"  # or "vanilla"
    activation: str = "swish"
    dim: int = 512
    intermediate_dim: int = 2048
    dropout: float = 0.0


@dataclasses.dataclass
class AttentionConfig:
    """Configuration class for the Attention module."""

    dim: int = 512
    num_heads: int = 4
    dropout: float = 0.0
    attention_window: Optional[Tuple[int, int]] = None
    causal: bool = False


@dataclasses.dataclass
class PositionalEncodingConfig:
    """Configuration class for the Positional Encoding module."""

    type: str = "rotary"  # or "absolute", "chunk-wise-absolute"
    dim: int = 512  # Hidden dimension if type is "absolute", or head dimension if type is "rotary"
    max_len: int = 512  # Maximum length of the input sequence, used for absolute positional encoding
    rope_theta: float = 10000.0  # Used for rotary positional encoding


@dataclasses.dataclass
class TransformerConfig:
    """Configuration class for the Transformer Encoder."""

    num_layers: int = 4
    dim: int = 512
    attention: AttentionConfig = AttentionConfig()
    feed_forward: FeedForwardConfig = FeedForwardConfig()
    positional_encoding: PositionalEncodingConfig = PositionalEncodingConfig()
    norm_type: str = "layernorm"  # or "rmsnorm"
    causal: bool = False
    residual_dropout: float = 0.0


@dataclasses.dataclass
class ConvModuleConfig:
    pass


@dataclasses.dataclass
class ConformerConfig(TransformerConfig):
    conv_module: ConvModuleConfig = ConvModuleConfig()


@dataclasses.dataclass
class EncoderConfig:
    """Configuration class for the Encoder.
    Waveform -> Mel Spectrogram -> Convolutional Subsampler -> Transformer Encoder
    """

    mel: MelSpectrogramConfig = MelSpectrogramConfig()
    subsampler: SubSamplerConfig = SubSamplerConfig()
    transformer: Union[TransformerConfig, ConformerConfig] = TransformerConfig()
    causal: bool = False
