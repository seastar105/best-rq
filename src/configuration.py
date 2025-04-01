import dataclasses
from typing import Union


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
class MLPConfig:
    pass


@dataclasses.dataclass
class AttnetionConfig:
    pass


@dataclasses.dataclass
class TransformerConfig:
    pass


@dataclasses.dataclass
class ConvModuleConfig:
    pass


@dataclasses.dataclass
class ConformerConfig:
    conv_module_config: ConvModuleConfig
    attention_config: AttnetionConfig
    mlp_config: MLPConfig
    causal: bool = False


@dataclasses.dataclass
class EncoderConfig:
    """Configuration class for the Encoder.
    Waveform -> Mel Spectrogram -> Convolutional Subsampler -> Transformer Encoder
    """

    mel_config: MelSpectrogramConfig
    subsampler_config: SubSamplerConfig
    transformer_config: Union[TransformerConfig, ConformerConfig]
    causal: bool = False
