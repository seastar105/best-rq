import dataclasses
from typing import Union


@dataclasses.dataclass
class MelSpectrogramConfig:
    pass


@dataclasses.dataclass
class SubSamplerConfig:
    pass


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
    pass


@dataclasses.dataclass
class EncoderConfig:
    """Configuration class for the Encoder.
        Waveform -> Mel Spectrogram -> Convolutional Subsampler -> Transformer Encoder

    Attributes:
        mel_config: Configuration for the mel spectrogram.
        subsampler_config: Configuration for the convolutional subsampler.
        transformer_config: Configuration for the transformer-based encoder.
        causal: Specifies whether the encoder is causal or not.
    """

    mel_config: MelSpectrogramConfig
    subsampler_config: SubSamplerConfig
    transformer_config: Union[TransformerConfig, ConformerConfig]
    causal: bool = False
