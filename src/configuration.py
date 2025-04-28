import dataclasses
from typing import Any, Optional, Tuple, Union


@dataclasses.dataclass
class RecursiveConfig:
    """
    Base class for configuration dataclasses.
    Overrides __setattr__ to recursively set attributes with the same name
    in nested dataclass instances that also inherit from RecursiveConfig.
    """

    def __setattr__(self, name: str, value: Any):
        # First, set the attribute on the current object as normal.
        # Use object.__setattr__ to avoid infinite recursion within this method.
        object.__setattr__(self, name, value)

        # Now, iterate through the fields of this dataclass instance
        for field in dataclasses.fields(self):
            # Get the value of the field (which might be a nested config)
            field_value = getattr(self, field.name)

            # Check if the field's value is an instance of our base class
            # (or any class inheriting from it) and if it has the attribute
            # 'name' that we are currently setting.
            if isinstance(field_value, RecursiveConfig) and hasattr(field_value, name):
                # Check if the attribute in the nested object is actually a field
                # This prevents attempting to set methods or other non-field attributes
                is_nested_field = False
                try:
                    # Check if 'name' is a defined field in the nested object's type
                    if name in {f.name for f in dataclasses.fields(field_value)}:
                        is_nested_field = True
                except TypeError:
                    # If field_value is not a dataclass type, fields() will raise TypeError
                    pass  # Keep is_nested_field as False

                if is_nested_field:
                    # Recursively call setattr on the nested object.
                    # This will trigger the same __setattr__ logic down the chain.
                    # print(f"Propagating '{name}' = {value} to {type(field_value).__name__}") # Optional: for debugging
                    setattr(field_value, name, value)


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
class SubSamplerConfig(RecursiveConfig):
    """Configuration class for the Convolution Subsampler."""

    type: str = "conv2d"  # or "conv1d"
    subsampling_factor: int = 4
    activation: str = "gelu"
    causal: bool = False
    dim_in: int = 80  # Mel Spectrogram channels
    dim: int = 512  # Output dimension
    num_channels: int = 512


@dataclasses.dataclass
class FeedForwardConfig(RecursiveConfig):
    """Configuration class for the ."""

    type: str = "glu"  # or "vanilla"
    activation: str = "swish"
    dim: int = 512
    intermediate_dim: int = 2048
    dropout: float = 0.0


@dataclasses.dataclass
class AttentionConfig(RecursiveConfig):
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
class TransformerConfig(RecursiveConfig):
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
class ConvModuleConfig(RecursiveConfig):
    """Configuration class for the Convolution module."""

    kernel_size: int = 9
    dim: int = 512
    activation: str = "swish"
    dropout: float = 0.0
    norm_type: str = "layernorm"
    causal: bool = False


@dataclasses.dataclass
class ConformerConfig(TransformerConfig):
    conv_module: ConvModuleConfig = ConvModuleConfig()


@dataclasses.dataclass
class EncoderConfig(RecursiveConfig):
    """Configuration class for the Encoder.
    Waveform -> Mel Spectrogram -> Convolutional Subsampler -> Transformer Encoder
    """

    mel: MelSpectrogramConfig = MelSpectrogramConfig()
    subsampler: SubSamplerConfig = SubSamplerConfig()
    transformer: Union[TransformerConfig, ConformerConfig] = TransformerConfig()
    causal: bool = False
