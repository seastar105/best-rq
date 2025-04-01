from lhotse import Fbank, FbankConfig

from src.configuration import MelSpectrogramConfig


def to_lhotse_config(config: MelSpectrogramConfig):
    cfg = FbankConfig(
        sampling_rate=config.sampling_rate,
        frame_length=config.window_length / config.sampling_rate,
        frame_shift=config.hop_length / config.sampling_rate,
        remove_dc_offset=config.remove_dc_offset,
        preemph_coeff=config.pre_emphasis_coefficient,
        window_type=config.window_type,
        dither=config.dither,
        snip_edges=config.snip_edges,
        num_filters=config.n_mels,
        torchaudio_compatible_mel_scale=False,
        norm_filters=False,
    )
    return cfg


def get_mel_transform(config: MelSpectrogramConfig):
    lhotse_config = to_lhotse_config(config)
    mel_transform = Fbank(lhotse_config)
    return mel_transform
