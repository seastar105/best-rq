import dataclasses
import importlib
import json
import random
import string
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import torch
from audiotools import AudioSignal
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.configuration import MelSpectrogramConfig
from src.data.fbank import get_mel_transform
from src.data.text_tokenizer import MLSuperbTokenizer


def upper_no_punc(text):
    return text.upper().translate(str.maketrans("", "", string.punctuation))


def import_from_string(path: str):
    module_name, func_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


@dataclasses.dataclass
class ASRDatasetConfig:
    jsonl_path: str
    tokenizer_path: Optional[str] = None
    mel_config: Optional[MelSpectrogramConfig] = MelSpectrogramConfig()
    sampling_rate: int = 16000
    max_duration: float = 20.0
    min_duration: float = 0.5
    blank_id: int = 0
    text_proc_fn: Optional[str] = None
    pad_to_max: bool = True
    max_target_length: int = 512
    token_list: Optional[Union[Path, str, Iterable[str]]] = None
    non_linguistic_symbols: Optional[Union[Path, str, Iterable[str]]] = None


class ASRDataset(Dataset):
    def __init__(self, config: ASRDatasetConfig):
        self.config = config
        if config.tokenizer_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        elif config.token_list is not None:
            self.tokenizer = MLSuperbTokenizer(
                token_list=config.token_list,
                non_linguistic_symbols=config.non_linguistic_symbols,
            )
        else:
            raise ValueError("Either tokenizer_path or token_list must be provided")

        with open(config.jsonl_path) as f:
            # assume each line has "audio_path" and "text" keys
            self.data = [json.loads(line) for line in f]

        self.text_proc_fn = None if config.text_proc_fn is None else import_from_string(config.text_proc_fn)
        assert config.mel_config is not None, "Currently only support raw audio inputs"
        self.mel_transform = None if config.mel_config is None else get_mel_transform(config.mel_config)

    def __len__(self):
        return len(self.data)

    def collate(self, batch):
        audios = [item["audio"] for item in batch]
        lengths = torch.LongTensor([audio.shape[-1] for audio in audios])
        max_len = (
            lengths.max().item()
            if not self.config.pad_to_max
            else int(self.config.max_duration * self.config.sampling_rate)
        )

        padded_audios = []
        for audio in audios:
            pad_len = max_len - audio.shape[-1]
            padded_audio = torch.nn.functional.pad(audio, (0, pad_len), value=0)
            padded_audios.append(padded_audio)
        audios = torch.stack(padded_audios, dim=0)  # (B, T)

        if self.mel_transform is not None:
            audios = self.mel_transform.extract_batch(audios, sampling_rate=self.config.sampling_rate)  # list of (T, C)
            hop_length = self.config.mel_config.hop_length
            lengths = lengths + (hop_length // 2)
            lengths = lengths // hop_length
            lengths = lengths.long()

        labels = [torch.LongTensor(item["labels"]).view(-1) for item in batch]
        max_label_len = (
            max([label.shape[-1] for label in labels]) if not self.config.pad_to_max else self.config.max_target_length
        )
        padded_labels = []
        for label in labels:
            pad_len = max_label_len - label.shape[-1]
            padded_label = torch.nn.functional.pad(label, (0, pad_len), value=self.config.blank_id)
            padded_labels.append(padded_label)
        labels = torch.stack(padded_labels, dim=0)  # (B, L)
        return {
            "audio": audios,
            "labels": labels,
            "lengths": lengths,
        }

    def __getitem__(self, idx):
        while True:
            item = self.data[idx]
            audio_path = item["audio_path"]
            text = item["text"]
            if self.text_proc_fn is not None:
                text = self.text_proc_fn(text)

            signal = AudioSignal(audio_path)
            if self.config.min_duration > signal.duration or self.config.max_duration < signal.duration:
                idx = random.randint(0, len(self.data) - 1)
                continue

            if signal.sample_rate != self.config.sampling_rate:
                signal = signal.resample(self.config.sampling_rate)

            if signal.num_channels > 1:
                signal = signal.to_mono()

            audio = signal.audio_data.squeeze()  # (T, )
            labels = self.tokenizer(text)["input_ids"]  # list of int
            break
        return {
            "audio": audio,
            "labels": labels,
        }
