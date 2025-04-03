import dataclasses
import importlib
import json
import random
import string
from typing import Optional

import torch
from audiotools import AudioSignal
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.configuration import MelSpectrogramConfig
from src.data.fbank import get_mel_transform


def upper_no_punc(text):
    return text.upper().translate(str.maketrans("", "", string.punctuation))


def import_from_string(path: str):
    module_name, func_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


@dataclasses.dataclass
class ASRDatasetConfig:
    jsonl_path: str
    tokenizer_path: str
    mel_config: Optional[MelSpectrogramConfig] = MelSpectrogramConfig()
    sampling_rate: int = 16000
    max_duration: float = 20.0
    min_duration: float = 0.5
    blank_id: int = 0
    text_proc_fn: str = "src.data.asr_dataset.upper_no_punc"


class ASRDataset(Dataset):
    def __init__(self, config: ASRDatasetConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)

        with open(config.jsonl_path) as f:
            # assume each line has "audio_path" and "text" keys
            self.data = [json.loads(line) for line in f]

        self.text_proc_fn = import_from_string(config.text_proc_fn)
        self.mel_transform = None
        if config.mel_config is not None:
            self.mel_transform = get_mel_transform(config.mel_config)

    def __len__(self):
        return len(self.data)

    def collate(self, batch):
        audios = [item["audio"] for item in batch]
        lengths = torch.LongTensor([audio.shape[-1] for audio in audios])
        max_len = lengths.max().item()

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

        labels = {"input_ids": [item["labels"] for item in batch]}
        try:
            labels = self.tokenizer.pad(labels, padding=True, return_tensors="pt").input_ids
        except Exception as e:
            print(f"Error during tokenization: {e}")
            print(f"Labels: {labels}")
            raise e
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
            labels = self.tokenizer(text).input_ids  # list of int
            break
        return {
            "audio": audio,
            "labels": labels,
        }
