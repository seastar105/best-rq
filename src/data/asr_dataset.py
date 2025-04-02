import dataclasses
import importlib
import json
import random
import string

import torch
from audiotools import AudioSignal
from torch.utils.data import Dataset
from transformers import AutoTokenizer


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

    def __len__(self):
        return len(self.data)

    def collate(self, batch):
        audios = [item["audio"] for item in batch]
        lengths = [audio.shape[-1] for audio in audios]
        max_len = max(lengths)

        padded_audios = []
        for audio in audios:
            pad_len = max_len - audio.shape[-1]
            padded_audio = torch.nn.functional.pad(audio, (0, pad_len), value=0)
            padded_audios.append(padded_audio)
        audios = torch.cat(padded_audios, dim=0)  # (B, T)

        labels = {"input_ids": [item["labels"] for item in batch]}
        labels = self.tokenizer.pad(labels, padding=True, return_tensors="pt").input_ids
        return {
            "audio": audios,
            "labels": labels,
            "lengths": torch.LongTensor(lengths),
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

            audio = signal.audio_data.squeeze(0)  # (1, T)
            labels = self.tokenizer(text, return_tensors="pt").input_ids.squeeze()  # (L, )
            break
        return {
            "audio": audio,
            "labels": labels,
        }
