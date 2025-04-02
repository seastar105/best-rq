from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoder.model import SpeechEncoder


class SpeechEncoderCTC(nn.Module):
    def __init__(self, encoder: SpeechEncoder, num_classes: int, blank_id: int = 0):
        super().__init__()
        self.encoder = encoder

        self.head = nn.Linear(encoder.config.transformer.dim, num_classes)
        self.pad_token_id = blank_id

        self.ctc = nn.CTCLoss(blank=blank_id)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        embs, lengths = self.encoder(x, lengths=lengths)
        logits = self.head(embs)
        loss = None

        if labels is not None:
            log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            input_lengths = lengths

            labels_mask = labels != self.pad_token_id
            targets = labels[labels_mask]
            target_lengths = labels_mask.sum(dim=1)

            loss = self.ctc(log_probs, targets, input_lengths, target_lengths)
        return logits, loss
