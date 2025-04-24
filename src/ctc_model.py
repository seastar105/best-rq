from contextlib import nullcontext
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoder.model import SpeechEncoder


# compared to torch.nn.CTCLoss, it is about 3x slower. Need to implement backward as stated in paper. hope it could be faster with XLA
def ctc_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: Optional[torch.Tensor] = None,
    target_lengths: Optional[torch.Tensor] = None,
    zero_infinity: bool = False,
    blank: int = 0,
):
    """Compute the CTC loss.

    Args:
        log_probs (torch.Tensor): (T, B, C) where B is the batch size, T is the input length,
            and C is the number of classes (including blank). log_probs = log_softmax(logits, dim=2).
        targets (torch.Tensor): Target sequences of shape (N, S) where S is the target length.
        input_lengths (torch.Tensor): Lengths of the input sequences of shape (N,).
        target_lengths (torch.Tensor): Lengths of the target sequences of shape (N,).
        zero_infinity (bool): If True, zero out the loss for invalid sequences.
        blank (int): blank label
    """
    T, B, C = log_probs.size()
    _, max_length = targets.shape
    S = 2 * max_length + 1

    if input_lengths is None:
        input_lengths = torch.full((B,), T, device=log_probs.device, dtype=torch.int64)

    if target_lengths is None:
        target_lengths = torch.full((B,), max_length, device=log_probs.device, dtype=torch.int64)

    INF = -1e5  # log(0) approx

    # extended labels with blank tokens, (B, S)
    labels = torch.full((B, S), blank).to(device=log_probs.device, dtype=torch.int64)
    for i in range(max_length):
        labels[:, 2 * i + 1] = targets[:, i]

    # (T, B, S)
    labels_prob = torch.gather(log_probs, 2, labels.unsqueeze(0).expand(T, -1, -1))

    # alpha
    log_alpha = torch.full((T, B, S), INF, dtype=log_probs.dtype, device=log_probs.device)
    # first time step, allowed only blank and first token
    log_alpha[0, :, :2] = labels_prob[0, :, :2]

    # alpha_curr(s) = (alpha_prev(s) + alpha_prev(s-1)) * prob(s)                   if s is blank or s == s-2
    #               = (alpha_prev(s) + alpha_prev(s-1) + alpha_prev(s-2)) * prob(s) else case
    # so, pre-compute mask to add alpha_prev(s-2)
    # transition_mask[i][j] is True, if labels[i][j] is not blank and labels[i][j] != labels[j-2] and j >= 2
    transition_mask = torch.zeros((B, S), device=log_probs.device).bool()
    transition_mask[:, 2:] = labels[:, 2:] != labels[:, :-2]  # s != s-2
    transition_mask[:, ::2] = False  # 0, 2, 4, ..., is blank

    for t in range(1, T):
        prev = log_alpha[t - 1]

        # alpha_prev(s)
        curr = prev.clone()

        # alpha_prev(s-1)
        shifted1 = torch.nn.functional.pad(prev[:, :-1], (1, 0), value=INF)
        curr = torch.logaddexp(curr, shifted1)

        # alpha_prev(s-2)
        shifted2 = torch.nn.functional.pad(prev[:, :-2], (2, 0), value=INF)
        curr = torch.where(transition_mask, torch.logaddexp(curr, shifted2), curr)

        # add log_prob
        log_alpha[t] = curr + labels_prob[t]

    # final prob is last token prob + last blank prob
    seq_probs = []
    for i in range(B):
        input_length = input_lengths[i]
        s1 = 2 * target_lengths[i]
        s2 = s1 - 1
        valid_s = []
        if s1 < S:
            valid_s.append(log_alpha[input_length - 1, i, s1])
        if s2 < S:
            valid_s.append(log_alpha[input_length - 1, i, s2])
        if len(valid_s) == 0:
            seq_prob = torch.tensor(INF, device=log_alpha.device, dtype=log_alpha.dtype)
        else:
            seq_prob = torch.logsumexp(torch.stack(valid_s), dim=0)
        if torch.isinf(seq_prob) and zero_infinity:
            seq_prob = torch.tensor(0.0, device=log_alpha.device, dtype=log_alpha.dtype)
        seq_probs.append(seq_prob)
    return (-torch.stack(seq_probs, dim=0) / target_lengths.float()).mean()


class SpeechEncoderCTC(nn.Module):
    def __init__(self, encoder: SpeechEncoder, num_classes: int, blank_id: int = 0, loss_impl: str = "torch"):
        super().__init__()
        self.encoder = encoder

        self.head = nn.Linear(encoder.config.transformer.dim, num_classes)
        self.pad_token_id = blank_id

        CTC_LOSS_IMPL = {
            "torch": torch.nn.CTCLoss(blank=blank_id, zero_infinity=True),
            "mine": partial(ctc_loss, blank=blank_id, zero_infinity=True),
        }
        self.ctc = CTC_LOSS_IMPL[loss_impl]
        self.num_classes = num_classes

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        embs, lengths = self.encoder(x, lengths=lengths)
        logits = self.head(embs)
        loss = None

        if labels is not None:
            # Disable autocast for CTC loss
            autocast_context = (
                nullcontext if not x.is_cuda else partial(torch.autocast, device_type="cuda", enabled=False)
            )
            with autocast_context():
                log_probs = F.log_softmax(logits.float(), dim=-1, dtype=torch.float32).transpose(0, 1)
                input_lengths = lengths

                labels_mask = labels != self.pad_token_id
                target_lengths = labels_mask.sum(dim=1)

                loss = self.ctc(log_probs, labels, input_lengths, target_lengths)
        return logits, loss
