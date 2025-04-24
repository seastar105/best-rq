from contextlib import nullcontext
from functools import partial

import jiwer
import rootutils
import torch
from torch.amp.grad_scaler import GradScaler
from tqdm.auto import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.configuration import EncoderConfig
from src.ctc_model import SpeechEncoderCTC
from src.data.asr_dataset import ASRDataset, ASRDatasetConfig
from src.encoder.model import SpeechEncoder

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(998244353)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_config = EncoderConfig()
    encoder = SpeechEncoder(encoder_config)

    dataset_config = ASRDatasetConfig(
        jsonl_path="/home/seastar105/Work/best-rq/mlsuperb2_train.jsonl",
        tokenizer_path=None,
        mel_config=encoder_config.mel,
        sampling_rate=encoder_config.mel.sampling_rate,
        pad_to_max=True,
        token_list="/home/seastar105/Work/best-rq/assets/ml_superb_tokens.txt",
        non_linguistic_symbols="/home/seastar105/Work/best-rq/assets/ml_superb_nlsyms.txt",
    )
    dataset = ASRDataset(dataset_config)
    valid_dataset_config = ASRDatasetConfig(
        jsonl_path="/home/seastar105/Work/best-rq/mlsuperb2_dev.jsonl",
        tokenizer_path=None,
        mel_config=encoder_config.mel,
        sampling_rate=encoder_config.mel.sampling_rate,
        pad_to_max=True,
        token_list="/home/seastar105/Work/best-rq/assets/ml_superb_tokens.txt",
        non_linguistic_symbols="/home/seastar105/Work/best-rq/assets/ml_superb_nlsyms.txt",
    )
    valid_dataset = ASRDataset(valid_dataset_config)
    tokenizer = dataset.tokenizer

    loss_impl = "torch"
    model = SpeechEncoderCTC(encoder, num_classes=len(dataset.tokenizer), loss_impl=loss_impl)

    batch_size = 32
    valid_batch_size = 32
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=dataset.collate, shuffle=True, num_workers=8
    )
    learning_rate = 1e-3
    epochs = 20
    total_steps = len(dataloader) * epochs
    log_interval = 100
    max_norm = 1.0

    # Decay only params with 2D shapes
    param_groups = [
        {"params": [p for p in model.parameters() if p.ndim > 1 and p.requires_grad], "weight_decay": 0.01},
        {"params": [p for p in model.parameters() if p.ndim == 1 and p.requires_grad], "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, total_steps=total_steps, pct_start=0.04
    )

    model.to(device)
    model.train()

    amp = True
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    running_losses = []
    running_norms = []
    autocast_context = (
        nullcontext
        if not torch.cuda.is_available()
        else partial(torch.autocast, device_type="cuda", enabled=amp, dtype=amp_dtype)
    )
    scaler = GradScaler("cuda", enabled=amp_dtype == torch.float16)

    step = 0
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params/1e6:.2f}M")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Total steps: {total_steps}")
    print(f"Using {amp_dtype} for AMP")

    is_first_validation = True
    refs = []

    for epoch in range(epochs):
        model.train()
        for batch_idx, batch in tqdm(
            enumerate(dataloader), total=len(dataloader), leave=False, desc=f"Epoch {epoch + 1}/{epochs}..."
        ):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            with autocast_context():
                logits, loss = model(batch["audio"], lengths=batch["lengths"], labels=batch["labels"])

            if amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
            elif amp and amp_dtype == torch.bfloat16:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()

            lr_scheduler.step()
            optimizer.zero_grad()

            running_norms.append(grad_norm)
            running_losses.append(loss.detach())
            step += 1

            if step % log_interval == 0:
                loss = torch.mean(torch.stack(running_losses))
                grad_norm = torch.mean(torch.stack(running_norms))
                learning_rate = optimizer.param_groups[0]["lr"]

                print(
                    f"Step {step} | Loss: {loss.item()} | Learning Rate: {learning_rate} | Grad Norm: {grad_norm.item()}"
                )

                running_losses = []
                running_norms = []

        model.eval()
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=valid_batch_size,
            collate_fn=valid_dataset.collate,
            shuffle=False,
            drop_last=True,
        )
        val_losses = []
        preds = []
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(valid_dataloader), total=len(valid_dataloader), leave=True, desc="Validating..."
            ):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                logits, loss = model(batch["audio"], lengths=batch["lengths"], labels=batch["labels"])
                lengths = (
                    batch["lengths"] + encoder_config.subsampler.subsampling_factor - 1
                ) // encoder_config.subsampler.subsampling_factor
                val_losses.append(loss)

                # remove padded regions
                pred = logits.argmax(dim=-1)
                for i in range(len(pred)):
                    p = pred[i, : lengths[i]]
                    preds.append(p)
                if is_first_validation:
                    for r in batch["labels"]:
                        r = r[r != dataset_config.blank_id]
                        refs.append(r)
        val_loss = torch.mean(torch.tensor(val_losses))
        random_idx = torch.randint(0, len(refs), (5,)).tolist()

        # decode
        def decode_pred(p):
            collapsed = []
            prev = None
            for i in p:
                if i != prev:
                    collapsed.append(i.item())
                prev = i
            collapsed = [i for i in collapsed if i != dataset_config.blank_id]
            return collapsed

        preds = [tokenizer.decode(decode_pred(p)) for p in tqdm(preds, desc="Decoding preds...")]
        if is_first_validation:
            refs = [tokenizer.decode(r) for r in tqdm(refs, desc="Decoding refs...")]
        is_first_validation = False
        for i in random_idx:
            print(f"Pred: '{preds[i]}' | Ref: '{refs[i]}'")
        wer = round(jiwer.wer(refs, preds) * 100, 2)
        cer = round(jiwer.cer(refs, preds) * 100, 2)  # type: ignore

        print(f"Epoch {epoch + 1} Validation Loss: {val_loss.item()} | CER: {cer:.2f} | WER: {wer:.2f}")


if __name__ == "__main__":
    main()
