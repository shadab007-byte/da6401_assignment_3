"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need".

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        # Build smooth target distribution
        # Start with uniform smoothing: eps / (V - 1) everywhere
        smooth_val = self.smoothing / (self.vocab_size - 1)
        with torch.no_grad():
            true_dist = torch.full(
                (target.size(0), self.vocab_size),
                fill_value=smooth_val,
                device=logits.device,
            )
            # Correct class gets confidence mass
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            # PAD positions get 0 everywhere
            true_dist[target == self.pad_idx] = 0.0

        # Log-softmax of logits
        log_probs = torch.log_softmax(logits, dim=-1)

        # KL-divergence (ignoring constant entropy term of target):
        # loss = -sum(true_dist * log_probs)
        loss = -(true_dist * log_probs).sum(dim=-1)

        # Mask out PAD positions
        non_pad_mask = (target != self.pad_idx)
        loss = loss[non_pad_mask].mean()
        return loss


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).
    """
    import wandb
    from tqdm import tqdm

    model.train() if is_train else model.eval()

    total_loss  = 0.0
    total_tokens = 0
    pad_idx = model.pad_idx

    context = torch.enable_grad() if is_train else torch.no_grad()
    mode_str = "Train" if is_train else "Val"

    with context:
        pbar = tqdm(data_iter, desc=f"Epoch {epoch_num} [{mode_str}]", leave=False)
        for src, tgt in pbar:
            src = src.to(device)   # [batch, src_len]
            tgt = tgt.to(device)   # [batch, tgt_len]

            # Decoder input: all tokens except last (shift right)
            tgt_input  = tgt[:, :-1]
            # Decoder target: all tokens except first (<sos> stripped)
            tgt_output = tgt[:, 1:]

            src_mask = make_src_mask(src, pad_idx)
            tgt_mask = make_tgt_mask(tgt_input, pad_idx)

            logits = model(src, tgt_input, src_mask, tgt_mask)
            # logits: [batch, tgt_len-1, vocab_size]

            batch_size, seq_len, vocab_size = logits.shape
            logits_flat  = logits.reshape(-1, vocab_size)
            targets_flat = tgt_output.reshape(-1)

            loss = loss_fn(logits_flat, targets_flat)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            # Accumulate (unscaled by batch so we can average later)
            non_pad = (targets_flat != pad_idx).sum().item()
            total_loss   += loss.item() * non_pad
            total_tokens += non_pad

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(total_tokens, 1)

    # Log to W&B
    try:
        import wandb
        log_dict = {
            f"{'train' if is_train else 'val'}_loss": avg_loss,
            "epoch": epoch_num,
        }
        if is_train and scheduler is not None:
            log_dict["lr"] = scheduler.get_last_lr()[0]
        wandb.log(log_dict)
    except Exception:
        pass

    print(f"Epoch {epoch_num} [{mode_str}] avg_loss={avg_loss:.4f}")
    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.
    """
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=model.pad_idx)
            logits   = model.decode(memory, src_mask, ys, tgt_mask)
            # Take the last token's prediction
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            if next_token.item() == end_symbol:
                break

    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
        tgt_vocab       : Vocabulary with lookup_token(idx) or itos[idx].
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).
    """
    import sacrebleu

    model.eval()
    pad_idx = model.pad_idx
    sos_idx = model.sos_idx
    eos_idx = model.eos_idx

    # Vocab accessor helper
    def idx_to_tok(idx):
        if hasattr(tgt_vocab, 'lookup_token'):
            return tgt_vocab.lookup_token(idx)
        elif hasattr(tgt_vocab, 'itos'):
            return tgt_vocab.itos.get(idx, "<unk>")
        else:
            return str(idx)

    hypotheses = []   # list of detokenized prediction strings
    references  = []  # list of detokenized reference strings

    with torch.no_grad():
        for src, tgt in test_dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            src_mask = make_src_mask(src, pad_idx)
            ys = greedy_decode(
                model, src, src_mask, max_len,
                start_symbol=sos_idx,
                end_symbol=eos_idx,
                device=device,
            )

            # Convert predicted indices to tokens (strip <sos>/<eos>/<pad>)
            pred_ids = ys[0].tolist()
            pred_tokens = []
            for idx in pred_ids:
                if idx == sos_idx:
                    continue
                if idx == eos_idx:
                    break
                if idx == pad_idx:
                    continue
                pred_tokens.append(idx_to_tok(idx))

            # Convert reference indices to tokens
            ref_ids = tgt[0].tolist()
            ref_tokens = []
            for idx in ref_ids:
                if idx == sos_idx:
                    continue
                if idx == eos_idx:
                    break
                if idx == pad_idx:
                    continue
                ref_tokens.append(idx_to_tok(idx))

            hypotheses.append(" ".join(pred_tokens))
            references.append(" ".join(ref_tokens))

    # sacrebleu corpus_bleu returns score in 0-100 range
    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    print(f"BLEU score: {bleu:.2f}")
    return bleu


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').
    """
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "model_config": {
            "src_vocab_size": model.src_vocab_size,
            "tgt_vocab_size": model.tgt_vocab_size,
            "d_model":        model.d_model,
            "N":              len(model.encoder.layers),
            "num_heads":      model.encoder.layers[0].self_attn.num_heads,
            "d_ff":           model.encoder.layers[0].ffn.linear1.out_features,
            "dropout":        model.encoder.layers[0].dropout.p,
        },
    }, path)
    print(f"[Checkpoint] Saved epoch {epoch} → {path}")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file.
        model     : Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    epoch = ckpt.get("epoch", 0)
    print(f"[Checkpoint] Loaded epoch {epoch} from {path}")
    return epoch


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.
    """
    import wandb
    from dataset import build_dataloaders
    from lr_scheduler import NoamScheduler

    # ── Hyperparameters ───────────────────────────────────────────────
    config = {
        "d_model":       256,
        "N":             3,
        "num_heads":     8,
        "d_ff":          512,
        "dropout":       0.1,
        "batch_size":    128,
        "num_epochs":    20,
        "warmup_steps":  4000,
        "label_smoothing": 0.1,
        "min_freq":      1,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── W&B ───────────────────────────────────────────────────────────
    wandb.init(
        project="da6401-assignment-3",
        entity="iitm_assigment",
        config=config,
    )

    # ── Dataset / vocab / dataloaders ─────────────────────────────────
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = build_dataloaders(
        batch_size=config["batch_size"],
        min_freq=config["min_freq"],
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        checkpoint_path=None,   # no pre-trained weights at start
        pad_idx=src_vocab.pad_idx,
        sos_idx=src_vocab.sos_idx,
        eos_idx=src_vocab.eos_idx,
    ).to(device)
    # Attach vocab to model for infer()
    model.src_vocab = src_vocab
    model.tgt_vocab = tgt_vocab

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )

    # ── Scheduler ─────────────────────────────────────────────────────
    scheduler = NoamScheduler(optimizer, d_model=config["d_model"], warmup_steps=config["warmup_steps"])

    # ── Loss ──────────────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=tgt_vocab.pad_idx,
        smoothing=config["label_smoothing"],
    )

    # ── Training loop ─────────────────────────────────────────────────
    best_val_loss = float("inf")
    for epoch in range(config["num_epochs"]):
        run_epoch(
            train_loader, model, loss_fn,
            optimizer, scheduler, epoch,
            is_train=True, device=device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn,
            None, None, epoch,
            is_train=False, device=device,
        )
        save_checkpoint(model, optimizer, scheduler, epoch, path="checkpoint.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, path="best_checkpoint.pt")

    # ── Final BLEU on test set ─────────────────────────────────────────
    load_checkpoint("best_checkpoint.pt", model)
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    wandb.log({"test_bleu": bleu})
    print(f"Final test BLEU: {bleu:.2f}")
    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()
