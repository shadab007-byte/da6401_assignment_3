"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)
    # Scaled dot-product scores: (..., seq_q, seq_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    attn_w = F.softmax(scores, dim=-1)
    # Replace NaN (from all-masked rows) with 0
    attn_w = torch.nan_to_num(attn_w, nan=0.0)

    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    # [batch, 1, 1, src_len]
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    batch_size, tgt_len = tgt.shape

    # Padding mask: [batch, 1, 1, tgt_len]
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)

    # Causal (look-ahead) mask: [1, 1, tgt_len, tgt_len]
    # upper triangle (above diagonal) = True (masked)
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)

    # Combine: mask out if PAD or future position
    return pad_mask | causal_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]

        Returns:
            output : shape [batch, seq_q, d_model]
        """
        batch_size = query.size(0)

        # Linear projections and split into heads
        # [batch, seq, d_model] -> [batch, seq, num_heads, d_k] -> [batch, num_heads, seq, d_k]
        def split_heads(x):
            x = x.view(batch_size, -1, self.num_heads, self.d_k)
            return x.transpose(1, 2)

        Q = split_heads(self.W_q(query))  # [batch, h, seq_q, d_k]
        K = split_heads(self.W_k(key))    # [batch, h, seq_k, d_k]
        V = split_heads(self.W_v(value))  # [batch, h, seq_k, d_k]

        # Scaled dot-product attention across all heads
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)
        # attn_output: [batch, h, seq_q, d_k]

        # Concatenate heads: [batch, seq_q, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        # Final linear
        return self.W_o(attn_output)


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute sinusoidal PE table: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # [1, max_len, d_model] — add batch dim
        pe = pe.unsqueeze(0)
        # Register as buffer (not a trainable parameter)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Post-LayerNorm ("Add & Norm") — as in original paper.
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        # Self-attention sub-layer
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN sub-layer
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn   = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn         = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1       = nn.LayerNorm(d_model)
        self.norm2       = nn.LayerNorm(d_model)
        self.norm3       = nn.LayerNorm(d_model)
        self.dropout     = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        # Masked self-attention
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        # Cross-attention
        cross_attn_out = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for German→English translation.

    All arguments have sensible defaults so that the autograder can call:
        model = Transformer()
    without any arguments.

    The __init__ also:
      - Loads spacy tokenizers (de_core_news_sm, en_core_web_sm)
      - Builds source/target vocabularies from Multi30k
      - Downloads trained weights from Google Drive (via gdown) and loads them
    """

    # ── Google Drive file ID for trained checkpoint ──────────────────
    GDRIVE_FILE_ID = "YOUR_GDRIVE_FILE_ID_HERE"   # <-- replace before submission

    def __init__(
        self,
        src_vocab_size: int = 8500,
        tgt_vocab_size: int = 6500,
        d_model:   int   = 256,
        N:         int   = 3,
        num_heads: int   = 8,
        d_ff:      int   = 512,
        dropout:   float = 0.1,
        checkpoint_path: str = "checkpoint.pt",
        pad_idx: int = 1,
        sos_idx: int = 2,
        eos_idx: int = 3,
    ) -> None:
        super().__init__()

        self.d_model        = d_model
        self.pad_idx        = pad_idx
        self.sos_idx        = sos_idx
        self.eos_idx        = eos_idx
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        # ── Build sub-modules ──────────────────────────────────────────
        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)

        self.encoder     = Encoder(enc_layer, N)
        self.decoder     = Decoder(dec_layer, N)

        self.src_embed   = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed   = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.src_pe      = PositionalEncoding(d_model, dropout)
        self.tgt_pe      = PositionalEncoding(d_model, dropout)
        self.proj        = nn.Linear(d_model, tgt_vocab_size)

        # ── Initialise parameters ─────────────────────────────────────
        self._init_weights()

        # ── Load tokenizers & vocabularies (needed for infer()) ───────
        self._load_tokenizers_and_vocab()

        # ── Download & load checkpoint ────────────────────────────────
        self._load_checkpoint(checkpoint_path)

    # ── Weight initialisation ─────────────────────────────────────────
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── Tokenizer / vocabulary setup ──────────────────────────────────
    def _load_tokenizers_and_vocab(self):
        """
        Load spacy tokenizers and build/restore vocabularies from Multi30k.
        This runs inside __init__ so that infer() works without extra setup.
        """
        import spacy
        from datasets import load_dataset

        # Load spacy models
        try:
            self.spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            os.system("python -m spacy download de_core_news_sm")
            self.spacy_de = spacy.load("de_core_news_sm")

        try:
            self.spacy_en = spacy.load("en_core_web_sm")
        except OSError:
            os.system("python -m spacy download en_core_web_sm")
            self.spacy_en = spacy.load("en_core_web_sm")

        # Tokenizer functions
        self._tokenize_de = lambda text: [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]
        self._tokenize_en = lambda text: [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

        # Try to load vocab from a saved file first; otherwise rebuild from dataset
        vocab_path = "vocab.pt"
        if os.path.exists(vocab_path):
            vocab_data = torch.load(vocab_path, map_location="cpu")
            self.src_stoi = vocab_data["src_stoi"]
            self.src_itos = vocab_data["src_itos"]
            self.tgt_stoi = vocab_data["tgt_stoi"]
            self.tgt_itos = vocab_data["tgt_itos"]
        else:
            self._build_vocab_from_dataset()
            torch.save({
                "src_stoi": self.src_stoi, "src_itos": self.src_itos,
                "tgt_stoi": self.tgt_stoi, "tgt_itos": self.tgt_itos,
            }, vocab_path)

        # Update embedding sizes if vocab was built dynamically
        actual_src = len(self.src_stoi)
        actual_tgt = len(self.tgt_stoi)
        if actual_src != self.src_vocab_size or actual_tgt != self.tgt_vocab_size:
            self.src_vocab_size = actual_src
            self.tgt_vocab_size = actual_tgt

    def _build_vocab_from_dataset(self):
        """Build vocab dictionaries from the Multi30k training set."""
        from datasets import load_dataset
        from collections import Counter

        dataset = load_dataset("bentrevett/multi30k", trust_remote_code=True)
        train_data = dataset["train"]

        special_tokens = ["<unk>", "<pad>", "<sos>", "<eos>"]

        src_counter = Counter()
        tgt_counter = Counter()
        for example in train_data:
            src_counter.update(self._tokenize_de(example["de"]))
            tgt_counter.update(self._tokenize_en(example["en"]))

        src_vocab = special_tokens + [w for w, _ in src_counter.most_common()]
        tgt_vocab = special_tokens + [w for w, _ in tgt_counter.most_common()]

        self.src_stoi = {w: i for i, w in enumerate(src_vocab)}
        self.src_itos = {i: w for i, w in enumerate(src_vocab)}
        self.tgt_stoi = {w: i for i, w in enumerate(tgt_vocab)}
        self.tgt_itos = {i: w for i, w in enumerate(tgt_vocab)}

    # ── Checkpoint download & load ────────────────────────────────────
    def _load_checkpoint(self, checkpoint_path: str):
        """
        Download checkpoint from Google Drive (if not already cached)
        and load weights into this model.
        """
        if not os.path.exists(checkpoint_path):
            try:
                gdown.download(
                    id=self.GDRIVE_FILE_ID,
                    output=checkpoint_path,
                    quiet=False,
                )
            except Exception as e:
                print(f"[Transformer] Could not download checkpoint: {e}")
                return

        if os.path.exists(checkpoint_path):
            try:
                ckpt = torch.load(checkpoint_path, map_location="cpu")
                state_dict = ckpt.get("model_state_dict", ckpt)
                self.load_state_dict(state_dict, strict=False)
                print(f"[Transformer] Loaded weights from {checkpoint_path}")
            except Exception as e:
                print(f"[Transformer] Could not load checkpoint: {e}")

    # ── AUTOGRADER HOOKS ──────────────────────────────────────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.
        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]
        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
        src_emb = self.src_pe(self.src_embed(src) * math.sqrt(self.d_model))
        return self.encoder(src_emb, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.
        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        tgt_emb = self.tgt_pe(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        dec_out = self.decoder(tgt_emb, memory, src_mask, tgt_mask)
        return self.proj(dec_out)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.
        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        """
        Translates a German sentence to English using greedy autoregressive decoding.

        Args:
            src_sentence: The raw German text.

        Returns:
            The fully translated English string, detokenized and clean.
        """
        self.eval()
        device = next(self.parameters()).device

        # Tokenize and numericalize source (German)
        tokens = self._tokenize_de(src_sentence)
        unk_idx = self.src_stoi.get("<unk>", 0)
        src_indices = [self.src_stoi.get(t, unk_idx) for t in tokens]
        src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)

        src_mask = make_src_mask(src_tensor, self.pad_idx)

        with torch.no_grad():
            memory = self.encode(src_tensor, src_mask)

            # Greedy decode
            ys = torch.tensor([[self.sos_idx]], dtype=torch.long, device=device)
            max_len = src_tensor.size(1) + 50

            for _ in range(max_len):
                tgt_mask = make_tgt_mask(ys, self.pad_idx)
                logits = self.decode(memory, src_mask, ys, tgt_mask)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_token], dim=1)
                if next_token.item() == self.eos_idx:
                    break

        # Convert indices back to words (skip <sos> and <eos>)
        token_ids = ys[0].tolist()
        words = []
        for idx in token_ids:
            if idx == self.sos_idx:
                continue
            if idx == self.eos_idx:
                break
            words.append(self.tgt_itos.get(idx, "<unk>"))

        return " ".join(words)
