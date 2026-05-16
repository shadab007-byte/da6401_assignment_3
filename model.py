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
import subprocess
import sys
import importlib
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#  SCALED DOT-PRODUCT ATTENTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    attn_w = F.softmax(scores, dim=-1)
    attn_w = torch.nan_to_num(attn_w, nan=0.0)
    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    batch_size, tgt_len = tgt.shape
    pad_mask    = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)
    return pad_mask | causal_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        def split(x):
            return x.view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        Q = split(self.W_q(query))
        K = split(self.W_k(key))
        V = split(self.W_v(value))
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out)


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):

    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):

    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#  FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):

    GDRIVE_CHECKPOINT_ID = "1hiBEx9oAnzS0TEzPc-jYBbbU3QceStS-"  
    GDRIVE_VOCAB_ID      = "1BCwr2tbr8KHky2FPDpGjNslKssq3Jdzj"       

    def __init__(
        self,
        src_vocab_size: int   = None,
        tgt_vocab_size: int   = None,
        d_model:        int   = 512,
        N:              int   = 6,
        num_heads:      int   = 8,
        d_ff:           int   = 2048,
        dropout:        float = 0.1,
        checkpoint_path: str  = None,
    ) -> None:
        super().__init__()

        
        self.pad_idx = 1
        self.sos_idx = 2
        self.eos_idx = 3

        
        self._load_spacy_tokenizers()

        
        self._download_vocab()
        src_vocab_size = len(self.src_stoi)
        tgt_vocab_size = len(self.tgt_stoi)
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        
        ckpt_path = checkpoint_path if checkpoint_path else "checkpoint.pt"
        if not os.path.exists(ckpt_path):
            try:
                print("[Transformer] Downloading checkpoint...")
                gdown.download(id=self.GDRIVE_CHECKPOINT_ID, output=ckpt_path, quiet=False)
            except Exception as e:
                print(f"[Transformer] Checkpoint download failed: {e}")

        if os.path.exists(ckpt_path):
            try:
                _ckpt = torch.load(ckpt_path, map_location="cpu")
                _cfg  = _ckpt.get("model_config", {})
                d_model   = _cfg.get("d_model",   d_model)
                N         = _cfg.get("N",         N)
                num_heads = _cfg.get("num_heads", num_heads)
                d_ff      = _cfg.get("d_ff",      d_ff)
                print(f"[Transformer] Config: d_model={d_model}, N={N}, "
                      f"num_heads={num_heads}, d_ff={d_ff}")
            except Exception as e:
                print(f"[Transformer] Could not read model_config: {e}")

        self.d_model = d_model

      
        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder   = Encoder(enc_layer, N)
        self.decoder   = Decoder(dec_layer, N)
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=self.pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=self.pad_idx)
        self.src_pe    = PositionalEncoding(d_model, dropout)
        self.tgt_pe    = PositionalEncoding(d_model, dropout)
        self.proj      = nn.Linear(d_model, tgt_vocab_size)

        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        
        if os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                self.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
                print("[Transformer] Weights loaded successfully.")
            except Exception as e:
                print(f"[Transformer] Could not load weights: {e}")

    @staticmethod
    def _install_spacy_model(name):
        URLS = {
            "de_core_news_sm": (
                "https://github.com/explosion/spacy-models/releases/download/"
                "de_core_news_sm-3.7.0/de_core_news_sm-3.7.0-py3-none-any.whl"
            ),
            "en_core_web_sm": (
                "https://github.com/explosion/spacy-models/releases/download/"
                "en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
            ),
        }
        r = subprocess.run(
            [sys.executable, "-m", "spacy", "download", name],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            url = URLS.get(name, "")
            if url:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", url, "--quiet"],
                    capture_output=True
                )

    def _load_spacy_tokenizers(self):
        import spacy
        for name, attr in [("de_core_news_sm", "spacy_de"), ("en_core_web_sm", "spacy_en")]:
            try:
                setattr(self, attr, spacy.load(name))
            except OSError:
                print(f"[Transformer] Installing {name}...")
                self._install_spacy_model(name)
                importlib.reload(spacy)
                setattr(self, attr, spacy.load(name))
        self._tok_de = lambda t: [x.text.lower() for x in self.spacy_de.tokenizer(t)]
        self._tok_en = lambda t: [x.text.lower() for x in self.spacy_en.tokenizer(t)]


    def _download_vocab(self):
        vocab_path = "vocab.pt"
        if not os.path.exists(vocab_path):
            try:
                print("[Transformer] Downloading vocab.pt...")
                gdown.download(id=self.GDRIVE_VOCAB_ID, output=vocab_path, quiet=False)
            except Exception as e:
                print(f"[Transformer] Vocab download failed: {e}")

        if os.path.exists(vocab_path):
            v = torch.load(vocab_path, map_location="cpu")
            self.src_stoi = v["src_stoi"]
            self.src_itos = v["src_itos"]
            self.tgt_stoi = v["tgt_stoi"]
            self.tgt_itos = v["tgt_itos"]
            print(f"[Transformer] Vocab: src={len(self.src_stoi)}, tgt={len(self.tgt_stoi)}")
        else:
            self._build_vocab()

    def _build_vocab(self):
        from datasets import load_dataset
        from collections import Counter
        print("[Transformer] Building vocab from Multi30k...")
        ds = load_dataset("bentrevett/multi30k", trust_remote_code=True)["train"]
        sc, tc = Counter(), Counter()
        for ex in ds:
            sc.update(self._tok_de(ex["de"]))
            tc.update(self._tok_en(ex["en"]))
        spec = ["<unk>", "<pad>", "<sos>", "<eos>"]
        sv = spec + [w for w, _ in sc.most_common()]
        tv = spec + [w for w, _ in tc.most_common()]
        self.src_stoi = {w: i for i, w in enumerate(sv)}
        self.src_itos = {i: w for i, w in enumerate(sv)}
        self.tgt_stoi = {w: i for i, w in enumerate(tv)}
        self.tgt_itos = {i: w for i, w in enumerate(tv)}
        torch.save({
            "src_stoi": self.src_stoi, "src_itos": self.src_itos,
            "tgt_stoi": self.tgt_stoi, "tgt_itos": self.tgt_itos,
        }, "vocab.pt")


    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.src_pe(self.src_embed(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(self, memory: torch.Tensor, src_mask: torch.Tensor,
               tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        x = self.tgt_pe(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        return self.proj(self.decoder(x, memory, src_mask, tgt_mask))

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    # infer() 

    def infer(self, src_sentence: str) -> str:
        self.eval()
        device = next(self.parameters()).device

        tokens   = self._tok_de(src_sentence)
        unk      = self.src_stoi.get("<unk>", 0)
        src_ids  = [self.src_stoi.get(t, unk) for t in tokens]
        src_t    = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = make_src_mask(src_t, self.pad_idx)

        with torch.no_grad():
            memory  = self.encode(src_t, src_mask)
            max_len = src_t.size(1) + 50
            ys = torch.tensor([[self.sos_idx]], dtype=torch.long, device=device)
            for _ in range(max_len):
                tgt_mask = make_tgt_mask(ys, self.pad_idx)
                logits   = self.decode(memory, src_mask, ys, tgt_mask)
                nxt      = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys       = torch.cat([ys, nxt], dim=1)
                if nxt.item() == self.eos_idx:
                    break

        words = []
        for idx in ys[0].tolist():
            if idx == self.sos_idx:
                continue
            if idx == self.eos_idx:
                break
            words.append(self.tgt_itos.get(idx, "<unk>"))
        return " ".join(words)
