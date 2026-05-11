"""
dataset.py — Multi30k Dataset Loading & Preprocessing
DA6401 Assignment 3: "Attention Is All You Need"

Loads bentrevett/multi30k from HuggingFace, tokenizes with spaCy,
builds vocabularies, and exposes a collate_fn-ready torch Dataset.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from typing import List, Tuple, Dict


# ══════════════════════════════════════════════════════════════════════
#  VOCABULARY
# ══════════════════════════════════════════════════════════════════════

class Vocabulary:
    """Simple word-level vocabulary with special tokens."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"

    def __init__(self, counter: Counter, min_freq: int = 1) -> None:
        self.special_tokens = [
            self.UNK_TOKEN,   # 0
            self.PAD_TOKEN,   # 1
            self.SOS_TOKEN,   # 2
            self.EOS_TOKEN,   # 3
        ]
        tokens = self.special_tokens + [
            word for word, freq in counter.most_common()
            if freq >= min_freq
        ]
        self.stoi: Dict[str, int] = {tok: i for i, tok in enumerate(tokens)}
        self.itos: Dict[int, str] = {i: tok for i, tok in enumerate(tokens)}

    def __len__(self) -> int:
        return len(self.stoi)

    @property
    def pad_idx(self) -> int:
        return self.stoi[self.PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.stoi[self.UNK_TOKEN]

    @property
    def sos_idx(self) -> int:
        return self.stoi[self.SOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.stoi[self.EOS_TOKEN]

    def numericalize(self, tokens: List[str]) -> List[int]:
        unk = self.unk_idx
        return [self.stoi.get(tok, unk) for tok in tokens]

    def lookup_token(self, idx: int) -> str:
        return self.itos.get(idx, self.UNK_TOKEN)


# ══════════════════════════════════════════════════════════════════════
#  MAIN DATASET CLASS
# ══════════════════════════════════════════════════════════════════════

class Multi30kDataset(Dataset):
    """
    Multi30k De→En dataset.

    Args:
        split       : 'train', 'validation', or 'test'
        src_vocab   : Pre-built Vocabulary for German (None = build from train)
        tgt_vocab   : Pre-built Vocabulary for English (None = build from train)
        min_freq    : Min token frequency for vocab inclusion (used when building)
    """

    def __init__(
        self,
        split: str = 'train',
        src_vocab: Vocabulary = None,
        tgt_vocab: Vocabulary = None,
        min_freq: int = 1,
    ) -> None:
        self.split = split

        # ── Load dataset from HuggingFace ─────────────────────────────
        from datasets import load_dataset
        raw = load_dataset("bentrevett/multi30k", trust_remote_code=True)
        self.raw_data = raw[split]

        # ── Load spaCy tokenizers ─────────────────────────────────────
        import spacy
        try:
            self._spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            os.system("python -m spacy download de_core_news_sm")
            self._spacy_de = spacy.load("de_core_news_sm")
        try:
            self._spacy_en = spacy.load("en_core_web_sm")
        except OSError:
            os.system("python -m spacy download en_core_web_sm")
            self._spacy_en = spacy.load("en_core_web_sm")

        # ── Build / assign vocabularies ───────────────────────────────
        if src_vocab is None or tgt_vocab is None:
            self.src_vocab, self.tgt_vocab = self.build_vocab(min_freq=min_freq)
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

        # ── Tokenize & numericalize all sentences ─────────────────────
        self.src_data, self.tgt_data = self.process_data()

    # ── Tokenizer helpers ─────────────────────────────────────────────

    def tokenize_de(self, text: str) -> List[str]:
        return [tok.text.lower() for tok in self._spacy_de.tokenizer(text)]

    def tokenize_en(self, text: str) -> List[str]:
        return [tok.text.lower() for tok in self._spacy_en.tokenizer(text)]

    # ── Vocabulary construction ───────────────────────────────────────

    def build_vocab(self, min_freq: int = 1) -> Tuple[Vocabulary, Vocabulary]:
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        src_counter = Counter()
        tgt_counter = Counter()
        for example in self.raw_data:
            src_counter.update(self.tokenize_de(example["de"]))
            tgt_counter.update(self.tokenize_en(example["en"]))

        src_vocab = Vocabulary(src_counter, min_freq=min_freq)
        tgt_vocab = Vocabulary(tgt_counter, min_freq=min_freq)
        return src_vocab, tgt_vocab

    # ── Numericalization ──────────────────────────────────────────────

    def process_data(self) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Convert German and English sentences into integer token lists using
        spacy and the defined vocabulary.
        """
        src_data = []
        tgt_data = []
        sos = self.src_vocab.sos_idx
        eos = self.src_vocab.eos_idx
        tgt_sos = self.tgt_vocab.sos_idx
        tgt_eos = self.tgt_vocab.eos_idx

        for example in self.raw_data:
            src_tokens = self.tokenize_de(example["de"])
            tgt_tokens = self.tokenize_en(example["en"])

            src_ids = [sos] + self.src_vocab.numericalize(src_tokens) + [eos]
            tgt_ids = [tgt_sos] + self.tgt_vocab.numericalize(tgt_tokens) + [tgt_eos]

            src_data.append(src_ids)
            tgt_data.append(tgt_ids)

        return src_data, tgt_data

    # ── PyTorch Dataset interface ──────────────────────────────────────

    def __len__(self) -> int:
        return len(self.src_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.src_data[idx], dtype=torch.long),
            torch.tensor(self.tgt_data[idx], dtype=torch.long),
        )


# ══════════════════════════════════════════════════════════════════════
#  COLLATE FUNCTION
# ══════════════════════════════════════════════════════════════════════

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_idx: int = 1):
    """
    Pad sequences in a batch to the same length.

    Args:
        batch   : List of (src_tensor, tgt_tensor) tuples.
        pad_idx : Padding token index (default 1 = <pad>).

    Returns:
        src_batch : [batch_size, max_src_len]
        tgt_batch : [batch_size, max_tgt_len]
    """
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.nn.utils.rnn.pad_sequence(
        src_batch, batch_first=True, padding_value=pad_idx
    )
    tgt_batch = torch.nn.utils.rnn.pad_sequence(
        tgt_batch, batch_first=True, padding_value=pad_idx
    )
    return src_batch, tgt_batch


# ══════════════════════════════════════════════════════════════════════
#  DATALOADER BUILDER
# ══════════════════════════════════════════════════════════════════════

def build_dataloaders(
    batch_size: int = 128,
    min_freq: int = 1,
):
    """
    Build train, validation, and test DataLoaders for Multi30k.

    Returns:
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    """
    print("Loading train split and building vocabularies...")
    train_dataset = Multi30kDataset(split='train', min_freq=min_freq)
    src_vocab = train_dataset.src_vocab
    tgt_vocab = train_dataset.tgt_vocab

    print("Loading validation split...")
    val_dataset  = Multi30kDataset(split='validation', src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    print("Loading test split...")
    test_dataset = Multi30kDataset(split='test',       src_vocab=src_vocab, tgt_vocab=tgt_vocab)

    pad_idx = src_vocab.pad_idx
    from functools import partial
    _collate = partial(collate_fn, pad_idx=pad_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=_collate)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=_collate)
    test_loader  = DataLoader(test_dataset,  batch_size=1,          shuffle=False, collate_fn=_collate)

    print(f"  src vocab size : {len(src_vocab)}")
    print(f"  tgt vocab size : {len(tgt_vocab)}")
    print(f"  train batches  : {len(train_loader)}")
    print(f"  val   batches  : {len(val_loader)}")
    print(f"  test  batches  : {len(test_loader)}")

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
