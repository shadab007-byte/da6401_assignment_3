# DA6401 Assignment 3 — Transformer for Machine Translation

Implementation of "Attention Is All You Need" (Vaswani et al., 2017) from scratch using PyTorch for German→English Neural Machine Translation on the Multi30k dataset.

## 📈 W&B Report

🔗 [View Full Report](https://wandb.ai/iitm_assigment/da6401-assignment-3/reports/da6401-assignment_3--VmlldzoxNjkxMjA4Mg)

## 🔗 Links

- **GitHub:** [shadab007-byte/da6401_assignment_3](https://github.com/shadab007-byte/da6401_assignment_3)
- **W&B Project Report:** : https://api.wandb.ai/links/iitm_assigment/g509q9cz](https://wandb.ai/iitm_assigment/da6401-assignment-3/reports/da6401-assignment_3--VmlldzoxNjkxMjA4Mg
- **Base Paper:** [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

## 📋 Overview

This assignment builds a full Transformer encoder-decoder architecture including:
- Scaled Dot-Product Attention and Multi-Head Attention (no `torch.nn.MultiheadAttention`)
- Sinusoidal Positional Encoding registered as a buffer
- Noam Learning Rate Scheduler with warmup
- Label Smoothing Loss
- Greedy decoding inference via `model.infer(german_sentence)`

## 📁 Project Structure

```
├── model.py          # Full Transformer architecture + infer()
├── train.py          # Training loop, greedy decode, BLEU evaluation, checkpointing
├── dataset.py        # Multi30k dataset loading, vocabulary, DataLoader
├── lr_scheduler.py   # Noam LR scheduler
└── README.md
```

## 🗂️ Dataset

**Multi30k** (bentrevett/multi30k from HuggingFace)
- Train: 29,000 pairs | Val: 1,014 pairs | Test: 1,000 pairs
- Language pair: German → English
- Tokenization: spaCy (`de_core_news_sm`, `en_core_web_sm`)

## 🏗️ Model Architecture

| Hyperparameter | Value |
|---|---|
| d_model | 512 |
| Encoder/Decoder layers (N) | 4 |
| Attention heads | 8 |
| d_ff | 1024 |
| Dropout | 0.1 |
| Warmup steps | 8000 |
| Label smoothing (ε) | 0.1 |

## 🚀 How to Run

### Install dependencies
```bash
pip install torch datasets spacy sacrebleu gdown wandb
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

### Train
```bash
python train.py
```

### Inference (autograder format)
```python
from model import Transformer
model = Transformer().to(device)
model.eval()
english = model.infer("Ein Mann sitzt auf einer Bank im Park.")
```

The model automatically downloads trained weights and vocabulary from Google Drive inside `__init__`.

## 📊 Results

| Metric | Score |
|---|---|
| Test BLEU (Greedy) | 37.80 |
| Test BLEU (Beam-4) | 38.77 |

## 🔬 W&B Experiments

| Section | Experiment | Key Finding |
|---|---|---|
| 2.1 | Noam vs Fixed LR | Noam achieves +5 BLEU over fixed LR=1e-4 |
| 2.2 | With vs Without 1/√dₖ | Without scaling: 2.5× larger, erratic gradient norms |
| 2.3 | Attention head analysis | 8 heads show distinct roles; Heads 2 & 7 are redundant |
| 2.4 | Sinusoidal vs Learned PE | Sinusoidal: 36.63 BLEU vs Learned: 34.77 BLEU |
| 2.5 | Label smoothing ε=0.1 vs 0.0 | Smoothing reduces overconfidence, improves generalisation |


## 👤 Author

Mohd Shadab | MTech Mathematics | IIT Madras
