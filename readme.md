# Character-Level MLP Language Model

A character-level language model built from scratch using a **Multi-Layer Perceptron (MLP)**, trained on a dataset of ~32,000 English names. Inspired by Andrej Karpathy's [makemore](https://github.com/karpathy/makemore) series.

## Overview

This project implements a neural network that learns to generate realistic-sounding names by predicting the next character given a fixed-size context window of previous characters. The model is trained purely using PyTorch primitives — no high-level abstractions.

## Architecture

| Component | Details |
|---|---|
| Vocabulary | 27 characters (a–z + `.` as delimiter) |
| Embedding | Each character → 30-dimensional vector |
| Context window | 8 previous characters |
| Input to hidden | 8 × 30 = 240 → 400 neurons |
| Activation | Tanh |
| Output | 400 → 27 logits (softmax over characters) |
| Loss | Cross-entropy |
| Optimizer | Vanilla SGD with learning rate decay |

## Dataset

- **Source:** `names.txt` — 32,033 English first names
- **Split:**
  - Train: 80% (~182,691 examples)
  - Dev: 10% (~22,793 examples)
  - Test: 10% (~22,662 examples)

## Training

- Mini-batch gradient descent with batch size **256~1000**
- **~150,000** training iterations
- Learning rate schedule: starts at `0.1`, decays to `0.01`
- Manual backpropagation using PyTorch autograd

## Results

| Split | Loss |
|---|---|
| Train (mini-batch) | ~2.15 |
| Dev | ~2.14 |
| Test | ~2.14 |



## Project Structure

```
character-level_MLP/
├── main.ipynb      # Full model: data loading, training, evaluation, sampling
├── names.txt       # Training dataset (32K names)
└── readme.md       # This file
```

## Usage

Open `main.ipynb` in Jupyter and run cells top to bottom:

1. **Load & tokenize** the names dataset
2. **Build train/dev/test splits**
3. **Initialize model parameters** (embeddings + MLP weights)
4. **Train** with mini-batch SGD
5. **Evaluate** loss on dev and test splits
6. **Sample** new names from the model

## Requirements

```
torch
matplotlib
```

Install with:

```bash
pip install torch matplotlib
```

## Key Concepts

- **Character embeddings:** Each of the 27 characters is mapped to a learnable dense vector, allowing the model to discover similarity between characters.
- **Context window:** Rather than processing one character at a time, the model sees the last 8 characters as context, flattened into a single input vector.
- **Mini-batch training:** Random subsets of the training data are used per step for faster, noisier gradient estimates — standard for large datasets.
- **Learning rate decay:** A higher initial LR allows fast early progress; decaying it later helps fine-tune convergence.

## References

- Bengio et al. (2003) — [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- Andrej Karpathy — [makemore (YouTube series)](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
