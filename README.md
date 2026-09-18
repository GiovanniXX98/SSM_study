# SSM_study: Fourier Space-Time State Space Model (FSTLLM 2.0)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Research Repository**: Investigating wave-theoretic dynamics and Fourier spectral modulation to build  **State Space Models (SSMs)** as alternative recurrent architectures.

---

## 🔬 Motivation & Research Vision

Modern Large Language Models (LLMs) face an efficiency bottleneck:
* **Standard Transformers** incur quadratic complexity $\mathcal{O}(S^2)$ during training and suffer from memory explosion in the Key-Value (KV) cache during autoregressive inference.
* **State Space Models (e.g., Mamba, S4, RWKV)** achieve linear training $\mathcal{O}(S)$ and constant inference $\mathcal{O}(1)$, but their real-valued state transitions can struggle with long-range harmonic phase coherence or micro-syntactic token dependencies.

**SSM_study** explores a wave-mechanics approach to State Space Models: **Fourier Space-Time LLM (FSTLLM 2.0)**. By mapping token representations onto continuous complex phasors ($e^{i \Phi}$) on the unit circle $\mathbb{S}^1$, tokens interact through constructive and destructive wave interference rather than static dot-products.

The primary goal of this repository is to systematically study, benchmark, and refine Fourier-based state-space mechanisms to develop **highly competitive, hardware-efficient SSMs**.

---

## ⚡ Core Architectural Pillars (v2.0)

```
                     ┌───────────────────────────────┐
                     │          Input Token          │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │    Local Depthwise Conv1D     │  ◄── Micro-syntactic mixing
                     └───────────────┬───────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
      ┌─────────────────────────────┐ ┌─────────────────────────────┐
      │   Grouped Key/Value Heads   │ │     Full Query Heads        │
      │  (num_kv_heads = 2, GQR)    │ │      (num_heads = 8)        │
      └──────────────┬──────────────┘ └──────────────┬──────────────┘
                     │                               │
                     ▼                               │
      ┌─────────────────────────────┐                │
      │  Holographic State Cache    │                │
      │  S_t = S_{t-1}·γ + K·V·e^-jΦ│                │
      └──────────────┬──────────────┘                │
                     │ (repeat_interleave)           │
                     ▼                               │
      ┌──────────────────────────────────────────────┴──────────────┐
      │      Selective Phase Demodulation & Resonant Readout        │
      └──────────────────────────────┬──────────────────────────────┘
                                     ▼
                     ┌───────────────────────────────┐
                     │      SwiGLU Feed-Forward      │
                     └───────────────────────────────┘
```

### 1. Holographic State Cache (True $\mathcal{O}(1)$ Constant Inference)
During text generation, FSTLLM 2.0 processes only the single latest token per step. The entire historical context is recursively preserved in a compact holographic state tensor:
$$S_t = S_{t-1} \odot \gamma_t + \left(K_t \odot V_t \odot e^{-j \Phi_t}\right)$$
* **Inference Cost**: strictly $\mathcal{O}(1)$ time and memory per token.
* **No KV Cache Growth**: memory consumption is completely independent of sequence length.

### 2. Grouped-Query Resonance (GQR)
Inspired by Grouped-Query Attention (GQA), FSTLLM 2.0 decouples complex wave generation from query readouts:
* A small set of macroscopic heads (`num_kv_heads`, e.g., 2) computes the carrier wave modulation and state updates.
* A larger set of query heads (`num_heads`, e.g., 8) listens to the duplicated state via `repeat_interleave`.
* **Impact**: drastically lowers parameter count and VRAM allocation without sacrificing expressivity.

### 3. Local Depthwise Conv1D Spatial Pre-Mixing
While Fourier phase modulation excels at long-range harmonic retrieval, short-range syntax (punctuation, articles, subwords) benefits from localized context.
* A depthwise causal 1D convolution (`kernel_size=4`) mixes each token with its immediate past before spectral projection.
* Convolution states are cached during autoregression to preserve the $\mathcal{O}(1)$ property.

---

## 📁 Repository Structure

This repository is intentionally kept clean, modular, and focused:

```
SSM_study/
├── model/
│   ├── __init__.py       # Package exports
│   └── model.py          # Pure PyTorch FSTLLM 2.0 implementation
├── train/
│   ├── __init__.py       # Package exports
│   └── train.py          # Training loop and O(1) decode benchmark
├── .gitignore            # Clean git rules
├── requirements.txt      # PyTorch and NumPy dependencies
└── README.md             # Project documentation and research roadmap
```

---

## 🚀 Quick Start

### 1. Prerequisites
Clone the repository and install dependencies:
```bash
git clone https://github.com/GiovanniXX98/SSM_study.git
cd SSM_study
pip install -r requirements.txt
```

### 2. Run Training & O(1) Generation Demo
The training script includes an automatic synthetic dataset fallback, allowing you to test training and generation immediately with zero setup:
```bash
python3 train/train.py --epochs 2 --d-model 256 --n-layers 4
```

### 3. Training on Custom Text
To train on your own corpus (e.g. Shakespeare, TinyStories, or code):
```bash
python3 train/train.py \
    --data-path /path/to/your/corpus.txt \
    --seq-len 256 \
    --batch-size 32 \
    --epochs 5 \
    --d-model 384 \
    --n-layers 6 \
    --num-heads 8 \
    --num-kv-heads 2
```

---

## 📊 Comparison Matrix

| Property | Standard Transformer | Mamba (S6) | RWKV-6 | **FSTLLM 2.0 (Ours)** |
| :--- | :---: | :---: | :---: | :---: |
| **Training Complexity** | $\mathcal{O}(S^2)$ | $\mathcal{O}(S)$ | $\mathcal{O}(S)$ | $\mathbf{\mathcal{O}(S)}$ |
| **Decode Complexity** | $\mathcal{O}(S)$ (expanding) | $\mathcal{O}(1)$ | $\mathcal{O}(1)$ | $\mathbf{\mathcal{O}(1)}$ |
| **KV Cache Footprint** | Grows linearly | Fixed state | Fixed state | **Fixed Holographic State** |
| **Phase / Harmonic Carrier** | None (learned dot-product) | Real discretization | Real decay | **Complex Unit Circle ($\mathbb{S}^1$)** |
| **Local Syntax Coupling** | Implicit | Conv1D | Time-mixing | **Depthwise Conv1D + GQR** |

---

## 🗺️ Ongoing Research & Roadmap

- [x] **v2.0 Architecture**: Holographic Cache, GQR, and Depthwise Conv1D.
- [ ] **Flash-Scan Kernel**: Custom Triton kernel for hardware-accelerated associative parallel prefix scan.
- [ ] **Needle-In-A-Haystack Benchmarking**: Stress testing phase retention up to 64k context windows.
- [ ] **BPE Tokenizer Integration**: Upgrading from character-level modeling to HuggingFace BPE/Tiktoken tokenization.
- [ ] **Scaling Laws**: Evaluating loss scaling vs parameter count up to 1B parameters against Mamba baselines.

---

## 📄 Citation & License

This project is open-source under the [MIT License](LICENSE).  
If you find this research useful in your study of State Space Models, please feel free to cite or star this repository.
