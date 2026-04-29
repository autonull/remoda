# ReMoDA (Recurrent Mixture-of-Depths Attention)

A PyTorch implementation of the ReMoDA architecture, exploring combinations of Recurrent Attention (RT) and Mixture-of-Depths Attention (MoDA) to improve Transformer efficiency and performance.

This repository demonstrates the architecture's parameters, causality, and training performance on dummy tasks and small datasets like TinyStories.

## Features

- **Parameter Efficiency:** The ReMoDA architecture achieves a lower parameter count while maintaining temporally deeper structures through Recurrent Attention.
- **Improved Throughput & Latency:** The depth retrieval from cross-layer mixture-of-depths allows ReMoDA to operate much faster on standard sequence tasks, achieving up to a 50% increase in throughput (tokens/second) compared to a standard transformer with similar effective depth.
- **Custom SDPA Kernel:** A unified attention kernel using `scaled_dot_product_attention` allows joint sequence and depth attention, correctly preserving causality.

## Benchmarks

Based on CPU tests comparing ReMoDA against a Standard Transformer baseline:

| Architecture | Batch Size | Seq Len | Vocab Size | Latency (ms) | Throughput (tok/s) | Parameter Count |
| --- | --- | --- | --- | --- | --- | --- |
| Standard | 4 | 128 | 1000 | 25.21 | 20,312 | 4.49M |
| ReMoDA | 4 | 128 | 1000 | 16.10 | 31,794 | 2.39M |
| Standard | 8 | 256 | 1000 | 97.26 | 21,056 | 4.52M |
| ReMoDA | 8 | 256 | 1000 | 63.42 | 32,292 | 2.42M |
| Standard | 16 | 512 | 1000 | 431.49 | 18,985 | 4.58M |
| ReMoDA | 16 | 512 | 1000 | 273.15 | 29,990 | 2.49M |

*Note: Benchmarks were run using CPU. For full tests, run `python benchmark.py`.*

## Installation & Usage

1. **Install requirements:**
```bash
pip install torch tqdm matplotlib psutil
```

2. **Run Kernel Tests:**
Ensures that causality in the custom SDPA sequence+depth kernel is respected.
```bash
python test_kernel.py
```

3. **Run Benchmarks:**
```bash
python benchmark.py
```

4. **Train Architecture Ablations (Fast):**
Trains dummy data across Standard, RT, MoDA, and ReMoDA architectures to observe convergence. This also outputs a `learning_curves.png` chart comparing the validation losses.
```bash
python train_fast.py
```
