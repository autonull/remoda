# 🚀 ReMoDA: Recurrent Mixture-of-Depths Attention 🚀

Welcome to the **ReMoDA** architecture repository! 🎉 This project explores an exciting combination of two powerful concepts: **Recurrent Attention (RT)** and **Mixture-of-Depths Attention (MoDA)**. Our goal is to push the boundaries of Transformer efficiency and performance! 🧠⚡

## 🌟 What makes ReMoDA unique?

ReMoDA is all about achieving **parameter efficiency** without sacrificing the model's ability to capture complex temporal patterns. By combining the strengths of its predecessors, it aims to deliver deeper architectural properties with fewer parameters! 📉

- 🔄 **Recurrent Memory Transformer (RMT):** Inspired by Recurrent Memory mechanisms, ReMoDA retains memory across steps using persistent KV states. This allows the model to achieve a temporally deeper structure while keeping the actual layer count (and parameters) low! [Read more about RMT here.](https://arxiv.org/abs/2207.06881)
- 🕳️ **Mixture-of-Depths (MoD):** Inspired by Mixture-of-Depths, ReMoDA uses cross-layer depth retrieval. It can selectively route and attend to historical layers (depth slots), allowing for dynamic compute allocation. [Read more about MoD here.](https://arxiv.org/abs/2404.02258)

By fusing these, ReMoDA can process standard sequence tasks **much faster** (higher throughput) and with **lower latency** than a standard transformer of similar effective depth.

## 🏗️ Architecture Diagram

Here is a simplified view of how ReMoDA processes tokens and manages its internal depth:

```text
    [Input Tokens]
          |
          v
+-------------------+
|  Token Embedding  |
+-------------------+
          |
          v
+-------------------------------------------------------+
| ReMoDA Layer (1)                                      |
|                                                       |
|  [Q] ---> [Unified Attention] <--- [Seq KV]           |
|                 ^                                     |
|                 |                                     |
|           [Depth KV Retrieval] (MoD Policy)           |
|                                                       |
|  ---> [MLP] ---> [Persistent KV State (RT)] ---> Save |
+-------------------------------------------------------+
          |
          v
+-------------------------------------------------------+
| ReMoDA Layer (2)                                      |
|                                                       |
|  [Q] ---> [Unified Attention] <--- [Seq KV]           |
|                 ^                                     |
|                 |                                     |
|           [Retrieve Layer 1 KV Cache]                 |
|                                                       |
|  ---> [MLP] ---> [Persistent KV State (RT)] ---> Save |
+-------------------------------------------------------+
          |
          v
     [Output Logits]
```

## 🛠️ Codebase Structure & Extensions

We've completely refactored the codebase to make it as modular and hackable as possible! 💻✨

- `config.py`: Contains the `ReMoDAConfig` for easy ablation toggles (turn RT or MoDA on/off!).
- `attention.py`: The `ReMoDAAttention` module handling unified sequence and depth attention.
- `layer.py`: The `ReMoDALayer` and `ReMoDAMLP` definitions.
- `model.py`: The core `ReMoDAModel` and task-specific wrappers (`ReMoDADecisionTransformer`, `ReMoDAForSequenceClassification`).
- `kernel.py`: Contains our custom SDPA kernel that properly preserves causality during joint sequence/depth attention.

You can easily extend this architecture! Want to try a different depth-selection policy? Jump into `attention.py`! Want to apply it to a new RL environment? Check out the wrappers in `model.py` and `experiments/rl_task.py`.

## 🚀 Installation & Usage

1. **Install requirements:**
```bash
pip install torch tqdm matplotlib psutil datasets transformers gymnasium
```

2. **Run Kernel Tests:**
Ensures that causality in the custom SDPA sequence+depth kernel is respected. ✅
```bash
python test_kernel.py
```

3. **Run Benchmarks:**
Compare ReMoDA against a Standard Transformer. ⏱️
```bash
python benchmark.py
```

4. **Train Architecture Ablations (Fast):**
Trains dummy data across Standard, RT, MoDA, and ReMoDA architectures to observe convergence. Outputs a beautiful `learning_curves.png` chart! 📈
```bash
python train_fast.py
```

5. **Evaluate Unified Framework:**
Run tests across RL (Decision Transformer on CartPole) and NLP (Sequence Classification on IMDb) tasks. 🏆
```bash
python evaluate_all.py
```

Get hacking and let's push the frontier of efficient attention together! 🚀🔥
