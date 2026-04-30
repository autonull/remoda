import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import time
import os
import numpy as np

from config import ReMoDAConfig
from model import ReMoDAModel

# We'll generate a dummy dataset that is just small enough to run fast on CPU
# but still acts as a valid test of the forward/backward pass and metric logging.
BATCH_SIZE = 8
SEQ_LEN = 64
VOCAB_SIZE = 1000
LEARNING_RATE = 1e-3
MAX_STEPS = 50
EVAL_STEPS = 25
EVAL_ITERS = 5
SEEDS = [42, 100, 1234]

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_config(arch_type: str) -> ReMoDAConfig:
    base = ReMoDAConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=64, # tiny for CPU
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=SEQ_LEN
    )

    if arch_type == "Standard":
        base.num_hidden_layers = 4
        base.use_rt_kv = False
        base.use_moda = False
    elif arch_type == "RT":
        base.num_hidden_layers = 2
        base.use_rt_kv = True
        base.use_moda = False
    elif arch_type == "MoDA":
        base.num_hidden_layers = 4
        base.use_rt_kv = False
        base.use_moda = True
        base.depth_slots = 1
    elif arch_type == "ReMoDA":
        base.num_hidden_layers = 2
        base.use_rt_kv = True
        base.use_moda = True
        base.depth_slots = 1

    return base

def get_dummy_batch():
    X = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    Y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    return X, Y

@torch.no_grad()
def evaluate(model):
    model.eval()
    losses = []
    for _ in range(EVAL_ITERS):
        X, Y = get_dummy_batch()
        loss, _ = model(X, labels=Y)
        losses.append(loss.item())
    model.train()
    avg_loss = sum(losses) / len(losses)
    return avg_loss, math.exp(avg_loss)

def train_model(arch_type: str, seed: int):
    set_seed(seed)
    # print(f"\nTraining Architecture: {arch_type} | Seed: {seed}")

    config = get_config(arch_type)
    model = ReMoDAModel(config).to('cpu')
    num_params = sum(p.numel() for p in model.parameters())

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    start_time = time.time()
    total_tokens = 0
    results = []

    for step in range(MAX_STEPS + 1):
        X, Y = get_dummy_batch()

        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(X, labels=Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_tokens += X.numel()

        if step % EVAL_STEPS == 0:
            val_loss, val_ppl = evaluate(model)
            elapsed = time.time() - start_time
            tps = total_tokens / elapsed if elapsed > 0 else 0
            # print(f"Step {step:04d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | Tokens/sec: {tps:.2f}")
            results.append((step, val_loss, val_ppl))

    return results, num_params

def main():
    import matplotlib.pyplot as plt

    architectures = ["Standard", "RT", "MoDA", "ReMoDA"]
    final_results = {}
    all_learning_curves_mean = {}
    all_learning_curves_std = {}
    all_steps = []

    print(f"\n{'='*50}\nStarting multi-seed training (Seeds: {SEEDS})\n{'='*50}")

    for arch in architectures:
        print(f"Training Architecture: {arch}...")
        arch_results = []
        params = 0

        for seed in SEEDS:
            results, params = train_model(arch, seed)
            arch_results.append(results)

        # Ensure all steps are the same
        steps = [r[0] for r in arch_results[0]]
        all_steps = steps

        # Aggregate results across seeds
        val_losses_across_seeds = [[r[1] for r in seed_res] for seed_res in arch_results]
        val_losses_mean = np.mean(val_losses_across_seeds, axis=0)
        val_losses_std = np.std(val_losses_across_seeds, axis=0)

        all_learning_curves_mean[arch] = val_losses_mean
        all_learning_curves_std[arch] = val_losses_std

        # Final PPL across seeds
        final_ppls = [seed_res[-1][2] for seed_res in arch_results]
        final_ppl_mean = np.mean(final_ppls)
        final_ppl_std = np.std(final_ppls)

        final_results[arch] = {
            "params": f"{params / 1e6:.4f}M",
            "final_val_ppl_mean": final_ppl_mean,
            "final_val_ppl_std": final_ppl_std
        }

    print(f"\n\n{'='*80}")
    print("FINAL ABLATION RESULTS (CPU Sandbox Run with Multi-Seed Aggregation)")
    print(f"{'='*80}")
    print(f"{'Architecture':<15} | {'Params':<10} | {'Final Val PPL (Mean ± Std)':<30}")
    print(f"{'-'*80}")
    for arch in architectures:
        res = final_results[arch]
        print(f"{arch:<15} | {res['params']:<10} | {res['final_val_ppl_mean']:.2f} ± {res['final_val_ppl_std']:.2f}")
    print(f"{'='*80}")

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    for arch in architectures:
        mean_losses = all_learning_curves_mean[arch]
        std_losses = all_learning_curves_std[arch]

        # Parse params in millions
        params_str = final_results[arch]['params']
        params_m = float(params_str.replace('M', ''))

        # Calculate effective compute: Steps * Params (M)
        effective_compute = [step * params_m for step in all_steps]

        p = plt.plot(effective_compute, mean_losses, marker='o', label=arch)
        color = p[0].get_color()
        plt.fill_between(effective_compute, mean_losses - std_losses, mean_losses + std_losses, color=color, alpha=0.2)

    plt.title("Validation Loss Normalized by Parameter Count (ReMoDA vs. Baselines)")
    plt.xlabel("Effective Compute (Steps × MParams)")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("learning_curves.png")
    print("\nSaved learning curve plot to 'learning_curves.png'.")

if __name__ == "__main__":
    main()
