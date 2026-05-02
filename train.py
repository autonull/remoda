import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import time
import os

from config import ReMoDAConfig
from model import ReMoDAModel
from data import get_dataloaders

# Training Hyperparameters
BATCH_SIZE = 8
SEQ_LEN = 128
LEARNING_RATE = 1e-3
MAX_STEPS = 250
EVAL_STEPS = 50
EVAL_ITERS = 10
NUM_WORKERS = 0

def get_config(arch_type: str, vocab_size: int) -> ReMoDAConfig:
    """Returns the requested configuration, holding parameters roughly equal."""
    base = ReMoDAConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=SEQ_LEN
    )

    if arch_type == "Standard":
        # 8 layers, no recurrence, no depth
        base.num_hidden_layers = 8
        base.use_rt_kv = False
        base.use_moda = False
    elif arch_type == "RT":
        # 4 layers (RT is temporally deeper), pure recurrence, no depth
        base.num_hidden_layers = 4
        base.use_rt_kv = True
        base.use_moda = False
    elif arch_type == "MoDA":
        # 8 layers, standard KV, depth retrieval enabled
        base.num_hidden_layers = 8
        base.use_rt_kv = False
        base.use_moda = True
        base.depth_slots = 2
    elif arch_type == "ReMoDA":
        # 4 layers (matches RT params), recurrence + depth retrieval
        base.num_hidden_layers = 4
        base.use_rt_kv = True
        base.use_moda = True
        base.depth_slots = 2
    else:
        raise ValueError("Unknown architecture")

    return base

@torch.no_grad()
def evaluate(model, val_loader, eval_iters):
    model.eval()
    losses = []
    for i, (X, Y) in enumerate(val_loader):
        if i >= eval_iters:
            break
        # Using CPU device explicitly for sandbox
        X, Y = X.to('cpu'), Y.to('cpu')
        loss, _ = model(X, labels=Y)
        losses.append(loss.item())

    model.train()
    avg_loss = sum(losses) / len(losses)
    return avg_loss, math.exp(avg_loss)

def train_model(arch_type: str, train_loader, val_loader, vocab_size):
    print(f"\n{'='*50}\nTraining Architecture: {arch_type}\n{'='*50}")

    config = get_config(arch_type, vocab_size)
    model = ReMoDAModel(config).to('cpu')

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f} M")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    train_iterator = iter(train_loader)

    start_time = time.time()
    total_tokens = 0

    results = []

    for step in range(MAX_STEPS + 1):
        try:
            X, Y = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            X, Y = next(train_iterator)

        X, Y = X.to('cpu'), Y.to('cpu')

        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(X, labels=Y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_tokens += X.numel()

        if step % EVAL_STEPS == 0:
            val_loss, val_ppl = evaluate(model, val_loader, EVAL_ITERS)
            elapsed = time.time() - start_time
            tps = total_tokens / elapsed if elapsed > 0 else 0

            print(f"Step {step:04d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | Tokens/sec: {tps:.2f}")
            results.append((step, val_loss, val_ppl))

    # Save the ReMoDA model at the end
    if arch_type == "ReMoDA":
        torch.save(model.state_dict(), "remoda_model.pt")
        print("Saved ReMoDA model to remoda_model.pt")

    return results, num_params

def main():
    print("Initializing DataLoaders (this may take a moment to fetch and tokenize)...")
    train_loader, val_loader, vocab_size = get_dataloaders(
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        max_samples=20000 # Use small subset for faster local sandbox testing
    )

    architectures = ["Standard", "RT", "MoDA", "ReMoDA"]
    final_results = {}

    for arch in architectures:
        results, params = train_model(arch, train_loader, val_loader, vocab_size)
        final_ppl = results[-1][2]
        final_results[arch] = {
            "params": f"{params / 1e6:.2f}M",
            "final_val_ppl": final_ppl
        }

    print(f"\n\n{'='*60}")
    print("FINAL ABLATION RESULTS (CPU Sandbox Run)")
    print(f"{'='*60}")
    print(f"{'Architecture':<15} | {'Params':<10} | {'Final Val PPL':<15}")
    print(f"{'-'*60}")
    for arch in architectures:
        res = final_results[arch]
        print(f"{arch:<15} | {res['params']:<10} | {res['final_val_ppl']:.2f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
