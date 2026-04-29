import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import time
import os

from model import ReMoDAConfig, ReMoDAModel

# We'll generate a dummy dataset that is just small enough to run fast on CPU
# but still acts as a valid test of the forward/backward pass and metric logging.
BATCH_SIZE = 8
SEQ_LEN = 64
VOCAB_SIZE = 1000
LEARNING_RATE = 1e-3
MAX_STEPS = 50
EVAL_STEPS = 25
EVAL_ITERS = 5

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

def train_model(arch_type: str):
    print(f"\n{'='*50}\nTraining Architecture: {arch_type}\n{'='*50}")

    config = get_config(arch_type)
    model = ReMoDAModel(config).to('cpu')
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f} M")

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
            print(f"Step {step:04d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | Tokens/sec: {tps:.2f}")
            results.append((step, val_loss, val_ppl))

    return results, num_params

def main():
    architectures = ["Standard", "RT", "MoDA", "ReMoDA"]
    final_results = {}

    for arch in architectures:
        results, params = train_model(arch)
        final_ppl = results[-1][2]
        final_results[arch] = {
            "params": f"{params / 1e6:.4f}M",
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
