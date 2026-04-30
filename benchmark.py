import torch
import time
import os
import gc
import psutil
import numpy as np
from config import ReMoDAConfig
from model import ReMoDAModel

NUM_TRIALS = 5

def get_config(arch_type: str, vocab_size: int, seq_len: int) -> ReMoDAConfig:
    base = ReMoDAConfig(
        vocab_size=vocab_size,
        hidden_size=256,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=seq_len
    )

    if arch_type == "Standard":
        base.num_hidden_layers = 4
        base.use_rt_kv = False
        base.use_moda = False
    elif arch_type == "ReMoDA":
        base.num_hidden_layers = 2
        base.use_rt_kv = True
        base.use_moda = True
        base.depth_slots = 2
    else:
        raise ValueError("Unknown architecture")

    return base

def benchmark_architecture(arch: str, batch_size: int, seq_len: int, vocab_size: int):
    # Free up memory
    gc.collect()

    config = get_config(arch, vocab_size, seq_len)
    model = ReMoDAModel(config)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())

    latencies = []
    throughputs = []
    memories = []

    for trial in range(NUM_TRIALS):
        # Clear memory between trials
        gc.collect()

        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Warmup
        for _ in range(5):
            _ = model(x)

        start_time = time.time()
        with torch.no_grad():
            for _ in range(50):
                _ = model(x)
        end_time = time.time()

        total_time = end_time - start_time
        passes = 50
        latency = total_time / passes
        throughput = (batch_size * seq_len * passes) / total_time

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)

        latencies.append(latency * 1000)
        throughputs.append(throughput)
        memories.append(memory_mb)

    return {
        "Params (M)": f"{num_params/1e6:.2f}M",
        "Latency Mean (ms)": np.mean(latencies),
        "Latency Std (ms)": np.std(latencies),
        "Throughput Mean (tok/s)": np.mean(throughputs),
        "Throughput Std (tok/s)": np.std(throughputs),
        "Memory Mean (MB)": np.mean(memories)
    }

def main():
    configs_to_test = [
        {"batch_size": 4, "seq_len": 128, "vocab_size": 1000},
        {"batch_size": 8, "seq_len": 256, "vocab_size": 1000},
        {"batch_size": 16, "seq_len": 512, "vocab_size": 1000},
    ]

    print(f"Benchmarking ReMoDA vs Standard Transformer (Aggregated over {NUM_TRIALS} trials)")
    print("=" * 115)

    for test_config in configs_to_test:
        bs = test_config['batch_size']
        sl = test_config['seq_len']
        vs = test_config['vocab_size']
        print(f"Batch Size: {bs}, Seq Len: {sl}, Vocab: {vs}")
        print("-" * 115)

        results = {}
        for arch in ["Standard", "ReMoDA"]:
            res = benchmark_architecture(arch, bs, sl, vs)
            results[arch] = res

        print(f"{'Architecture':<15} | {'Params':<10} | {'Latency (ms)':<25} | {'Throughput (tok/s)':<30} | {'Memory (MB)':<15}")
        print("-" * 115)
        for arch, res in results.items():
            latency_str = f"{res['Latency Mean (ms)']:.2f} ± {res['Latency Std (ms)']:.2f}"
            throughput_str = f"{res['Throughput Mean (tok/s)']:.2f} ± {res['Throughput Std (tok/s)']:.2f}"
            memory_str = f"{res['Memory Mean (MB)']:.2f}"
            print(f"{arch:<15} | {res['Params (M)']:<10} | {latency_str:<25} | {throughput_str:<30} | {memory_str:<15}")
        print("=" * 115)

if __name__ == "__main__":
    main()
