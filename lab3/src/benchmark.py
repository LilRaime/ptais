import time
import numpy as np
import torch

def run_warmup(model, tensors, iterations=15):
    model.eval()
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    n = len(tensors)
    with torch.no_grad():
        for i in range(iterations):
            _ = model(tensors[i % n])

def benchmark_inference(model, tensors, iterations=100, warmup_iterations=15):
    run_warmup(model, tensors, iterations=warmup_iterations)
    model.eval()
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    n = len(tensors)

    latencies = []
    with torch.no_grad():
        for i in range(iterations):
            tensor = tensors[i % n]
            t_start = time.perf_counter()
            _ = model(tensor)
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000.0)

    latencies = np.array(latencies, dtype=np.float64)
    mean_ms = float(np.mean(latencies))
    std_ms = float(np.std(latencies))
    median_ms = float(np.median(latencies))
    p90_ms = float(np.percentile(latencies, 90))
    p99_ms = float(np.percentile(latencies, 99))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return {
        "iterations": iterations,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "median_ms": median_ms,
        "p90_ms": p90_ms,
        "p99_ms": p99_ms,
        "fps": fps,
        "latencies": latencies,
    }
