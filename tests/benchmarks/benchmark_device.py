#!/usr/bin/env python3
"""Benchmark MPS vs CPU device performance for DQN training."""

import json
import psutil
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import torch


@dataclass
class BenchmarkResult:
    device: str
    steps: int
    wall_time_seconds: float
    steps_per_second: float
    peak_memory_gb: float


def run_training(device: str, steps: int) -> BenchmarkResult:
    """Run training directly on specified device."""
    torch.manual_seed(7)

    device_obj = torch.device(device)

    # Create model and optimizer
    model = torch.nn.Sequential(
        torch.nn.Linear(84*84*4, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 18)
    ).to(device_obj)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.00025)
    process = psutil.Process()

    batch_size = 32
    state_dim = 84 * 84 * 4
    peak_memory = 0

    start = monotonic()
    for step in range(steps):
        states = torch.randn(batch_size, state_dim, device=device_obj)
        targets = torch.randn(batch_size, 18, device=device_obj)

        outputs = model(states)
        loss = torch.nn.functional.mse_loss(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mem_mb = process.memory_info().rss / 1024 / 1024
        peak_memory = max(peak_memory, mem_mb)

        if (step + 1) % max(1, steps // 10) == 0:
            print(f"  {device.upper()} Step {step+1}/{steps}", flush=True)

    end = monotonic()

    return BenchmarkResult(
        device=device,
        steps=steps,
        wall_time_seconds=end - start,
        steps_per_second=steps / (end - start),
        peak_memory_gb=peak_memory / 1024
    )


def main():
    """Run device efficiency benchmark."""
    print("=== DQN Device Efficiency Benchmark ===")
    print(f"Testing: MPS vs CPU")
    print(f"Machine: M4 (10 cores, 24GB RAM)")
    print()

    # Check device availability
    has_mps = torch.backends.mps.is_available()
    has_cuda = torch.cuda.is_available()

    print(f"Available devices:")
    print(f"  CUDA: {has_cuda}")
    print(f"  MPS:  {has_mps}")
    print(f"  CPU:  True")
    print()

    if not has_mps:
        print("WARNING: MPS not available on this machine")
        return

    # Run benchmarks
    steps = 10_000  # 10K steps for quick comparison
    results = []

    for device in ["cpu", "mps"]:
        print(f"Running benchmark: {device.upper()}... (10K steps)")
        result = run_training(device, steps)
        results.append(result)
        print(f"  Wall time: {result.wall_time_seconds:.1f}s")
        print(f"  Throughput: {result.steps_per_second:.0f} steps/sec")
        print(f"  Peak memory: {result.peak_memory_gb:.2f} GB")
        print()

    # Analysis
    print("=== Results ===")
    cpu_result = next(r for r in results if r.device == "cpu")
    mps_result = next(r for r in results if r.device == "mps")

    speedup = cpu_result.steps_per_second / mps_result.steps_per_second
    print(f"MPS speedup vs CPU: {speedup:.2f}x")

    if speedup > 1.3:
        print("✓ MPS is significantly faster. Recommend using MPS for training.")
        recommended_device = "mps"
    elif speedup > 0.8:
        print("~ MPS is comparable to CPU. Either is fine.")
        recommended_device = "mps"  # Prefer GPU if comparable
    else:
        print("✗ CPU is faster. Recommend using CPU for training.")
        recommended_device = "cpu"

    print()
    print(f"Recommended device: {recommended_device.upper()}")

    # Save results
    output_file = Path(__file__).parent / "benchmark_device_results.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "benchmark": "device_efficiency",
                "steps": steps,
                "results": [
                    {
                        "device": r.device,
                        "wall_time_seconds": r.wall_time_seconds,
                        "steps_per_second": r.steps_per_second,
                        "peak_memory_gb": r.peak_memory_gb,
                    }
                    for r in results
                ],
                "recommended_device": recommended_device,
            },
            indent=2,
        )
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
