#!/usr/bin/env python3
"""Benchmark parallel training efficiency: how many runs can we do simultaneously?"""

import json
import psutil
import torch
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class ParallelBenchmarkResult:
    parallel_count: int
    wall_time_seconds: float
    avg_steps_per_second: float
    peak_total_memory_gb: float
    efficiency: float


def run_single_training(run_id: int, steps: int, device: str = "mps") -> dict:
    """Run one training session (called in parallel via ThreadPool)."""
    torch.manual_seed(7 + run_id)
    device_obj = torch.device(device)

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

    end = monotonic()

    return {
        "run_id": run_id,
        "steps": steps,
        "wall_time_seconds": end - start,
        "steps_per_second": steps / (end - start),
        "peak_memory_gb": peak_memory / 1024
    }


def benchmark_parallel_count(num_parallel: int, steps: int) -> ParallelBenchmarkResult:
    """Run multiple training sessions in parallel and measure efficiency."""
    print(f"  Launching {num_parallel} parallel runs ({steps:,} steps each)...")

    start = monotonic()

    with ThreadPoolExecutor(max_workers=num_parallel) as executor:
        futures = [
            executor.submit(run_single_training, i, steps)
            for i in range(num_parallel)
        ]
        results = [f.result() for f in as_completed(futures)]

    end = monotonic()
    wall_time = end - start

    # Aggregate results
    total_steps = sum(r["steps"] for r in results)
    avg_throughput = total_steps / wall_time
    peak_memory = max(r["peak_memory_gb"] for r in results)

    return ParallelBenchmarkResult(
        parallel_count=num_parallel,
        wall_time_seconds=wall_time,
        avg_steps_per_second=avg_throughput,
        peak_total_memory_gb=peak_memory,
        efficiency=0
    )


def main():
    """Run parallel efficiency benchmark."""
    print("=== DQN Parallel Training Benchmark ===")
    print(f"Testing: How many runs can we parallelize?")
    print(f"Machine: M4 (10 cores, 24GB RAM)")
    print()

    steps = 10_000  # Steps per run (10K = ~10 sec, allows quick testing)
    parallel_counts = [1, 2, 3, 4]

    results = []

    # Run baseline first (1 run)
    print(f"Baseline: Single run ({steps:,} steps)")
    baseline = benchmark_parallel_count(1, steps)
    results.append(baseline)
    print(f"  Wall time: {baseline.wall_time_seconds:.1f}s")
    print(f"  Throughput: {baseline.avg_steps_per_second:.0f} steps/sec")
    print(f"  Peak memory: {baseline.peak_total_memory_gb:.2f} GB")
    print()

    baseline_throughput = baseline.avg_steps_per_second

    # Run parallel tests
    for num_parallel in parallel_counts[1:]:
        print(f"Test: {num_parallel} parallel runs ({steps:,} steps each)")
        result = benchmark_parallel_count(num_parallel, steps)
        result.efficiency = result.avg_steps_per_second / (baseline_throughput * num_parallel)
        results.append(result)
        print(f"  Wall time: {result.wall_time_seconds:.1f}s")
        print(f"  Total throughput: {result.avg_steps_per_second:.0f} steps/sec")
        print(f"  Efficiency: {result.efficiency:.1%}")
        print(f"  Peak memory (process): {result.peak_total_memory_gb:.2f} GB")
        print()

    # Analysis
    print("=== Analysis ===")
    for result in results[1:]:
        if result.peak_total_memory_gb > 20:
            print(f"⚠ {result.parallel_count} runs: Memory high ({result.peak_total_memory_gb:.1f}GB)")
        elif result.efficiency < 0.7:
            print(f"⚠ {result.parallel_count} runs: Efficiency poor ({result.efficiency:.1%})")
        else:
            print(f"✓ {result.parallel_count} runs: OK (mem={result.peak_total_memory_gb:.1f}GB, eff={result.efficiency:.1%})")

    # Recommendation
    safe_results = [r for r in results if r.peak_total_memory_gb < 20 and r.efficiency > 0.7]
    if safe_results:
        safe_count = max(r.parallel_count for r in safe_results)
    else:
        safe_count = 1
    print()
    print(f"Recommended parallel count: {safe_count}")
    print(f"This gives ~{safe_count}x speedup for 50M runs (from ~73 hours to ~{73//safe_count}h per batch)")

    # Save results
    output_file = Path(__file__).parent / "benchmark_parallel_results.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "benchmark": "parallel_efficiency",
                "steps_per_run": steps,
                "results": [
                    {
                        "parallel_count": r.parallel_count,
                        "wall_time_seconds": r.wall_time_seconds,
                        "avg_steps_per_second": r.avg_steps_per_second,
                        "peak_memory_gb": r.peak_total_memory_gb,
                        "efficiency": r.efficiency,
                    }
                    for r in results
                ],
                "recommended_parallel_count": safe_count,
            },
            indent=2,
        )
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
