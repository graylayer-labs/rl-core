# DQN Training Efficiency Testing

Before running 25 × 50M step qualifying runs, we benchmark to ensure optimal compute utilization.

## Test Suite

### 1. Device Efficiency (`benchmark_device.py`)
**What:** MPS vs CPU throughput comparison.

**Run:**
```bash
uv run python tests/benchmarks/benchmark_device.py
```

**What it measures:**
- Steps/second on MPS
- Steps/second on CPU
- Peak memory for each
- Recommends faster device

**Expected duration:** ~3-5 minutes total
**Output:** `benchmark_device_results.json`

### 2. Parallel Efficiency (`benchmark_parallel.py`)
**What:** How many training runs can we parallelize safely?

**Run:**
```bash
uv run python tests/benchmarks/benchmark_parallel.py
```

**What it measures:**
- Wall time for 1, 2, 3, 4 parallel runs (10K steps each)
- Throughput scaling (efficiency vs sequential)
- Peak memory per process
- Recommends safe parallelization level

**Expected duration:** ~3-5 minutes total (tests run in parallel)
**Output:** `benchmark_parallel_results.json`

## Testing Strategy

```
Step 1: Run device_efficiency benchmark
  ↓
  Decision: Use MPS or CPU?
  ↓
Step 2: Run parallel_efficiency benchmark
  ↓
  Decision: Run 1, 2, 3, or 4 in parallel?
  ↓
Step 3: Validation run (5M steps × 25 configs)
  Use optimal device + parallel count from benchmarks
  ↓
  Decision: Does throughput match predictions? Memory OK?
  ↓
Step 4: Full 50M runs
  Use same configuration at scale
```

## Results Interpretation

### From `benchmark_device_results.json`
Look for:
- `speedup`: How much faster is MPS?
  - > 1.3x: Use MPS (significant win)
  - 1.0-1.3x: Either works, use MPS for consistency
  - < 1.0x: Use CPU

### From `benchmark_parallel_results.json`
Look for:
- `efficiency`: Speedup per additional run
  - 1 run → 2 runs: should be ~1.8x-1.9x (small overhead)
  - 2 runs → 3 runs: should be ~2.8x-2.9x
  - 3 runs → 4 runs: may drop to ~3.2x-3.5x (saturation starting)
  - If < 2.0x for 2 runs, memory/CPU contention too high
- `peak_memory_gb`: Peak process memory
  - Should stay < 8GB per process (4 processes = 32GB, leaving headroom)
  - If > 10GB per process, reduce parallelism

## Validation Run

After benchmarks complete, run validation:

```bash
# Will use optimal config from benchmarks
# This takes 2-3x parallel_count × 100K steps time
uv run python tests/benchmarks/run_validation.py
```

(Script to be created after benchmarks succeed)

## Typical Outcomes

**Scenario A: MPS Faster, 2x Parallelism**
- Device: MPS
- Parallel: 2 runs at a time
- Time for 50M per run: ~90 hours → ~45 hours per batch
- 25 runs = 13 batches × 45h = ~580 hours = ~24 days

**Scenario B: CPU Only, 3x Parallelism**
- Device: CPU
- Parallel: 3 runs
- Time for 50M per run: ~110 hours → ~37 hours per batch
- 25 runs = 9 batches × 37h = ~330 hours = ~14 days

## Success Criteria

Run benchmarks successfully if:
- [ ] `benchmark_device.py` completes without error
- [ ] `benchmark_parallel.py` completes without error
- [ ] Results show clear recommendations
- [ ] Recommended memory usage < 20GB
- [ ] Recommended parallelism ≥ 2 (otherwise months of wall time)

## Notes

- These benchmarks use **synthetic training** (random data), not actual Atari environments
  - This is intentional: tests compute efficiency, not environment overhead
  - Actual runs will be slower (~5-10% more) due to environment interactions
- Results are specific to M4 Mac; different hardware will have different characteristics
- If either benchmark fails, investigate before proceeding to validation
