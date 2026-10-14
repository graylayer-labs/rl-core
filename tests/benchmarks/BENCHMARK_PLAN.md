# DQN Training Efficiency Benchmarks

Before running 25 × 50M step runs, validate compute strategy with targeted tests.

## Benchmarks to Run

### 1. Device Efficiency (MPS vs CPU)
**Goal:** Determine if MPS (GPU) is actually faster than CPU for DQN.

- Run: 100K steps, single game (BeamRider), seed 7
- Test configurations:
  - MPS device
  - CPU device
- Metrics collected:
  - Wall clock time
  - Steps/second
  - Peak memory usage
  - Is MPS actually faster? By how much?

**Duration:** ~20 min per device (~40 min total)

### 2. Batch Size Impact
**Goal:** Find optimal batch_size for throughput without memory bloat.

- Run: 100K steps, single game, MPS (or faster device from benchmark 1)
- Test configurations:
  - batch_size=32 (default)
  - batch_size=64
  - batch_size=128
- Metrics:
  - Steps/second
  - Peak memory
  - Wall time
  - Does larger batch = better throughput or just more memory?

**Duration:** ~60 min total

### 3. Parallel Efficiency
**Goal:** Find optimal number of parallel runs before memory/CPU saturation.

- Run: 100K steps, single game, optimal device from benchmark 1
- Test configurations:
  - 1 run (baseline)
  - 2 runs in parallel
  - 3 runs in parallel
  - 4 runs in parallel (if memory allows)
- Metrics per run:
  - Steps/second
  - Peak memory per process
  - Total system memory usage
  - Wall time (should be ~same for all since they run parallel)

**Duration:** 100K per run × 4 configs = but parallel, so ~100-120 min

### 4. Full Validation Run (if benchmarks successful)
**Goal:** Confirm 5M-step validation is feasible.

- Run: 5M steps × 25 configs (5 games × 5 seeds)
- Parallelization: Based on benchmark 3 findings
- Expected time: ~6-8 hours wall time

## Success Criteria

- [ ] MPS is clearly faster than CPU (>1.5x), or CPU is comparable and MPS isn't worth complexity
- [ ] Identified optimal batch_size (throughput plateau, memory OK)
- [ ] Identified safe parallel count (e.g., "run 2 in parallel = 2x speedup, 3+ = saturation")
- [ ] Peak memory never exceeds 20GB (leaving 4GB headroom on 24GB)
- [ ] Can confidently estimate time for 50M runs

## Output Artifacts

Each benchmark creates:
- `benchmark_<name>_results.json` with raw metrics
- Console report with recommendations
