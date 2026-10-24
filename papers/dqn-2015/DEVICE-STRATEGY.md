# Device Utilization Strategy

## Benchmark: MPS vs CPU on 1M steps

**Purpose:** Determine optimal device configuration for 50M runs.

**Test:**
```bash
# Run on MPS
python device-comparison.py mps

# Run on CPU
python device-comparison.py cpu

# Run both sequentially
python device-comparison.py both
```

**What to measure:**
- Wall clock time for 1M steps
- Steps/second throughput
- Peak memory usage
- Stability (any errors?)

**Expected results based on synthetic benchmark:**
- MPS: ~34 steps/sec (synthetic) → possibly ~20-25 steps/sec (with Atari overhead)
- CPU: ~25 steps/sec (synthetic) → possibly ~15-20 steps/sec (with Atari overhead)
- Ratio: MPS should be ~1.3-1.5x faster

---

## Hybrid Approaches (Advanced)

If single-device is bottlenecked, we could split work:

### Option A: MPS for NN, CPU for Environment
```
Environment (CPU) → Observation
Preprocess (CPU) → Frame stack
Model forward/backward (MPS) → Q-values, loss
Replay buffer (CPU) → Sampling
```
**Tradeoff:** Data movement CPU↔MPS adds overhead. Probably slower than pure MPS.

### Option B: MPS for both, but pipeline
```
MPS-A: Environment preprocessing (async)
MPS-B: Model training (concurrent)
```
**Reality:** M4 has one GPU. This is just single-threaded work on one device.

### Option C: CPU bottleneck elimination
```
Profile where time goes:
- Environment step: X%
- Preprocessing: Y%
- Model training: Z%
```
If training is <50% of time, environment is the bottleneck, not the model.
Optimization: vectorized envs or faster preprocessing (e.g., GPU preprocessing).

---

## Recommendation

**Start here:** Run the 1M comparison first. If MPS is clearly faster (>20% speedup), use MPS exclusively for 50M runs. If similar, doesn't matter much.

**Hybrid only if:** Profile shows training is <30% of wall time. Then GPU preprocessing might help.

**Most likely outcome:** MPS ~30% faster, use MPS, accept 75-hour runs.
