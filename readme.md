# Mini-batch Playground — README (Beginner-friendly)

---

## Contents

- Overview  
- Requirements  
- Files  
- How to run  
- Function-by-function explanation  
- Shapes cheat-sheet (why it matters)  
- **Understanding the output and timing** *(added & explained)*  
- Quick experiments to try (copy-paste)  
- Typical errors & debugging tips  
- Simple **fixes** & code snippets to make comparisons fairer and timings accurate  
- Further reading / next steps

---

## Overview

This repo contains a small, educational experiment that compares three ways to train a simple linear model `ŷ = w·x + b`:

- **Batch Gradient Descent** (full dataset per update)  
- **Stochastic Gradient Descent (SGD)** (one-sample updates)  
- **Mini-batch Gradient Descent** (small-group updates, e.g. `bs=32`)

Goal: learn how the algorithms differ in loss behavior, runtime, and final model parameters.

---

## Requirements

- Python 3.7+  
- `numpy`  
- `matplotlib`  

Install:

```bash
pip install numpy matplotlib
```

---

## Files

- `minibatch_fixed.py` — main script implementing and comparing the three algorithms.

---

## How to run

```bash
python minibatch_fixed.py
```

The script prints final learned parameters, final training loss, runtimes and shows a plot of training loss vs epoch for each method.

---

## Function-by-function explanation

(kept short here — full explanations are already present in the code comments; the most important pieces:)

- `set_seed(seed=0)` — sets NumPy RNG for reproducible runs.
- `generate_linear_data(n_samples=2000, noise_std=1.0)` — creates `X` with shape `(n,1)` and `y = true_w * X + true_b + noise`.
- `mse_loss(y_true, y_pred)` — returns scalar MSE.
- `predict_linear(X, w, b)` — computes `X.dot(w) + b`. If `X` is `(n, d)` and `w` is `(d, 1)` result is `(n, 1)`.
- `batch_gd(...)` — full-dataset gradient descent; gradients use factor `2.0 / n`.
- `sgd(...)` — per-sample SGD; typical pattern: shuffle indices, slice `xi = X[i:i+1]` to keep shapes `(1,1)`, compute per-sample gradient and update.
- `minibatch_gd(...)` — like SGD but with slices `xb = X[idx]` of size `bs`, gradient uses `2.0 / xb.shape[0]`.

> **Shape rule of thumb**: keep `X` 2-D (`(n,d)`) and `w` 2-D (`(d,1)`) to avoid broadcasting surprises. Use `X[i:i+1]` to get a `(1, d)` slice.

---

## Shapes cheat-sheet (very important)

- `X`: `(n_samples, d)` — generated as `(n, 1)` in this project  
- `w`: `(d, 1)` — kept as 2-D to make `X.dot(w)` predictable  
- `b`: scalar (Python float) — broadcasting adds it to every row  
- `y`, `y_pred`: `(n_samples, 1)`  

---

## Understanding the output and timing  *(added section)*

### Example output (typical)
```
len(losses_batch) = 100
len(losses_mini)  = 100
len(losses_sgd)   = 25

True params: w=3.0, b=2.0

Batch GD:   w=3.0390, b=1.7318, final loss=1.068601
SGD:        w=2.9912, b=1.9156, final loss=1.013989
Mini-batch: w=3.0429, b=1.9828, final loss=1.002311

Timing: batch_gd took 0.063 s
Timing: SGD took 2.640 s (epochs=25)
Timing: Mini-batch took 0.452 s
```

### What each part means
- `len(losses_*)` — number of recorded epoch losses returned by each trainer. Note: `sgd` was run with fewer epochs in this run, hence shorter list.
- `True params` — the parameters used to **generate** the synthetic data (`true_w = 3.0, true_b = 2.0`).
- `Batch GD`, `SGD`, `Mini-batch` lines — final learned `w`, `b`, and final MSE (training loss).
- **Timings** — time taken (wall-clock) for each training routine as measured during this run.

### Interpretation & intuition
- **Final loss ≈ 1.0**: you generated `noise_std = 1.0`. The irreducible variance from noise is ≈1.0, so MSE near 1.0 is expected and means the model learned the signal well.
- **Mini-batch had the lowest loss (~1.002)**: this shows mini-batch here reached parameters closest to the true values — a typical result because mini-batch balances stable gradients and useful stochasticity.
- **Batch had a slightly higher loss**: batch-GD performed only `epochs` updates (one update per epoch). Because mini-batch/SGD perform many more updates (many batches or per-sample updates), they had more opportunities to refine parameters given the learning rates used.
- **Timing differences**:
  - `batch_gd` is fastest per epoch here because it performs a few **large, vectorized** NumPy operations (fast C code).
  - `sgd` is slowest because it updates **per-sample in Python** (many Python loops calling NumPy), which is costly.
  - `mini-batch` sits in between: it does repeated vectorized ops per batch, so has more Python overhead than batch but much less than per-sample SGD.
- **Important**: In CPU+NumPy experiments with a small dataset, full-batch may be fastest per epoch. In real ML on GPUs and large datasets, mini-batch usually gives the best wall-clock training time and is the standard approach.

### Key takeaway
| Method | Update per | Stability | Convergence Speed | Typical Use |
|---------|-------------|------------|--------------------|--------------|
| Batch GD | Full dataset | Very stable | Slow for large data | Small datasets / teaching |
| SGD | Single sample | Noisy | Fast but unstable | Large streaming data |
| Mini-batch GD | Small groups (e.g. 32) | Balanced | Usually best overall | Deep learning, GPUs |

Mini-batch Gradient Descent gives you **the best trade-off** — efficient updates, smoother loss curve, and scalable training.

---

## Quick experiments to try (copy-paste)

1. **Standardize `X` first**
```python
X = (X - X.mean(axis=0)) / X.std(axis=0)
```

2. **Vary noise**
```python
X, y, _, _ = generate_linear_data(n_samples=2000, noise_std=3.0)
```

3. **Batch size sweep**
```python
for bs in [1, 8, 32, 128, X.shape[0]]:
    w, b, losses = minibatch_gd(X, y, lr=0.01, epochs=60, batch_size=bs)
    print('bs=', bs, 'final loss=', losses[-1])
```

4. **Plot `w` & `b` trajectories** (inside training loop collect `w` each epoch and plot).

5. **Time-to-threshold test**
```python
import time
target_loss = 1.01

t0 = time.perf_counter()
w_b, b_b, losses_b = batch_gd(X, y, lr=0.01, epochs=200)
t_batch = time.perf_counter() - t0
epoch_to_target_batch = next((i for i,l in enumerate(losses_b,1) if l < target_loss), None)

t0 = time.perf_counter()
w_m, b_m, losses_m = minibatch_gd(X, y, lr=0.01, epochs=200, batch_size=32)
t_mini = time.perf_counter() - t0
epoch_to_target_mini = next((i for i,l in enumerate(losses_m,1) if l < target_loss), None)

print("batch time:", t_batch, "epochs to target:", epoch_to_target_batch)
print("mini time: ", t_mini,  "epochs to target:", epoch_to_target_mini)
```

---

## Typical errors & debugging tips

- **Shape errors**: Print `X.shape`, `w.shape`, `y.shape`. Use `X[i:i+1]` to preserve `(1,d)`.
- **Loss blows up / `nan`**: Lower `lr` (learning rate) and/or reduce init scale `* 0.1`.
- **Plotting mismatch**: If `losses` lists have different lengths, pad or align them before plotting (the script currently pads `losses_sgd` to the main `epochs` for plotting).

---

## Simple fixes & improved code snippets (paste into your script)

### 1) **Measure each training function separately (accurate timing)**

Replace your timing block in `main()` with:
```python
import time

t0 = time.perf_counter()
w_batch, b_batch, losses_batch = batch_gd(X, y, lr=0.01, epochs=epochs)
t_batch = time.perf_counter() - t0

t0 = time.perf_counter()
w_sgd, b_sgd, losses_sgd = sgd(X, y, lr=0.005, epochs=epochs//4)
t_sgd = time.perf_counter() - t0

t0 = time.perf_counter()
w_mini, b_mini, losses_mini = minibatch_gd(X, y, lr=0.01, epochs=epochs, batch_size=32)
t_mini = time.perf_counter() - t0

print(f"Timing: batch_gd took {t_batch:.3f} s")
print(f"Timing: SGD took {t_sgd:.3f} s (epochs={epochs//4})")
print(f"Timing: Mini-batch took {t_mini:.3f} s")
```

### 2) **Make SGD gradient scaling explicit** (option A: leave as per-sample gradient but know lr must be small; option B: normalize per-sample by `n` to directly compare step sizes)

**Option A** — per-sample gradient (keep current behavior but be aware `lr` should be smaller):
```python
# in sgd()
dw = (2.0 / 1) * xi.T.dot(y_pred - yi)  # identical to 2.0 * xi.T.dot(...)
db = float((2.0 / 1) * np.sum(y_pred - yi))
```

**Option B** — scale per-sample gradient to match full-batch normalization (useful for fair comparisons if you want each update size comparable to batch updates):
```python
# in sgd() if you want to normalize by total samples n (less common)
dw = (2.0 / n) * xi.T.dot(y_pred - yi)
db = float((2.0 / n) * np.sum(y_pred - yi))
```
> If you pick B, then you should increase SGD epochs (or learning rate) because each per-sample update becomes much smaller.

### 3) **Pad losses for plotting (already in script but keep this pattern)**
```python
if len(losses_sgd) < epochs:
    losses_sgd += [losses_sgd[-1]] * (epochs - len(losses_sgd))
```

### 4) **Standardize features (recommended)**  
Add before training:
```python
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
```

---

## Further reading / next steps

- Why mini-batches help in deep learning (hardware + optimization dynamics)  
- Try the same experiment with PyTorch or TensorFlow and a GPU — you’ll see mini-batch advantages in runtime.  
- Read about adaptive optimizers (Adam, RMSprop) and batch-normalization.

---

## Short takeaway (one paragraph)

You have one dataset (generated using `true_w = 3.0` and `true_b = 2.0`) and a model that starts with random parameters `w` and `b=0.0`. The model updates `w` and `b` using different update rules. Mini-batch GD typically gives the best practical trade-off (stable, efficient updates) and in your experiments reaches loss near the noise floor (~1.0). Timing differences are heavily influenced by whether operations are vectorized in C (fast) or done in Python loops (slow): full-batch uses very few large vectorized ops and looks fast per epoch in NumPy/CPU, but mini-batch/SGD often win overall on time-to-convergence in larger, GPU-based settings.

---

If you want, I can also create a ready-to-run version of `minibatch_fixed.py` that includes the accurate timing, standardized `X` option, and an optional flag to normalize SGD gradients. Say the word and I will produce that file too.
