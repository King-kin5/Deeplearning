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

