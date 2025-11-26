# Full-batch vs Mini-batch Linear Regression (README)

## Overview

This repository contains a NumPy-based experiment that trains a simple linear regression model on synthetic data using two different training strategies:

1. **Full-batch gradient descent** — compute gradients on the entire dataset and perform one update per epoch.
2. **Mini-batch stochastic gradient descent (SGD)** — compute gradients on small shuffled batches and update parameters many times per epoch.

The goal is to compare convergence behaviour (MSE, parameter error) and wall-clock performance between the two approaches. The data is generated from the linear model `y = 3x + 2 + noise` where `noise ~ N(0, 1)`.

---

## Requirements

* Python 3.8+
* NumPy
* Matplotlib

Install dependencies (example):

```bash
pip install numpy matplotlib
```

---

## Files

* `train_compare.py` — main script that generates data, trains using both methods, prints progress, and plots loss curves.
* `README.md` — this file (explanation and output interpretation).

---

## How to run

1. Clone or copy the script to a folder.
2. Ensure dependencies are installed.
3. Run with:

```bash
python train_compare.py
```

Notes:

* By default the script uses `N = 1_000_000` samples and `epochs = 100`. Reduce `N` (e.g., `100_000`) while iterating locally to speed up testing.
* Adjust `batch_size`, `lr` and `epochs` at the top of the script to experiment.

---

## Brief code walkthrough

The script is organized into these blocks:

1. **Imports and utilities** — `time`, `numpy`, `matplotlib` and a helper `compute_mse()` that returns the MSE for given parameters and data.

2. **Data generation**

```py
np.random.seed(0)
N = 1_000_000
x = np.random.uniform(-5, 5, size=(N,1))
w_true, b_true = 3.0, 2.0
y = w_true * x + b_true + np.random.normal(scale=1.0, size=(N,1))
```

* Creates `N` input samples uniformly in `[-5, 5]` and builds noisy targets from the true linear function.

3. **Hyperparameters** — `lr`, `epochs`, `batch_size`, `print_every`.

4. **Full-batch training function `train_full_batch`**

* Initializes `w, b = 0.0`.
* For each epoch:

  * Compute predictions `y_pred = w * X + b` and residuals.
  * Compute MSE `loss = mean((y_pred - y)**2)`.
  * Compute gradients using the entire dataset:

    ```py
    grad_w = (2/N) * sum(x * (y_pred - y))
    grad_b = (2/N) * sum(y_pred - y)
    ```
  * Update parameters `w -= lr * grad_w` and `b -= lr * grad_b`.
  * Optionally print progress (MSE, w, b, and their absolute errors vs true values).

5. **Mini-batch training function `train_minibatch`**

* Similar initialization of `w, b`.
* Each epoch:

  * Shuffle indices using `np.random.permutation(N)`.
  * Break data into `batch_size` chunks; for each batch:

    * Compute batch predictions and batch MSE.
    * Compute gradients on the batch (divide by batch size `m`):

      ```py
      grad_w = (2/m) * sum(xb * (yb_pred - yb))
      grad_b = (2/m) * sum(yb_pred - yb)
      ```
    * Update parameters immediately (SGD style).
  * After all batches, compute epoch-average MSE by weighted sum of batch losses divided by `N`.
  * Optionally print progress.

6. **Run both trainers** — call both functions on the same dataset, compute final full-data MSE using `compute_mse()` for an apples-to-apples comparison, and store loss histories for plotting.

7. **Plotting** — draw loss curves for both methods on a log y-scale and show a short numeric summary of true parameters vs learned parameters.

---

## Output explanation

When you run the script it prints progress lines for both training methods and a final summary. Here's how to read and interpret those outputs.

### Example printed lines (format)

```
[FullBatch] epoch  10 | MSE 6.590975 | w=2.516561 (err 0.483439) | b=0.366754 (err 1.633246)
[MiniBatch] epoch  10 | MSE 0.998526 | w=2.997737 (err 0.002263) | b=2.004618 (err 0.004618)
```

Each printed line includes:

* The method (FullBatch or MiniBatch).
* The epoch number.
* Epoch-average **MSE** (mean squared error) for that epoch.
* Current learned parameters `w` and `b` and their absolute error from the true values (`err = abs(learned - true)`).

### What the numbers mean

* **MSE**: measures average squared difference between model predictions and true `y`. For this dataset the irreducible noise variance is `1.0` (we added noise with `scale=1.0`), so the best achievable MSE is about `1.0` for large `N`.
* **Parameter errors**: show how close the learned `w` and `b` are to the true values (`3.0` and `2.0`).

### Why mini-batch can look "better" after the same number of epochs

* **Updates per epoch differ**: Full-batch does **1 update per epoch**. Mini-batch with `batch_size = 1024` does about `N/1024` updates per epoch (for `N = 1_000_000`, that's ~977 updates per epoch). Therefore after the same number of epochs, mini-batch has performed many more parameter updates, usually moving parameters farther toward the optimum.

* **Mini-batch noise helps early convergence**: stochastic updates can accelerate early progress and help the optimizer escape slow directions. However, they introduce noise which can make loss curves less smooth.

* **Computational overhead**: In pure NumPy+Python, many small batches increase Python-level overhead (indexing/slicing and function calls), so mini-batch might be **slower in wall-clock time** than full-batch for CPU-only NumPy code. In optimized frameworks or GPU setups, mini-batch is often faster wall-clock.

### Interpreting example summary lines

At the end of the script a summary is printed like:

```
True params:       w=3.0, b=2.0
Full-batch params: w=3.000430 (err=0.000430), b=1.734308 (err=0.265692)
Mini-batch params: w=2.998877 (err=0.001123), b=2.002673 (err=0.002673)
```

* This tells you the mini-batch solution is much closer to the true intercept `b` than the full-batch solution after the same number of epochs. The main cause is the huge difference in the number of updates per epoch.

### MSE vs noise floor

* If your MSE is near `1.0` (the noise variance), your model is essentially optimal — it explains the linear signal and remaining error is just the random noise.
* If MSE is significantly above `1.0`, you may need more updates (increase epochs), a larger learning rate (careful), or better optimizer settings.

---

## Quick closed-form check (ordinary least squares)

For this linear regression with a single feature you can compute the exact least-squares solution (no training loop required):

```py
x_flat = x.ravel()
y_flat = y.ravel()
x_mean = x_flat.mean()
y_mean = y_flat.mean()
w_closed = np.sum((x_flat - x_mean) * (y_flat - y_mean)) / np.sum((x_flat - x_mean)**2)
b_closed = y_mean - w_closed * x_mean
mse_closed = np.mean((w_closed * x + b_closed - y)**2)
print("Closed-form: w=", w_closed, " b=", b_closed, " MSE=", mse_closed)
```

This should give `w` very close to `3.0`, `b` very close to `2.0`, and `MSE` close to the noise variance (≈ 1.0).

---

## Suggestions & next experiments

* Try different `lr` values for each method. Mini-batch often uses a slightly smaller learning rate than full-batch.
* Try different `batch_size` values — larger batches reduce Python overhead but give fewer updates per epoch.
* Add momentum or use Adam for faster and more stable convergence.
* Plot `w` and `b` vs epoch to visualize parameter convergence.
* Increase `epochs` for full-batch if you want to reach the same number of updates as mini-batch.

