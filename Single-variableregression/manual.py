import time
import numpy as np
import matplotlib.pyplot as plt

# ---------- data ----------
np.random.seed(0)
N = 1000000          
x = np.random.uniform(-5, 5, size=(N, 1))
w_true, b_true = 3.0, 2.0
y = w_true * x + b_true + np.random.normal(scale=1.0, size=(N, 1))

# ---------- training hyperparams ----------
lr = 0.05
epochs = 100
print_every = 10     

# ---------- helper ----------
def compute_mse(w, b, X, Y):
    pred = w * X + b
    err = pred - Y
    return float(np.mean(err ** 2))

# ---------- Full-batch gradient descent ----------
def train_full_batch(X, Y, lr=0.01, epochs=100, print_every=10):
    N = X.shape[0]
    w, b = 0.0, 0.0
    loss_history = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        y_pred = w * X + b
        errors = y_pred - Y               # shape (N,1)
        loss = float(np.mean(errors ** 2))
        # gradients (computed on whole dataset)
        grad_w = (2.0 / N) * np.sum(X * errors)
        grad_b = (2.0 / N) * np.sum(errors)
        # update
        w -= lr * grad_w
        b -= lr * grad_b
        loss_history.append(loss)
        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            w_err = abs(w - w_true)
            b_err = abs(b - b_true)
            print(f"[FullBatch] epoch {epoch:3d} | MSE {loss:.6f} | w={w:.6f} (err {w_err:.6f}) | b={b:.6f} (err {b_err:.6f})")
    dt = time.time() - t0
    print(f"[FullBatch] finished in {dt:.2f}s")
    return w, b, loss_history

# ---------- Mini-batch SGD ----------
def train_minibatch(X, Y, lr=0.01, epochs=100, batch_size=1024, print_every=10):
    N = X.shape[0]
    w, b = 0.0, 0.0
    loss_history = []
    t0 = time.time()
    # Precompute number of batches
    n_batches = int(np.ceil(N / batch_size))
    for epoch in range(1, epochs + 1):
        # shuffle indices each epoch
        perm = np.random.permutation(N)
        epoch_loss_sum = 0.0   # to compute epoch-average loss
        for i in range(n_batches):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            xb = X[idx]
            yb = Y[idx]
            y_pred_b = w * xb + b
            errors_b = y_pred_b - yb
            batch_loss = float(np.mean(errors_b ** 2))
            epoch_loss_sum += batch_loss * xb.shape[0] 
            # gradients computed on this batch
            m = xb.shape[0]
            grad_w = (2.0 / m) * np.sum(xb * errors_b)
            grad_b = (2.0 / m) * np.sum(errors_b)
            # update immediately (SGD style)
            w -= lr * grad_w
            b -= lr * grad_b
        epoch_loss = epoch_loss_sum / N
        loss_history.append(epoch_loss)
        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            w_err = abs(w - w_true)
            b_err = abs(b - b_true)
            print(f"[MiniBatch] epoch {epoch:3d} | MSE {epoch_loss:.6f} | w={w:.6f} (err {w_err:.6f}) | b={b:.6f} (err {b_err:.6f})")
    dt = time.time() - t0
    print(f"[MiniBatch] finished in {dt:.2f}s")
    return w, b, loss_history

# ---------- run both ----------
print("Starting Full-batch training...")
w_fb, b_fb, loss_fb = train_full_batch(x, y, lr=lr, epochs=epochs, print_every=print_every)
mse_fb = compute_mse(w_fb, b_fb, x, y)
print(f"Full-batch final: w={w_fb:.6f}, b={b_fb:.6f}, MSE(full)={mse_fb:.6f}\n")

print("Starting Mini-batch training...")
batch_size = 1024
w_mb, b_mb, loss_mb = train_minibatch(x, y, lr=lr, epochs=epochs, batch_size=batch_size, print_every=print_every)
mse_mb = compute_mse(w_mb, b_mb, x, y)
print(f"Mini-batch final: w={w_mb:.6f}, b={b_mb:.6f}, MSE(full)={mse_mb:.6f}\n")

# ---------- compare & plot ----------
plt.figure(figsize=(8,5))
plt.plot(loss_fb, label="Full-batch")
plt.plot(loss_mb, label=f"Mini-batch (bs={batch_size})")
plt.yscale("log")
plt.xlabel("epoch")
plt.ylabel("MSE (log scale)")
plt.title("Training loss: full-batch vs mini-batch")
plt.legend()  
plt.grid(True, linestyle=":", alpha=0.4)
plt.show()

# Print a final summary of parameter errors
print("Summary:")
print(f" True params:       w={w_true:.6f}, b={b_true:.6f}")
print(f" Full-batch params: w={w_fb:.6f} (err={abs(w_fb-w_true):.6f}), b={b_fb:.6f} (err={abs(b_fb-b_true):.6f})")
print(f" Mini-batch params: w={w_mb:.6f} (err={abs(w_mb-w_true):.6f}), b={b_mb:.6f} (err={abs(b_mb-b_true):.6f})")
