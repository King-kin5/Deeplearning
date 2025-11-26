# minibatch_fixed.py
import numpy as np
import matplotlib.pyplot as plt
import time

def set_seed(seed=0):
    np.random.seed(seed)

def generate_linear_data(n_samples=20000, noise_std=1.0):
    X = np.random.uniform(-3, 3, size=(n_samples, 1))
    true_w, true_b = 3.0, 2.0
    noise = np.random.normal(0, noise_std, size=(n_samples, 1))
    y = true_w * X + true_b + noise
    return X, y, true_w, true_b

def mse_loss(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def predict_linear(X, w, b):
    return X.dot(w) + b

def batch_gd(X, y, lr=0.01, epochs=100):
    n = X.shape[0]
    w = np.random.randn(1,1) * 0.1
    b = 0.0
    losses = []
    for epoch in range(epochs):
        y_pred = predict_linear(X, w, b)
        losses.append(mse_loss(y, y_pred))
        # gradients (averaged over whole dataset)
        dw = (2.0 / n) * X.T.dot(y_pred - y)           # shape (1,1)
        db = float((2.0 / n) * np.sum(y_pred - y))    # scalar Python float
        # updates
        w -= lr * dw
        b -= lr * db
    return w, b, losses

def sgd(X, y, lr=0.005, epochs=30):
    n = X.shape[0]
    w = np.random.randn(1,1) * 0.1
    b = 0.0
    losses = []
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        for i in perm:
            xi = X[i:i+1]   # shape (1,1)
            yi = y[i:i+1]
            y_pred = predict_linear(xi, w, b)
            # per-sample gradient (batch size = 1)
            dw = 2.0 * xi.T.dot(y_pred - yi)    # (1,1)
            db = float(2.0 * np.sum(y_pred - yi))
            w -= lr * dw
            b -= lr * db
        # measure full-training loss at end of epoch
        losses.append(mse_loss(y, predict_linear(X, w, b)))
    return w, b, losses

def minibatch_gd(X, y, lr=0.01, epochs=100, batch_size=32):
    n = X.shape[0]
    w = np.random.randn(1,1) * 0.1
    b = 0.0
    losses = []
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X[idx], y[idx]
            y_pred = predict_linear(xb, w, b)
            dw = (2.0 / xb.shape[0]) * xb.T.dot(y_pred - yb)
            db = float((2.0 / xb.shape[0]) * np.sum(y_pred - yb))
            w -= lr * dw
            b -= lr * db
        losses.append(mse_loss(y, predict_linear(X, w, b)))
    return w, b, losses

def main():
    set_seed(1)
    X, y, true_w, true_b = generate_linear_data(n_samples=2000, noise_std=1.0)

    epochs = 100

    # batch GD
    t0 = time.perf_counter()
    w_batch, b_batch, losses_batch = batch_gd(X, y, lr=0.01, epochs=epochs)
    t_batch = time.perf_counter() - t0

    # SGD: fewer epochs is common for per-sample updates; make it explicit
    sgd_epochs = epochs // 4
    t0 = time.perf_counter()
    w_sgd, b_sgd, losses_sgd = sgd(X, y, lr=0.005, epochs=sgd_epochs)
    t_sgd = time.perf_counter() - t0

    # mini-batch GD
    t0 = time.perf_counter()
    w_mini, b_mini, losses_mini = minibatch_gd(X, y, lr=0.01, epochs=epochs, batch_size=32)
    t_mini = time.perf_counter() - t0

    print(f"Timing: batch_gd took {t_batch:.3f} s")
    print(f"Timing: SGD took {t_sgd:.3f} s (epochs={sgd_epochs})")
    print(f"Timing: Mini-batch took {t_mini:.3f} s")

    # quick checks before plotting
    print("len(losses_batch) =", len(losses_batch))
    print("len(losses_mini)  =", len(losses_mini))
    print("len(losses_sgd)   =", len(losses_sgd))

    print(f"\nTrue params: w={true_w}, b={true_b}\n")
    print("Batch GD:   w={:.4f}, b={:.4f}, final loss={:.6f}".format(float(w_batch.squeeze()), float(b_batch), losses_batch[-1]))
    print("SGD:        w={:.4f}, b={:.4f}, final loss={:.6f}".format(float(w_sgd.squeeze()), float(b_sgd), losses_sgd[-1]))
    print("Mini-batch: w={:.4f}, b={:.4f}, final loss={:.6f}".format(float(w_mini.squeeze()), float(b_mini), losses_mini[-1]))

    # ensure lengths match (pad SGD if needed so we can plot on same x-axis)
    if len(losses_sgd) < epochs:
        losses_sgd += [losses_sgd[-1]] * (epochs - len(losses_sgd))

    plt.figure(figsize=(9,5))
    plt.plot(range(1, epochs+1), losses_batch, label='Batch GD (full)')
    plt.plot(range(1, epochs+1), losses_mini, label='Mini-batch GD (bs=32)')
    plt.plot(range(1, epochs+1), losses_sgd, label=f'SGD (per-sample, epochs={sgd_epochs})')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (training)")
    plt.title("Linear regression: Batch vs SGD vs Mini-batch")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
