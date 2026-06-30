# pyright: reportMissingImports=false, reportUndefinedVariable=false

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


base_dir = Path(__file__).resolve().parent
raw = np.loadtxt(f"{base_dir}/train_accuracy.csv", delimiter=",")

window = 50
kernel = np.ones(window) / window
padded = np.pad(raw, (window // 2, window - 1 - window // 2), mode="edge")
smooth = np.convolve(padded, kernel, mode="valid")

x = np.arange(0, 720100, 100)

mask = x <= 60000
x = x[mask]
raw = raw[mask]
smooth = smooth[mask]

plt.rcParams.update({
    "figure.figsize": (12, 7),
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

plt.plot(x, raw, label="raw", linewidth=2)
plt.plot(x, smooth, label="smoothed", linewidth=2)

# plt.title("Accuracy on MNIST test data during training")
plt.xlabel("batch")
plt.ylabel("accuracy")

ax = plt.gca()
ax.set_yticks(np.arange(0, 1.01, 0.1))

plt.grid(True)
plt.legend(); plt.show()
