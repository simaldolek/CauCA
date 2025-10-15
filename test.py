#!/usr/bin/env python3
# ============================================================
# PLOT LATENT (Z) AND OBSERVED (X) TIME SERIES & RELATIONSHIPS
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- User parameters ----------
base_dir = Path("generatedSyntheticData/healthy_controls")  # <-- change to match your cfg["experiment"]["base_output_dir"]
subject_id = 0  # choose which subject to visualize

# ---------- Load data ----------
Z_path = base_dir / f"sub_{subject_id:02d}_Z.npy"
X_path = base_dir / f"sub_{subject_id:02d}_X.npy"

Z = np.load(Z_path)
X = np.load(X_path)

T, K = Z.shape
print(f"Loaded subject {subject_id}: {K} variables, {T} time points")

# ---------- 1. Plot latent variables over time ----------
plt.figure(figsize=(10, 5))
for i in range(K):
    plt.plot(Z[:, i], label=f"Z{i+1}")
plt.xlabel("Time (t)")
plt.ylabel("Latent activity")
plt.title(f"Latent variables (Z) for subject {subject_id:02d}")
plt.legend()
plt.tight_layout()
plt.show()

# ---------- 2. Plot observed variables over time ----------
plt.figure(figsize=(10, 5))
for i in range(K):
    plt.plot(X[:, i], label=f"X{i+1}")
plt.xlabel("Time (t)")
plt.ylabel("Observed signal")
plt.title(f"Observed variables (X = f(Z)) for subject {subject_id:02d}")
plt.legend()
plt.tight_layout()
plt.show()

# ---------- 3. Plot pairwise scatter plots ----------
pairs_to_plot = [(0, 1), (1, 2), (0, 2)]  # (Z1 vs Z2, Z2 vs Z3, etc.)

fig, axes = plt.subplots(1, len(pairs_to_plot), figsize=(15, 4))
for idx, (i, j) in enumerate(pairs_to_plot):
    ax = axes[idx]
    ax.scatter(Z[:, i], Z[:, j], s=10, alpha=0.6, label=f"Z{i+1} vs Z{j+1}", color="royalblue")
    ax.set_xlabel(f"Z{i+1}")
    ax.set_ylabel(f"Z{j+1}")
    ax.set_title(f"Z{i+1} vs Z{j+1}")
plt.suptitle(f"Pairwise latent relationships (subject {subject_id:02d})")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, len(pairs_to_plot), figsize=(15, 4))
for idx, (i, j) in enumerate(pairs_to_plot):
    ax = axes[idx]
    ax.scatter(X[:, i], X[:, j], s=10, alpha=0.6, label=f"X{i+1} vs X{j+1}", color="darkorange")
    ax.set_xlabel(f"X{i+1}")
    ax.set_ylabel(f"X{j+1}")
    ax.set_title(f"X{i+1} vs X{j+1}")
plt.suptitle(f"Pairwise observed relationships (subject {subject_id:02d})")
plt.tight_layout()
plt.show()

# ---------- 4. Optional: overlay Z vs X for one node ----------
node_id = 0  # pick a node (e.g., first variable)
plt.figure(figsize=(10, 4))
plt.plot(Z[:, node_id], label=f"Z{node_id+1} (latent)", color="steelblue")
plt.plot(X[:, node_id], label=f"X{node_id+1} (observed)", color="darkorange", alpha=0.7)
plt.xlabel("Time (t)")
plt.ylabel("Signal amplitude")
plt.title(f"Comparison: Z{node_id+1} vs X{node_id+1}")
plt.legend()
plt.tight_layout()
plt.show()


