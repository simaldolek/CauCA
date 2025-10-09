#!/usr/bin/env python3
# ============================================================
# SYNTHETIC MULTI-SUBJECT DATA GENERATOR (CauCA-based)
# ============================================================

import numpy as np, os, yaml
from pathlib import Path
from tqdm import tqdm
from typing import Tuple


def sample_time_lagged_graph(K, edge_prob, max_lag, w_scale, seed):
    rng = np.random.default_rng(seed)
    W_bin = (rng.random((K, K)) < edge_prob).astype(float)
    W_bin = np.tril(W_bin, k=-1)
    np.fill_diagonal(W_bin, 0.0)
    L = np.zeros_like(W_bin, dtype=int)
    lag_mask = W_bin == 1
    if lag_mask.any():
        L[lag_mask] = rng.integers(1, max_lag + 1, lag_mask.sum())
    W = W_bin * rng.normal(0, w_scale, size=W_bin.shape)
    return W, L

def simulate_latents(W, L, T, sigma_eps=0.2, seed=0):
    rng = np.random.default_rng(seed)
    K = W.shape[0]
    max_lag = int(L.max())
    Z = np.zeros((T + max_lag, K))
    for t in range(max_lag, T + max_lag):
        eps = rng.normal(0, sigma_eps, K)
        for i in range(K):
            parents = np.nonzero(W[i])[0]
            if parents.size == 0:
                Z[t, i] = eps[i]
            else:
                s = sum(W[i, j] * Z[t - L[i, j], j] for j in parents)
                Z[t, i] = np.tanh(s) + eps[i]
    return Z[max_lag:]

def sample_invertible_matrix(d, rng, scale_range=(0.7, 1.3)):
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    s = rng.uniform(scale_range[0], scale_range[1], size=d)
    return Q @ np.diag(s)

def sigma_elem(x, eps=0.5):
    return x + eps * np.tanh(x)

class InvertibleMLP:
    def __init__(self, d, M=3, eps=0.5, seed=0):
        rng = np.random.default_rng(seed)
        self.As = [sample_invertible_matrix(d, rng) for _ in range(M)]
        self.eps = eps
    def forward(self, X):
        H = X
        for A in self.As:
            H = H @ A.T
            H = sigma_elem(H, self.eps)
        return H

# ------------------ PTSD intervention ------------------
def intervene_graph(W: np.ndarray, L: np.ndarray, cfg: dict, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exaggerated intervention: randomly remove a fraction of existing edges and
    add a fraction of brand-new edges, preserving DAG (j<i).
    Returns new (W_ptsd, L_ptsd).
    """
    rng = np.random.default_rng(seed)
    W_new, L_new = W.copy(), L.copy()

    # ----- Removal -----
    existing = np.nonzero(W_new != 0)         # (i_idx, j_idx)
    n_edges = len(existing[0])
    rem_frac = float(cfg.get("removal_fraction", 0.0))
    n_remove = int(round(rem_frac * n_edges))

    removed = []
    if n_remove > 0 and n_edges > 0:
        pick = rng.choice(np.arange(n_edges), size=min(n_remove, n_edges), replace=False)
        for k in pick:
            i, j = existing[0][k], existing[1][k]
            removed.append((int(i), int(j)))
            W_new[i, j] = 0.0
            L_new[i, j] = 0

    # ----- Addition (respect DAG: only j<i) -----
    add_frac = float(cfg.get("addition_fraction", 0.0))
    n_add = int(round(add_frac * n_edges))  # relative to original edge count for clarity

    added = []
    if n_add > 0:
        K = W_new.shape[0]
        # candidate zeros that still respect DAG lower-triangular structure
        cand_i, cand_j = np.where((W_new == 0.0) & (np.tril(np.ones_like(W_new), k=-1).astype(bool)))
        n_cand = len(cand_i)
        if n_cand > 0:
            n_add_eff = min(n_add, n_cand)
            # randomly pick new zero slots to flip to edges
            pick = rng.choice(np.arange(n_cand), size=n_add_eff, replace=False)
            w_scale_new = float(cfg.get("new_edge_w_scale", 0.6))
            max_lag_new = int(cfg.get("new_edge_max_lag", max(1, int(L.max()) if L.size else 3)))
            for k in pick:
                i, j = int(cand_i[k]), int(cand_j[k])
                # add weight and lag
                W_new[i, j] = rng.normal(0.0, w_scale_new)
                L_new[i, j] = rng.integers(1, max_lag_new + 1)
                added.append((i, j))

    return W_new, L_new


# ------------------ main function ------------------
def generate_group(group_name, W, L, cfg, base_seed):
    K = cfg["neural_model"]["K"]
    T = int(round((cfg["acquisition"]["minutes"] * 60) / cfg["acquisition"]["TR"]))
    out_dir = Path(cfg["experiment"]["base_output_dir"]) / group_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for subj in tqdm(range(cfg["experiment"]["n_subjects"]), desc=f"Generating {group_name}"):
        seed = base_seed + subj
        Z = simulate_latents(W, L, T, sigma_eps=cfg["neural_model"]["sigma_eps"], seed=seed)
        f = InvertibleMLP(K, M=cfg["mixing"]["layers"], eps=cfg["mixing"]["eps_sigma"], seed=seed + 100)
        X = f.forward(Z)
        np.save(out_dir / f"sub_{subj:02d}_Z.npy", Z)
        np.save(out_dir / f"sub_{subj:02d}_X.npy", X)

# ------------------ entrypoint ------------------
if __name__ == "__main__":
    with open("configs/synthetic_groups.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    base_seed = cfg["experiment"]["random_seed"]
    base_out = Path(cfg["experiment"]["base_output_dir"])

    # ---------------- HEALTHY CONTROLS ----------------
    hc_dir = base_out / "healthy_controls" / "base_graph"
    hc_dir.mkdir(parents=True, exist_ok=True)

    W_hc_path, L_hc_path = hc_dir / "W_base.npy", hc_dir / "L_base.npy"

    # create or load healthy control base graph
    if not W_hc_path.exists():
        W_hc, L_hc = sample_time_lagged_graph(
            cfg["neural_model"]["K"],
            cfg["neural_model"]["edge_prob"],
            cfg["neural_model"]["max_lag"],
            cfg["neural_model"]["w_scale"],
            base_seed
        )
        np.save(W_hc_path, W_hc)
        np.save(L_hc_path, L_hc)
        print(f" Saved healthy base graph → {hc_dir}")
    else:
        W_hc, L_hc = np.load(W_hc_path), np.load(L_hc_path)
        print(f" Loaded existing healthy base graph from {hc_dir}")

    generate_group("healthy_controls", W_hc, L_hc, cfg, base_seed + 1000)

    # ---------------- PTSD GROUP ----------------
    ptsd_dir = base_out / "ptsd" / "base_graph"
    ptsd_dir.mkdir(parents=True, exist_ok=True)

    W_ptsd_path, L_ptsd_path = ptsd_dir / "W_base.npy", ptsd_dir / "L_base.npy"

    # create or load PTSD base graph (intervention on healthy graph)
    if not W_ptsd_path.exists():
        W_ptsd, L_ptsd = intervene_graph(
            W_hc, L_hc,
            cfg["intervention"],
            seed=base_seed + cfg["intervention"]["seed_offset"]
        )
        np.save(W_ptsd_path, W_ptsd)
        np.save(L_ptsd_path, L_ptsd)
        print(f" Saved PTSD base graph → {ptsd_dir}")
    else:
        W_ptsd, L_ptsd = np.load(W_ptsd_path), np.load(L_ptsd_path)
        print(f" Loaded existing PTSD base graph from {ptsd_dir}")

    generate_group("ptsd", W_ptsd, L_ptsd, cfg, base_seed + 2000)
