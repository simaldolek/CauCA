#!/usr/bin/env python3
# Single-file synthetic fMRI generator (causal latents -> HRF -> BOLD)

import argparse
import numpy as np
from scipy.signal import fftconvolve

# ----- graph_sampler.py -----
def sample_time_lagged_graph(K: int,
                             edge_prob: float = 0.15,
                             max_lag: int = 3,
                             w_scale: float = 0.8,
                             seed: int = 0):
    """
    Returns:
      W: (K,K) weights for influences j -> i (zero diag).
      L: (K,K) integer lags >=1 where W[i,j]!=0, else 0.
    Notes:
      Small edge_prob + max_lag keep it stable; we also scale weights.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(0, w_scale, size=(K, K)) * (rng.random((K, K)) < edge_prob)
    np.fill_diagonal(W, 0.0)
    # Bias towards stability: shrink spectral radius-ish
    W *= 0.5 / max(1.0, np.linalg.norm(W, ord=2))
    L = np.zeros_like(W, dtype=int)
    # Assign lags (only where edges exist)
    lag_mask = (W != 0)
    if lag_mask.any():
        L[lag_mask] = rng.integers(1, max_lag + 1, size=lag_mask.sum())
    return W, L

# ----- noise_generator.py -----
def ar1_noise(T: int, K: int, phi: float = 0.4, sigma: float = 0.5, seed: int = 1):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, sigma, size=(T, K))
    for t in range(1, T):
        e[t] += phi * e[t-1]
    return e

def white_noise(T: int, P: int, sigma: float = 0.1, seed: int = 2):
    rng = np.random.default_rng(seed)
    return rng.normal(0, sigma, size=(T, P))

# ----- scm.py -----
def simulate_latents(W, L, T: int,
                     nonlin: str = "tanh",
                     exo_noise=None,
                     seed: int = 3):
    """
    Simulate latent time series Z using causal, lagged influences.
    Z[t,i] = f( sum_j W[i,j]*Z[t-L[i,j],j] + u[t,i] ), with t >= maxLag
    """
    rng = np.random.default_rng(seed)
    K = W.shape[0]
    maxLag = int(L.max()) if L.size else 0
    Z = np.zeros((T, K))
    if exo_noise is None:
        exo_noise = ar1_noise(T, K, phi=0.3, sigma=0.7, seed=seed+1)

    def f(x):
        if nonlin == "tanh":
            return np.tanh(x)
        if nonlin == "relu":
            return np.maximum(0, x)
        return x  # identity

    # Simulate
    for t in range(maxLag, T):
        drive = exo_noise[t]
        if maxLag > 0:
            for i in range(K):
                s = 0.0
                parents = np.where(W[i] != 0)[0]
                if parents.size:
                    for j in parents:
                        lag = L[i, j]
                        if lag > 0 and t - lag >= 0:
                            s += W[i, j] * Z[t - lag, j]
                Z[t, i] = f(s + drive[i])
        else:
            Z[t] = f(drive)
    return Z

# ----- mixing_function.py -----
def linear_mixer(Z, P: int, nonlin: str = "identity",
                 obs_sigma: float = 0.05, seed: int = 4):
    """
    Observed X = g(M @ Z^T)^T + eps, where M ∈ R^{P×K}.
    Returns X ∈ R^{T×P}, M.
    """
    rng = np.random.default_rng(seed)
    T, K = Z.shape
    M = rng.normal(0, 1.0 / np.sqrt(K), size=(P, K))
    Y = Z @ M.T  # (T,K)@(K,P) -> (T,P)
    if nonlin == "tanh":
        Y = np.tanh(Y)
    elif nonlin == "relu":
        Y = np.maximum(0, Y)
    X = Y + white_noise(T, P, sigma=obs_sigma, seed=seed+1)
    return X, M

# ----- data_module.py (simple wrapper) -----
class CausalGenerator:
    def __init__(self, K=12, P=20, T=2000, seed=0):
        self.K, self.P, self.T, self.seed = K, P, T, seed

    def generate(self):
        W, L = sample_time_lagged_graph(self.K, seed=self.seed)
        Z = simulate_latents(W, L, self.T, nonlin="tanh", seed=self.seed+10)
        X, M = linear_mixer(Z, self.P, nonlin="identity", obs_sigma=0.05, seed=self.seed+20)
        return dict(Z=Z, X=X, W=W, L=L, M=M)

# ----- HRF / fMRI bits -----
# --- HRF: SPM-like double gamma (discrete-time) ---
def spm_hrf(tr: float, dt: float = 0.1, duration: float = 32.0,
            p1=6, p2=16, k1=1.0, k2=1/6):
    """
    Returns an HRF sampled at 'dt' (finer grid), intended to be used
    before downsampling to TR. Shape is roughly 0..duration/dt.
    """
    t = np.arange(0, duration, dt)
    # gamma pdfs
    def g(t, p):  # shape p, scale 1
        from scipy.special import gamma
        return (t**(p-1) * np.exp(-t)) / gamma(p)
    h = k1 * g(t, p1) - k2 * g(t, p2)
    h /= np.sum(h)  # normalize area
    return t, h

def resample_to_TR(signal_dt, dt, TR, T_out):
    """
    Downsample a dt-sampled signal to TR frames (nearest index).
    """
    n = int(round(TR / dt))
    idx = np.arange(0, T_out) * n
    idx = np.clip(idx, 0, len(signal_dt)-1)
    return signal_dt[idx]

# --- Low-frequency drift (1/f-like) ---
def pinkish_drift(T, strength=0.3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=T)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(T, d=1.0)
    filt = 1.0 / np.maximum(freqs, 1e-3)
    Y = X * filt
    y = np.fft.irfft(Y, n=T)
    y = strength * (y / np.std(y))
    return y

# --- Motion spikes ---
def motion_spikes(T, n_spikes=3, amp=1.0, seed=0):
    rng = np.random.default_rng(seed)
    if T < 20 or n_spikes <= 0:
        return np.zeros(T)
    times = rng.choice(np.arange(10, T-10), size=min(n_spikes, max(1, T//50)), replace=False)
    s = np.zeros(T)
    for t in times:
        s[t] += amp
        if t+1 < T: s[t+1] += 0.5*amp
        if t+2 < T: s[t+2] += 0.25*amp
    return s

# --- Full fMRI generator on top of the previous SCM ---
def generate_fmri_bold(W, L,
                       minutes=10.0,
                       TR=2.0,
                       neural_dt=0.1,
                       nonlin="tanh",
                       ar_phi=0.4,
                       ar_sigma=0.25,
                       obs_sigma=0.15,
                       global_strength=0.2,
                       drift_strength=0.3,
                       spike_count=3,
                       burn_in_s=20.0,
                       seed=123):
    """
    Returns dict with BOLD (T×K), neural Z_fast (n_fast × K), HRF, params.
    Pipeline:
      - Simulate neural latents at dt=neural_dt with extra burn-in
      - Convolve with HRF, downsample to TR
      - Add AR(1), pink drift, global, motion spikes, white noise
      - Mild spatial cross-talk via near-identity M
    """
    rng = np.random.default_rng(seed)
    K = W.shape[0]
    T_out = int(np.floor(minutes*60.0 / TR))            # fMRI volumes
    T_fast = int(np.ceil((minutes*60.0 + burn_in_s) / neural_dt))  # neural steps incl burn-in

    # 1) simulate fast neural latents (reuse SCM nonlin & AR exogenous noise)
    exo = ar1_noise(T_fast, K, phi=0.3, sigma=0.7, seed=seed+1)
    Z_fast_full = simulate_latents(W, L, T_fast, nonlin=nonlin, exo_noise=exo, seed=seed+2)

    # Drop burn-in before HRF/downsampling
    burn_idx = int(round(burn_in_s / neural_dt))
    Z_fast = Z_fast_full[burn_idx:]

    # 2) HRF convolution per node at fast rate, then downsample to TR
    _, h = spm_hrf(TR, dt=neural_dt, duration=32.0)
    T_fast_eff = Z_fast.shape[0]
    BOLD = np.zeros((T_out, K))
    for k in range(K):
        conv = fftconvolve(Z_fast[:, k], h, mode='full')[:T_fast_eff]
        BOLD[:, k] = resample_to_TR(conv, dt=neural_dt, TR=TR, T_out=T_out)

    # 3) Add realistic BOLD noise components
    ar = ar1_noise(T_out, K, phi=ar_phi, sigma=ar_sigma, seed=seed+3)
    g = pinkish_drift(T_out, strength=global_strength, seed=seed+4)[:, None]  # (T,1)
    drift = np.stack([pinkish_drift(T_out, strength=drift_strength, seed=seed+5+i)
                      for i in range(K)], axis=1)
    spikes = motion_spikes(T_out, n_spikes=spike_count, amp=1.0, seed=seed+6)[:, None]
    spike_scales = rng.lognormal(mean=0.0, sigma=0.2, size=(1, K))
    spikes = spikes * spike_scales
    eps = white_noise(T_out, K, sigma=obs_sigma, seed=seed+7)

    X = BOLD + ar + g + drift + spikes + eps

    # Optional mild spatial cross-talk (ROI mixing)
    M = np.eye(K) + 0.05 * rng.normal(0, 1/np.sqrt(K), size=(K, K))
    X_obs = X @ M.T

    return dict(
        BOLD=X_obs,          # observed ROI time series (T_out × K)
        BOLD_clean=BOLD,     # convolved, downsampled, pre-noise
        Z_fast=Z_fast,       # fast neural latents after burn-in (≈ minutes*60/neural_dt × K)
        W=W, L=L, M=M,
        TR=TR, minutes=minutes, neural_dt=neural_dt,
        components=dict(ar=ar, global=g, drift=drift, spikes=spikes, white=eps),
        hrf=h
    )

# ---------- Command-line entry ----------
def main():
    parser = argparse.ArgumentParser(description="Synthetic healthy fMRI BOLD (100 ROIs).")
    parser.add_argument("--K", type=int, default=100, help="Number of ROIs (default: 100)")
    parser.add_argument("--minutes", type=float, default=10.0, help="Duration in minutes (default: 10)")
    parser.add_argument("--TR", type=float, default=0.8, help="Repetition time in seconds (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    args = parser.parse_args()

    # 1) sample a latent graph among K ROIs
    W, L = sample_time_lagged_graph(args.K, edge_prob=0.05, max_lag=3, w_scale=0.6, seed=7)

    # 2) generate BOLD
    sim = generate_fmri_bold(
        W, L,
        minutes=args.minutes,
        TR=args.TR,
        neural_dt=0.1,
        nonlin="tanh",
        ar_phi=0.35,
        ar_sigma=0.2,
        obs_sigma=0.1,
        global_strength=0.15,
        drift_strength=0.25,
        spike_count=2,
        burn_in_s=20.0,
        seed=args.seed
    )

    X = sim["BOLD"]         # (T × K)
    Z = sim["Z_fast"]       # fast neural sources
    TR = sim["TR"]

    # Print quick summary
    print(f"BOLD shape: {X.shape}  |  Z_fast shape: {Z.shape}  |  TR: {TR:.3f}s")
    print(f"W/L shapes: {sim['W'].shape}/{sim['L'].shape}  |  M shape: {sim['M'].shape}")

    # Save outputs
    np.save("bold.npy", X)
    np.save("bold_clean.npy", sim["BOLD_clean"])
    np.save("latents_fast.npy", Z)
    np.save("W.npy", sim["W"])
    np.save("L.npy", sim["L"])
    np.save("M.npy", sim["M"])
    print("Saved: bold.npy, bold_clean.npy, latents_fast.npy, W.npy, L.npy, M.npy")

if __name__ == "__main__":
    main()
