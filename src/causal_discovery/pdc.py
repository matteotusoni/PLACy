#! Baccalà & Sameshima, 2001

import numpy as np
from numpy.linalg import inv, pinv
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Iterable
import os

# =========================
# VAR estimate via _lag_design (same style as your code)
# =========================

def _lag_design(process: np.ndarray, p: int):
    """
    Build the lag design matrix for OLS.
    process: (n_vars, T)
    Returns:
      X  : (T-p, 1 + n_vars*p)  [intercept + (var-major) p lags blocks]
      Ys : list of length n_vars, each Y_j with shape (T-p,)
    """
    n_vars, T = process.shape
    rows = T - p
    X = np.ones((rows, 1 + n_vars * p), dtype=float)
    col = 1
    for v in range(n_vars):
        for k in range(1, p + 1):
            X[:, col] = process[v, p - k: T - k]
            col += 1
    Ys = [process[j, p:T].astype(float) for j in range(n_vars)]
    return X, Ys


def _estimate_var_coeffs(process: np.ndarray, p: int) -> Tuple[np.ndarray, float]:
    """
    Estimate VAR(p) with OLS, equation-by-equation, reusing _lag_design.
    Returns:
      A_list: array with shape (p, n_vars, n_vars) = coefficient matrices A_k
              such that X_t = sum_k A_k X_{t-k} + eps_t
      sigma2_mean: mean residual variance across equations (not used by PDC but handy)
    """
    n_vars, T = process.shape
    X, Ys = _lag_design(process, p)
    XtX = X.T @ X
    XtX_inv = inv(XtX)
    n, q = X.shape

    A_list = np.zeros((p, n_vars, n_vars), dtype=float)
    sigmas = []
    for i in range(n_vars):
        y = Ys[i]
        beta = XtX_inv @ (X.T @ y)
        resid = y - X @ beta
        sigma2 = float((resid @ resid) / (n - q))
        sigmas.append(sigma2)

        # reconstruct A_k: row i, col j for each lag k
        # columns in X: [intercept] + for var j=0..n-1: (lag1..lagp)
        for j in range(n_vars):
            base = 1 + j * p
            for k in range(1, p + 1):
                A_list[k - 1, i, j] = beta[base + (k - 1)]

    return A_list, float(np.mean(sigmas))


def _A_bar_of_omega(A_list: np.ndarray, omega: float) -> np.ndarray:
    """
    Ā(ω) = I - sum_{k=1}^p A_k e^{-i ω k}
    """
    p, n, _ = A_list.shape
    Abar = np.eye(n, dtype=complex)
    for k in range(1, p + 1):
        Abar -= A_list[k - 1] * np.exp(-1j * omega * k)
    return Abar


def pdc_spectrum(process: np.ndarray,
                 p: int = 1,
                 n_freqs: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the PDC(ω) spectrum for all pairs (j->i).
    Returns:
      omegas : (m,) frequencies in radians
      PDC    : (n, n, m) with PDC[i,j,ℓ] = PDC_{j->i}(ω_ℓ)
    """
    n_vars, T = process.shape
    assert T > p + 5, "Time series too short for the chosen lag order."
    eps = 1e-6
    omegas = np.linspace(eps, np.pi - eps, n_freqs)
    A_list, _ = _estimate_var_coeffs(process, p)

    P = np.zeros((n_vars, n_vars, n_freqs), dtype=float)
    for idx, om in enumerate(omegas):
        Abar = _A_bar_of_omega(A_list, om)  # (n,n) complex
        # column normalization (driver j)
        denom = np.sqrt(np.sum(np.abs(Abar)**2, axis=0, keepdims=True)) + 1e-12
        P[:, :, idx] = np.abs(Abar) / denom
    return omegas, P


def plot_pdc_spectrum_pair(process: np.ndarray,
                           causing: int,
                           caused: int,
                           p: int = 1,
                           n_freqs: int = 256,
                           title: str = None,
                           save: bool = False,
                           out_dir: str = "plot_pdc",
                           show: bool = True) -> str:
    """
    Plot PDC for a single pair (causing -> caused).
    """
    omegas, P = pdc_spectrum(process, p=p, n_freqs=n_freqs)
    vals = P[caused, causing, :]

    plt.figure()
    plt.plot(omegas, vals, label=f"PDC {causing}→{caused}")
    plt.xlabel("Frequency ω (radians)")
    plt.ylabel("PDC")
    if title is None:
        title = f"PDC(ω): {causing} → {caused} (p={p}, m={n_freqs})"
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    filepath = ""
    if save:
        os.makedirs(out_dir, exist_ok=True)
        fname = f"pdc_{causing}_to_{caused}_p{p}_n{n_freqs}.png"
        filepath = os.path.join(out_dir, fname)
        plt.savefig(filepath, dpi=200)
        if not show:
            plt.close()

    if show:
        plt.show()
    elif not save:
        plt.close()

    return filepath


# =========================
# Bandwise test via surrogates (time-shift)
# =========================

def _circular_shift(x: np.ndarray, shift: int) -> np.ndarray:
    return np.roll(x, shift)

def _band_indices(omegas: np.ndarray, wmin: float, wmax: float) -> np.ndarray:
    return np.where((omegas >= wmin) & (omegas <= wmax))[0]

def pdc_causality(process: np.ndarray,
                  p: int = 1,
                  n_freqs: int = 256,
                  bands: Tuple[Tuple[float, float], ...] = ((1e-3, np.pi/3),
                                                            (np.pi/3, 2*np.pi/3),
                                                            (2*np.pi/3, np.pi-1e-3)),
                  alpha: float = 0.05,
                  stat: str = "max",          # "max" or "mean" of PDC within band
                  n_perm: int = 200,          # time-shift surrogates on the driver
                  min_coverage_bands: float = 0.0,  # required fraction of significant bands
                  auto_plot: bool = False,
                  out_dir: str = "plot_pdc",
                  show_plots: bool = False) -> np.ndarray:
    """
    Estimate VAR(p), compute PDC(ω), and decide edges per band using surrogate (driver shift) tests.
    Band decision: upper-tail p_perm for the chosen statistic; significant if p_perm < alpha.
    Global decision: set edge j->i to 1 if the fraction of significant bands >= min_coverage_bands
                     (if 0 -> at least one band is enough).
    Returns: A_bin (n_vars x n_vars).
    """
    n_vars, T = process.shape
    assert T > p + 5, "Time series too short."
    eps = 1e-6
    omegas, P = pdc_spectrum(process, p=p, n_freqs=n_freqs)  # P[i,j,ℓ]

    # Prepare plotting
    if auto_plot:
        os.makedirs(out_dir, exist_ok=True)

    # Precompute band indices
    band_idxs = []
    for (wmin, wmax) in bands:
        idx = _band_indices(omegas, max(eps, wmin), min(np.pi - eps, wmax))
        band_idxs.append(idx)

    # Band statistic (observed)
    def band_stat(vals: np.ndarray, idxs: np.ndarray) -> float:
        if idxs.size == 0:
            return 0.0
        if stat == "mean":
            return float(np.mean(vals[idxs]))
        else:
            return float(np.max(vals[idxs]))

    A_bin = np.zeros((n_vars, n_vars), dtype=int)

    # Surrogates: for each pair, shift only the driver j and recompute PDC for that pair
    rng = np.random.default_rng(0)

    for caused in range(n_vars):
        for causing in range(n_vars):
            if causing == caused:
                continue

            # observed
            pdc_curve = P[caused, causing, :]  # (m,)
            stats_obs = [band_stat(pdc_curve, idx) for idx in band_idxs]

            # surrogate distribution (upper-tail)
            perm_stats = np.zeros((len(bands), n_perm), dtype=float)
            for b, idxs in enumerate(band_idxs):
                if idxs.size == 0:
                    perm_stats[b, :] = 0.0
                    continue
                for r in range(n_perm):
                    s = int(rng.integers(low=1, high=T-1))
                    x_shifted = _circular_shift(process[causing], s)

                    # More correct (but costlier) version: re-estimate VAR with the shifted driver
                    proc_perm = process.copy()
                    proc_perm[causing] = x_shifted
                    A_list_perm, _ = _estimate_var_coeffs(proc_perm, p)

                    # PDC just for this pair/frequency indices
                    vals_perm = np.empty_like(idxs, dtype=float)
                    for ii, om_idx in enumerate(idxs):
                        om = omegas[om_idx]
                        Abar_perm = _A_bar_of_omega(A_list_perm, om)
                        denom_perm = np.sqrt(np.sum(np.abs(Abar_perm)**2, axis=0)) + 1e-12
                        vals_perm[ii] = np.abs(Abar_perm[caused, causing]) / denom_perm[causing]
                    perm_stats[b, r] = float(np.max(vals_perm) if stat == "max" else np.mean(vals_perm))

            # per-band p-values (upper-tail): p = (1 + #surrogates >= stat_obs) / (1 + n_perm)
            pvals_bands = []
            for b, s_obs in enumerate(stats_obs):
                ge = np.sum(perm_stats[b, :] >= s_obs)
                pval = (1.0 + ge) / (1.0 + n_perm)
                pvals_bands.append(pval)

            # global decision for this pair
            sig_bands = sum(p < alpha for p in pvals_bands)
            denom = sum(len(idx) > 0 for idx in band_idxs)
            if denom == 0:
                A_bin[causing, caused] = 0
            else:
                needed = max(1, int(np.ceil(min_coverage_bands * denom))) if min_coverage_bands > 0 else 1
                A_bin[causing, caused] = int(sig_bands >= needed)

            # optional plotting
            if auto_plot:
                plt.figure()
                plt.plot(omegas, pdc_curve, label=f"PDC {causing}→{caused}")
                # highlight bands
                for (wmin, wmax) in bands:
                    plt.axvspan(wmin, wmax, alpha=0.1)
                plt.xlabel("Frequency ω (radians)")
                plt.ylabel("PDC")
                plt.title(f"PDC(ω) & bands: {causing}→{caused}")
                plt.legend()
                plt.tight_layout()
                os.makedirs(out_dir, exist_ok=True)
                fname = f"pdc_{causing}_to_{caused}.png"
                plt.savefig(os.path.join(out_dir, fname), dpi=200)
                if not show_plots:
                    plt.close()

    np.fill_diagonal(A_bin, 0)
    return A_bin
