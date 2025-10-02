#! Korzeniewska et al. (2003).

import numpy as np
from numpy.linalg import inv, pinv
import matplotlib.pyplot as plt
from typing import Tuple
import os

# ------------------------------------------------------------
# Utility: lag design (paste here if you haven't imported it elsewhere)
# ------------------------------------------------------------
def _lag_design(process: np.ndarray, p: int):
    """
    Build the lag design matrix for OLS.
    process: (n_vars, T)
    Returns:
      X  : (T-p, 1 + n_vars*p)  [intercept + (var-major) p-lag blocks]
      Ys : list of length n_vars with each Y_j shaped (T-p,)
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

# ------------------------------------------------------------
# VAR(p) via OLS equation-by-equation
# ------------------------------------------------------------
def _estimate_var_coeffs(process: np.ndarray, p: int):
    """
    Returns:
      A_list      : array (p, n_vars, n_vars) with the A_k coefficients
      sigma2_mean : mean residual variance (optional/useful)
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

def _H_of_omega(A_list: np.ndarray, omega: float) -> np.ndarray:
    """
    H(ω) = Ā(ω)^{-1}. Use pinv if ill-conditioned.
    """
    Abar = _A_bar_of_omega(A_list, omega)
    try:
        if np.linalg.cond(Abar) < 1e12:
            return inv(Abar)
        else:
            return pinv(Abar)
    except np.linalg.LinAlgError:
        return pinv(Abar)

# ------------------------------------------------------------
# DTF spectrum for all pairs
# ------------------------------------------------------------
def dtf_spectrum(process: np.ndarray,
                 p: int = 1,
                 n_freqs: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute DTF(ω) for all pairs (j->i).
    Returns:
      omegas : (m,) frequencies (radians)
      DTF    : (n, n, m) with DTF[i,j,ℓ] = DTF_{j->i}(ω_ℓ)
    """
    n_vars, T = process.shape
    assert T > p + 5, "Time series too short for the chosen lag order."
    eps = 1e-6
    omegas = np.linspace(eps, np.pi - eps, n_freqs)
    A_list, _ = _estimate_var_coeffs(process, p)

    D = np.zeros((n_vars, n_vars, n_freqs), dtype=float)
    for idx, om in enumerate(omegas):
        H = _H_of_omega(A_list, om)  # (n,n) complex
        # row normalization (destination i)
        denom = np.sqrt(np.sum(np.abs(H)**2, axis=1, keepdims=True)) + 1e-12
        D[:, :, idx] = np.abs(H) / denom
    return omegas, D

def plot_dtf_spectrum_pair(process: np.ndarray,
                           causing: int,
                           caused: int,
                           p: int = 1,
                           n_freqs: int = 256,
                           title: str = None,
                           save: bool = False,
                           out_dir: str = "plot_dtf",
                           show: bool = True) -> str:
    """
    Plot DTF(ω) for a single pair (causing -> caused).
    """
    omegas, D = dtf_spectrum(process, p=p, n_freqs=n_freqs)
    vals = D[caused, causing, :]

    plt.figure()
    plt.plot(omegas, vals, label=f"DTF {causing}→{caused}")
    plt.xlabel("Frequency ω (radians)")
    plt.ylabel("DTF")
    if title is None:
        title = f"DTF(ω): {causing} → {caused} (p={p}, m={n_freqs})"
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    filepath = ""
    if save:
        os.makedirs(out_dir, exist_ok=True)
        fname = f"dtf_{causing}_to_{caused}_p{p}_n{n_freqs}.png"
        filepath = os.path.join(out_dir, fname)
        plt.savefig(filepath, dpi=200)
        if not show:
            plt.close()
    if show:
        plt.show()
    elif not save:
        plt.close()
    return filepath

# ------------------------------------------------------------
# Bandwise test with surrogates (time-shift of the driver)
# ------------------------------------------------------------
def _circular_shift(x: np.ndarray, shift: int) -> np.ndarray:
    return np.roll(x, shift)

def _band_indices(omegas: np.ndarray, wmin: float, wmax: float) -> np.ndarray:
    return np.where((omegas >= wmin) & (omegas <= wmax))[0]

def dtf_causality(process: np.ndarray,
                  p: int = 1,
                  n_freqs: int = 256,
                  bands: Tuple[Tuple[float, float], ...] = ((1e-3, np.pi/3),
                                                            (np.pi/3, 2*np.pi/3),
                                                            (2*np.pi/3, np.pi-1e-3)),
                  alpha: float = 0.05,
                  stat: str = "max",           # "max" or "mean" of DTF within band
                  n_perm: int = 200,           # time-shift surrogates on the driver
                  min_coverage_bands: float = 0.0,  # fraction of bands required as significant (0 => at least 1)
                  auto_plot: bool = False,
                  out_dir: str = "plot_dtf",
                  show_plots: bool = False) -> np.ndarray:
    """
    Estimate VAR(p), compute DTF(ω), and decide edges per band via surrogate (driver-shift) tests.
    Band decision: upper-tail p_perm on the chosen statistic ("max" or "mean"); significant if p_perm < alpha.
    Global decision: set j->i = 1 if the fraction of significant bands >= min_coverage_bands
                     (if 0 -> at least one band is enough).
    Return: A_bin (n_vars x n_vars).
    """
    n_vars, T = process.shape
    assert T > p + 5, "Time series too short."
    eps = 1e-6
    omegas, D = dtf_spectrum(process, p=p, n_freqs=n_freqs)  # D[i,j,ℓ]

    # Prepare plotting
    if auto_plot:
        os.makedirs(out_dir, exist_ok=True)

    # Bands -> indices
    band_idxs = []
    for (wmin, wmax) in bands:
        idx = _band_indices(omegas, max(eps, wmin), min(np.pi - eps, wmax))
        band_idxs.append(idx)

    # Band statistic (observed)
    def band_stat(vals: np.ndarray, idxs: np.ndarray) -> float:
        if idxs.size == 0:
            return 0.0
        return float(np.max(vals[idxs]) if stat == "max" else np.mean(vals[idxs]))

    A_bin = np.zeros((n_vars, n_vars), dtype=int)
    rng = np.random.default_rng(0)

    for caused in range(n_vars):
        for causing in range(n_vars):
            if causing == caused:
                continue

            # observed
            dtf_curve = D[caused, causing, :]
            stats_obs = [band_stat(dtf_curve, idx) for idx in band_idxs]

            # surrogates (re-estimate VAR for each surrogate for this pair)
            perm_stats = np.zeros((len(band_idxs), n_perm), dtype=float)
            for b, idxs in enumerate(band_idxs):
                if idxs.size == 0:
                    perm_stats[b, :] = 0.0
                    continue
                for r in range(n_perm):
                    s = int(rng.integers(low=1, high=T-1))
                    proc_perm = process.copy()
                    proc_perm[causing] = _circular_shift(proc_perm[causing], s)

                    # estimate VAR for surrogate and compute DTF for this pair only (efficient)
                    A_list_perm, _ = _estimate_var_coeffs(proc_perm, p)
                    vals_perm = np.empty_like(idxs, dtype=float)
                    for ii, om_idx in enumerate(idxs):
                        om = omegas[om_idx]
                        H_perm = _H_of_omega(A_list_perm, om)
                        denom = np.sqrt(np.sum(np.abs(H_perm)**2, axis=1)) + 1e-12
                        vals_perm[ii] = np.abs(H_perm[caused, causing]) / denom[caused]
                    perm_stats[b, r] = float(np.max(vals_perm) if stat == "max" else np.mean(vals_perm))

            # per-band p-values (upper-tail, add-one smoothing)
            pvals_bands = []
            for b, s_obs in enumerate(stats_obs):
                ge = np.sum(perm_stats[b, :] >= s_obs)
                pval = (1.0 + ge) / (1.0 + n_perm)
                pvals_bands.append(pval)

            # global decision
            sig_bands = sum(p < alpha for p in pvals_bands)
            denom = sum(len(idx) > 0 for idx in band_idxs)
            if denom == 0:
                A_bin[causing, caused] = 0
            else:
                needed = max(1, int(np.ceil(min_coverage_bands * denom))) if min_coverage_bands > 0 else 1
                A_bin[causing, caused] = int(sig_bands >= needed)

            # optional plot
            if auto_plot:
                plt.figure()
                plt.plot(omegas, dtf_curve, label=f"DTF {causing}→{caused}")
                for (wmin, wmax) in bands:
                    plt.axvspan(wmin, wmax, alpha=0.1)
                plt.xlabel("Frequency ω (radians)")
                plt.ylabel("DTF")
                plt.title(f"DTF(ω) & bands: {causing}→{caused}")
                plt.legend()
                plt.tight_layout()
                fname = f"dtf_{causing}_to_{caused}.png"
                plt.savefig(os.path.join(out_dir, fname), dpi=200)
                if not show_plots:
                    plt.close()

    np.fill_diagonal(A_bin, 0)
    return A_bin
