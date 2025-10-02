import numpy as np
from numpy.linalg import inv, pinv
import matplotlib.pyplot as plt
from scipy.signal import csd, welch, get_window
from typing import Tuple, List, Optional
import os

# ============================================================
# Spectral matrix estimation (Welch)
# ============================================================

def compute_spectral_matrix_welch(
    process: np.ndarray,
    fs: float = 1.0,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    window: str = "hann",
    detrend: Optional[str] = None,
    average: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the cross-spectral density matrix S(ω) using Welch's method.

    Parameters
    ----------
    process : (n_vars, T) array
    fs : sampling frequency
    nperseg, noverlap, window, detrend, average : scipy.signal parameters

    Returns
    -------
    freqs : (m,) array in [0, fs/2]
    S : (m, n_vars, n_vars) complex-Hermitian cross-spectral matrix at each frequency
    """
    n_vars, T = process.shape
    if noverlap is None:
        noverlap = nperseg // 2

    # Precompute autospectra and cross-spectra
    freqs, Sxx = welch(process[0], fs=fs, window=window, nperseg=nperseg,
                       noverlap=noverlap, detrend=detrend, average=average, return_onesided=True)
    m = len(freqs)
    S = np.zeros((m, n_vars, n_vars), dtype=complex)

    # Autospectra
    for i in range(n_vars):
        _, Pii = welch(process[i], fs=fs, window=window, nperseg=nperseg,
                       noverlap=noverlap, detrend=detrend, average=average, return_onesided=True)
        S[:, i, i] = Pii

    # Cross-spectra
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            _, Pij = csd(process[i], process[j], fs=fs, window=window,
                         nperseg=nperseg, noverlap=noverlap, detrend=detrend,
                         average=average, return_onesided=True)
            S[:, i, j] = Pij
            S[:, j, i] = np.conjugate(Pij)

    return freqs, S


# ============================================================
# Geweke nonparametric spectrum (pairwise or conditional)
# ============================================================

def _schur_conditional_scalar(Sw: np.ndarray, target_idx: int, cond_idx: List[int]) -> float:
    """
    Conditional spectrum S_{tt · cond} at a single frequency:
    S_tt - S_tC S_CC^{-1} S_Ct
    """
    if len(cond_idx) == 0:
        return float(np.real(Sw[target_idx, target_idx]))
    S_tt = Sw[target_idx, target_idx]
    S_tC = Sw[target_idx, :][cond_idx]                  # shape (k,)
    S_CC = Sw[np.ix_(cond_idx, cond_idx)]               # shape (k,k)
    S_Ct = Sw[:, target_idx][cond_idx]                  # shape (k,)
    # use pinv for numerical stability
    inv_CC = inv(S_CC) if np.linalg.cond(S_CC) < 1e12 else pinv(S_CC)
    val = S_tt - S_tC @ inv_CC @ S_Ct
    return float(np.real(val))

def _geweke_F_one_freq(Sw: np.ndarray, x: int, y: int, cond: List[int]) -> float:
    """
    Geweke F_{x->y|cond} at a single frequency given full spectral matrix Sw (n x n).
    F = log( S_{yy·cond} / S_{yy·(cond ∪ {x})} )
    """
    cond_set = list(cond)
    Syy_cond = _schur_conditional_scalar(Sw, target_idx=y, cond_idx=cond_set)
    Syy_xcond = _schur_conditional_scalar(Sw, target_idx=y, cond_idx=cond_set + [x])
    num = max(Syy_cond, 1e-300)
    den = max(Syy_xcond, 1e-300)
    return float(np.log(num / den))

def geweke_nonparametric_spectrum(
    process: np.ndarray,
    fs: float = 1.0,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    window: str = "hann",
    condition_on_rest: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute nonparametric Geweke spectral causality F_{j->i}(ω) for all pairs.

    Parameters
    ----------
    process : (n_vars, T)
    condition_on_rest : if True, compute F_{j->i | Z} with Z = all vars except {i,j};
                        if False, compute bivariate F_{j->i}.

    Returns
    -------
    freqs : (m,)
    Fspec : (n_vars, n_vars, m) with Fspec[i, j, l] = F_{j->i}(ω_l)
    """
    n_vars, _ = process.shape
    freqs, S = compute_spectral_matrix_welch(
        process, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window
    )
    m = len(freqs)
    Fspec = np.zeros((n_vars, n_vars, m), dtype=float)

    for l in range(m):
        Sw = S[l, :, :]  # (n,n) at frequency l
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j: 
                    continue
                cond = [k for k in range(n_vars) if k not in (i, j)] if condition_on_rest else []
                Fspec[i, j, l] = _geweke_F_one_freq(Sw, x=j, y=i, cond=cond)

    return freqs, Fspec


# ============================================================
# Plot for a single pair
# ============================================================

def plot_geweke_nonparametric_pair(
    process: np.ndarray,
    causing: int,
    caused: int,
    fs: float = 1.0,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    window: str = "hann",
    title: Optional[str] = None,
    save: bool = False,
    out_dir: str = "plot_geweke_np",
    show: bool = True,
    condition_on_rest: bool = False,
) -> str:
    """
    Plot F_{causing->caused}(ω) using the nonparametric Geweke measure.
    """
    freqs, Fspec = geweke_nonparametric_spectrum(
        process, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window,
        condition_on_rest=condition_on_rest
    )
    vals = Fspec[caused, causing, :]

    plt.figure()
    plt.plot(freqs, vals, label=f"Geweke NP {causing}→{caused}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Geweke F (nats)")
    if title is None:
        title = f"Nonparametric Geweke: {causing}→{caused} (nperseg={nperseg})"
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    filepath = ""
    if save:
        os.makedirs(out_dir, exist_ok=True)
        fname = f"geweke_np_{causing}_to_{caused}_nseg{nperseg}.png"
        filepath = os.path.join(out_dir, fname)
        plt.savefig(filepath, dpi=200)
        if not show:
            plt.close()

    if show:
        plt.show()
    elif not save:
        plt.close()

    return filepath


# ============================================================
# Bandwise testing with phase-randomization surrogates
# ============================================================

def _phase_randomize(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Phase-randomize a real-valued time series (preserve power spectrum; destroy cross-relations).
    """
    T = len(x)
    Xf = np.fft.rfft(x)
    # keep DC and Nyquist real; randomize phases for bins 1..-2
    phases = rng.uniform(0, 2*np.pi, size=Xf.shape)
    phases[0] = 0.0
    if T % 2 == 0:  # Nyquist bin exists
        phases[-1] = 0.0
    Xf_rand = np.abs(Xf) * np.exp(1j * phases)
    xr = np.fft.irfft(Xf_rand, n=T)
    return xr.astype(float)

def _band_indices(freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    return np.where((freqs >= fmin) & (freqs <= fmax))[0]

def geweke_nonparametric_causality(
    process: np.ndarray,
    fs: float = 1.0,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    window: str = "hann",
    bands: Optional[Tuple[Tuple[float, float], ...]] = None,
    alpha: float = 0.05,
    stat: str = "max",                 # "max" or "mean" of F in band
    n_surrogates: int = 200,
    min_coverage_bands: float = 0.0,   # fraction of significant bands required (0 -> at least one)
    condition_on_rest: bool = False,
    auto_plot: bool = False,
    out_dir: str = "plot_geweke_np",
    show_plots: bool = False,
) -> np.ndarray:
    """
    Nonparametric spectral Granger (Geweke) graph via bandwise surrogate testing.

    Procedure:
      1) Estimate S(ω) with Welch; compute F_{j->i}(ω) (pairwise or conditional).
      2) For each band, compute a band statistic (max or mean of F).
      3) Build null via phase-randomized surrogates (preserve univariate spectra).
      4) A directed edge j->i is 1 if the fraction of significant bands >= min_coverage_bands
         (if 0 -> at least one band is significant).

    Returns
    -------
    A_bin : (n_vars, n_vars) binary adjacency matrix.
    """
    n_vars, T = process.shape
    assert T > nperseg, "Series too short for the chosen nperseg."

    # Frequency grid and F spectrum
    eps = 1e-9
    freqs, Fspec = geweke_nonparametric_spectrum(
        process, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window,
        condition_on_rest=condition_on_rest
    )  # shape Fspec[i,j,f]

    # Default: split [0, fs/2] into 3 equal bands (excluding DC-only band edges)
    if bands is None:
        fmax = freqs[-1]
        edges = np.linspace(freqs[1], fmax - 1e-6, 4)  # avoid DC at 0
        bands = tuple((edges[k], edges[k+1]) for k in range(3))

    band_idxs = [ _band_indices(freqs, fmin, fmax) for (fmin, fmax) in bands ]

    def band_stat(vals: np.ndarray, idxs: np.ndarray) -> float:
        if idxs.size == 0:
            return 0.0
        return float(np.max(vals[idxs]) if stat == "max" else np.mean(vals[idxs]))

    A_bin = np.zeros((n_vars, n_vars), dtype=int)
    rng = np.random.default_rng(0)

    # Precompute denominator (bands with at least one freq bin)
    denom_per_pair = sum(len(idx) > 0 for idx in band_idxs)

    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue

            # Observed band stats
            F_curve = Fspec[i, j, :]
            stats_obs = [band_stat(F_curve, idx) for idx in band_idxs]

            # Surrogate distribution: phase-randomize all series independently
            perm_stats = np.zeros((len(band_idxs), n_surrogates), dtype=float)
            for r in range(n_surrogates):
                proc_perm = np.vstack([_phase_randomize(process[v], rng) for v in range(n_vars)])
                # Recompute F_{j->i}(ω) for the surrogate (pairwise or conditional)
                _, Fspec_perm = geweke_nonparametric_spectrum(
                    proc_perm, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window,
                    condition_on_rest=condition_on_rest
                )
                F_perm_curve = Fspec_perm[i, j, :]
                for b, idxs in enumerate(band_idxs):
                    perm_stats[b, r] = band_stat(F_perm_curve, idxs)

            # Upper-tail p-values per band with add-one smoothing
            pvals_bands = []
            for b, s_obs in enumerate(stats_obs):
                ge = int(np.sum(perm_stats[b, :] >= s_obs))
                pval = (1.0 + ge) / (1.0 + n_surrogates)
                pvals_bands.append(pval)

            sig_bands = sum(p < alpha for p in pvals_bands)
            denom = denom_per_pair if denom_per_pair > 0 else 1
            needed = max(1, int(np.ceil(min_coverage_bands * denom))) if min_coverage_bands > 0 else 1
            A_bin[j, i] = int(sig_bands >= needed)  # note: j->i

            # Optional plot
            if auto_plot:
                os.makedirs(out_dir, exist_ok=True)
                plt.figure()
                plt.plot(freqs, F_curve, label=f"Geweke NP {j}→{i}")
                for (fmin, fmax) in bands:
                    plt.axvspan(fmin, fmax, alpha=0.1)
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Geweke F (nats)")
                plt.title(f"Geweke NP & bands: {j}→{i}")
                plt.legend()
                plt.tight_layout()
                fname = f"geweke_np_{j}_to_{i}.png"
                plt.savefig(os.path.join(out_dir, fname), dpi=200)
                if not show_plots:
                    plt.close()

    np.fill_diagonal(A_bin, 0)
    return A_bin
