import numpy as np
from numpy.linalg import inv
from typing import List, Tuple, Dict, Optional
from statsmodels.tsa.api import VAR
from scipy.stats import f as f_dist


def _lag_design_two_series(y: np.ndarray, x: np.ndarray, p: int):
    """
    Build lagged design matrix for the regression:
        y_t = c + sum_{l=1}^p a_yy,l * y_{t-l} + sum_{l=1}^p a_yx,l * x_{t-l} + e_t

    Returns:
      Y : (T-p,)
      X : (T-p, 1 + 2p)  [intercept | p lags of y | p lags of x]
    """
    T = y.shape[0]
    rows = T - p
    Y = y[p:].copy()
    X = np.ones((rows, 1 + 2*p), dtype=float)
    # y lags
    for l in range(1, p+1):
        X[:, 1 + (l-1)] = y[p-l:T-l]
    # x lags
    for l in range(1, p+1):
        X[:, 1 + p + (l-1)] = x[p-l:T-l]
    return Y, X


def _ols(Y: np.ndarray, X: np.ndarray):
    """Simple OLS: beta, residual variance, (X'X)^(-1)"""
    XtX = X.T @ X
    XtX_inv = inv(XtX)
    beta = XtX_inv @ (X.T @ Y)
    resid = Y - X @ beta
    dof = max(len(Y) - X.shape[1], 1)
    sigma2 = float((resid @ resid) / dof)
    return beta, sigma2, XtX_inv, dof


def _frequency_restriction_matrix(p: int, omega: float, k_trend: int = 1) -> np.ndarray:
    """
    Construct the restriction matrix R(omega) to test the null of no causality
    from x → y at frequency omega in a regression:
      [c | a_yy,1..p | a_yx,1..p]

    H0: sum_l a_yx,l * cos(l*omega) = 0
        sum_l a_yx,l * sin(l*omega) = 0
    """
    R = np.zeros((2, k_trend + 2*p))
    for l in range(1, p+1):
        col = k_trend + p + (l-1)
        R[0, col] = np.cos(l * omega)
        R[1, col] = np.sin(l * omega)
    return R


def _band_test_pvalue(beta: np.ndarray, sigma2: float, XtX_inv: np.ndarray, dof: int,
                      p: int, band: Tuple[float, float], n_grid: int = 16) -> float:
    """
    Scan the frequency band [w1, w2] with n_grid points and take
    the minimum p-value (conservative). Returns the 'band-level' p-value.
    """
    w1, w2 = band
    if w1 < 0.0:
        w1 = 0.0
    if w2 > 0.5:
        w2 = 0.5
    if w2 <= w1:
        return 1.0

    omegas = np.linspace(w1, w2, n_grid)
    best_p = 1.0
    for w in omegas:
        R = _frequency_restriction_matrix(p, w, k_trend=1)
        Var_beta = sigma2 * XtX_inv
        RVR = R @ Var_beta @ R.T
        try:
            RVR_inv = inv(RVR)
        except np.linalg.LinAlgError:
            continue
        r = R @ beta
        W = float(r.T @ (RVR_inv @ r))
        q = R.shape[0]  # 2 restrictions
        F = W / q
        pval = 1.0 - f_dist.cdf(F, q, dof)
        if pval < best_p:
            best_p = pval
    return best_p


def frequency_band_causality_graph(
    process: np.ndarray,
    maxlags: int = 5,
    bands: Optional[List[Tuple[float, float]]] = None,
    alpha: float = 0.05,
    n_grid_per_band: int = 16,
    aggregate: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Build frequency-band Granger causality graphs from multivariate time series (n_vars, T).

    Parameters
    ----------
    process : array (n_vars, T)
    maxlags : VAR order (p)
    bands   : list of tuples (f_min, f_max) in [0, 0.5]; default: low/mid/high
    alpha   : significance threshold
    n_grid_per_band : number of frequency grid points per band
    aggregate : if True, also return the OR across all bands

    Returns
    -------
    dict with keys:
      - 'band_0', 'band_1', ... : binary adjacency matrices (n_vars, n_vars)
      - 'aggregate' (optional): logical OR of all bands
    """
    if bands is None:
        # Example: long, medium, short frequency bands
        bands = [(0.00, 0.06), (0.06, 0.20), (0.20, 0.50)]

    n_vars, T = process.shape
    results: Dict[str, np.ndarray] = {}

    band_graphs = []
    for b_idx, band in enumerate(bands):
        A = np.zeros((n_vars, n_vars), dtype=int)
        for caused in range(n_vars):
            for causing in range(n_vars):
                if caused == causing:
                    continue

                data_pair = process[[causing, caused]].T  # (T, 2)
                try:
                    model = VAR(data_pair)
                    _ = model.fit(maxlags=maxlags, ic=None, trend='c')
                except Exception:
                    try:
                        _ = model.fit(maxlags=maxlags, trend='c')
                    except Exception:
                        continue

                y = data_pair[:, 1]  # 'caused'
                x = data_pair[:, 0]  # 'causing'
                try:
                    Y, X = _lag_design_two_series(y, x, maxlags)
                    beta, sigma2, XtX_inv, dof = _ols(Y, X)
                except Exception:
                    continue

                p_band = _band_test_pvalue(beta, sigma2, XtX_inv, dof,
                                           p=maxlags, band=band, n_grid=n_grid_per_band)
                if p_band < alpha:
                    A[causing, caused] = 1

        results[f'band_{b_idx}'] = A
        band_graphs.append(A)

    if aggregate and len(band_graphs) > 0:
        agg = np.zeros_like(band_graphs[0])
        for G in band_graphs:
            agg |= G
        results['aggregate'] = agg

    return results


def geweke_spectral_causality_bandgraph(
    process: np.ndarray, maxlags: int = 5, n_freqs: int = 128
) -> np.ndarray:
    """
    Drop-in replacement for your original 'geweke_spectral_causality_old' function.
    Returns a single binary adjacency matrix (aggregated over frequency bands).
    """
    out = frequency_band_causality_graph(
        process, maxlags=maxlags,
        bands=[(0.00, 0.06), (0.06, 0.20), (0.20, 0.50)],
        alpha=0.05, n_grid_per_band=16, aggregate=True
    )
    return out['aggregate']
