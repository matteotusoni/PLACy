from causal_ccm.causal_ccm import ccm
import numpy as np


def check_convergence_stats(ccm_values, window_size=3, threshold=0.01):
    rolling_mean = np.convolve(ccm_values, np.ones(window_size)/window_size, mode='valid')
    diff = np.abs(np.diff(rolling_mean))
    return np.all(diff[-window_size:] < threshold)


def perform_ccm(process: np.ndarray, max_lags: int, filter_size: int, stride: int) -> np.ndarray:
    n_vars = len(process)
    process = np.asarray([
        np.convolve(process[i], np.ones(filter_size) / filter_size, mode='valid')[::stride]
        for i in range(n_vars)
    ])

    cm = np.zeros((n_vars, n_vars))
    L_range = range(25, process.shape[-1], 250)
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j: continue
            Xhat_My = []
            for L in L_range:
                ccm_XY = ccm(process[i], process[j], tau=max_lags, E=2, L=L)
                Xhat_My.append(ccm_XY.causality()[0])
                p_valueXY = ccm_XY.causality()[1]
            if p_valueXY < 0.05 and check_convergence_stats(Xhat_My):
                cm[i, j] = 1
    return cm


def perform_ccm_freq(
    fft: np.ndarray,
    max_lags: int,
    filter_size: int,
    stride: int,
    E: int = 2,
    p_threshold: float = 0.05,
    convergence_window: int = 3,
    convergence_threshold: float = 0.01,
    L_range: list = None
) -> np.ndarray:
    
    λ = fft[0::2]
    α = fft[1::2]
    N= λ.shape[0]

    # Preprocessing: filter + stride
    λ_proc = np.asarray([
        np.convolve(λ[i], np.ones(filter_size) / filter_size, mode='valid')[::stride]
        for i in range(N)
    ])
    α_proc = np.asarray([
        np.convolve(α[i], np.ones(filter_size) / filter_size, mode='valid')[::stride]
        for i in range(N)
    ])

    if L_range is None:
        L_range = range(25, λ_proc.shape[-1], 250)

    G = np.zeros((N, N), dtype=int)

    for i in range(N):
        target = λ_proc[i]

        causing_λ = [λ_proc[j] for j in range(N) if j != i]
        λ_indices = [j for j in range(N) if j != i]
        all_causes = causing_λ + list(α_proc)

        for idx, cause in enumerate(all_causes):
            Xhat_My = []
            for L in L_range:
                ccm_XY = ccm(target, cause, tau=max_lags, E=E, L=L)
                score, p_value = ccm_XY.causality()
                Xhat_My.append(score)

            if p_value < p_threshold and check_convergence_stats(Xhat_My, convergence_window, convergence_threshold):
                if idx < len(causing_λ):
                    j = λ_indices[idx]
                    G[i, j] = 1

    return G
