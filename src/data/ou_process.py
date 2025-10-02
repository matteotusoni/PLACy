import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
import os
import colorednoise as cn


def ou_process(
    n_vars: int, length: int, prob_edge: float, lag: int, causal_strength: float, post_causality: bool = False, 
    σ_g = .5, σ_b: float = 0., tau_c = 1 / 0.7, 
    noise_exponent: float = 2, decay_exponent: float = 1, causal_exponent: float = 1, dt: int = 0.01,
    stationary: bool = True, plot: bool = False
) -> Dict[str, np.ndarray]:
    
    causal_matrix = np.triu((np.random.rand(n_vars, n_vars) < prob_edge).astype(int), k=1)
    strength_matrix = causal_strength * causal_matrix
    lag_matrix = lag * causal_matrix

    # causal_matrix[i, j] means i has causal influence on j (i -> j)

    ou = np.ones((n_vars, length))
    if not stationary:
        ou *= 100

    η = np.random.normal(loc=0, scale=σ_g, size=(n_vars, length))
    η += np.asarray([cn.powerlaw_psd_gaussian(noise_exponent, length) for _ in range(n_vars)]) * σ_b

    for l in range(1, length):
        decay_term = - np.sign(ou[:, l - 1]) * np.power(np.abs(ou[:, l - 1]), decay_exponent) / tau_c

        if not post_causality and l > lag:
            decay_term += causal_strength * np.dot(causal_matrix.T, np.sign(ou[:, l-lag]) * np.power(np.abs(ou[:, l-lag]), causal_exponent))

        ou[:, l] = ou[:, l - 1] + dt * decay_term + np.sqrt(dt) * η[:, l]


    if post_causality:
        for i, j in zip(*np.where(causal_matrix == 1)):
            lag_idx = lag_matrix[i, j]
            if lag_idx > 0 and lag_idx < length:
                ou[j, lag_idx:] += strength_matrix[i, j] * np.sign(ou[i, :-lag_idx]) * np.power(np.abs(ou[i, :-lag_idx]), causal_exponent)

    ou = ou[:, n_vars*lag:]
    ou = (ou.T / ou[:, 0]).T

    if plot:
        path = "plots/process"
        os.makedirs(path, exist_ok=True) 
        plt.figure(figsize=(12, 5))
        plt.plot(ou.T, label=range(n_vars))
        plt.legend()
        plt.savefig(f'{path}/ou_process.pdf', bbox_inches='tight')
        plt.close()

    return dict(
        causal_matrix=causal_matrix,
        strength_matrix=strength_matrix,
        lag_matrix=lag_matrix,
        process=ou
    )