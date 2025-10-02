'''
Used to save synthetic data for the experiments on Rhino.
'''

import numpy as np
from typing import Dict
import os
import colorednoise as cn
import pandas as pd
from multiprocessing import Pool


def ou_process(
    causal_matrix, n_vars: int, length: int, lag: int, causal_strength: float, post_causality: bool = False, 
    σ_g = .5, σ_b: float = 0., tau_c = 1 / 0.7, 
    noise_exponent: float = 2, decay_exponent: float = 1, causal_exponent: float = 1, dt: int = 0.01,
    stationary: bool = True, plot: bool = False
) -> Dict[str, np.ndarray]:
    
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

    ou = (ou.T / ou[:, 0]).T

    return dict(
        causal_matrix=causal_matrix,
        strength_matrix=strength_matrix,
        lag_matrix=lag_matrix,
        process=ou
    )

def ou_process_stocklike(
    causal_matrix, n_vars: int, length: int, lag: int, causal_strength: float, post_causality: bool = False, 
    σ_g = .5, σ_b: float = 0., tau_c = 1 / 0.7, 
    noise_exponent: float = 2, decay_exponent: float = 1, causal_exponent: float = 1, dt: int = 0.01,
    stationary=True, plot = False
) -> Dict[str, np.ndarray]:
    
    strength_matrix = causal_strength * causal_matrix
    lag_matrix = lag * causal_matrix

    # causal_matrix[i, j] means i has causal influence on j (i -> j)

    ou = np.ones((n_vars, length))
    if not stationary:
        ou *= 100

    η = np.random.normal(loc=0, scale=σ_g, size=(n_vars, length))
    η_2 = np.random.normal(loc=0, scale=σ_g, size=(n_vars, length))
    η += np.asarray([cn.powerlaw_psd_gaussian(noise_exponent, length) for _ in range(n_vars)]) * σ_b

    for l in range(1, length):
        decay_term = 1 - np.sign(ou[:, l - 1]) * np.power(np.abs(ou[:, l - 1]), decay_exponent) / tau_c

        if not post_causality and l > lag:
            decay_term += causal_strength * np.dot(causal_matrix.T, np.sign(ou[:, l-lag]) * np.power(np.abs(ou[:, l-lag]), causal_exponent))

        ou[:, l] = ou[:, l - 1] + dt * decay_term + np.sqrt(dt) * η_2[:, l]*ou[:, l - 1] + np.sqrt(dt) * η[:, l]

    if post_causality:
        for i, j in zip(*np.where(causal_matrix == 1)):
            lag_idx = lag_matrix[i, j]
            if lag_idx > 0 and lag_idx < length:
                ou[j, lag_idx:] += strength_matrix[i, j] * np.sign(ou[i, :-lag_idx]) * np.power(np.abs(ou[i, :-lag_idx]), causal_exponent)

    ou = (ou.T / ou[:, 0]).T

    return dict(
        causal_matrix=causal_matrix,
        strength_matrix=strength_matrix,
        lag_matrix=lag_matrix,
        process=ou
    )

def generate_data(args):
    n_vars, causal_strength, s_g, s_b, seed, prob_edge, stationary, stock_like = args

    np.random.seed(seed)
    causal_matrix = np.triu((np.random.rand(n_vars, n_vars) < prob_edge).astype(int), k=1)

    if stock_like:
        process_train = ou_process_stocklike(
            causal_matrix=causal_matrix, n_vars=n_vars, length=10000, lag=5, causal_strength=causal_strength, post_causality=True,
            σ_g=s_g, σ_b=s_b, tau_c=5, noise_exponent=2, decay_exponent=1, causal_exponent=1, stationary=stationary
        )['process']

        process_test = ou_process_stocklike(
            causal_matrix=causal_matrix, n_vars=n_vars, length=10000, lag=5, causal_strength=causal_strength, post_causality=True,
            σ_g=s_g, σ_b=s_b, tau_c=5, noise_exponent=2, decay_exponent=1, causal_exponent=1, stationary=stationary
        )['process']
    else:
        process_train = ou_process(
            causal_matrix=causal_matrix, n_vars=n_vars, length=10000, lag=5, causal_strength=causal_strength, post_causality=True,
            σ_g=s_g, σ_b=s_b, tau_c=5, noise_exponent=2, decay_exponent=1, causal_exponent=1, stationary=stationary
        )['process']

        process_test = ou_process(
            causal_matrix=causal_matrix, n_vars=n_vars, length=10000, lag=5, causal_strength=causal_strength, post_causality=True,
            σ_g=s_g, σ_b=s_b, tau_c=5, noise_exponent=2, decay_exponent=1, causal_exponent=1, stationary=stationary
        )['process']

    data_type = 'stat' if stationary else 'nonstat'
    if stock_like:
        data_type += '_stocklike'
    folder = f'generated_data/n_vars={n_vars}_causal_strength={causal_strength}_s_g={s_g}_s_b={s_b}_{data_type}_seed={seed}'
    os.makedirs(folder, exist_ok=True)
    np.save(f'{folder}/adj_matrix.npy', causal_matrix)

    numbers = np.arange(50)
    repetitions = 200
    index = np.repeat(numbers, repetitions)

    process_train = pd.DataFrame(process_train.T)
    process_train = pd.concat([pd.Series(index), process_train], axis=1)
    process_train.to_csv(f'{folder}/train.csv', index=False, header=False)

    process_test = pd.DataFrame(process_test.T)
    process_test = pd.concat([pd.Series(index), process_test], axis=1)
    process_test.to_csv(f'{folder}/test.csv', index=False, header=False)


if __name__ == "__main__":
    prob_edge = 0.3

    tasks = []
    for n_vars in [5]:
        for causal_strength in [1.]:
            for s_g in [1.]:
                for s_b in [.1]:
                    for stationary in [True, False]:
                        for stock_like in [True, False]:
                            for seed in range(10):
                                tasks.append((n_vars, causal_strength, s_g, s_b, seed, prob_edge, stationary, stock_like))
    print(f"Total tasks: {len(tasks)}")

    with Pool(processes=64) as pool:
        pool.map(generate_data, tasks)