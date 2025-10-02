from typing import Dict
import argparse
import json
import os
import numpy as np
from data.ou_process_stocklike import ou_process_stocklike
from data.ou_process import ou_process
from data.airquality import airquality_process_loader
from data.rivers import rivers_process_loader
from causal_discovery.granger import granger
from causal_discovery.placy import PLACy
from causal_discovery.pcmci import pcmciplus
from causal_discovery.rcv_varlingam import rcv_varlingam
from causal_discovery.ccm import perform_ccm
from causal_discovery.pcmci_omega import pcmci_omega
from causal_discovery.BC import geweke_spectral_causality_bandgraph
from causal_discovery.pdc import pdc_causality
from causal_discovery.dtf import dtf_causality
from causal_discovery.gew_non_par import geweke_nonparametric_causality

# from causal_discovery.dynotears import dynotears


def main(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    if args.dataset == 'synthetic':
        if args.stock_like:
            d: Dict[str, np.ndarray] = ou_process_stocklike(
                n_vars=args.n_vars, length=args.length, 
                prob_edge=args.prob_edge, lag=args.lag, causal_strength=args.causal_strength, post_causality=args.post_causality,
                σ_g=args.s_g, σ_b=args.s_b, tau_c=args.tau_c, 
                noise_exponent=args.noise_exponent, decay_exponent=args.decay_exponent, causal_exponent=args.causal_exponent, 
                stationary=args.stationary, plot=args.debug
            )
        else:
            d: Dict[str, np.ndarray] = ou_process(
                n_vars=args.n_vars, length=args.length, 
                prob_edge=args.prob_edge, lag=args.lag, causal_strength=args.causal_strength, post_causality=args.post_causality,
                σ_g=args.s_g, σ_b=args.s_b, tau_c=args.tau_c, 
                noise_exponent=args.noise_exponent, decay_exponent=args.decay_exponent, causal_exponent=args.causal_exponent, 
                stationary=args.stationary, plot=args.debug
            )

    elif args.dataset == 'AirQuality':
        d = airquality_process_loader(args.seed)
    
    elif args.dataset == 'Rivers':
        d = rivers_process_loader(args.seed)

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    process, cm = d['process'], d['causal_matrix']
    
    if args.method == 'Granger': 
        cm_est_time: np.ndarray = granger(process, max_lags=args.max_lags)
        cm_est_freq: np.ndarray = PLACy(process, max_lags=args.max_lags, window_length=args.window_length, stride=args.stride)

    elif args.method == 'PCMCI':
        cm_est_time: np.ndarray = pcmciplus(process, tau_max=args.max_lags).max(-1)
        np.fill_diagonal(cm_est_time, 0)

    elif args.method == 'RCV-VarLiNGAM':
        k, seq_len = 7, 300
        τ_c, τ_v = .7, .4
        cm_est_time = rcv_varlingam(process, k=k, seq_len=seq_len, τ_c=τ_c, τ_v=τ_v, max_lags=args.max_lags)

    elif args.method == 'CCM-Filtering':
        filter_size, stride = 5, 1
        cm_est_time = perform_ccm(process, max_lags=args.max_lags, filter_size=filter_size, stride=stride)

    # elif args.method == 'Dynotears':
    #     cm_est_time = dynotears(process, max_lags=args.max_lags)

    elif args.method == 'PCMCIomega':
        cm_est_time = pcmci_omega(process, tau_max=args.max_lags)

    elif args.method == 'BCGeweke':
        cm_est_time = geweke_spectral_causality_bandgraph(process, maxlags=args.max_lags, n_freqs= 128)
    
    elif args.method == 'pdc':
        cm_est_time = pdc_causality(process, p=2, n_freqs=128, alpha=0.05, n_perm=100)
        
    elif args.method == 'dtf':
        cm_est_time = dtf_causality(process, p=2, n_freqs=128, alpha=0.05, n_perm=100)

    elif args.method == 'gewelenp':
        cm_est_time = geweke_nonparametric_causality(process, fs=1.0, nperseg=128, alpha=0.05, n_surrogates=100)
    
    
    to_save = dict(
        cm=cm.tolist(),
        cm_est_time=cm_est_time.tolist(),
        cm_est_freq=cm_est_freq.tolist() if args.method == 'Granger' else None
    )

    path = f'results/{args.method}'
    if args.dataset == 'synthetic':
        data_type = 'stat' if args.stationary else 'nonstat'
        if args.stock_like:
            data_type += '_stocklike'
        path += f'_{data_type}/{args.dir}/' + \
            f'n_vars={args.n_vars}/' + \
            f'causal_strength={args.causal_strength}/' + \
            f's_g={args.s_g}/' + \
            f's_b={args.s_b}/'
    else:
        path += f'_{args.dataset}'
    os.makedirs(path, exist_ok=True)
    with open(f'{path}/seed={args.seed}.json', 'w') as f:
        json.dump(to_save, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Seed
    parser.add_argument('-s', '--seed', default=1, type=int)

    # Saving results
    parser.add_argument('--dir', default='', type=str)
    parser.add_argument('--debug', action='store_true')

    # Data 
    parser.add_argument('--dataset', default='synthetic', type=str)

    # Generation of the process
    parser.add_argument('--n_vars', default=5, type=int)
    parser.add_argument('--length', default=5000, type=int)
    parser.add_argument('--prob_edge', default=.5, type=float)
    parser.add_argument('--lag', default=5, type=int)
    parser.add_argument('--causal_strength', default=1., type=float)
    parser.add_argument('--s_g', default=1.0, type=float)
    parser.add_argument('--s_b', default=0.0, type=float)
    parser.add_argument('--tau_c', default=5., type=float)
    parser.add_argument('--noise_exponent', default=2., type=float)
    parser.add_argument('--decay_exponent', default=1., type=float)
    parser.add_argument('--causal_exponent', default=1, type=float)
    parser.add_argument('--post_causality', default=True, type=bool)
    parser.add_argument('--stationary', default=True, type=lambda x: x == 'True')
    parser.add_argument('--stock_like', default=True, type=lambda x: x == 'True')

    # Causal discovery
    parser.add_argument('--method', default='Granger', type=str)
    parser.add_argument('--max_lags', default=20, type=int)

    # FFT
    parser.add_argument('--window_length', default=50, type=int)
    parser.add_argument('--stride', default=1, type=int)
    
    main(parser.parse_args())
