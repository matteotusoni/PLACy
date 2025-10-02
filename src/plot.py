import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from itertools import product
import os
import json
from tqdm import tqdm
from multiprocessing import Pool


def tnr_score(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp)


def getPath(method, d):
    return f'results/{method}_{d["data_type"]}/n_vars={d["n_vars"]}/causal_strength={d["causal_strength"]}/s_g={d["s_g"]}/s_b={d["s_b"]}'


def make_plot(results_f1, results_tnr, methods, ind_var, ind_values):
    ind_var2label = {
        'data_type': 'Dataset',
        'n_vars': 'Number of variables',
        'causal_strength': 'Causal strength',
        's_g': r'Gausian noise $\sigma_g$',
        's_b': r'Brownian noise $\sigma_b$',
    }

    def ind_val2label(ind_val):
        if type(ind_val) == str:
            return {
                'stat': r'${\rm{OU}}(\sigma_g^m = 0)$',
                'stat_stocklike': r'${\rm{OU}}(\sigma_g^m > 0)$',
                'nonstat': r'$\widehat{\rm{OU}}(\sigma_g^m = 0)$',
                'nonstat_stocklike': r'$\widehat{\rm{OU}}(\sigma_g^m > 0)$',
            }[ind_val]
        else:
            return ind_val

    n = sum(len(methods[m]) for m in methods)
    base_x = np.arange(len(ind_values)) * (n+1)  # Base positions for x-axis labels
    base_x
    offsets = np.arange(n)

    pos = np.asarray([x + offsets - np.median(offsets) for x in base_x]).T

    i = 0
    method2x = dict()
    for m in methods:
        for tf in methods[m]:
            method2x[f'{m}_{tf}'] = pos[i]
            i += 1

    method2c = {m: f'C{i}' for i, m in enumerate(method2x)}
    for title in tqdm(results_f1[ind_var]):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
        for m in methods:
            for tf in methods[m]:
                data_time = np.asarray([results_f1[ind_var][title][inv_value][f'{m}_{tf}'] for inv_value in ind_values])
                add_label = True
                for j, x_time in enumerate(data_time):
                    x_time = x_time[~np.isnan(x_time)]
                    c = method2c[f'{m}_{tf}']
                    ax1.boxplot(
                        x_time, positions=method2x[f'{m}_{tf}'][[j]], label=f'{m}_{tf}' if add_label else None, 
                        patch_artist=True, widths=.6, 
                        showmeans=True, meanprops=dict(marker='o', markerfacecolor='black', markersize=5, markeredgecolor='black'),
                        boxprops=dict(facecolor=c), medianprops=dict(color='black')
                    )
                    add_label = False
                data_time = np.asarray([results_tnr[ind_var][title][inv_value][f'{m}_{tf}'] for inv_value in ind_values])
                for j, x_time in enumerate(data_time):
                    x_time = x_time[~np.isnan(x_time)]
                    c = method2c[f'{m}_{tf}']
                    ax2.boxplot(
                        x_time, positions=method2x[f'{m}_{tf}'][[j]], label=f'{m}_{tf}' if add_label else None, 
                        patch_artist=True, widths=.6, 
                        showmeans=True, meanprops=dict(marker='o', markerfacecolor='black', markersize=5, markeredgecolor='black'),
                        boxprops=dict(facecolor=c), medianprops=dict(color='black')
                    )

        ax1.set_ylabel('F1', fontsize=22)
        ax2.set_ylabel('TNR', fontsize=22)
        ax1.set_xlabel(None)
        ax2.set_xlabel(ind_var2label[ind_var], fontsize=22)
        ax1.set_xticks([])
        ax2.set_xticks(base_x, [ind_val2label(v) for v in ind_values], fontsize=18)
        
        for ax in [ax1, ax2]:
            ax.set_yticks(np.arange(0, 1.1, .25), labels=np.arange(0, 1.1, .25).round(1), fontsize=18)
            ax.set_ylim([-.05, 1.05])
            for pos in base_x[:-1]:
                ax.axvline(x=pos+(n+1)/2, color='gray', linestyle='--', linewidth=0.5)
        
        title = title.replace('{', '').replace('}', '').replace("'", '').replace(', ', ' - ')
        fig.tight_layout()
        folder = f'plots/ind_var={ind_var}'
        savename = title.replace(': ', '=').replace(' - ', '_')
        os.makedirs(folder, exist_ok=True)
        fig.savefig(f'{folder}/{savename}.pdf', bbox_inches='tight')
        plt.close(fig)


def open_data(ind_var, ind_values, other_variables):
    methods = {
        'Granger': ['time', 'freq'], 
        'PCMCI': ['time'],
        'CCM-Filtering': ['time'],
        'RCV-VarLiNGAM': ['time'],
        'Rhino': ['time'],
        'Dynotears': ['time'],
        'PCMCIomega': ['time'],
    }

    zero_division = np.nan

    results_f1 = dict()
    results_tnr = dict()

    results_f1[ind_var] = dict()
    results_tnr[ind_var] = dict()
    other_values = [other_variables[v] for v in other_variables]
    combinations = list(product(*other_values))
    for combo in tqdm(combinations):
        title = {list(other_variables.keys())[i]: v for i, v in enumerate(combo)}
        results_f1[ind_var][str(title)] = dict()
        results_tnr[ind_var][str(title)] = dict()

        for ind_value in ind_values:
            results_f1[ind_var][str(title)][ind_value] = dict()
            results_tnr[ind_var][str(title)][ind_value] = dict()
            for method in methods:
                for tf in methods[method]:
                    results_f1[ind_var][str(title)][ind_value][f'{method}_{tf}'] = list()
                    results_tnr[ind_var][str(title)][ind_value][f'{method}_{tf}'] = list()
                    path = getPath(method, title | {ind_var: ind_value})
                    for seed in range(100):
                        filename = f'{path}/seed={seed}.json'
                        if os.path.exists(filename):
                            with open(filename, 'r') as f:
                                data = json.load(f)
                            cm_est_tf = np.asarray(data[f'cm_est_{tf}']).ravel()
                            cm_true = np.zeros_like(cm_est_tf) if 'causal_strength=0.0' in path else np.asarray(data['cm']).ravel()
                            f1 = np.nan if 'causal_strength=0.0' in path else f1_score(cm_true, cm_est_tf, zero_division=zero_division)
                            tnr = tnr_score(cm_true, cm_est_tf)
                        else:
                            f1 = np.nan
                            tnr = np.nan
                        results_f1[ind_var][str(title)][ind_value][f'{method}_{tf}'].append(f1)
                        results_tnr[ind_var][str(title)][ind_value][f'{method}_{tf}'].append(tnr)

    make_plot(results_f1, results_tnr, methods, ind_var, ind_values)


def main():
    
    variables = {
        'n_vars': [5],  # 10, 20, 40
        'causal_strength': [.5,],  # .0, .2, 10.
        's_g': [1.],
        's_b': [0., .1, .5, 1.],  # , 2., 5.
        'data_type': ['stat', 'stat_stocklike', 'nonstat', 'nonstat_stocklike']
    }

    legend_elements = [
        plt.Rectangle((0,0), 1, 1, facecolor='C0', edgecolor='black', label='Granger'),
        plt.Rectangle((0,0), 1, 1, facecolor='C1', edgecolor='black', label='PLaCy'),
        plt.Rectangle((0,0), 1, 1, facecolor='C2', edgecolor='black', label='PCMCI'),
        plt.Rectangle((0,0), 1, 1, facecolor='C3', edgecolor='black', label='CCM-Filtering'),
        plt.Rectangle((0,0), 1, 1, facecolor='C4', edgecolor='black', label='RCV-VarLiNGAM'),
        plt.Rectangle((0,0), 1, 1, facecolor='C5', edgecolor='black', label='Rhino'),
        plt.Rectangle((0,0), 1, 1, facecolor='C6', edgecolor='black', label='DYNOTEARS'),
        plt.Rectangle((0,0), 1, 1, facecolor='C7', edgecolor='black', label=r'$PCMCI_{\Omega}$')
    ]
    fig = plt.figure(figsize=(len(legend_elements), 1))
    plt.legend(handles=legend_elements, loc='center', ncol=len(legend_elements), fontsize=22)
    plt.tight_layout()
    plt.axis('off')
    os.makedirs('plots', exist_ok=True)
    fig.savefig('plots/legend_PLaCy.pdf', bbox_inches='tight')
    plt.close(fig)

    l = list()
    for ind_var, ind_values in variables.items():
        l.append((ind_var, ind_values, {k: v for k, v in variables.items() if k != ind_var}))

    with Pool(len(l)) as p:
        p.starmap(open_data, l)

if __name__ == '__main__':
    main()
