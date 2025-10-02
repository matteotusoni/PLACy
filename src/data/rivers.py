import numpy as np
from typing import Dict
import glob
import os


def rivers_process_loader(seed) -> Dict[str, np.ndarray]:
    
    path = "data/Rivers"
    csv_files = glob.glob(os.path.join(path, '*.csv'))

    # sort files: train → validation → test
    def sort_key(x):
        if 'train' in x:
            return 0
        elif 'validation' in x:
            return 1
        elif 'test' in x:
            return 2
        else:
            return 3

    csv_files_sorted = sorted(csv_files, key=sort_key)

    arrays = []
    for file in csv_files_sorted:
        data = np.genfromtxt(file, delimiter=',', skip_header=1, usecols=range(1, 7))
        arrays.append(data)

    combined_array = np.vstack(arrays)

    num_vars = 6
    causal_matrix = np.zeros((num_vars, num_vars))
    causal_matrix[3, 0] = 1  # dillingen_precipitation → dillingen
    causal_matrix[4, 1] = 1  # kempten_precipitation → kempten
    causal_matrix[5, 2] = 1  # lenggries_precipitation → lenggries
    causal_matrix[1, 0] = 1  # kempten → dillingen (direct hydrological flow)
    causal_matrix[2, 0] = 1  # lenggries → dillingen (via Danube, considered direct from a hydrological standpoint)

    lenght = 500
    if len(combined_array) < (seed+1) *lenght+1  or (seed+1) *lenght+1 >= 9000:
        exit("No more experiment to analyze")
    
    return dict(
        causal_matrix=causal_matrix,
        process=combined_array.T[:, seed*lenght:(seed+1)*lenght],
    )

