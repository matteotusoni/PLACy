import numpy as np
from typing import Dict
import glob
import os


def airquality_process_loader(seed) -> Dict[str, np.ndarray]:

    path = "data/AirQuality"
    csv_files = glob.glob(os.path.join(path, '*.csv'))
    
    arrays = []
    csv_files = glob.glob(os.path.join(path, '*.csv'))
    csv_files_sorted = sorted(csv_files, key=lambda x: 0 if 'train' in x else 1)

    arrays = []
    for file in csv_files_sorted:
        data = np.genfromtxt(file, delimiter=',', skip_header=1)
        arrays.append(data)

    combined_array = np.vstack(arrays)

    # Load the .npy causal matrix
    npy_file = 'gc_real.npy'  
    full_path = os.path.join(path, npy_file)
    causal_matrix = np.load(full_path)
    np.fill_diagonal(causal_matrix, 0)

    length = 500
    if len(combined_array) < (seed+1) *length:
        exit("No more experiment to analyze")

    return dict(
        causal_matrix = causal_matrix,
        process=combined_array.T[1:,seed*length:(seed+1)*length],
    )
