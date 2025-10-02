import numpy as np
from statsmodels.tsa.api import VAR
from analysis.fft_with_fit import fft_with_fit


def PLACy(process: np.ndarray, max_lags: int, window_length, stride, signif: float = .05) -> np.ndarray:
    
    data_freq: np.array = fft_with_fit(process, window_length=window_length, stride=stride)

    n_vars = len(data_freq) // 2
    causality_matrix = np.zeros((n_vars, n_vars))
    
    for caused in range(n_vars):
        for causing in range(n_vars):
            if caused == causing:
                continue

            try:
                selected_data = data_freq[caused*2], data_freq[causing*2], data_freq[causing*2+1]
                selected_data = np.column_stack(selected_data) 

                model = VAR(selected_data)
                results = model.fit(maxlags=max_lags)
                
                test_result = results.test_causality(
                    caused=[0], causing=[1, 2], kind='wald', signif=signif
                )
                
                if test_result.pvalue < signif:
                    causality_matrix[causing, caused] = 1

            except Exception as e:
                print(f"Error {causing} → {caused}: {e}")
                continue

    return causality_matrix

