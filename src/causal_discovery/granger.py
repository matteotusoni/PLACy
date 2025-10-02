import numpy as np
from statsmodels.tsa.api import VAR



def granger(process: np.ndarray, max_lags: int, signif: float = .05) -> np.ndarray:
    n_vars = len(process)
    causality_matrix = np.zeros((n_vars, n_vars))

    for caused in range(n_vars):
        for causing in range(n_vars):
            if caused == causing:
                continue
            
            try:
                model = VAR(process[[causing, caused]].T)
                results = model.fit(maxlags=max_lags)
                
                test_result = results.test_causality(
                    caused=1, causing=0, kind='wald', signif=signif
                )
                
                if test_result.pvalue < signif:
                    causality_matrix[causing, caused] = 1

            except Exception as e:
                print(f"Error {causing} → {caused}: {e}")
                continue

    return causality_matrix
