import numpy as np
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI


def pcmciplus(ts: np.ndarray, tau_max: int) -> np.ndarray:
    dataframe = pp.DataFrame(ts.T)
    pcmci_parcorr = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr())
    results = pcmci_parcorr.run_pcmciplus(tau_max=tau_max, tau_min=1, pc_alpha=0.05)
    return (results['graph'] == '-->').astype(int)
