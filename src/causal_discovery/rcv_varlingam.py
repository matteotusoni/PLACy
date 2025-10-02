from lingam import VARLiNGAM, DirectLiNGAM
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def rcv_varlingam(process: np.ndarray, max_lags: int, k: int, seq_len: int, τ_c: float, τ_v: float) -> np.ndarray:

    # Cross-validation
    def getGraph(data):
        m = VARLiNGAM(lags=max_lags, random_state=0).fit(data.T)
        adjacency_matrices = m.adjacency_matrices_
        b0 = np.expand_dims(DirectLiNGAM().fit(m.residuals_).adjacency_matrix_, 0)
        g = np.concatenate((b0, adjacency_matrices), axis=0).max(0)
        return g

    g0 = getGraph(process)

    rnd_i = np.random.randint(0, process.shape[1] - seq_len, size=k)
    g_is = np.asarray([getGraph(process[:, i:i+seq_len]) for i in rnd_i])

    consistency = np.mean([np.sign(g0) == np.sign(g_i) for g_i in g_is], 0)
    variability = np.std(g_is, 0) / (np.abs(g0) + 1e-15)

    return np.triu(((consistency > τ_c) & (variability < τ_v)).astype(int), k=1)
