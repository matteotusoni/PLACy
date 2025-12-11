import numpy as np
from statsmodels.tsa.api import VAR
from analysis.fft_with_fit import fft_with_fit


def fill_nan_with_previous(arr):
    arr = np.asarray(arr, dtype=float)

    for i in range(len(arr)):
        if np.isnan(arr[i]):
            if i == 0:
                arr[i] = 0.0
            else:
                arr[i] = arr[i - 1]
    return arr


def _find_cycle(adj: np.ndarray):
    n = adj.shape[0]
    visited = [0] * n   # 0 = unvisited, 1 = visiting, 2 = done
    stack = []

    def dfs(v):
        visited[v] = 1
        stack.append(v)

        for w in range(n):
            if adj[v, w]:
                if visited[w] == 0:
                    res = dfs(w)
                    if res is not None:
                        return res
                elif visited[w] == 1:
                    # Found a back edge → cycle from w to v
                    idx = stack.index(w)
                    cycle_nodes = stack[idx:] + [w]
                    return cycle_nodes

        stack.pop()
        visited[v] = 2
        return None

    for v in range(n):
        if visited[v] == 0:
            res = dfs(v)
            if res is not None:
                return res

    return None


def make_acyclic(adj: np.ndarray, pvalues: np.ndarray) -> np.ndarray:

    adj = adj.copy().astype(int)

    while True:
        cycle_nodes = _find_cycle(adj)
        if cycle_nodes is None:
            # No cycles → adjacency is acyclic
            break

        # Convert cycle nodes to a list of directed edges (u -> v)
        edges_in_cycle = []
        for i in range(len(cycle_nodes) - 1):
            u = cycle_nodes[i]
            v = cycle_nodes[i + 1]
            edges_in_cycle.append((u, v))

        # Choose the edge with the highest p-value to remove
        worst_edge = None
        worst_p = -1.0

        for u, v in edges_in_cycle:
            p = pvalues[u, v]
            if np.isnan(p):
                # If p-value missing, treat as very non-significant
                p = 1.0
            if p > worst_p:
                worst_p = p
                worst_edge = (u, v)

        if worst_edge is None:
            # Fallback: if for some reason no edge was selected, drop first
            u, v = edges_in_cycle[0]
        else:
            u, v = worst_edge

        # Remove that edge
        adj[u, v] = 0

    return adj


def Ac_PLACy(process: np.ndarray,
          max_lags: int,
          window_length,
          stride,
          signif: float = 0.05) -> np.ndarray:

    data_freq: np.ndarray = fft_with_fit(
        process,
        window_length=window_length,
        stride=stride
    )

    n_vars = len(data_freq) // 2
    causality_matrix = np.zeros((n_vars, n_vars), dtype=int)
    pvalue_matrix = np.full((n_vars, n_vars), np.nan, dtype=float)

    for caused in range(n_vars):
        for causing in range(n_vars):
            if caused == causing:
                continue

            try:
                selected_data = (
                    data_freq[caused * 2],
                    data_freq[causing * 2],
                    data_freq[causing * 2 + 1],
                )
                selected_data = np.column_stack(selected_data)

                model = VAR(selected_data)
                results = model.fit(maxlags=max_lags)

                test_result = results.test_causality(
                    caused=[0], causing=[1, 2], kind='wald', signif=signif
                )

                p = float(test_result.pvalue)
                pvalue_matrix[causing, caused] = p

                if p < signif:
                    causality_matrix[causing, caused] = 1

            except Exception as e:
                print(f"Error {causing} → {caused}: {e}")
                continue

    # Enforce acyclicity by removing least significant edges in cycles
    causality_matrix = make_acyclic(causality_matrix, pvalue_matrix)

    return causality_matrix
