# PLACy: Robust Causal Discovery in Real-World Time Series with Power-Laws

🏆 **ICML 2026 Spotlight (Top 2%)**

**Matteo Tusoni, Giuseppe Masi, Andrea Coletta, Aldo Glielmo, Viviana Arrigoni, Novella Bartolini**

**Robust Causal Discovery in Real-World Time Series with Power-Laws**

Accepted at the **International Conference on Machine Learning (ICML 2026)** as a **Spotlight** paper.

📄 Paper: https://arxiv.org/abs/2507.12257

---

## Installation

```bash
pip install placy
```

---

## Quick Start

A complete walkthrough is available in:

📓 **[example.ipynb](example.ipynb)**

---

## Development Setup

```bash
conda create --name CausalDiscovery python=3.11.11
conda activate CausalDiscovery
pip install -r requirements.txt
```

---

## Reproducing the Paper Experiments

Run an experiment on synthetic data:

```bash
python src/run.py \
    -s SEED \
    --n_vars N_VARS \
    --length LENGTH \
    --causal_strength C \
    --method METHOD \
    --window_length W \
    --stride S
```

Run an experiment on real data:

```bash
python src/run.py \
    -s SEED \
    --dataset DATASET_NAME \
    --method METHOD \
    --window_length W \
    --stride S
```

---

## Benchmark Dataset

We provide a free benchmark dataset generated using the same simulation environment described in the paper. The datasets are based on Ornstein–Uhlenbeck (OU) processes and are intended as a baseline for evaluating causal discovery algorithms.

The benchmark includes both stationary and stock-like OU process simulations (with multiplicative noise), together with their corresponding ground-truth causal graphs.

Users can either:

* Use the pre-generated datasets provided as `.zip` archives in the main directory.
* Generate custom datasets using `dataset.ipynb`, which imports the `ou_process` and `ou_process_stocklike` functions and allows simulation parameters to be modified as needed.

Each dataset contains:

* Multivariate time series (`process.csv`)
* Ground-truth adjacency matrix (`adj_matrix.npy`)
* Metadata describing the generation parameters (`metadata.json`)


📓 **[Benchmark.zip](Benchmark.zip)**


---

## Citation

If you use PLACy in your research, please cite:

```bibtex
@inproceedings{tusoni2026placy,
  title={Robust Causal Discovery in Real-World Time Series with Power-Laws},
  author={Tusoni, Matteo and Masi, Giuseppe and Coletta, Andrea and Glielmo, Aldo and Arrigoni, Viviana and Bartolini, Novella},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```
