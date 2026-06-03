# PLACy: Robust Causal Discovery in Real-World Time Series with Power-Laws

🏆 **ICML 2026 Spotlight (Top 2%)**

**Matteo Tusoni, Gianmarco Masi, Andrea Coletta, Alessandro Glielmo, Valerio Arrigoni, Nicola Bartolini**

**Robust Causal Discovery in Real-World Time Series with Power-Laws**

Accepted at the **International Conference on Machine Learning (ICML 2026)** as a **Spotlight** paper.

📄 Paper: https://arxiv.org/abs/2507.12257

---

## Installation

Create the environment:

```bash
conda create --name CausalDiscovery python=3.11.11
conda activate CausalDiscovery
pip install -r requirements.txt
```

---

## Quick Start

A complete walkthrough is available in:

📓 **[example.ipynb](example.ipynb)**

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

## Citation

If you use PLACy in your research, please cite:

```bibtex
@article{tusoni2025placy,
  title={Robust Causal Discovery in Real-World Time Series with Power-Laws},
  author={Tusoni, Matteo and Masi, Gianmarco and Coletta, Andrea and Glielmo, Alessandro and Arrigoni, Valerio and Bartolini, Nicola},
  journal={International Conference on Machine Learning (ICML)},
  year={2026}
}
```
