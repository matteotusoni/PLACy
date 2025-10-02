# PLACy: Robust Causal Discovery in Real-World Time Series with Power-Laws

## How to Run

Create the environment
```
conda create --name CausalDiscovery python=3.11.11
conda activate CausalDiscovery
pip install -r requirements.txt
```

Run an experiment on synthetic data:
```
python src/run.py -s SEED --n_vars N_VARS --length LENGTH --causal_strength C  --method METHOD --window_length W --stride S
```

Run an experiment on real data:
```
python src/run.py -s SEED --dataset DATASET_NAME --method METHOD --window_length W --stride S
```

If you use this code, please cite the following paper:

Tusoni, M., Masi, G., Coletta, A., Glielmo, A., Arrigoni, V., & Bartolini, N. (2025).  
**Robust Causal Discovery in Real‑World Time Series with Power‑Laws**.  
Available at: [https://arxiv.org/abs/2507.12257](https://arxiv.org/abs/2507.12257)
