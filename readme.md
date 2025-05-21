# Contextual RL & COIN — Reproducible Code-base

This repository accompanies our paper on **contextual reinforcement learning** with the COIN generative model.  
It contains:

├── basic_crl.py # fixed-context CECE example
├── coin.py # COIN particle-filter implementation (by Changmin Yu)
├── environments.py # parametrisable Gymnasium envs (custom)
├── rl.py # tabular & contextual Q-learning agents
├── basics.ipynb # basic testing of rl modules and new coin approaches
├── figures.ipynb # all training and evaluation of the paper's figures in a jupyter notebook format.
├── utils/
│ ├── clustering.py
│ ├── distribution_utils.py
│ ├── general_utils.py
│ └── plot_utils.py
├── figures/ # auto-generated figures land here
└── models/ # pre-trained models


---

## Quick-start (pip + virtualenv)

```bash
# 1. create & activate an isolated environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. install packaged dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```


