# Contextual RL & COIN — Reproducible Code-base

This repository accompanies our paper on **contextual reinforcement learning** with the COIN
(COntextual INference) generative model.

The COIN model itself is **not vendored here**. It comes from the external `realtimecoin` package,
which is pinned in `requirements.txt` and installed with the rest of the dependencies; the notebooks
import it directly. (Earlier revisions carried a local `coin.py` port — that file and its private
helpers have been removed.)

```text
.
├── environments.py      # parametrisable Gymnasium environments (custom)
├── rl.py                # tabular & contextual (COIN-conditioned) RL agents
├── figures.ipynb        # reproduces the paper figures
├── basics.ipynb         # templates & demos for COIN, the environments and the agents
├── utils/
│   └── plot_utils.py    # context-probability plotting helpers
├── figures/             # generated figures (committed)
├── models/              # trained model / results checkpoints (committed)
├── video/               # rendered rollout videos and GIFs
├── requirements.txt
└── readme.md
```

---

## Notebooks

### `figures.ipynb`

Reproduces the figures for the IEEE Transactions on Cognitive and Developmental Systems paper:

| Section | Content |
| --- | --- |
| Figure 1 | Contextualisation — COIN responsibilities over a perturbation sequence |
| Figure 2 | Training — COIN-Q vs. a single-policy baseline |
| Figure 3 | Generalization — performance across several control tasks |
| Figure 4 | Curriculum — curriculum vs. no-curriculum learning on Mountain Car |
| Appendix A1 | Training Mountain Car with PPO |
| Appendix A2 | COIN with cues |

### `basics.ipynb`

Worked templates and demos rather than paper results: a COIN template (offline batch and real-time
per-trial use), a Gymnasium/Q-learning walkthrough, contextual COIN-Q training, and the video cells
that render the clips in `video/`.

### Cached results (`TRAIN` flag)

Each figure section of `figures.ipynb` defines a `TRAIN` flag. With `TRAIN = False` (the committed
default) the section **loads the cached arrays from `models/`** and only redraws the plots, which
runs in seconds. Set `TRAIN = True` to re-run the underlying training — this is slow, and it
overwrites the checkpoints in `models/` and the figures in `figures/`. Both directories hold
committed generated output, so a plot-only pass should leave the working tree clean.

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

Then launch the notebooks:

```bash
jupyter lab
```
