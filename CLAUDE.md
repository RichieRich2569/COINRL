# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code accompanying a paper on contextual reinforcement learning with the COIN
(COntextual INference) generative model, targeting IEEE Transactions on Cognitive and
Developmental Systems. The deliverables are the paper figures — most work happens in
`figures.ipynb` and the Python modules it imports.

## Setup and commands

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
jupyter lab                        # run the notebooks
```

- The COIN model is NOT in this repo. It comes from the external `realtimecoin` package,
  pinned in `requirements.txt` as `realtimecoin @ git+https://github.com/RichieRich2569/real_time_md_coin.git@py-v0.1.0`.
  The `@ref` is mandatory (the default branch is the MATLAB reference with no `pyproject.toml`).
  For local development instead: `pip install -e "../real_time_md_coin/python"`.
- `pytest` is listed in requirements but there are currently no test files in the repo.
- Current working branch is `realtime-coinrl` (conversion of everything to the real-time
  COIN API); `main` is the PR target.

## Architecture

Three Python layers feed two notebooks:

- **`environments.py`** — parametrisable Gymnasium classic-control envs (`Custom{MountainCar,CartPole,Pendulum,Acrobot}Env`), built on a `TimeLimitMixin` that replaces the `TimeLimit` wrapper. The `*XEnv` variants (`MountainCarXEnv`, `CartPoleXEnv`, `AcrobotXEnv`) use `_PadToAcrobotInterfaceMixin` to pad every env to a common observation/action interface so one agent can run across tasks (Figure 3).
- **`rl.py`** — all agents. Tabular: `QLearningAgent`, `COINQLearningAgent` (a database of Q-tables soft-weighted by COIN context responsibilities), `EmbodiedCOINQLearningAgent`. Neural: `PPOAgent`, `COINPPOAgent`, `EmbodiedCOINPPOAgent`, `AmortisedCOINPPOAgent` (+ `ContingencyEncoder`, `CoinPredictive` for amortised inference). Agents don't own the COIN model — they accept context-probability vectors (`p_context`); the notebooks run COIN and feed responsibilities in. **Before changing `AmortisedCOINPPOAgent`, read `.claude/skills/amortised-coin/SKILL.md`** — the working log of what is implemented, what was built and deleted (and why), every experiment run with its numbers, and the named failure modes. It exists to stop re-implementing removed mechanisms and re-deriving measured conclusions.
- **`baselines/`** — Figure 2 comparison methods (CMDP-Q, Context-QL, HM-MDP, O-TempLe), each exposing a `run_single_rep_*` entry point consumed by `figures.ipynb` via `multiprocess`. Each module's docstring states its paper source and the Figure-2 parity rules it must follow (30×30 discretisation, α=0.1, γ=0.99, ε 1.0→0.01 decayed 0.999/episode, 200-step episodes). Keep new baselines on those parity rules.

### realtimecoin ↔ agent layout convention

`realtimecoin` queries return fixed-width `(max_contexts + 1,)` vectors in a *global* frame:
known contexts at indices `0..k-1`, novel at index `k`, padding above. The agents expect
`[known..., nan..., novel-last]` with `np.nan` marking uninstantiated slots. Convert with
`rl.coin_context_vector` / `rl.coin_context_trace` — don't hand raw query vectors to agents. `realtimecoin` is imported lazily inside functions
so `rl.py` stays importable without it.

### Notebook conventions

- **`figures.ipynb`** reproduces the paper figures (Fig 1 contextualisation, Fig 2 training vs. baselines, Fig 3 generalisation, Fig 4 curriculum, appendices). Each figure section defines a `TRAIN` flag, committed as `TRAIN = False`: it then loads cached arrays from `models/` and only redraws plots (seconds). `TRAIN = True` re-runs training (slow, parallelised with `multiprocess`) and **overwrites** the committed checkpoints in `models/` and figures in `figures/`. A plot-only pass should leave the working tree clean — treat unexpected diffs in `models/` or `figures/` as a red flag.
- **`basics.ipynb`** is templates/demos (COIN usage, Q-learning walkthrough, video rendering into `video/`), not paper results.
- `models/` file naming encodes the figure: `fig{N}_{method}.npz`.
