# Figure 3 v2 — second-machine handoff

State of the amortised COIN-PPO campaign as of 2026-09-01, and the run matrix for a
second machine. Written for a Claude Code session (or a human) picking this up cold.

## Where the campaign stands

Task 4 was swapped from InvertedCartPole (gravity flip) to MirrorCartPole
(force_mag=-10) after the gravity-flip pair proved dynamically inseparable under
blocked arrival. Since then, seven mechanisms landed in `rl.py` (all behind
default-off kwargs, 70 tests green in `test_rl.py`):

- `value_coef` — differentiable-responsibility value term (routing-boundary refiner),
- `repel_coef` — value-surprise repulsion (evicts a task parked on an established
  context whose head's value error explodes vs its running baseline),
- `observe_value` — **2-D COIN observation** `(z, segment-mean episodic return)` via
  the realtimecoin MD pipeline; the return channel births contexts on performance
  collapse before the old head is overwritten,
- `encoder_reward` — raw reward in encoder features + decoder reward-prediction head,
- `decoder_residual` — `s' = s + f(s,a,z)`,
- `rail_coef` — tanh-saturation hinge,
- `z_channel_noise` — noisy z channel forcing inter-task code gaps to a chosen scale.

COIN-side: `prior_mean_retention=0.9995` (novelty support must cover the code range;
default stationary sd is only 0.027) and `FLOOR=0.05`. Known artefact: the *scalar*
COIN pipeline's likelihood underflows beyond ~38σ surprise (linear-space pdf) —
the MD path (used by `observe_value`) is immune.

Since the first handoff (machine-1 findings, all committed):

- Single-delta pilots: wide axis FAILED (crowding relocates merges), undirected
  repulsion FAILED (pushes codes into the tanh rail and pins them), **2-D
  observation SUCCEEDED at train time** (first-ever five distinct contexts; CP and
  Mirror on separate heads via the return-crash birth) but eval collapsed -- the
  z-marginal cannot distinguish contexts whose codes stayed merged, and the value
  channel is a TRANSIENT discriminator (ceilings coincide for mastered tasks).
- New mechanisms since: `episodic_value_steps` (one value-gradient encoder step per
  completed episode -- realized raw return-to-go targets, routing from the segment's
  prefix-so-far = eval-rehearsal evidence, `value_pi_source="stationary"` keeps the
  error signal alive so z diverges until the z-marginal alone routes; plus a
  non-finite-gradient guard) and the raw-return fix for `observe_value` (shaped MC
  returns had inflated its value coordinate ~+0.5).
- STRATEGY CHANGE after the machine-2 crash: NO full pilots until miniatures pass.
  `scripts/pair_stream_test.py` is the workhorse: a two-task block stream with
  expert-bootstrapped heads (policy+value injected at context birth -- diagnostic
  only), per-rollout probe-code traces, and a real `evaluate_identifying` stage,
  ~1-2 h per run with explicit divergence and eval-separation gates printed at the
  end.

## Setup

```bash
git clone <repo> && cd COINRL && git checkout realtime-coinrl
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
# REQUIRED override: use the local real_time_md_coin clone, not the pinned tag —
# the MD pipeline and retention behaviour were verified against it.
pip install -e "<path-to>/real_time_md_coin/python"
python -m pytest test_rl.py -q     # expect 70 passed
```

## Run matrix v2 (post-crash, miniatures only — supersedes everything below)

Machine 1 is already running: E1 = gravflip CP/InvCP pair miniature
(`FIG3_TASK4=gravflip python scripts/pair_stream_test.py --tasks 3 4 --rollouts 50`),
E2 = the Mirror twin (no env var), E3 = MC/FlatMC (`--tasks 0 2`), plus the one
pre-existing full pilot 10b (raw-return verification). Machine 2 should take the
**seed-robustness and coverage arms** of the same miniatures — everything is a
1–2 h run:

1. E1/E2/E3 at seeds 1 and 2 (edit `rl.seed_everything(0)` calls via a `--seed`
   argument if you add one, or run as-is first to reproduce machine 1's seed 0).
2. Remaining pairs at seed 0: `--tasks 0 1` (MC/Acrobot), `--tasks 1 2`
   (Acrobot/FlatMC) — completes the pairwise matrix for the left cluster.
3. An ablation arm of E1: `value_pi_source="predicted"` (edit the constructor line
   in the script) — quantifies what the eval-faithful prior buys.

Do NOT run full pilots or baseline re-runs. Commit each result npz + log with a
descriptive message and push; machine 1 integrates.

## Run matrix v1 (SUPERSEDED — kept for reference)

The runner is `scripts/run_phase.py` (executes a temp copy of figures.ipynb cells
30–37 with substitutions; the repo notebook is never modified). Gate evaluation:
`scripts/eval_pilot.py [path-to-npz]`. Each seed writes
`models/fig3_amortised_s{seed}.npz`, so parallel seeds do not collide.

The **synthesis stack** constructor substitution used below (NOTE 2026-09-01: 
`repel_coef` REMOVED from the default stack — machine-1 pilot 9 showed the undirected
repulsion pushes codes into the tanh rail and pins them there (margin 0.8 > distance
to the rail keeps the hinge active on a saturated, unmovable code). Machine-1 pilot
10 showed the 2-D observation alone achieves 5 distinct train-time contexts. Keep
repulsion OFF until it is made direction-aware):

```
--sub "same_task_rollout=True, **f3.PPO_KWARGS)::same_task_rollout=True, value_coef=1e-3, decoder_residual=True, encoder_reward=True, rail_coef=1.0, z_channel_noise=0.4, observe_value=True, **f3.PPO_KWARGS)"
```

and the shared substitutions for every run:

```
--flags TRAIN_AMORTISED --last 37
--sub "ANCHOR_COEF, ANCHOR_WINDOW, ANCHOR_WARMUP = 1.0, 64, 0::ANCHOR_COEF, ANCHOR_WINDOW, ANCHOR_WARMUP = 0.0, 64, 0"
--sub "KL_COEF, ENCODER_LR, REPLAY_CAPACITY = 0.0, 3e-4, 512::KL_COEF, ENCODER_LR, REPLAY_CAPACITY = 0.0, 1.5e-4, 512"
--sub "FLOOR = 0.01         # <- set from Phase 0 (pooled within-task posterior sd ~0.007)::FLOOR = 0.05         # jump absorption; 16x under channel-forced gaps"
--sub "CTX_IDS = [0, 1, 2, 3, 4, 5, 6, \"novel\"]::CTX_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, \"novel\"]"
--sub "coin = RealTimeCOIN(rng=seed, sigma_motor_noise=float(floor),::coin = RealTimeCOIN(rng=seed, sigma_motor_noise=float(floor), prior_mean_retention=0.9995, state_dim=2, process_noise_covariance=np.diag([0.0089**2, 0.01**2]),"
```

Seed selection: `--sub "FIG3_SEEDS = FIG3_ALL_SEEDS   # pilot gates passed -> full 5-seed protocol::FIG3_SEEDS = [FIG3_ALL_SEEDS[N]]"` with N = 0/1/2.
Runtimes: 2–5 h per seed; ~1 core each, so seeds run in parallel comfortably.

### P1 — Gravity-flip rescue (run FIRST; may cancel all baseline work)

Set `FIG3_TASK4=gravflip` in the shell before launching (the kernel inherits it;
`baselines/fig3_common.py` then restores the original InvertedCartPole as task 4).
Run the synthesis stack, seeds 0–2. Hypothesis: the gravity-flip pair was
inseparable by DYNAMICS signal, but the 2-D return channel discriminates it
(CP↔InvCP cross-transfer is 56–77 vs 200, so the return collapses on the switch and
COIN births the context). **If this passes the gates, the committed gravflip
baselines (`models/fig3_{single_ppo,ewc,owl,rppo,llirl}_s*.npz`) and oracles are
valid again and NO baseline re-runs are needed.**

### P2 — Mirror synthesis, seeds 0–2 (no env var)

Same commands without `FIG3_TASK4`. This is the head-to-head fallback if P1 fails.

### P3 — Ablations (seed 0 only, either task set, whichever P-line won)

- `value_coef=0` under the synthesis stack,
- `z_channel_noise=0` under the synthesis stack.

### HOLD — baseline re-runs

Do NOT start the 25 Mirror baseline reps until P1's verdict: a P1 pass makes them
unnecessary.

## Gates (per seed)

From `scripts/eval_pilot.py`: one context per task at train time (no fragmentation,
no shared heads — especially CP vs task 4), the block-end diagonal near oracles
(MC −84, Acrobot −61, FlatMC −42, CP 200, task4 200), retention of earlier tasks in
the final column, and eval routing to the OWN head (w > 0.9). The campaign rule:
nothing is adopted without 3 passing seeds.

## Reporting back

Commit result npz files + executed-notebook logs to `realtime-coinrl` and push;
machine 1 coordinates through this branch. Keep `models/fig3_amortised_s*.npz`
seed-named and note the config (P1/P2/ablation) in the commit message.
