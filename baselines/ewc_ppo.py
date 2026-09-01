"""
ewc_ppo.py

**PPO + online Elastic Weight Consolidation** -- the regularisation-based continual-learning
baseline for Figure 3 v2 (``fig3_amortised_plan.md``, Part I section 3.2 and section 8).

Paper source
------------
Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks*, PNAS 114(13),
2017 -- the quadratic Fisher penalty. Online (single-anchor, bounded-memory) variant:
Schwarz et al., *Progress & Compress: A scalable framework for continual learning*,
ICML 2018, eq. (8): ``F_bar <- gamma_ewc * F_bar + F_k``.

Mechanism
---------
One actor-critic runs straight through the blocked stream, exactly like the single-PPO
floor, with one addition. At the end of task block ``k`` the current weights are anchored,
``theta* <- theta``, and a **diagonal Fisher** estimates how much each weight mattered for
the block just finished,

    F_i = E_{s ~ block, a ~ pi_theta(.|s)} [ ( d log pi_theta(a|s) / d theta_i )^2 ] ,

accumulated into a single running matrix ``F_bar <- gamma_ewc * F_bar + F_k``
(``gamma_ewc = 0.95``). Every later PPO minibatch then minimises

    L = L_PPO(theta)  +  (lam / 2) * sum_i F_bar_i (theta_i - theta*_i)^2 ,

injected through :func:`fig3_common.ppo_update`'s ``extra_loss`` hook, so the PPO terms
themselves are bit-for-bit the ones every other Figure-3 method uses. Intuitively: a
quadratic leash whose stiffness per weight is that weight's importance to the past.

What is GRANTED
---------------
**Block boundaries.** EWC cannot exist without them -- it needs to know *when* to snapshot
``theta*`` and estimate ``F_k``. They are handed to it directly from the blocked schedule
(:func:`fig3_common.blocked_schedule`), a handicap in the baseline's favour that the paper
states. Nothing else: no task identity, no test-time signal.

Limitation probed
-----------------
The plasticity-stability dilemma of a *single shared network protected by weight anchoring*.
The leash that preserves MountainCar is the same leash that impairs learning CartPole, and
-- decisively -- there is **no test-time task inference at all**: one parameter vector must
serve all five tasks simultaneously, so evaluation is a plain greedy PPO policy
(:class:`fig3_common.GreedyPPOPolicy`). Our method sidesteps the dilemma by *routing*
instead of *constraining*.

Fisher / anchoring choice (documented per the plan's request)
-------------------------------------------------------------
Both networks are anchored, each with importances from **its own** output distribution --
the standard practical reading of Kirkpatrick's ``F = E[(d log p(y|x) / d theta)^2]`` when
the model has two heads:

* **Policy parameters** -- the exact Fisher above. ``log pi`` *is* the model's predictive
  log-likelihood, so this is the cited formulation verbatim. With only three discrete
  actions the expectation over ``a`` is taken **exactly**,
  ``F_i = sum_a pi(a|s) (d log pi(a|s)/d theta_i)^2``, rather than by sampling one action
  (``fisher_mode="exact"``, the default) -- same estimator, no Monte-Carlo noise, three
  backward passes per state. ``fisher_mode="empirical"`` instead uses the single stored
  behaviour action (the "empirical Fisher"), which is what most RL EWC code ships and is
  the only option for continuous actions.
* **Value parameters** -- the critic emits no distribution over actions, so a Fisher from
  ``log pi`` is identically zero there and would leave the critic completely unleashed
  (it would then drag the shared advantage signal across blocks). Reading the critic as the
  mean of a unit-variance Gaussian predictive ``N(V_theta(s), 1)``, the same Fisher formula
  gives ``F_i = E_s[(d V_theta(s) / d theta_i)^2]`` -- one extra backward pass per state.
  Set ``anchor_value=False`` for the policy-only variant.

States for the expectation are drawn from the block's **last ``fisher_rollouts=5``
rollouts** (plan section 8), subsampled to ``fisher_samples=512`` transitions with a
seeded generator; the stored pool is cleared at each boundary so a block's Fisher only ever
sees that block's data.

``lam`` is a kwarg (default 100.0) so the pilot grid ``{10, 100, 1000}`` (plan section 8) is
a cheap sweep from the notebook.

Conventions
-----------
Figure-2/Figure-3 baseline conventions throughout: module-level pool entry point
:func:`run_single_rep_ewc_ppo` with its heavy imports inside so it pickles for
``multiprocess``; every rollout-shape and eval parameter overridable with the spec default;
``time.perf_counter`` timings for the compute table. All protocol machinery -- schedule,
env factories, rollout collection, GAE, PPO update, evaluation, seeding -- comes from
:mod:`baselines.fig3_common`; nothing is re-derived here and ``rl.py`` is untouched.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from baselines.fig3_common import (
    BLOCK_SIZES,
    EVAL_EPISODES,
    EVAL_MAX_STEPS,
    MAX_EPISODE_STEPS,
    MB_SIZE,
    MINI_EPOCHS,
    N_SEGMENTS,
    N_TASKS,
    PPO_KWARGS,
    SEG_STEPS,
    TASK_NAMES,
    GreedyPPOPolicy,
    blocked_schedule,
    checkpoints_from,
    close_envs,
    collect_rollout,
    count_parameters,
    evaluate_all_tasks,
    make_task_envs,
    ppo_update,
    seed_everything,
    train_env_seeds,
)

__all__ = ["OnlineEWC", "run_single_rep_ewc_ppo",
           "DEFAULT_LAMBDA", "DEFAULT_GAMMA_EWC",
           "DEFAULT_FISHER_ROLLOUTS", "DEFAULT_FISHER_SAMPLES"]


#: Penalty strength ``lam`` in ``(lam/2) * sum_i F_bar_i (theta_i - theta*_i)^2``.
#: Pilot grid in the notebook: {10, 100, 1000} (plan section 8).
DEFAULT_LAMBDA = 100.0

#: Online-Fisher decay ``F_bar <- gamma_ewc * F_bar + F_k`` (Schwarz et al. 2018).
DEFAULT_GAMMA_EWC = 0.95

#: Rollouts at the end of each block whose transitions feed the Fisher estimate.
DEFAULT_FISHER_ROLLOUTS = 5

#: Transitions subsampled from that pool for the (per-sample-gradient) Fisher estimate.
DEFAULT_FISHER_SAMPLES = 512


# ======================================================================================
# Online EWC state
# ======================================================================================

class OnlineEWC:
    """
    Online-EWC bookkeeping for one :class:`rl.PPOAgent`: running diagonal Fisher, weight
    anchor, and the differentiable penalty handed to
    :func:`fig3_common.ppo_update`'s ``extra_loss`` hook.

    The object owns no networks -- it keys everything off the agent's ``named_parameters``
    (``"policy.<name>"`` / ``"value.<name>"``), so the agent stays a stock ``PPOAgent``.

    Args:
        agent (rl.PPOAgent): The agent whose parameters are protected.
        lam (float): Penalty strength ``lam``. Default :data:`DEFAULT_LAMBDA`.
        gamma_ewc (float): Online-Fisher decay. Default :data:`DEFAULT_GAMMA_EWC`.
        anchor_value (bool): Anchor the critic too, with importances from
            ``(d V(s)/d theta)^2`` (see the module docstring). Default ``True``.
        fisher_mode (str): ``"exact"`` -- expectation over actions taken analytically,
            ``sum_a pi(a|s) (grad log pi(a|s))^2`` (discrete actions only, the default);
            ``"empirical"`` -- single stored behaviour action per state.
        fisher_samples (int): Transitions subsampled per Fisher estimate.
        rng (np.random.Generator, optional): Generator for that subsampling; a seeded
            default keeps a repetition reproducible.

    Attributes:
        n_consolidations (int): Number of block boundaries consolidated so far. The penalty
            is inactive (and :meth:`penalty` is never asked for) while this is 0.
        fisher (dict): ``key -> torch.Tensor`` running ``F_bar``.
        anchor (dict): ``key -> torch.Tensor`` the anchored ``theta*``.
    """

    def __init__(self, agent, lam: float = DEFAULT_LAMBDA,
                 gamma_ewc: float = DEFAULT_GAMMA_EWC, anchor_value: bool = True,
                 fisher_mode: str = "exact", fisher_samples: int = DEFAULT_FISHER_SAMPLES,
                 rng: Optional[np.random.Generator] = None):
        import torch

        if fisher_mode not in ("exact", "empirical"):
            raise ValueError(f"fisher_mode must be 'exact' or 'empirical', got {fisher_mode!r}")

        self._torch = torch
        self.agent = agent
        self.lam = float(lam)
        self.gamma_ewc = float(gamma_ewc)
        self.anchor_value = bool(anchor_value)
        # An exact expectation over actions needs a finite action set.
        self.fisher_mode = "empirical" if agent.action_continuous else fisher_mode
        self.fisher_samples = int(fisher_samples)
        self.rng = np.random.default_rng() if rng is None else rng

        self.n_consolidations = 0
        self.fisher: Dict[str, Any] = {}
        self.anchor: Dict[str, Any] = {}

        # Transitions of the current block's most recent rollouts (obs, act), FIFO.
        self._pool: List[Tuple[Any, np.ndarray]] = []

    # ---------------------------------------------------------------- parameter views
    def _policy_params(self) -> List[Tuple[str, Any]]:
        return [("policy." + n, p) for n, p in self.agent.policy.named_parameters()
                if p.requires_grad]

    def _value_params(self) -> List[Tuple[str, Any]]:
        if not self.anchor_value:
            return []
        return [("value." + n, p) for n, p in self.agent.value_net.named_parameters()
                if p.requires_grad]

    def _all_params(self) -> List[Tuple[str, Any]]:
        return self._policy_params() + self._value_params()

    # ---------------------------------------------------------------- experience pool
    def observe_rollout(self, batch, keep: int = DEFAULT_FISHER_ROLLOUTS) -> None:
        """
        Remember one rollout's ``(obs, act)`` for the next Fisher estimate.

        Only the most recent ``keep`` rollouts of the *current* block are retained (plan
        section 8: "the block's last 5 rollouts"); :meth:`consolidate` empties the pool.
        No copy is made -- ``fig3_common.collect_rollout`` hands back freshly built CPU
        tensors that nothing else mutates.
        """
        self._pool.append((batch.obs, np.asarray(batch.act)))
        if len(self._pool) > int(keep):
            del self._pool[:len(self._pool) - int(keep)]

    @property
    def pool_size(self) -> int:
        """Transitions currently pooled for the next Fisher estimate."""
        return int(sum(o.shape[0] for o, _ in self._pool))

    # ---------------------------------------------------------------- Fisher / anchor
    def _estimate_fisher(self) -> Dict[str, Any]:
        """
        Diagonal Fisher over the pooled transitions (see the module docstring for the
        estimator and why the critic gets its own).

        Returns:
            dict: ``key -> torch.Tensor`` of the same shapes as the anchored parameters.
        """
        torch = self._torch
        agent = self.agent

        pol_params = self._policy_params()
        val_params = self._value_params()
        fisher = {k: torch.zeros_like(p) for k, p in pol_params + val_params}
        if not self._pool:
            return fisher

        obs = torch.cat([o for o, _ in self._pool]).to(agent.device)
        acts = np.concatenate([a for _, a in self._pool])
        n_pool = obs.shape[0]
        n = min(int(self.fisher_samples), n_pool)
        idx = self.rng.choice(n_pool, size=n, replace=False)

        pol_tensors = [p for _, p in pol_params]
        val_tensors = [p for _, p in val_params]

        for i in idx:
            obs_i = obs[int(i):int(i) + 1]

            # ---- policy: F = E_a[(d log pi(a|s)/d theta)^2] ----
            if agent.action_continuous:
                mu = agent.policy(obs_i)
                std = agent.log_std.exp().expand_as(mu)
                dist = torch.distributions.Normal(mu, std)
                act_i = torch.as_tensor(np.asarray(acts[int(i)]), dtype=torch.float32,
                                        device=agent.device).view(1, -1)
                terms = [(1.0, dist.log_prob(act_i).sum())]
            else:
                logp_all = torch.log_softmax(agent.policy(obs_i), dim=-1)[0]
                if self.fisher_mode == "exact":
                    probs = logp_all.exp().detach()
                    terms = [(float(probs[a]), logp_all[a]) for a in range(logp_all.shape[0])]
                else:
                    terms = [(1.0, logp_all[int(acts[int(i)])])]

            for j, (weight, scalar) in enumerate(terms):
                grads = torch.autograd.grad(scalar, pol_tensors,
                                            retain_graph=(j < len(terms) - 1))
                for (k, _), g in zip(pol_params, grads):
                    fisher[k] += weight * g.detach().pow(2)

            # ---- value: F = E_s[(d V(s)/d theta)^2] (unit-variance Gaussian predictive) ----
            if val_tensors:
                v = agent.value_net(obs_i).squeeze()
                grads = torch.autograd.grad(v, val_tensors)
                for (k, _), g in zip(val_params, grads):
                    fisher[k] += g.detach().pow(2)

        for k in fisher:
            fisher[k] /= float(n)
        return fisher

    def consolidate(self) -> Dict[str, float]:
        """
        Consolidate at a (granted) block boundary: estimate ``F_k`` from the pooled
        transitions, apply ``F_bar <- gamma_ewc * F_bar + F_k``, re-anchor
        ``theta* <- theta``, and clear the pool.

        Returns:
            dict: ``block_fisher_norm`` (sum of ``F_k``), ``fisher_norm`` (sum of the
            updated ``F_bar``), ``n_samples`` (transitions the estimate averaged over),
            ``seconds``.
        """
        torch = self._torch
        t0 = time.perf_counter()

        n_used = min(int(self.fisher_samples), self.pool_size)
        f_new = self._estimate_fisher()

        if not self.fisher:
            self.fisher = {k: v.clone() for k, v in f_new.items()}
        else:
            for k, v in f_new.items():
                self.fisher[k] = self.gamma_ewc * self.fisher[k] + v

        with torch.no_grad():
            self.anchor = {k: p.detach().clone() for k, p in self._all_params()}

        self._pool.clear()
        self.n_consolidations += 1

        return {"block_fisher_norm": float(sum(float(v.sum()) for v in f_new.values())),
                "fisher_norm": float(sum(float(v.sum()) for v in self.fisher.values())),
                "n_samples": int(n_used),
                "seconds": float(time.perf_counter() - t0)}

    # ---------------------------------------------------------------- penalty
    @property
    def active(self) -> bool:
        """``True`` once at least one block boundary has been consolidated."""
        return self.n_consolidations > 0 and self.lam != 0.0

    def penalty(self, agent=None):
        """
        ``(lam/2) * sum_i F_bar_i (theta_i - theta*_i)^2`` as a differentiable scalar.

        Signature matches :func:`fig3_common.ppo_update`'s ``extra_loss`` contract
        (``extra_loss(agent) -> torch.Tensor``); the ``agent`` argument is accepted and
        ignored -- the penalty always refers to the agent this object was built on.
        """
        torch = self._torch
        total = None
        for k, p in self._all_params():
            f = self.fisher.get(k)
            a = self.anchor.get(k)
            if f is None or a is None:
                continue
            term = (f * (p - a).pow(2)).sum()
            total = term if total is None else total + term
        if total is None:                        # pragma: no cover - active implies params
            return torch.zeros((), device=self.agent.device)
        return 0.5 * self.lam * total


# ======================================================================================
# Pool entry point
# ======================================================================================

def run_single_rep_ewc_ppo(rep_id, lam=DEFAULT_LAMBDA, gamma_ewc=DEFAULT_GAMMA_EWC,
                           fisher_rollouts=DEFAULT_FISHER_ROLLOUTS,
                           fisher_samples=DEFAULT_FISHER_SAMPLES, fisher_mode="exact",
                           anchor_value=True, n_rollouts_per_block=BLOCK_SIZES,
                           n_segments=N_SEGMENTS, seg_steps=SEG_STEPS,
                           mini_epochs=MINI_EPOCHS, mb_size=MB_SIZE,
                           eval_episodes=EVAL_EPISODES, eval_max_steps=EVAL_MAX_STEPS,
                           max_episode_steps=MAX_EPISODE_STEPS, ppo_kwargs=None,
                           progress=True, return_agent=False):
    """
    One repetition of **PPO + online EWC** on the Figure 3 v2 blocked stream.

    Identical to :func:`fig3_common.run_single_rep_single_ppo` -- same schedule, same
    rollout shape, same PPO hypers, same frozen evaluation -- with the EWC penalty added to
    every minibatch loss and a Fisher consolidation at each granted block boundary. The
    difference between this run and the single-PPO floor is therefore attributable to EWC
    alone.

    Pool-ready in the Figure-2 style: module-level, heavy imports inside, plain numpy/dict
    return.

    Args:
        rep_id (int): Repetition index, used directly as the seed (torch, numpy, env
            streams, Fisher subsampling).
        lam (float): EWC penalty strength. Default 100.0; pilot grid {10, 100, 1000}.
        gamma_ewc (float): Online-Fisher decay ``F_bar <- gamma_ewc * F_bar + F_k``.
            Default 0.95.
        fisher_rollouts (int): Rollouts at the end of each block pooled for the Fisher
            estimate. Default 5.
        fisher_samples (int): Transitions subsampled from that pool. Default 512.
        fisher_mode (str): ``"exact"`` (default; expectation over the 3 actions taken
            analytically) or ``"empirical"`` (stored behaviour action only).
        anchor_value (bool): Anchor the critic as well as the policy. Default ``True``.
        n_rollouts_per_block (sequence of int): Rollouts per task block, in
            :data:`fig3_common.TASKS` order. Default (300, 100, 50, 50, 50). Block ends are
            both the EWC consolidation points and the eval checkpoints.
        n_segments, seg_steps (int): Rollout shape (default 8 x 256 = 2048 transitions).
        mini_epochs, mb_size (int): PPO minibatch schedule.
        eval_episodes, eval_max_steps (int): Frozen-eval budget per heatmap cell.
        max_episode_steps (int): Env time limit (200 everywhere in Figure 3).
        ppo_kwargs (dict, optional): Overrides merged onto :data:`fig3_common.PPO_KWARGS`.
        progress (bool): Show a ``tqdm`` bar.
        return_agent (bool): Also return ``(agent, ewc)`` for debugging.

    Returns:
        dict: The :func:`fig3_common.run_single_rep_single_ppo` schema
        (``A_raw`` ``(5, n_ckpt, eval_episodes)``, ``train_returns``, ``task_ids``,
        ``policy_loss``/``value_loss``/``entropy``, ``collect_seconds``/``update_seconds``/
        ``rollout_seconds``/``eval_seconds``/``total_seconds``, ``checkpoints``,
        ``block_sizes``, ``env_steps``, ``n_params``, ``task_names``, ``method``, ``seed``,
        ``meta``) with ``method = "ewc_ppo"``, plus these EWC-specific extras:

        ==========================  ====================================================
        key                         value
        ==========================  ====================================================
        ``lam``                     ``float`` penalty strength used
        ``gamma_ewc``               ``float`` online-Fisher decay used
        ``penalty``                 ``float64 (n_rollouts,)`` value of the EWC penalty on
                                    the last minibatch of that rollout's update (0 before
                                    the first consolidation)
        ``fisher_norm``             ``float64 (n_blocks,)`` sum of ``F_bar`` after each
                                    block's consolidation
        ``block_fisher_norm``       ``float64 (n_blocks,)`` sum of that block's own ``F_k``
        ``fisher_samples_used``     ``int64 (n_blocks,)`` transitions each estimate averaged
        ``fisher_seconds``          ``float64 (n_blocks,)`` wall-clock per consolidation
                                    (excluded from ``rollout_seconds``, included in
                                    ``total_seconds``)
        ``consolidation_rollouts``  ``int64 (n_blocks,)`` 1-based rollout index of each
                                    consolidation (== ``checkpoints``)
        ==========================  ====================================================
    """
    # Heavy imports inside the function (Figure-2 pool convention).
    import numpy as _np
    from tqdm.auto import tqdm

    from rl import PPOAgent

    _NT, _TN, _PK = N_TASKS, TASK_NAMES, PPO_KWARGS

    seed = int(rep_id)
    seed_everything(seed)

    blocks = _np.asarray(list(n_rollouts_per_block), dtype=int)
    schedule = blocked_schedule(blocks)
    ckpts = checkpoints_from(blocks)
    n_rollouts = int(schedule.size)
    n_blocks = int(ckpts.size)
    hypers = dict(_PK)
    if ppo_kwargs:
        hypers.update(ppo_kwargs)

    # One agent for the whole stream (built on a CartPole env purely to read the shared
    # padded 6-d / Discrete(3) interface -- every Figure-3 task exposes the same one).
    proto_env = make_task_envs(3, 1, seed=seed, max_episode_steps=max_episode_steps)[0]
    agent = PPOAgent(proto_env, **hypers)
    proto_env.close()

    ewc = OnlineEWC(agent, lam=lam, gamma_ewc=gamma_ewc, anchor_value=anchor_value,
                    fisher_mode=fisher_mode, fisher_samples=fisher_samples,
                    rng=_np.random.default_rng(seed + 4_242))

    seeds = train_env_seeds(seed, n_rollouts, n_segments)

    A_raw = _np.full((_NT, n_blocks, int(eval_episodes)), _np.nan, dtype=_np.float64)
    train_returns = _np.full(n_rollouts, _np.nan)
    reward_per_step = _np.full(n_rollouts, _np.nan)
    seg_returns = _np.full((n_rollouts, int(n_segments)), _np.nan)
    pol_loss = _np.full(n_rollouts, _np.nan)
    val_loss = _np.full(n_rollouts, _np.nan)
    ent = _np.full(n_rollouts, _np.nan)
    penalty = _np.zeros(n_rollouts)
    t_collect = _np.zeros(n_rollouts)
    t_update = _np.zeros(n_rollouts)
    t_eval = _np.zeros(n_blocks)
    t_fisher = _np.zeros(n_blocks)
    fisher_norm = _np.full(n_blocks, _np.nan)
    block_fisher_norm = _np.full(n_blocks, _np.nan)
    fisher_used = _np.zeros(n_blocks, dtype=_np.int64)

    t_start = time.perf_counter()
    bar = tqdm(range(n_rollouts), desc=f"EWC-PPO rep {rep_id}", disable=not progress)
    ckpt_i = 0

    for r in bar:
        task = int(schedule[r])

        t0 = time.perf_counter()
        envs = make_task_envs(task, int(n_segments), seed=int(seeds[r]),
                              max_episode_steps=max_episode_steps)
        batch = collect_rollout(agent, envs, int(seg_steps), task_id=task)
        close_envs(envs)
        t1 = time.perf_counter()

        # The EWC penalty is added to *every* minibatch loss; it is simply absent until the
        # first block boundary has been consolidated.
        stats = ppo_update(agent, batch, int(mini_epochs), int(mb_size),
                           extra_loss=ewc.penalty if ewc.active else None)
        t2 = time.perf_counter()

        # Keep this rollout in the pool the next consolidation will draw its states from.
        ewc.observe_rollout(batch, keep=int(fisher_rollouts))

        t_collect[r], t_update[r] = t1 - t0, t2 - t1
        train_returns[r] = stats["mean_episode_return"]
        reward_per_step[r] = stats["mean_reward_per_step"]
        seg_returns[r, :batch.seg_returns.size] = batch.seg_returns
        pol_loss[r], val_loss[r], ent[r] = (
            stats["policy_loss"], stats["value_loss"], stats["entropy"])
        penalty[r] = stats["penalty"]

        if progress:
            bar.set_postfix(task=_TN[task], ret=f"{train_returns[r]:.1f}",
                            pen=f"{penalty[r]:.2e}")

        # ---- block end: GRANTED boundary -> consolidate, then frozen eval ----
        if ckpt_i < n_blocks and (r + 1) == int(ckpts[ckpt_i]):
            info = ewc.consolidate()          # before eval: consolidation never moves theta

            te = time.perf_counter()
            policy = GreedyPPOPolicy(agent)   # no task inference exists -- that is the point
            cell = evaluate_all_tasks(
                lambda _t: policy, int(eval_episodes), int(eval_max_steps), seed,
                max_episode_steps=max_episode_steps)
            agent.policy.train()
            elapsed = time.perf_counter() - te

            # A zero-length block makes two checkpoints coincide; fill both from the one
            # evaluation rather than silently leaving a nan column.
            first = True
            while ckpt_i < n_blocks and (r + 1) == int(ckpts[ckpt_i]):
                A_raw[:, ckpt_i, :] = cell
                t_eval[ckpt_i] = elapsed
                t_fisher[ckpt_i] = info["seconds"] if first else 0.0
                fisher_norm[ckpt_i] = info["fisher_norm"]
                block_fisher_norm[ckpt_i] = info["block_fisher_norm"]
                fisher_used[ckpt_i] = info["n_samples"]
                elapsed = 0.0
                first = False
                ckpt_i += 1

    total = time.perf_counter() - t_start

    result = {
        "A_raw": A_raw,
        "train_returns": train_returns,
        "train_reward_per_step": reward_per_step,
        "seg_returns": seg_returns,
        "task_ids": schedule.astype(_np.int64),
        "policy_loss": pol_loss,
        "value_loss": val_loss,
        "entropy": ent,
        "collect_seconds": t_collect,
        "update_seconds": t_update,
        "rollout_seconds": t_collect + t_update,
        "eval_seconds": t_eval,
        "total_seconds": float(total),
        "checkpoints": ckpts.astype(_np.int64),
        "block_sizes": blocks.astype(_np.int64),
        "env_steps": (_np.arange(1, n_rollouts + 1, dtype=_np.int64)
                      * int(n_segments) * int(seg_steps)),
        "n_params": count_parameters(agent),
        "task_names": list(_TN),
        "method": "ewc_ppo",
        "seed": seed,
        # ---- EWC-specific ----
        "lam": float(lam),
        "gamma_ewc": float(gamma_ewc),
        "penalty": penalty,
        "fisher_norm": fisher_norm,
        "block_fisher_norm": block_fisher_norm,
        "fisher_samples_used": fisher_used,
        "fisher_seconds": t_fisher,
        "consolidation_rollouts": ckpts.astype(_np.int64),
        "meta": {"seed": seed, "n_segments": int(n_segments),
                 "seg_steps": int(seg_steps),
                 "rollout_steps": int(n_segments) * int(seg_steps),
                 "mini_epochs": int(mini_epochs), "mb_size": int(mb_size),
                 "max_episode_steps": int(max_episode_steps),
                 "eval_episodes": int(eval_episodes),
                 "eval_max_steps": int(eval_max_steps),
                 "block_sizes": blocks.tolist(), "checkpoints": ckpts.tolist(),
                 "lam": float(lam), "gamma_ewc": float(gamma_ewc),
                 "fisher_rollouts": int(fisher_rollouts),
                 "fisher_samples": int(fisher_samples),
                 "fisher_mode": ewc.fisher_mode, "anchor_value": bool(anchor_value),
                 "granted": "block boundaries",
                 **hypers},
    }
    if return_agent:
        return result, agent, ewc
    return result
