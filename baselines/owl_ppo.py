"""
owl_ppo.py

**Multi-head PPO + UCB1 bandit head selection** (OWL-style) -- the structural
continual-learning baseline for Figure 3 v2 (``fig3_amortised_plan.md``, Part I
section 3.3 and section 8).

Paper source
------------
Kessler, Parker-Holder, Ball, Zohren & Roberts, *Same State, Different Task: Continual
Reinforcement Learning without Interference*, AAAI 2022 ("OWL"). OWL keeps one policy head
per task on a shared feature extractor and, when the task label is withheld at test time,
selects the head with a multi-armed bandit over episode returns. We reimplement it in-stack
on our own ``rl.PPOAgent`` so that performance *and* wall-clock are comparable with every
other Figure-3 method; following the plan's spec each head is a fully independent
actor-critic (no shared trunk), which is the interference-free limit of OWL's design and
the exact structural analogue of our own PPO head database.

Mechanism
---------
*Training.* Five independent actor-critics. Each rollout is routed to the head named by the
**true task label** from the blocked schedule; only that head collects the rollout and only
that head is updated. Forgetting is zero by construction -- the other four heads are simply
not touched.

*Evaluation.* No label. Per (task, checkpoint) heatmap cell a **fresh UCB1 bandit** runs
over the five heads across that cell's evaluation episodes: each arm is played once
(warm-up), then

    k_t = argmax_k  mu_k + c * sqrt( 2 ln t / n_k )

with ``mu_k`` the mean normalised return of head ``k`` so far, ``n_k`` its play count,
``t`` the total plays, and ``c = 1.0`` (kwarg). The whole episode is played with the chosen
head's greedy policy; the episode's return is the bandit's reward. **The heatmap cell is
the mean over all evaluation episodes, warm-up included** -- the exploration cost is part of
the score, which is the honest reading of "how well does this method do when it is not told
the task". ``chosen_heads`` and the raw per-episode returns are both stored so the appendix
can split warm-up from post-warm-up.

Bandit reward normalisation
---------------------------
UCB1 assumes rewards in ``[0, 1]``, so raw returns are mapped through a fixed affine
function per env family -- ``(r - lo) / (hi - lo)``, clipped to ``[0, 1]`` -- using the
bounds the 200-step cap and the per-step reward structure impose:

* MountainCar / Flat MountainCar / Acrobot (reward ``-1`` per step): ``lo, hi = -200, 0``,
  i.e. ``(r + 200) / 200``;
* CartPole / Inverted CartPole (reward ``+1`` per step): ``lo, hi = 0, 200``, i.e.
  ``r / 200``.

See :data:`RETURN_BOUNDS`. The map is strictly monotone, so it never changes *which* head
looks best -- only the scale on which the ``c * sqrt(2 ln t / n_k)`` exploration bonus
trades off against the estimated means. Override per cell with the
``return_bounds_for_task`` kwarg (e.g. a single family-agnostic ``(-200, 200)`` map).

What is GRANTED
---------------
**Task labels during training** (the routing signal) and **a fixed head count of 5**
matching the number of tasks. Both are handicaps in the baseline's favour, stated in the
paper.

Limitation probed
-----------------
*Identification.* The bandit needs **whole episodes** of reward feedback to discover the
right head, and pays exploration cost forever after -- whereas COIN identifies the context
from tens of *steps* of reward-free sensorimotor evidence. It also cannot detect novelty:
there is no sixth arm to instantiate.

Conventions
-----------
Figure-2/Figure-3 baseline conventions throughout: module-level pool entry point
:func:`run_single_rep_owl_ppo` with its heavy imports inside so it pickles for
``multiprocess``; every rollout-shape and eval parameter overridable with the spec default;
``time.perf_counter`` timings for the compute table. All protocol machinery comes from
:mod:`baselines.fig3_common`; ``rl.py`` is untouched.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence, Tuple

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
    EvalPolicy,
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

__all__ = ["UCB1HeadPolicy", "run_single_rep_owl_ppo",
           "RETURN_BOUNDS", "DEFAULT_UCB_C"]


#: UCB1 exploration constant (plan section 8: ``c = 1``).
DEFAULT_UCB_C = 1.0

#: ``task_idx -> (lo, hi)`` raw-return bounds used to normalise the bandit reward to
#: ``[0, 1]``. Fixed per env family, from the 200-step cap and the per-step reward
#: (``-1`` for the MountainCar/Acrobot family, ``+1`` for the CartPole family) -- see the
#: module docstring. Order matches :data:`fig3_common.TASKS`.
RETURN_BOUNDS: Dict[int, Tuple[float, float]] = {
    0: (-200.0, 0.0),      # MountainCar
    1: (-200.0, 0.0),      # Acrobot
    2: (-200.0, 0.0),      # FlatMountainCar
    3: (0.0, 200.0),       # CartPole
    4: (0.0, 200.0),       # MirrorCartPole
}


# ======================================================================================
# Evaluation policy: UCB1 over the head database
# ======================================================================================

class UCB1HeadPolicy(EvalPolicy):
    """
    Frozen evaluation policy that picks *which head to play* with UCB1, one arm pull per
    evaluation episode.

    The harness (:func:`fig3_common.evaluate_on_task`) calls ``reset()`` at the start of
    every episode -- that is where the arm is chosen -- and ``episode_end(ep_return)`` when
    it finishes -- that is where the bandit is updated. ``act`` just defers to the chosen
    head's greedy policy, so the whole episode is played by a single head.

    A fresh instance is built per (task, checkpoint) cell, so no bandit knowledge leaks
    between cells (matching :func:`fig3_common.evaluate_all_tasks`'s contract).

    Args:
        heads (sequence): The :class:`rl.PPOAgent` heads, in task order.
        c (float): UCB1 exploration constant. Default :data:`DEFAULT_UCB_C`.
        bounds (tuple): ``(lo, hi)`` raw-return bounds for the affine map to ``[0, 1]``.

    Attributes:
        chosen (list[int]): Head index played in each episode so far, in order.
        rewards (list[float]): Normalised bandit reward per episode, in order.
        counts (np.ndarray): ``n_k`` play counts.
        means (np.ndarray): ``mu_k`` mean normalised return per head.
    """

    def __init__(self, heads: Sequence[Any], c: float = DEFAULT_UCB_C,
                 bounds: Tuple[float, float] = (-200.0, 0.0)):
        self.policies = [GreedyPPOPolicy(h) for h in heads]
        self.n_heads = len(self.policies)
        self.c = float(c)
        self.lo, self.hi = float(bounds[0]), float(bounds[1])
        self._span = max(self.hi - self.lo, 1e-12)

        self.counts = np.zeros(self.n_heads, dtype=np.int64)
        self.means = np.zeros(self.n_heads, dtype=np.float64)
        self.t = 0
        self.current = 0
        self.chosen: List[int] = []
        self.rewards: List[float] = []

    # ---------------------------------------------------------------- bandit
    def _select(self) -> int:
        """UCB1: play every arm once, then ``argmax_k mu_k + c*sqrt(2 ln t / n_k)``."""
        unplayed = np.flatnonzero(self.counts == 0)
        if unplayed.size:                       # warm-up, arms in index order
            return int(unplayed[0])
        bonus = self.c * np.sqrt(2.0 * np.log(max(self.t, 1)) / self.counts)
        return int(np.argmax(self.means + bonus))

    def normalise(self, ep_return: float) -> float:
        """Raw episode return -> bandit reward in ``[0, 1]`` (see the module docstring)."""
        return float(np.clip((float(ep_return) - self.lo) / self._span, 0.0, 1.0))

    # ---------------------------------------------------------------- EvalPolicy hooks
    def reset(self) -> None:
        self.current = self._select()

    def act(self, obs):
        return self.policies[self.current].act(obs)

    def episode_end(self, ep_return: float) -> None:
        k = int(self.current)
        r = self.normalise(ep_return)
        self.counts[k] += 1
        self.means[k] += (r - self.means[k]) / float(self.counts[k])
        self.t += 1
        self.chosen.append(k)
        self.rewards.append(r)


# ======================================================================================
# Pool entry point
# ======================================================================================

def run_single_rep_owl_ppo(rep_id, c=DEFAULT_UCB_C, return_bounds_for_task=None,
                           n_rollouts_per_block=BLOCK_SIZES, n_segments=N_SEGMENTS,
                           seg_steps=SEG_STEPS, mini_epochs=MINI_EPOCHS, mb_size=MB_SIZE,
                           eval_episodes=EVAL_EPISODES, eval_max_steps=EVAL_MAX_STEPS,
                           max_episode_steps=MAX_EPISODE_STEPS, ppo_kwargs=None,
                           progress=True, return_agent=False):
    """
    One repetition of **multi-head PPO + UCB1 head selection** on the Figure 3 v2 blocked
    stream.

    Same schedule, rollout shape, PPO hypers and eval budget as
    :func:`fig3_common.run_single_rep_single_ppo`; the only differences are that the rollout
    is routed to the head named by the true task label (granted) and that evaluation must
    find the head with a bandit instead of being told.

    Pool-ready in the Figure-2 style: module-level, heavy imports inside, plain numpy/dict
    return.

    Args:
        rep_id (int): Repetition index, used directly as the seed (torch, numpy, env
            streams).
        c (float): UCB1 exploration constant. Default 1.0.
        return_bounds_for_task (dict or callable, optional): ``task_idx -> (lo, hi)`` raw
            return bounds for the bandit-reward normalisation. Default
            :data:`RETURN_BOUNDS`.
        n_rollouts_per_block (sequence of int): Rollouts per task block, in
            :data:`fig3_common.TASKS` order. Default (300, 100, 50, 50, 50). One head per
            entry; the cumulative sums are the eval checkpoints.
        n_segments, seg_steps (int): Rollout shape (default 8 x 256 = 2048 transitions).
        mini_epochs, mb_size (int): PPO minibatch schedule.
        eval_episodes, eval_max_steps (int): Frozen-eval budget per heatmap cell. Note the
            first ``n_heads`` episodes of every cell are UCB1 warm-up.
        max_episode_steps (int): Env time limit (200 everywhere in Figure 3).
        ppo_kwargs (dict, optional): Overrides merged onto :data:`fig3_common.PPO_KWARGS`.
        progress (bool): Show a ``tqdm`` bar.
        return_agent (bool): Also return the list of heads (debugging).

    Returns:
        dict: The :func:`fig3_common.run_single_rep_single_ppo` schema
        (``A_raw`` ``(5, n_ckpt, eval_episodes)``, ``train_returns``, ``task_ids``,
        ``policy_loss``/``value_loss``/``entropy``, ``collect_seconds``/``update_seconds``/
        ``rollout_seconds``/``eval_seconds``/``total_seconds``, ``checkpoints``,
        ``block_sizes``, ``env_steps``, ``n_params`` (summed over **all** heads),
        ``task_names``, ``method``, ``seed``, ``meta``) with ``method = "owl_ppo"``, plus
        these OWL-specific extras:

        ============================  ==================================================
        key                           value
        ============================  ==================================================
        ``chosen_heads``              ``int64 (n_tasks, n_ckpt, eval_episodes)`` head
                                      played in each evaluation episode
        ``bandit_rewards``            ``float64 (n_tasks, n_ckpt, eval_episodes)``
                                      normalised return fed to UCB1
        ``c``                         ``float`` exploration constant used
        ``n_warmup_episodes``         ``int`` leading episodes of every cell that are
                                      pure UCB1 warm-up (= ``n_heads``)
        ``A_warmup_mean``             ``float64 (n_tasks, n_ckpt)`` mean raw return over
                                      the warm-up episodes only
        ``A_post_warmup_mean``        ``float64 (n_tasks, n_ckpt)`` mean raw return after
                                      warm-up (the "bandit has converged" view)
        ``head_hit_rate``             ``float64 (n_tasks, n_ckpt)`` fraction of
                                      post-warm-up episodes that played the head trained
                                      on that task (chance = ``1/n_heads``)
        ``head_ids``                  ``int64 (n_rollouts,)`` head trained per rollout
                                      (== ``task_ids``: routing is by the true label)
        ``head_update_counts``        ``int64 (n_heads,)`` PPO updates received per head
        ``n_heads``                   ``int``
        ``return_bounds``             ``float64 (n_tasks, 2)`` normalisation map used
        ============================  ==================================================
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
    n_ckpts = int(ckpts.size)
    n_heads = int(blocks.size)
    hypers = dict(_PK)
    if ppo_kwargs:
        hypers.update(ppo_kwargs)

    if return_bounds_for_task is None:
        bounds_fn = lambda t: RETURN_BOUNDS[int(t)]
    elif callable(return_bounds_for_task):
        bounds_fn = return_bounds_for_task
    else:
        bounds_fn = lambda t: return_bounds_for_task[int(t)]

    # One independent actor-critic per task. Built on a CartPole env purely to read the
    # shared padded 6-d / Discrete(3) interface -- every Figure-3 task exposes the same one.
    proto_env = make_task_envs(3, 1, seed=seed, max_episode_steps=max_episode_steps)[0]
    heads = [PPOAgent(proto_env, **hypers) for _ in range(n_heads)]
    proto_env.close()

    seeds = train_env_seeds(seed, n_rollouts, n_segments)

    A_raw = _np.full((_NT, n_ckpts, int(eval_episodes)), _np.nan, dtype=_np.float64)
    chosen = _np.full((_NT, n_ckpts, int(eval_episodes)), -1, dtype=_np.int64)
    bandit_r = _np.full((_NT, n_ckpts, int(eval_episodes)), _np.nan, dtype=_np.float64)
    train_returns = _np.full(n_rollouts, _np.nan)
    reward_per_step = _np.full(n_rollouts, _np.nan)
    seg_returns = _np.full((n_rollouts, int(n_segments)), _np.nan)
    pol_loss = _np.full(n_rollouts, _np.nan)
    val_loss = _np.full(n_rollouts, _np.nan)
    ent = _np.full(n_rollouts, _np.nan)
    t_collect = _np.zeros(n_rollouts)
    t_update = _np.zeros(n_rollouts)
    t_eval = _np.zeros(n_ckpts)
    head_updates = _np.zeros(n_heads, dtype=_np.int64)

    t_start = time.perf_counter()
    bar = tqdm(range(n_rollouts), desc=f"OWL-PPO rep {rep_id}", disable=not progress)
    ckpt_i = 0

    for r in bar:
        task = int(schedule[r])
        # GRANTED: the true task label routes the rollout. Only this head sees the data,
        # so the other four are untouched -- zero forgetting by construction.
        head = heads[task % n_heads]

        t0 = time.perf_counter()
        envs = make_task_envs(task, int(n_segments), seed=int(seeds[r]),
                              max_episode_steps=max_episode_steps)
        batch = collect_rollout(head, envs, int(seg_steps), task_id=task)
        close_envs(envs)
        t1 = time.perf_counter()

        stats = ppo_update(head, batch, int(mini_epochs), int(mb_size))
        t2 = time.perf_counter()
        head_updates[task % n_heads] += 1

        t_collect[r], t_update[r] = t1 - t0, t2 - t1
        train_returns[r] = stats["mean_episode_return"]
        reward_per_step[r] = stats["mean_reward_per_step"]
        seg_returns[r, :batch.seg_returns.size] = batch.seg_returns
        pol_loss[r], val_loss[r], ent[r] = (
            stats["policy_loss"], stats["value_loss"], stats["entropy"])

        if progress:
            bar.set_postfix(task=_TN[task], head=task % n_heads,
                            ret=f"{train_returns[r]:.1f}")

        # ---- checkpoint: frozen eval, NO task label -- a fresh UCB1 bandit per cell ----
        if ckpt_i < n_ckpts and (r + 1) == int(ckpts[ckpt_i]):
            te = time.perf_counter()
            bandits = {t: UCB1HeadPolicy(heads, c=c, bounds=bounds_fn(t))
                       for t in range(_NT)}
            cell = evaluate_all_tasks(
                lambda t: bandits[int(t)], int(eval_episodes), int(eval_max_steps), seed,
                max_episode_steps=max_episode_steps)
            for h in heads:
                h.policy.train()
            elapsed = time.perf_counter() - te

            cell_heads = _np.asarray([bandits[t].chosen for t in range(_NT)],
                                     dtype=_np.int64)
            cell_rew = _np.asarray([bandits[t].rewards for t in range(_NT)],
                                   dtype=_np.float64)

            # A zero-length block makes two checkpoints coincide; fill both from the one
            # evaluation rather than silently leaving a nan column.
            while ckpt_i < n_ckpts and (r + 1) == int(ckpts[ckpt_i]):
                A_raw[:, ckpt_i, :] = cell
                chosen[:, ckpt_i, :] = cell_heads
                bandit_r[:, ckpt_i, :] = cell_rew
                t_eval[ckpt_i] = elapsed
                elapsed = 0.0
                ckpt_i += 1

    total = time.perf_counter() - t_start

    # Warm-up split for the appendix (plan section 8: the heatmap cell itself is the mean
    # over *all* episodes; these are the decomposition, not the headline number).
    n_warm = min(n_heads, int(eval_episodes))
    A_warm = _np.nanmean(A_raw[:, :, :n_warm], axis=2)
    if int(eval_episodes) > n_warm:
        A_post = _np.nanmean(A_raw[:, :, n_warm:], axis=2)
        post_heads = chosen[:, :, n_warm:]
        own = _np.arange(_NT).reshape(_NT, 1, 1) % n_heads
        hit = (post_heads == own).mean(axis=2).astype(_np.float64)
    else:                                        # pragma: no cover - tiny smoke configs
        A_post = _np.full((_NT, n_ckpts), _np.nan)
        hit = _np.full((_NT, n_ckpts), _np.nan)

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
        "n_params": int(sum(count_parameters(h) for h in heads)),
        "task_names": list(_TN),
        "method": "owl_ppo",
        "seed": seed,
        # ---- OWL-specific ----
        "chosen_heads": chosen,
        "bandit_rewards": bandit_r,
        "c": float(c),
        "n_warmup_episodes": int(n_warm),
        "A_warmup_mean": A_warm,
        "A_post_warmup_mean": A_post,
        "head_hit_rate": hit,
        "head_ids": (schedule % n_heads).astype(_np.int64),
        "head_update_counts": head_updates,
        "n_heads": int(n_heads),
        "return_bounds": _np.asarray([bounds_fn(t) for t in range(_NT)],
                                     dtype=_np.float64),
        "meta": {"seed": seed, "n_segments": int(n_segments),
                 "seg_steps": int(seg_steps),
                 "rollout_steps": int(n_segments) * int(seg_steps),
                 "mini_epochs": int(mini_epochs), "mb_size": int(mb_size),
                 "max_episode_steps": int(max_episode_steps),
                 "eval_episodes": int(eval_episodes),
                 "eval_max_steps": int(eval_max_steps),
                 "block_sizes": blocks.tolist(), "checkpoints": ckpts.tolist(),
                 "c": float(c), "n_heads": int(n_heads),
                 "n_warmup_episodes": int(n_warm),
                 "granted": "true task labels at train time; fixed head count = n_tasks",
                 **hypers},
    }
    if return_agent:
        return result, heads
    return result
