"""
fig3_common.py

Shared harness for the **Figure 3 v2** experiment (`fig3_amortised_plan.md`), plus the
single-head PPO floor (plan Part I, section 3.1).

Figure 3 v2 trains one agent sequentially on five classic-control tasks that share a
padded 6-d observation / 3-action interface (`environments.*XEnv`), presented in blocks,
hardest first, with no cue and no task label. Every method in the figure -- ours and the
four baselines (EWC, OWL, recurrent PPO, LLIRL) -- must run on the *identical* protocol so
that performance **and** wall-clock are comparable (plan section 8). This module is that
protocol, factored out once:

* the task table and its env factories (plan Part I, section 1),
* the blocked schedule and its eval checkpoints (plan section 5, decision 10),
* the rollout shape -- 8 segments x 256 steps, every segment from a genuine
  ``env.reset()``, 8 fresh seeded envs per rollout (plan section 2 and section 5,
  decisions 2 and 3),
* a PPO update that matches :meth:`rl.PPOAgent.train_step` term for term but consumes the
  segment-structured rollout, with the no-carry GAE rule (plan section 6, item 3):
  each segment bootstraps from *its own* final observation, never the next segment's,
* the frozen evaluation protocol (100 episodes x 200 steps per task per checkpoint),
  storing **raw** returns -- normalisation against the per-task oracle happens at plot
  time in ``figures.ipynb`` (plan section 4),
* ``time.perf_counter`` timing hooks for the compute table (plan section 4).

Baseline modules import from here rather than re-deriving any of it. Following the
Figure-2 convention (see ``baselines/cmdp_q.py``), the pool entry point
:func:`run_single_rep_single_ppo` is a module-level function with its heavy imports
inside, so it pickles cleanly for ``multiprocess``.

PPO hyper-parameters are fixed across every Figure-3 method (plan section 8):
``gamma=0.995``, ``ent_coef=0.01``, ``lr=3e-4``, nets ``rl._MLP(6, ., 64)``,
``mini_epochs=10``, ``mb_size=64``.

Nothing here touches ``rl.py``; the agents are used as-is.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ======================================================================================
# 1. Task table  (plan Part I, section 1 -- order is load-bearing: it is the block order)
# ======================================================================================

#: Episode cap for every Figure-3 task. 200 keeps segments at 256 steps, i.e. the
#: validated fixed8 rollout structure (plan section 5, decision 3).
MAX_EPISODE_STEPS = 200

#: Shared padded interface (``_PadToAcrobotInterfaceMixin``).
OBS_DIM = 6
ACT_DIM = 3


def _make_mountaincar(**kwargs):
    from environments import MountainCarXEnv
    return MountainCarXEnv(amplitude=1.0, **kwargs)


def _make_acrobot(**kwargs):
    from environments import AcrobotXEnv
    return AcrobotXEnv(**kwargs)


def _make_flat_mountaincar(**kwargs):
    from environments import MountainCarXEnv
    return MountainCarXEnv(amplitude=0.0, **kwargs)


def _make_cartpole(**kwargs):
    from environments import CartPoleXEnv
    return CartPoleXEnv(gravity=9.8, **kwargs)


def _make_mirror_cartpole(**kwargs):
    from environments import CartPoleXEnv
    # force_mag = -10: the action-to-force mapping is reversed, so the same (s, a)
    # produces the opposite cart acceleration at EVERY state -- the sensorimotor
    # contingency reversal COIN's motor-adaptation heritage models. It replaced the
    # gravity flip (2026-08-31): the gravity term ~ g*sin(theta) vanishes at theta=0,
    # exactly where the data concentrates, which made the pair's per-segment dynamics
    # margin ~160x weaker and forced a CP/InvCP code merge under blocked arrival
    # (parking at CP's code was the arriving task's true loss optimum). The mirror
    # task keeps difficulty parity: the optimal policy is CartPole's, mirrored.
    return CartPoleXEnv(force_mag=-10.0, **kwargs)


#: ``[(name, factory), ...]`` in Figure-3 block order. Index into this list is the *task
#: index* used everywhere (schedules, ``A_raw`` rows, diagnostics). Each factory takes
#: keyword arguments forwarded to the env constructor (``max_episode_steps``,
#: ``render_mode``) and returns a **fresh** env -- the custom envs compute derived
#: constants once in ``__init__``, so envs are built, never mutated.
TASKS: List[Tuple[str, Callable[..., Any]]] = [
    ("MountainCar", _make_mountaincar),
    ("Acrobot", _make_acrobot),
    ("FlatMountainCar", _make_flat_mountaincar),
    ("CartPole", _make_cartpole),
    ("MirrorCartPole", _make_mirror_cartpole),
]


def _make_gravflip_cartpole(**kwargs):
    from environments import CartPoleXEnv
    return CartPoleXEnv(gravity=-9.8, **kwargs)


# FIG3_TASK4=gravflip restores the original InvertedCartPole as task 4 -- the pair the
# dynamics-only encoder could never split (its signal ~ g*sin(theta) vanishes where the
# data lives). The 2-D value-observation stack discriminates it through the RETURN
# channel instead (CP<->InvCP cross-transfer is 56-77 vs 200), so if that pilot passes,
# the committed gravflip baselines and oracles are valid again and no baseline re-run
# is needed. Env var rather than an edit so two machines can run different task sets
# from one commit.
if os.environ.get("FIG3_TASK4", "").lower() in ("gravflip", "invcp", "inverted"):
    TASKS[4] = ("InvertedCartPole", _make_gravflip_cartpole)

TASK_NAMES: List[str] = [name for name, _ in TASKS]
N_TASKS: int = len(TASKS)

#: Potential-based energy shaping for MountainCar (task 0) during
#: TRAINING only, applied identically for every method. PPO cannot learn MountainCar's
#: sparse reward at the 200-step cap (verified: 400 oracle rollouts flat at -200, and the
#: no-carry 256-step segments cap in-stream episodes regardless of the env limit), so the
#: training envs add F = gamma*Phi(s') - Phi(s) with Phi = SHAPING_COEF * mechanical
#: energy -- policy-invariant (Ng et al., 1999). The coefficient comes from a pilot grid
#: {1e4, 3e4, 6e4} at the exact block budget (300 rollouts x 2048 steps): 3e4 and 6e4
#: break through to ~-110 raw eval, 1e4 does not; 6e4 breaks through earliest. Only
#: MountainCar itself is shaped: Flat MountainCar (amplitude 0) has no potential term,
#: and its pure kinetic bonus was found to DESTROY learning (oracle -45 -> -200) --
#: practical optimisation dominates the asymptotic invariance guarantee, and FlatMC
#: learns fine unshaped. Evaluation always runs on UNSHAPED envs and reports raw returns.
SHAPING_COEF: float = 60000.0
SHAPING_GAMMA: float = 0.995
SHAPED_TASKS: Tuple[int, ...] = (0,)


def make_task_env(task_idx: int, seed: Optional[int] = None,
                  max_episode_steps: int = MAX_EPISODE_STEPS, train: bool = True):
    """
    Build one fresh env for task ``task_idx``.

    Args:
        task_idx (int): Index into :data:`TASKS`.
        seed (int, optional): If given, the env is reset once with this seed and its
            action space seeded with it, so the whole subsequent episode stream is
            reproducible. (Callers still call ``env.reset()`` themselves; an unseeded
            reset simply continues the seeded RNG stream.)
        max_episode_steps (int): Time-limit cap (``TimeLimitMixin``).
        train (bool): Training envs for the MountainCar family carry the potential-based
            energy shaping (:data:`SHAPING_COEF`); evaluation envs (``train=False``)
            never do, so every reported return is raw.

    Returns:
        gymnasium.Env: A padded ``*XEnv`` with rendering off.
    """
    name, factory = TASKS[int(task_idx)]
    kwargs = {}
    if train and int(task_idx) in SHAPED_TASKS:
        kwargs.update(shaping_coef=SHAPING_COEF, shaping_gamma=SHAPING_GAMMA)
    env = factory(max_episode_steps=int(max_episode_steps), render_mode="none", **kwargs)
    if seed is not None:
        env.reset(seed=int(seed))
        env.action_space.seed(int(seed))
    return env


def make_task_envs(task_idx: int, n_envs: int, seed: Optional[int] = None,
                   max_episode_steps: int = MAX_EPISODE_STEPS,
                   train: bool = True) -> List[Any]:
    """
    Build ``n_envs`` fresh envs for one task -- one per rollout segment.

    Env ``j`` is seeded with ``seed + j`` when ``seed`` is given, so the eight segments of
    a rollout are independent but reproducible. ``train`` as in :func:`make_task_env`.
    """
    return [make_task_env(task_idx, None if seed is None else int(seed) + j,
                          max_episode_steps, train=train)
            for j in range(int(n_envs))]


# ======================================================================================
# 2. Blocked schedule  (plan section 5, decision 10; section 7)
# ======================================================================================

#: Rollouts per block, in :data:`TASKS` order: MC 300 -> Acrobot 100 -> Flat MC 50 ->
#: CartPole 50 -> Inverted CartPole 50 = 550 rollouts total.
BLOCK_SIZES: Tuple[int, ...] = (300, 100, 50, 50, 50)

#: Cumulative rollout counts at which every method is evaluated on all five tasks:
#: (300, 400, 450, 500, 550). Checkpoint ``t`` ends block ``t`` -- entry ``t`` is a
#: **1-based rollout count**, so the checkpoint fires after rollout index
#: ``CHECKPOINTS[t] - 1``.
CHECKPOINTS: Tuple[int, ...] = tuple(int(c) for c in np.cumsum(BLOCK_SIZES))

#: Total rollouts in the blocked stream.
N_ROLLOUTS: int = int(sum(BLOCK_SIZES))


def blocked_schedule(block_sizes: Sequence[int] = BLOCK_SIZES) -> np.ndarray:
    """
    Expand block sizes into the per-rollout task-index stream.

    Args:
        block_sizes: Rollouts per block, one entry per task, in :data:`TASKS` order.

    Returns:
        np.ndarray: ``int`` array of length ``sum(block_sizes)``; element ``r`` is the
        task index trained on rollout ``r``.
    """
    return np.repeat(np.arange(len(block_sizes), dtype=int),
                     np.asarray(block_sizes, dtype=int))


def checkpoints_from(block_sizes: Sequence[int] = BLOCK_SIZES) -> np.ndarray:
    """1-based cumulative rollout counts at which to evaluate (block ends)."""
    return np.cumsum(np.asarray(block_sizes, dtype=int))


# ======================================================================================
# 3. Rollout shape and PPO hypers  (plan section 2, section 8)
# ======================================================================================

#: Segments per rollout.
N_SEGMENTS = 8
#: Steps per segment.
SEG_STEPS = 256
#: Transitions per rollout (``N_SEGMENTS * SEG_STEPS``).
ROLLOUT_STEPS = N_SEGMENTS * SEG_STEPS

#: PPO minibatch schedule, identical for every Figure-3 method.
MINI_EPOCHS = 10
MB_SIZE = 64

#: Constructor kwargs for :class:`rl.PPOAgent`, identical for every Figure-3 method
#: (plan section 8). ``gamma``/``ent_coef``/``lr`` are the values the plan pins; the rest
#: are the defaults the old Figure-3 notebook cells spelled out, kept explicit so a
#: change to ``rl.PPOAgent``'s defaults cannot silently move the baseline.
#: Network width is ``rl._MLP``'s default ``hidden=64`` on the padded 6-d interface.
PPO_KWARGS: Dict[str, Any] = {
    "gamma": 0.995, "gae_lambda": 0.95, "clip_eps": 0.2, "lr": 3e-4,
    "ent_coef": 0.01, "vf_coef": 0.5, "device": "cpu",
}

#: Frozen-evaluation budget per (task, checkpoint) cell.
EVAL_EPISODES = 100
EVAL_MAX_STEPS = 200


# ======================================================================================
# 4. Seeding
# ======================================================================================

#: Offsets keeping the training-env, evaluation-env and torch/numpy seed streams of one
#: repetition disjoint.
TRAIN_SEED_OFFSET = 10_007
EVAL_SEED_OFFSET = 900_000


def seed_everything(seed: int) -> None:
    """
    Seed torch, numpy and the stdlib RNG for one repetition.

    Kept local to this module on purpose: it must not depend on a helper landing in
    ``rl.py``. If ``rl.seed_everything`` exists it is called too, so a future rl-side
    helper (which may seed more state) is honoured rather than bypassed.
    """
    import random

    import numpy as _np
    import torch

    seed = int(seed)
    random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():          # pragma: no cover - CPU-only in practice
        torch.cuda.manual_seed_all(seed)

    try:
        import rl as _rl
        _rl_seed = getattr(_rl, "seed_everything", None)
    except Exception:                       # pragma: no cover - rl always importable
        _rl_seed = None
    if callable(_rl_seed):
        _rl_seed(seed)


def train_env_seeds(rep_seed: int, n_rollouts: int, n_segments: int = N_SEGMENTS,
                    ) -> np.ndarray:
    """
    Base env seed for every rollout of one repetition.

    ``seeds[r]`` seeds segment 0 of rollout ``r``; segment ``j`` gets ``seeds[r] + j``
    (see :func:`make_task_envs`). Drawn from a repetition-specific generator so two reps
    never share initial states, and strided by ``n_segments`` so seeds never collide.

    Returns:
        np.ndarray: ``int64`` array of shape ``(n_rollouts,)``.
    """
    rng = np.random.default_rng(int(rep_seed) + TRAIN_SEED_OFFSET)
    raw = rng.integers(0, 2 ** 31 - 1 - int(n_segments), size=int(n_rollouts))
    return raw.astype(np.int64)


def eval_env_seed(rep_seed: int, task_idx: int) -> int:
    """
    Seed for the evaluation env of one (repetition, task) pair.

    Deliberately independent of the checkpoint: every checkpoint evaluates a task from the
    *same* 100 initial states, so a heatmap row compares like with like. Episode ``e``
    resets with ``eval_env_seed(...) + e`` (:func:`evaluate_on_task`), which makes the
    initial-state sequence policy-independent as well.
    """
    return int(EVAL_SEED_OFFSET + 1_000 * int(rep_seed) + int(task_idx))


# ======================================================================================
# 5. Rollout collection  (plan section 2: 8 segments x 256 steps, no state carry-over)
# ======================================================================================

@dataclass
class Rollout:
    """
    One collected rollout, kept segment-structured.

    ``N = n_segments * seg_steps`` transitions, stored flat in segment order, so segment
    ``s`` occupies ``slice(s * seg_steps, (s + 1) * seg_steps)`` (:meth:`segment_slice`).
    Advantages are computed per segment because, with no state carry-over, a segment
    boundary is a genuine episode boundary (plan section 6, item 3).

    Attributes:
        obs (torch.Tensor): ``[N, obs_dim]`` float32, CPU.
        next_obs (torch.Tensor): ``[N, obs_dim]`` float32, CPU -- the observation the env
            returned, stored **before** any auto-reset, so episode-final transitions are
            intact. Baselines that fit dynamics (LLIRL) read this.
        act (np.ndarray): ``[N]`` int64 (discrete) actions actually sent to the env.
        logp (torch.Tensor): ``[N]`` float32 behaviour log-probabilities.
        rew (np.ndarray): ``[N]`` float64 rewards.
        val (np.ndarray): ``[N]`` float64 value estimates at the acting step.
        done (np.ndarray): ``[N]`` bool, ``terminated or truncated``.
        last_obs (torch.Tensor): ``[n_segments, obs_dim]`` -- each segment's bootstrap
            observation (post-auto-reset, matching :meth:`rl.PPOAgent.train_step`; the
            done mask neutralises it when the segment ended terminally).
        seg_returns (np.ndarray): ``[n_segments]`` mean return of episodes that *ended*
            in that segment; ``nan`` where none did.
        ep_returns (List[float]): every completed episode return, in order.
        task_id (int, optional): Task the rollout came from, for diagnostics.
    """
    obs: Any
    next_obs: Any
    act: np.ndarray
    logp: Any
    rew: np.ndarray
    val: np.ndarray
    done: np.ndarray
    last_obs: Any
    seg_returns: np.ndarray
    ep_returns: List[float] = field(default_factory=list)
    n_segments: int = N_SEGMENTS
    seg_steps: int = SEG_STEPS
    task_id: Optional[int] = None

    def segment_slice(self, s: int) -> slice:
        """Flat-buffer slice of segment ``s``."""
        return slice(int(s) * self.seg_steps, (int(s) + 1) * self.seg_steps)

    @property
    def n_steps(self) -> int:
        return self.n_segments * self.seg_steps

    @property
    def mean_episode_return(self) -> float:
        """Mean over completed episodes in the rollout; ``nan`` if none completed."""
        return float(np.mean(self.ep_returns)) if self.ep_returns else float("nan")


def collect_segment(agent, env, n_steps: int = SEG_STEPS,
                    policy_fn: Optional[Callable] = None,
                    transition_cb: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Collect one segment: ``n_steps`` transitions starting from a genuine ``env.reset()``.

    No state is carried in from anywhere (plan section 5, decision 2) -- uniform treatment
    across methods and no cross-task state leakage. Episodes that end inside the segment
    auto-reset, exactly as :meth:`rl.PPOAgent.train_step` does.

    Args:
        agent: A :class:`rl.PPOAgent` (or subclass). Used for ``_flatten_obs``, ``_act``
            and ``value_net`` unless ``policy_fn`` overrides the acting.
        env: A fresh env for this segment.
        n_steps (int): Transitions to collect.
        policy_fn (callable, optional): ``policy_fn(obs_t) -> (action, logp, value)`` for
            baselines whose acting is not plain ``agent._act`` (routed heads, recurrent
            policies). ``logp`` must be a scalar torch tensor, ``value`` a float. Default
            is vanilla PPO acting under ``torch.no_grad()``.
        transition_cb (callable, optional): Called after every env step as
            ``transition_cb(i, obs_t, action, reward, next_obs_t, done)`` *before* the next
            action is chosen, so a stateful policy (GRU hidden state, running latent
            posterior) can update itself in-loop.

    Returns:
        dict: ``obs``/``next_obs`` (lists of CPU tensors), ``act``, ``logp``, ``rew``,
        ``val``, ``done`` (lists), ``last_obs`` (tensor), ``ep_returns`` (list of floats).
    """
    import torch

    def _default_policy(obs_t):
        with torch.no_grad():
            value = agent.value_net(obs_t).squeeze().item()
            action, logp, _, _ = agent._act(obs_t)
        return action, logp, value

    act_fn = policy_fn if policy_fn is not None else _default_policy

    obs_t = agent._flatten_obs(env.reset()[0])
    obs_l, next_l, act_l, logp_l, rew_l, val_l, done_l = [], [], [], [], [], [], []
    ep_returns: List[float] = []
    ep_ret = 0.0

    for i in range(int(n_steps)):
        action, logp, value = act_fn(obs_t)
        next_obs, reward, done, trunc, _ = env.step(action)

        # Store next_obs BEFORE any reset -- letting the reset overwrite it corrupts every
        # episode-final transition with no visible symptom (same rule as rl.py).
        next_obs_t = agent._flatten_obs(next_obs)
        obs_l.append(obs_t.detach().cpu())
        next_l.append(next_obs_t.detach().cpu())
        act_l.append(action)
        logp_l.append(logp.detach().cpu())
        rew_l.append(float(reward))
        val_l.append(float(value))
        done_l.append(bool(done or trunc))

        if transition_cb is not None:
            transition_cb(i, obs_t, action, float(reward), next_obs_t, bool(done or trunc))

        ep_ret += float(reward)
        if done or trunc:
            next_obs, _ = env.reset()
            ep_returns.append(ep_ret)
            ep_ret = 0.0
        obs_t = agent._flatten_obs(next_obs)

    return {"obs": obs_l, "next_obs": next_l, "act": act_l, "logp": logp_l,
            "rew": rew_l, "val": val_l, "done": done_l, "last_obs": obs_t.detach().cpu(),
            "ep_returns": ep_returns}


def collect_rollout(agent, envs: Sequence[Any], seg_steps: int = SEG_STEPS,
                    policy_fn: Optional[Callable] = None,
                    transition_cb: Optional[Callable] = None,
                    task_id: Optional[int] = None) -> Rollout:
    """
    Collect one full rollout: ``len(envs)`` segments of ``seg_steps`` steps.

    Every segment starts from its own ``env.reset()`` on its own fresh env (plan
    section 2). All segments come from the same task in the blocked protocol; ``envs`` is
    what decides that, so an interleaved protocol is expressible with the same call.

    Args:
        agent: A :class:`rl.PPOAgent`-like agent (see :func:`collect_segment`).
        envs: One fresh env per segment -- build with :func:`make_task_envs`.
        seg_steps (int): Steps per segment.
        policy_fn, transition_cb: Forwarded to :func:`collect_segment`. ``transition_cb``
            receives the *within-segment* step index.
        task_id (int, optional): Recorded on the returned :class:`Rollout`.

    Returns:
        Rollout: Segment-structured buffer ready for :func:`ppo_update`.
    """
    import torch

    obs_l, next_l, act_l, logp_l, rew_l, val_l, done_l = [], [], [], [], [], [], []
    last_obs, seg_returns, ep_returns = [], [], []

    for env in envs:
        seg = collect_segment(agent, env, seg_steps, policy_fn, transition_cb)
        obs_l += seg["obs"]
        next_l += seg["next_obs"]
        act_l += seg["act"]
        logp_l += seg["logp"]
        rew_l += seg["rew"]
        val_l += seg["val"]
        done_l += seg["done"]
        last_obs.append(seg["last_obs"])
        ep_returns += seg["ep_returns"]
        seg_returns.append(float(np.mean(seg["ep_returns"])) if seg["ep_returns"]
                           else float("nan"))

    act_arr = np.asarray(act_l)
    if act_arr.dtype.kind in "iub":
        act_arr = act_arr.astype(np.int64)

    return Rollout(
        obs=torch.stack(obs_l),
        next_obs=torch.stack(next_l),
        act=act_arr,
        logp=torch.stack(logp_l).float(),
        rew=np.asarray(rew_l, dtype=np.float64),
        val=np.asarray(val_l, dtype=np.float64),
        done=np.asarray(done_l, dtype=bool),
        last_obs=torch.stack(last_obs),
        seg_returns=np.asarray(seg_returns, dtype=np.float64),
        ep_returns=ep_returns,
        n_segments=len(envs),
        seg_steps=int(seg_steps),
        task_id=None if task_id is None else int(task_id),
    )


def close_envs(envs: Sequence[Any]) -> None:
    """Close a list of envs, ignoring envs that dislike being closed twice."""
    for env in envs:
        try:
            env.close()
        except Exception:                   # pragma: no cover
            pass


# ======================================================================================
# 6. PPO update  (mirrors rl.PPOAgent.train_step, with the no-carry GAE rule)
# ======================================================================================

def compute_advantages(agent, batch: Rollout) -> Tuple[Any, Any]:
    """
    Per-segment GAE, normalised over the whole rollout.

    **No-carry bootstrap** (plan section 6, item 3): with ``carry_state=False`` every
    segment is a fresh episode, so segment ``s`` bootstraps from *its own* final
    observation. Using the next segment's would leak a value estimate across an episode
    (and, in an interleaved protocol, across tasks).

    Advantages are normalised across the whole rollout, not per segment -- per-segment
    normalisation was measured to be a regression in ``rl.AmortisedCOINPPOAgent``.

    Returns:
        (adv, ret): float32 tensors ``[N]`` on ``agent.device``; ``adv`` is normalised.
    """
    import torch

    adv_parts, ret_parts = [], []
    with torch.no_grad():
        for s in range(batch.n_segments):
            sl = batch.segment_slice(s)
            last_val = agent.value_net(
                batch.last_obs[s].to(agent.device)).squeeze().item()
            a, r = agent._compute_advantages(
                list(batch.rew[sl]), list(batch.val[sl]),
                list(batch.done[sl].astype(float)), last_val)
            adv_parts.append(a)
            ret_parts.append(r)

    adv = torch.cat(adv_parts).to(agent.device).float()
    ret = torch.cat(ret_parts).to(agent.device).float()
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, ret


def ppo_update(agent, batch: Rollout, mini_epochs: int = MINI_EPOCHS,
               mb_size: int = MB_SIZE,
               extra_loss: Optional[Callable] = None) -> Dict[str, float]:
    """
    Run the PPO clipped-surrogate update on a collected rollout.

    Term for term identical to :meth:`rl.PPOAgent.train_step`'s optimisation block
    (clip, ``vf_coef``, ``ent_coef``, minibatch schedule) -- only the rollout collection
    differs, so the comparison to the in-notebook PPO stays honest.

    Args:
        agent: The :class:`rl.PPOAgent` whose ``policy``/``value_net``/``optim`` are
            updated.
        batch (Rollout): From :func:`collect_rollout`.
        mini_epochs (int): Passes over the rollout.
        mb_size (int): Minibatch size.
        extra_loss (callable, optional): ``extra_loss(agent) -> torch.Tensor`` added to
            every minibatch loss. This is the hook the EWC baseline uses for its
            ``(lambda/2) * sum_i F_i (theta_i - theta*_i)^2`` penalty; it must be
            differentiable w.r.t. the agent's parameters.

    Returns:
        dict: ``policy_loss``, ``value_loss``, ``entropy``, ``penalty`` (0.0 when
        ``extra_loss`` is None), ``mean_episode_return``, ``mean_reward_per_step``.
    """
    import torch

    adv, ret = compute_advantages(agent, batch)

    obs_t = batch.obs.to(agent.device)
    if agent.action_continuous:
        act_t = torch.as_tensor(np.asarray(batch.act), dtype=torch.float32,
                                device=agent.device)
    else:
        act_t = torch.as_tensor(batch.act, dtype=torch.long, device=agent.device)
    old_logp = batch.logp.to(agent.device)

    n = batch.n_steps
    actor_loss = critic_loss = entropy = None
    penalty_val = 0.0

    for _ in range(int(mini_epochs)):
        idxs = torch.randperm(n)
        for start in range(0, n, int(mb_size)):
            mb = idxs[start:start + int(mb_size)]
            b_obs, b_act = obs_t[mb], act_t[mb]

            if agent.action_continuous:
                mu = agent.policy(b_obs)
                std = agent.log_std.exp().expand_as(mu)
                dist = torch.distributions.Normal(mu, std)
                new_logp = dist.log_prob(b_act).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
            else:
                dist = torch.distributions.Categorical(logits=agent.policy(b_obs))
                new_logp = dist.log_prob(b_act)
                entropy = dist.entropy().mean()

            ratio = torch.exp(new_logp - old_logp[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1 - agent.clip_eps, 1 + agent.clip_eps) * adv[mb]
            actor_loss = -torch.min(surr1, surr2).mean()

            value_pred = agent.value_net(b_obs).squeeze(-1)
            critic_loss = (ret[mb] - value_pred).pow(2).mean()

            loss = actor_loss + agent.vf_coef * critic_loss - agent.ent_coef * entropy
            if extra_loss is not None:
                penalty = extra_loss(agent)
                loss = loss + penalty
                penalty_val = float(penalty.detach().item())

            agent.optim.zero_grad()
            loss.backward()
            agent.optim.step()

    # Keeps PPOAgent.evaluate's CPU shadow in sync (rl.PPOAgent._get_eval_nets_cpu).
    agent._weights_version += 1

    return {"policy_loss": float(actor_loss.item()),
            "value_loss": float(critic_loss.item()),
            "entropy": float(entropy.item()),
            "penalty": penalty_val,
            "mean_episode_return": batch.mean_episode_return,
            "mean_reward_per_step": float(np.mean(batch.rew))}


def count_parameters(agent) -> int:
    """
    Trainable parameter count of a PPO-family agent, for the compute table
    (plan section 4). Sums ``policy``/``value_net`` plus, for head-database agents, every
    per-context module held on the agent.
    """
    import torch.nn as nn

    seen, total = set(), 0
    for attr in vars(agent).values():
        for module in ([attr] if isinstance(attr, nn.Module)
                       else list(attr.values()) if isinstance(attr, dict)
                       else list(attr) if isinstance(attr, (list, tuple)) else []):
            if isinstance(module, nn.Module) and id(module) not in seen:
                seen.add(id(module))
                total += sum(p.numel() for p in module.parameters() if p.requires_grad)
    return int(total)


# ======================================================================================
# 7. Frozen evaluation  (plan section 4 and section 7: 5 tasks x 5 checkpoints x 100 eps)
# ======================================================================================

class EvalPolicy:
    """
    Base class for a frozen evaluation policy on one task.

    Subclass this in a baseline module and override what you need; the harness calls the
    hooks in this order per episode::

        reset()
        for each step:  act(obs) -> action;  observe(obs, a, r, next_obs, done)
        episode_end(ep_return)

    ``observe`` exists for methods that identify the task online (LLIRL's per-segment CRP
    assignment, a recurrent hidden state); ``episode_end`` for methods whose identification
    is episode-granular (OWL's UCB1 bandit over heads). All learning must already be
    frozen -- the harness never calls an optimiser.
    """

    def reset(self) -> None:
        """Called at the start of every evaluation episode."""

    def act(self, obs) -> Any:
        """Return the (deterministic) action for ``obs``."""
        raise NotImplementedError

    def observe(self, obs, action, reward: float, next_obs, done: bool) -> None:
        """Called after every environment step."""

    def episode_end(self, ep_return: float) -> None:
        """Called once an evaluation episode has finished."""


class GreedyPPOPolicy(EvalPolicy):
    """
    Deterministic ``argmax`` policy of a :class:`rl.PPOAgent` (discrete actions) or its
    mean action (continuous). Networks are put in ``eval()`` mode and never updated.
    """

    def __init__(self, agent):
        import torch

        self.agent = agent
        self.torch = torch
        self.net = agent.policy
        self.net.eval()

    def act(self, obs):
        torch = self.torch
        with torch.inference_mode():
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32),
                                    device=self.agent.device).view(-1)
            out = self.net(obs_t)
            if self.agent.action_continuous:
                return np.clip(out.cpu().numpy().astype(np.float32),
                               self.agent.act_low_np, self.agent.act_high_np)
            return int(torch.argmax(out).item())


def evaluate_on_task(policy: EvalPolicy, task_idx: int, n_episodes: int = EVAL_EPISODES,
                     max_steps: int = EVAL_MAX_STEPS, seed: Optional[int] = None,
                     max_episode_steps: int = MAX_EPISODE_STEPS) -> np.ndarray:
    """
    Frozen evaluation of one policy on one task.

    Episode ``e`` resets with ``seed + e``, so the 100 initial states are fixed for a
    (repetition, task) pair and identical at every checkpoint -- and, because the classic
    -control envs draw randomness only at reset, independent of the policy being tested.

    Args:
        policy (EvalPolicy): Already-frozen policy for this task.
        task_idx (int): Index into :data:`TASKS`.
        n_episodes (int): Episodes to run.
        max_steps (int): Hard step cap per episode (the env's own 200 cap truncates first).
        seed (int, optional): Base env seed; see :func:`eval_env_seed`.
        max_episode_steps (int): Env time limit.

    Returns:
        np.ndarray: ``float64`` array ``(n_episodes,)`` of **raw** episode returns.
        Normalisation against the per-task oracle happens at plot time (plan section 4).
    """
    env = make_task_env(task_idx, None, max_episode_steps, train=False)
    returns = np.empty(int(n_episodes), dtype=np.float64)

    for e in range(int(n_episodes)):
        if seed is None:
            obs, _ = env.reset()
        else:
            obs, _ = env.reset(seed=int(seed) + e)
        policy.reset()
        ep_ret = 0.0
        for _ in range(int(max_steps)):
            action = policy.act(obs)
            next_obs, reward, done, trunc, _ = env.step(action)
            policy.observe(obs, action, float(reward), next_obs, bool(done or trunc))
            ep_ret += float(reward)
            obs = next_obs
            if done or trunc:
                break
        policy.episode_end(ep_ret)
        returns[e] = ep_ret

    env.close()
    return returns


def evaluate_all_tasks(policy_for_task: Callable[[int], EvalPolicy],
                       n_episodes: int = EVAL_EPISODES, max_steps: int = EVAL_MAX_STEPS,
                       rep_seed: int = 0, task_indices: Optional[Sequence[int]] = None,
                       max_episode_steps: int = MAX_EPISODE_STEPS) -> np.ndarray:
    """
    Evaluate on all five tasks at one checkpoint.

    Args:
        policy_for_task (callable): ``policy_for_task(task_idx) -> EvalPolicy``. Called
            once per task, so a method that carries per-task identification state (a
            fresh UCB1 bandit, a reset latent posterior) gets a clean instance per cell.
            A single-policy method just returns the same object every time.
        n_episodes (int): Episodes per task.
        max_steps (int): Step cap per episode.
        rep_seed (int): Repetition seed; env seeds come from :func:`eval_env_seed`.
        task_indices (sequence, optional): Subset of tasks (default: all five, in order).

    Returns:
        np.ndarray: ``float64`` ``(n_tasks, n_episodes)`` of raw returns -- one column
        block of the ``A_raw[5 tasks, 5 checkpoints, n_episodes]`` matrix callers assemble.
    """
    idxs = list(range(N_TASKS)) if task_indices is None else list(task_indices)
    out = np.empty((len(idxs), int(n_episodes)), dtype=np.float64)
    for row, task_idx in enumerate(idxs):
        out[row] = evaluate_on_task(
            policy_for_task(int(task_idx)), int(task_idx), n_episodes, max_steps,
            eval_env_seed(rep_seed, task_idx), max_episode_steps)
    return out


# ======================================================================================
# 8. Entry point: single-head PPO floor  (plan Part I, section 3.1)
# ======================================================================================

def run_single_rep_single_ppo(rep_id, n_rollouts_per_block=BLOCK_SIZES,
                              n_segments=N_SEGMENTS, seg_steps=SEG_STEPS,
                              mini_epochs=MINI_EPOCHS, mb_size=MB_SIZE,
                              eval_episodes=EVAL_EPISODES, eval_max_steps=EVAL_MAX_STEPS,
                              max_episode_steps=MAX_EPISODE_STEPS, ppo_kwargs=None,
                              progress=True, return_agent=False):
    """
    One repetition of the **single-head PPO floor** for Figure 3 v2.

    A single actor-critic trained straight through the blocked stream, with no context
    machinery of any kind, evaluated on all five tasks at every block end. It establishes
    the catastrophic-forgetting floor: each new block overwrites the last, so the heatmap
    should be bright only on the diagonal (plan Part I, section 3.1).

    Pool-ready in the Figure-2 style: module-level, heavy imports inside, plain
    numpy/dict return.

    Args:
        rep_id (int): Repetition index; used directly as the seed (torch, numpy, env
            streams). Old Figure-3 reps were unseeded -- this fixes that
            (plan section 6, item 5).
        n_rollouts_per_block (sequence of int): Rollouts per task block, in :data:`TASKS`
            order. Default :data:`BLOCK_SIZES` = (300, 100, 50, 50, 50). Checkpoints are
            the cumulative sums, so shrinking this for a smoke test shrinks the
            checkpoints with it.
        n_segments (int): Segments per rollout (default 8).
        seg_steps (int): Steps per segment (default 256) -- 8 x 256 = 2048 transitions.
        mini_epochs (int), mb_size (int): PPO minibatch schedule.
        eval_episodes (int), eval_max_steps (int): Frozen-eval budget per heatmap cell.
        max_episode_steps (int): Env time limit, 200 everywhere in Figure 3.
        ppo_kwargs (dict, optional): Overrides merged onto :data:`PPO_KWARGS`.
        progress (bool): Show a ``tqdm`` bar.
        return_agent (bool): Also return the trained agent (debugging; not picklable-cheap).

    Returns:
        dict: See the module report / the keys below.

        ==========================  ====================================================
        key                         value
        ==========================  ====================================================
        ``A_raw``                   ``float64 (n_tasks, n_checkpoints, eval_episodes)``
                                    raw per-episode eval returns
        ``train_returns``           ``float64 (n_rollouts,)`` mean completed-episode
                                    return per rollout (``nan`` if none completed)
        ``train_reward_per_step``   ``float64 (n_rollouts,)``
        ``seg_returns``             ``float64 (n_rollouts, n_segments)`` per-segment mean
        ``task_ids``                ``int64 (n_rollouts,)`` task trained per rollout
        ``policy_loss``             ``float64 (n_rollouts,)``
        ``value_loss``              ``float64 (n_rollouts,)``
        ``entropy``                 ``float64 (n_rollouts,)``
        ``collect_seconds``         ``float64 (n_rollouts,)`` wall-clock, rollout only
        ``update_seconds``          ``float64 (n_rollouts,)`` wall-clock, PPO update only
        ``rollout_seconds``         ``float64 (n_rollouts,)`` collect + update
        ``eval_seconds``            ``float64 (n_checkpoints,)``
        ``total_seconds``           ``float`` whole repetition
        ``checkpoints``             ``int64 (n_checkpoints,)`` 1-based rollout counts
        ``block_sizes``             ``int64 (n_tasks,)``
        ``env_steps``               ``int64 (n_rollouts,)`` cumulative env steps trained
        ``n_params``                ``int`` trainable parameters
        ``task_names``              ``list[str]``
        ``method``                  ``"single_ppo"``
        ``seed``                    ``int``
        ``meta``                    ``dict`` of the hypers actually used
        ==========================  ====================================================
    """
    # Heavy imports inside the function (Figure-2 pool convention). Everything else comes
    # from this module's own globals -- unpickling this function in a worker imports the
    # module, so they are always present.
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
    hypers = dict(_PK)
    if ppo_kwargs:
        hypers.update(ppo_kwargs)

    # One agent for the whole stream. Built on a CartPole env purely to read the shared
    # padded interface (6-d Box obs, Discrete(3)) -- every task exposes the same one.
    proto_env = make_task_envs(3, 1, seed=seed, max_episode_steps=max_episode_steps)[0]
    agent = PPOAgent(proto_env, **hypers)
    proto_env.close()

    seeds = train_env_seeds(seed, n_rollouts, n_segments)

    A_raw = _np.full((_NT, int(ckpts.size), int(eval_episodes)), _np.nan, dtype=_np.float64)
    train_returns = _np.full(n_rollouts, _np.nan)
    reward_per_step = _np.full(n_rollouts, _np.nan)
    seg_returns = _np.full((n_rollouts, int(n_segments)), _np.nan)
    pol_loss = _np.full(n_rollouts, _np.nan)
    val_loss = _np.full(n_rollouts, _np.nan)
    ent = _np.full(n_rollouts, _np.nan)
    t_collect = _np.zeros(n_rollouts)
    t_update = _np.zeros(n_rollouts)
    t_eval = _np.zeros(int(ckpts.size))

    t_start = time.perf_counter()
    bar = tqdm(range(n_rollouts), desc=f"single-PPO rep {rep_id}", disable=not progress)
    ckpt_i = 0

    for r in bar:
        task = int(schedule[r])

        t0 = time.perf_counter()
        envs = make_task_envs(task, int(n_segments), seed=int(seeds[r]),
                              max_episode_steps=max_episode_steps)
        batch = collect_rollout(agent, envs, int(seg_steps), task_id=task)
        close_envs(envs)
        t1 = time.perf_counter()

        stats = ppo_update(agent, batch, int(mini_epochs), int(mb_size))
        t2 = time.perf_counter()

        t_collect[r], t_update[r] = t1 - t0, t2 - t1
        train_returns[r] = stats["mean_episode_return"]
        reward_per_step[r] = stats["mean_reward_per_step"]
        seg_returns[r, :batch.seg_returns.size] = batch.seg_returns
        pol_loss[r], val_loss[r], ent[r] = (
            stats["policy_loss"], stats["value_loss"], stats["entropy"])

        if progress:
            bar.set_postfix(task=_TN[task], ret=f"{train_returns[r]:.1f}")

        # ---- checkpoint: frozen eval on all five tasks (one shared policy) ----
        if ckpt_i < ckpts.size and (r + 1) == int(ckpts[ckpt_i]):
            te = time.perf_counter()
            policy = GreedyPPOPolicy(agent)
            cell = evaluate_all_tasks(
                lambda _t: policy, int(eval_episodes), int(eval_max_steps), seed,
                max_episode_steps=max_episode_steps)
            agent.policy.train()
            elapsed = time.perf_counter() - te
            # A zero-length block makes two checkpoints coincide; fill both from the one
            # evaluation rather than silently leaving a nan column.
            while ckpt_i < ckpts.size and (r + 1) == int(ckpts[ckpt_i]):
                A_raw[:, ckpt_i, :] = cell
                t_eval[ckpt_i] = elapsed
                elapsed = 0.0
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
        "method": "single_ppo",
        "seed": seed,
        "meta": {"seed": seed, "n_segments": int(n_segments),
                 "seg_steps": int(seg_steps), "rollout_steps": int(n_segments) * int(seg_steps),
                 "mini_epochs": int(mini_epochs), "mb_size": int(mb_size),
                 "max_episode_steps": int(max_episode_steps),
                 "eval_episodes": int(eval_episodes), "eval_max_steps": int(eval_max_steps),
                 "block_sizes": blocks.tolist(), "checkpoints": ckpts.tolist(),
                 **hypers},
    }
    if return_agent:
        return result, agent
    return result
