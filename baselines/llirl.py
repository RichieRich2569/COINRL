"""
llirl.py

LLIRL (Lifelong Incremental Reinforcement Learning) baseline for the **Figure 3 v2**
experiment (`fig3_amortised_plan.md`, Part I section 3.5, Part II section 8).

**Source.** Wang, Chen & Dong, *"Lifelong Incremental Reinforcement Learning with Online
Bayesian Inference"*, IEEE Transactions on Neural Networks and Learning Systems 33(8), 2021.

**Mechanism.** LLIRL is the one published method in the comparison that also *instantiates
new contexts online with no external signal*, which makes it the direct rival to our
Bayesian-nonparametric contextual inference. It maintains a mixture of "environment
clusters" under a Chinese Restaurant Process. Each cluster ``k`` owns

* a small dynamics model ``f_k(s, a) -> s'`` (``rl._MLP(6 + 3, 6, hidden=64)``), and
* a policy -- here one :class:`rl.PPOAgent`, so the head is literally the same network and
  the same update every other Figure-3 method runs.

For each new chunk of experience ``D`` the CRP posterior over cluster identity is

    p(c = k   | D)  ~  n_k    / (n - 1 + alpha)  *  p(D | f_k)      (existing cluster)
    p(c = new | D)  ~  alpha  / (n - 1 + alpha)  *  p(D | f_0)      (new cluster, prior model)

with a fixed-variance Gaussian observation model

    p(D | f_k)  =  prod_t  N( s'_t ; f_k(s_t, a_t), sigma^2 I ).

Assignment is **hard EM**: the winning cluster's count increments, its dynamics model
trains on the chunk, and its policy trains on the rollout; a new-cluster win spawns both.

**GRANTED: nothing.** Like ours, it runs the raw unlabelled stream -- no boundaries, no
labels, no cues.

**Limitation probed: the quality of the inference itself.** LLIRL makes hard,
point-estimate assignments at segment granularity under a memoryless CRP prior, with the
segment log-likelihood summed over 256 x 6 Gaussian terms -- so the likelihood swamps the
prior and the assignment is effectively an argmax over dynamics fit. COIN instead maintains
a full posterior: soft responsibilities updated every *step*, the encoder's uncertainty
propagated into the assignment, and a *learned Markov transition structure between
contexts*. Head to head, this is "structured Bayesian contextual inference vs plain CRP-EM
clustering". The headline diagnostic is the inferred cluster count against the true 5.

Conventions chosen here (all documented at their point of use, all overridable):

* **Delta targets.** ``f_k`` predicts ``s' - s``. The first pilot used raw ``s'``
  targets and collapsed to ``K = 1``: against O(1) raw states a freshly-initialised
  ``f_0`` is catastrophically wrong on *every* task, so the trained cluster always won
  and a new table could structurally never open (``sigma`` cancels from the comparison).
  With delta targets the zero-ish prediction of a fresh net is a *competitive* baseline:
  a cluster beats it on the task it was fitted to and loses to it where its confident
  predictions are systematically wrong, which is exactly the two-sided contest the CRP
  spawn decision needs. Note the scalar ``sigma`` still sums errors across state dims in
  physical units, so small-scale signatures (the O(1e-3) MountainCar amplitude term) can
  stay invisible -- that is the published method's fixed-sigma formulation, reported
  as-is.
* **One fixed prior model ``f_0`` per repetition**, never trained; a new cluster starts as
  a deep copy of it and is then trained on the winning segment. Re-drawing ``f_0`` per
  segment would make ``p(D | f_0)`` a noisy quantity and the spawn decision partly a
  coin-flip on initialisation.
* **Segment-granular assignment, rollout-granular policy update** (the plan's "the winning
  cluster's dynamics net trains on the segment and its PPO head trains on the rollout").
  All 8 segments of a rollout are assigned in order, each training its winner's dynamics
  model immediately (so later segments are judged against updated models); the **majority**
  winner over those 8 assignments then takes the PPO update on the whole 2048-step rollout.
  Ties are broken by summed posterior mass, then by lowest cluster index.
* **Advantages from the training head's own value net.** ``Rollout.val`` was filled by the
  *acting* head, so it is recomputed under the majority head before
  :func:`fig3_common.ppo_update`. The behaviour log-probabilities are left alone -- they
  are the correct importance-sampling reference for the PPO ratio.
* **Acting uses the head that won the previous rollout** (head 0 on the first rollout).
  A rollout must be collected before it can be assigned, so a one-rollout lag is intrinsic
  to segment-granular assignment; this is the plan's reading of it.
* **New-cluster head cloning mirrors the repo's novel-head convention**
  (:meth:`rl.COINPPOAgent._instantiate_context_net`): the networks are deep-copied from the
  most recently used head and given a *fresh* Adam optimiser.
* At evaluation everything is frozen and no cluster is ever spawned; log-likelihoods
  accumulate step by step over the episode's transitions and the acting head is the
  running argmax. That is *more* generous than training (step-granular rather than
  segment-granular identification), deliberately so.

Following the Figure-2 convention (see ``baselines/cmdp_q.py``), the pool entry point
:func:`run_single_rep_llirl` is a module-level function with its heavy imports inside, so
it pickles cleanly for ``multiprocess``.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np

from baselines.fig3_common import (
    ACT_DIM, BLOCK_SIZES, EVAL_EPISODES, EVAL_MAX_STEPS, MAX_EPISODE_STEPS, MB_SIZE,
    MINI_EPOCHS, N_SEGMENTS, N_TASKS, PPO_KWARGS, SEG_STEPS, TASK_NAMES, EvalPolicy,
    blocked_schedule, checkpoints_from, close_envs, collect_rollout, count_parameters,
    evaluate_all_tasks, make_task_envs, ppo_update, seed_everything, train_env_seeds,
)

#: CRP concentration. Larger ``alpha`` -> more clusters. The plan sweeps {0.5, 1} in the
#: notebook; 1.0 is the default (and makes "first customer" tie-break to the existing table).
ALPHA = 1.0

#: Fixed observation standard deviation of the Gaussian dynamics likelihood.
#:
#: In practice this, not ``alpha``, is LLIRL's sensitivity knob here, and the pilot should
#: sweep it alongside ``alpha``. The segment log-likelihood sums 256 x 6 Gaussian terms, so
#: it outweighs the CRP prior by orders of magnitude and the spawn decision reduces to
#: "does the best existing dynamics model beat the untrained prior model on this segment,
#: scaled by 1 / (2 sigma^2)". At ``sigma = 0.1`` on 8 x 256 rollouts the answer is almost
#: always yes -- one cluster's model adapts to each new block within a rollout or two and
#: absorbs it (an observed collapse to K = 2 on a 12-rollout probe, with only Acrobot,
#: whose state ranges differ most, opening a second table). That under-segmentation *is*
#: the limitation the plan wants on record, but the pilot should establish it is the
#: method's behaviour and not a mis-set constant.
SIGMA = 0.05

#: Hidden width of each cluster's dynamics model (matches ``rl._MLP``'s default and the
#: width of every other network in Figure 3).
DYN_HIDDEN = 64

#: Dynamics-model fitting on a winning segment: full-batch MSE, this many gradient steps.
DYN_EPOCHS = 5
DYN_LR = 1e-3


# ======================================================================================
# 1. Dynamics likelihood helpers
# ======================================================================================

def dyn_features(obs, act, act_dim: int = ACT_DIM):
    """
    Build the dynamics-model input ``[s_t, a_t one-hot]`` -> ``[L, obs_dim + act_dim]``.

    Args:
        obs: ``[L, obs_dim]`` float tensor.
        act: ``[L]`` integer tensor/array of discrete actions.
        act_dim (int): Number of discrete actions (3 on the padded Figure-3 interface).

    Returns:
        torch.Tensor: ``[L, obs_dim + act_dim]`` float32 on ``obs``'s device.
    """
    import torch
    import torch.nn.functional as F

    obs = obs.float()
    a = torch.as_tensor(np.asarray(act), dtype=torch.long, device=obs.device)
    return torch.cat([obs, F.one_hot(a, int(act_dim)).float()], dim=-1)


def gaussian_log_lik(pred, target, sigma: float) -> float:
    """
    ``sum_t sum_d log N(target[t, d]; pred[t, d], sigma^2)`` -- the segment log-likelihood
    exactly as the plan writes it (a **sum** over transitions, not a per-transition mean).

    Keeping the sum is faithful to LLIRL and is also the source of its weakness: with
    256 x 6 terms the likelihood dominates the CRP prior by orders of magnitude, so the
    assignment reduces to "which dynamics model fits best". Everything is done in log space
    and the caller subtracts the max before exponentiating.
    """
    import torch

    with torch.no_grad():
        sq = (pred - target).pow(2).sum().item()
    n = int(target.numel())
    s2 = float(sigma) ** 2
    return -0.5 * n * float(np.log(2.0 * np.pi * s2)) - 0.5 * sq / s2


# ======================================================================================
# 2. Cluster bookkeeping
# ======================================================================================

class _Cluster:
    """One CRP cluster: a dynamics model (with its own Adam) and a PPO head."""

    def __init__(self, dyn, dyn_optim, head, count: int = 1):
        self.dyn = dyn
        self.dyn_optim = dyn_optim
        self.head = head
        self.count = int(count)


class LLIRLMixture:
    """
    The CRP mixture of {dynamics model, PPO head} clusters.

    Holds the never-trained prior model ``f_0``, the cluster list, and the CRP counts. It
    does the assignment, the dynamics fitting and the spawning; the PPO update is left to
    :func:`fig3_common.ppo_update` so the policy learning is bit-for-bit the shared one.

    Args:
        proto_env: An env exposing the shared padded interface; used only to construct
            :class:`rl.PPOAgent` heads (kept alive for the whole repetition because heads
            are spawned lazily).
        ppo_kwargs (dict): Hypers for every head (:data:`fig3_common.PPO_KWARGS`).
        alpha (float): CRP concentration.
        sigma (float): Observation sd of the dynamics likelihood.
        dyn_hidden (int), dyn_lr (float), dyn_epochs (int): Dynamics-model configuration.
        device (str): Torch device.
    """

    def __init__(self, proto_env, ppo_kwargs: Dict[str, Any], alpha: float = ALPHA,
                 sigma: float = SIGMA, dyn_hidden: int = DYN_HIDDEN, dyn_lr: float = DYN_LR,
                 dyn_epochs: int = DYN_EPOCHS, device: str = "cpu"):
        import copy

        import torch.optim as optim

        from rl import PPOAgent, _MLP

        self._copy = copy
        self._optim = optim
        self._PPOAgent = PPOAgent
        self._MLP = _MLP

        self.proto_env = proto_env
        self.ppo_kwargs = dict(ppo_kwargs)
        self.alpha, self.sigma = float(alpha), float(sigma)
        self.dyn_hidden, self.dyn_lr = int(dyn_hidden), float(dyn_lr)
        self.dyn_epochs = int(dyn_epochs)
        self.device = device

        self.obs_dim = int(np.prod(proto_env.observation_space.shape))
        self.act_dim = int(proto_env.action_space.n)

        #: The prior model f_0. Fixed for the repetition and never trained -- see the module
        #: docstring on why it is not re-drawn per segment.
        self.prior_dyn = _MLP(self.obs_dim + self.act_dim, self.obs_dim,
                              self.dyn_hidden).to(device)
        for p in self.prior_dyn.parameters():
            p.requires_grad_(False)
        self.prior_dyn.eval()

        self.clusters: List[_Cluster] = []
        # Seat the first customer at table 0 (count 1) so the very first assignment is a
        # genuine CRP draw: with count 0 the existing-cluster prior would be identically
        # zero and a new cluster would always win.
        self._spawn(source_head=None, count=1)
        #: Index of the head that acted on the most recent rollout (and, before the first
        #: assignment, the head that will act on it).
        self.acting_idx = 0

    # ------------------------------------------------------------------ construction
    def _new_head(self, source_head=None):
        """
        A fresh :class:`rl.PPOAgent` head, optionally initialised from ``source_head``.

        Mirrors :meth:`rl.COINPPOAgent._instantiate_context_net`: the *networks* are copied,
        the optimiser is **not** -- a new context starts from the donor's weights with fresh
        Adam moments.
        """
        head = self._PPOAgent(self.proto_env, **self.ppo_kwargs)
        if source_head is not None:
            head.policy.load_state_dict(source_head.policy.state_dict())
            head.value_net.load_state_dict(source_head.value_net.state_dict())
        return head

    def _spawn(self, source_head=None, count: int = 1) -> int:
        """Create a cluster: dynamics = deep copy of ``f_0``, head cloned from
        ``source_head``. Returns the new cluster's index."""
        dyn = self._copy.deepcopy(self.prior_dyn).to(self.device)
        for p in dyn.parameters():
            p.requires_grad_(True)
        dyn.train()
        dyn_optim = self._optim.Adam(dyn.parameters(), lr=self.dyn_lr)
        self.clusters.append(_Cluster(dyn, dyn_optim, self._new_head(source_head),
                                      count=int(count)))
        return len(self.clusters) - 1

    # ------------------------------------------------------------------ inference
    @property
    def counts(self) -> np.ndarray:
        return np.array([c.count for c in self.clusters], dtype=np.int64)

    def log_crp_prior(self) -> np.ndarray:
        """
        ``[K + 1]`` log CRP prior, novel table last::

            log n_k - log(seated + alpha)      and      log alpha - log(seated + alpha)

        (``seated = sum_k n_k`` is the ``n - 1`` of the plan's formula.)
        """
        counts = self.counts.astype(np.float64)
        seated = counts.sum()
        denom = seated + self.alpha
        with np.errstate(divide="ignore"):
            lp = np.log(counts) - np.log(denom)
        return np.concatenate([lp, [np.log(self.alpha) - np.log(denom)]])

    def segment_log_lik(self, feats, targets) -> np.ndarray:
        """``[K + 1]`` dynamics log-likelihoods of one segment, prior model ``f_0`` last."""
        import torch

        out = np.empty(len(self.clusters) + 1, dtype=np.float64)
        with torch.no_grad():
            for k, c in enumerate(self.clusters):
                out[k] = gaussian_log_lik(c.dyn(feats), targets, self.sigma)
            out[-1] = gaussian_log_lik(self.prior_dyn(feats), targets, self.sigma)
        return out

    def assign_segment(self, feats, targets):
        """
        CRP hard-EM assignment of one segment, then fit the winner's dynamics model on it.

        Returns:
            (idx, post, dyn_mse): winning cluster index (a *new* index if the novel table
            won), the normalised posterior over ``[K + 1]`` tables **before** any spawn,
            and the winner's final full-batch MSE on the segment.
        """
        log_post = self.log_crp_prior() + self.segment_log_lik(feats, targets)
        post = np.exp(log_post - log_post.max())
        post = post / post.sum()

        # argmax puts the novel table last, so an exact tie resolves to an existing cluster
        # -- the conservative choice for a method whose headline number is cluster count.
        j = int(np.argmax(log_post))
        if j == len(self.clusters):
            idx = self._spawn(source_head=self.clusters[self.acting_idx].head, count=1)
        else:
            idx = j
            self.clusters[idx].count += 1

        return idx, post, self.fit_dynamics(idx, feats, targets)

    def fit_dynamics(self, idx: int, feats, targets) -> float:
        """Full-batch MSE fit of cluster ``idx``'s dynamics model on one segment.

        Returns the loss of the **last** gradient step (the segment's fitted MSE)."""
        c = self.clusters[idx]
        loss_val = float("nan")
        for _ in range(self.dyn_epochs):
            loss = (c.dyn(feats) - targets).pow(2).mean()
            c.dyn_optim.zero_grad()
            loss.backward()
            c.dyn_optim.step()
            loss_val = float(loss.detach().item())
        return loss_val

    # ------------------------------------------------------------------ diagnostics
    def n_params(self) -> int:
        """Total trainable parameters across all clusters (heads + dynamics models) plus
        the prior model -- the compute-table figure, which *grows with K*."""
        total = sum(p.numel() for p in self.prior_dyn.parameters())
        for c in self.clusters:
            total += count_parameters(c.head)
            total += sum(p.numel() for p in c.dyn.parameters() if p.requires_grad)
        return int(total)


# ======================================================================================
# 3. Frozen evaluation
# ======================================================================================

class LLIRLEvalPolicy(EvalPolicy):
    """
    LLIRL's own online assignment, run frozen on an evaluation episode.

    Everything is frozen: no dynamics fitting, no PPO update, and **no new clusters**
    (a CRP spawn at test time would be a learning event). Per episode the running
    per-cluster log-likelihood is reset to zero and the acting head is the argmax of

        log n_k - log(seated + alpha)  +  sum_{t < now} log N(s'_t; f_k(s_t, a_t), sigma^2)

    recomputed after every step, so identification sharpens within the episode. Before the
    first transition the argmax is the CRP prior alone, i.e. the most-used cluster --
    LLIRL has no other zero-shot signal.

    Note this is *more* generous than the training-time procedure, which commits at
    256-step segment granularity; the eval protocol gives LLIRL step-granular
    identification so that the heatmap measures its models rather than its chunking.
    """

    def __init__(self, mixture: LLIRLMixture):
        import torch

        self.torch = torch
        self.mixture = mixture
        self.act_dim = mixture.act_dim
        self.sigma = mixture.sigma
        self.log_prior = mixture.log_crp_prior()[:-1]     # drop the novel table
        self.dyns = [c.dyn for c in mixture.clusters]
        self.heads = [c.head for c in mixture.clusters]
        for dyn in self.dyns:
            dyn.eval()
        for head in self.heads:
            head.policy.eval()
        self.loglik = np.zeros(len(self.dyns), dtype=np.float64)
        self.idx = int(np.argmax(self.log_prior))

    def reset(self) -> None:
        self.loglik[:] = 0.0
        self.idx = int(np.argmax(self.log_prior))

    def act(self, obs):
        torch = self.torch
        head = self.heads[self.idx]
        with torch.inference_mode():
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32),
                                    device=head.device).view(-1)
            return int(torch.argmax(head.policy(obs_t)).item())

    def observe(self, obs, action, reward: float, next_obs, done: bool) -> None:
        torch = self.torch
        with torch.inference_mode():
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32),
                                    device=self.mixture.device).view(1, -1)
            tgt = torch.as_tensor(np.asarray(next_obs, dtype=np.float32) -
                                  np.asarray(obs, dtype=np.float32),
                                  device=self.mixture.device).view(1, -1)
            feats = dyn_features(obs_t, [int(action)], self.act_dim)
            for k, dyn in enumerate(self.dyns):
                self.loglik[k] += gaussian_log_lik(dyn(feats), tgt, self.sigma)
        self.idx = int(np.argmax(self.log_prior + self.loglik))


# ======================================================================================
# 4. Entry point
# ======================================================================================

def run_single_rep_llirl(rep_id, n_rollouts_per_block=BLOCK_SIZES, n_segments=N_SEGMENTS,
                         seg_steps=SEG_STEPS, mini_epochs=MINI_EPOCHS, mb_size=MB_SIZE,
                         alpha=ALPHA, sigma=SIGMA, dyn_hidden=DYN_HIDDEN,
                         dyn_epochs=DYN_EPOCHS, dyn_lr=DYN_LR,
                         eval_episodes=EVAL_EPISODES, eval_max_steps=EVAL_MAX_STEPS,
                         max_episode_steps=MAX_EPISODE_STEPS, ppo_kwargs=None,
                         progress=True, return_agent=False):
    """
    One repetition of the **LLIRL (CRP mixture)** baseline for Figure 3 v2.

    Mirrors :func:`fig3_common.run_single_rep_single_ppo` step for step -- same blocked
    stream, same 8 x 256 rollout, same checkpoints, same frozen eval -- with the single
    actor-critic replaced by a CRP mixture of {dynamics model, PPO head} clusters
    (:class:`LLIRLMixture`). Per rollout: collect with the previous rollout's winning head,
    assign each of the ``n_segments`` segments and fit its winner's dynamics model, then run
    the shared :func:`fig3_common.ppo_update` on the majority winner's head.

    Pool-ready in the Figure-2 style: module-level, heavy imports inside, plain numpy/dict
    return.

    Args:
        rep_id (int): Repetition index, used directly as the seed (torch, numpy, env
            streams), exactly as in the single-PPO floor.
        n_rollouts_per_block (sequence of int): Rollouts per task block in
            :data:`fig3_common.TASKS` order; checkpoints are its cumulative sums, so
            shrinking it for a smoke test shrinks the checkpoints with it.
        n_segments (int), seg_steps (int): Rollout shape (default 8 x 256 = 2048). The
            segment is also LLIRL's assignment granularity.
        mini_epochs (int), mb_size (int): PPO minibatch schedule for the winning head.
        alpha (float): CRP concentration (plan sweeps {0.5, 1}).
        sigma (float): Observation sd of the Gaussian dynamics likelihood -- in practice the
            knob that actually moves the inferred cluster count; see :data:`SIGMA`.
        dyn_hidden (int), dyn_epochs (int), dyn_lr (float): Dynamics-model configuration.
        eval_episodes (int), eval_max_steps (int): Frozen-eval budget per heatmap cell.
        max_episode_steps (int): Env time limit (200 everywhere in Figure 3).
        ppo_kwargs (dict, optional): Overrides merged onto :data:`fig3_common.PPO_KWARGS`.
        progress (bool): Show a ``tqdm`` bar.
        return_agent (bool): Also return the :class:`LLIRLMixture` (debugging).

    Returns:
        dict: The schema of :func:`fig3_common.run_single_rep_single_ppo` (``A_raw``,
        ``train_returns``, ``train_reward_per_step``, ``seg_returns``, ``task_ids``,
        ``policy_loss``, ``value_loss``, ``entropy``, ``collect_seconds``,
        ``update_seconds``, ``rollout_seconds``, ``eval_seconds``, ``total_seconds``,
        ``checkpoints``, ``block_sizes``, ``env_steps``, ``n_params``, ``task_names``,
        ``method`` = ``"llirl"``, ``seed``, ``meta``), plus the CRP extras:

        ==========================  ====================================================
        key                         value
        ==========================  ====================================================
        ``n_clusters``              ``int64 (n_rollouts,)`` clusters after each rollout --
                                    the headline trajectory against the true 5
        ``n_clusters_final``        ``int``
        ``assignments``             ``int64 (n_rollouts, n_segments)`` winning cluster per
                                    segment
        ``cluster_counts``          ``int64 (n_clusters_final,)`` final CRP counts n_k
        ``acting_head``             ``int64 (n_rollouts,)`` head that collected the rollout
        ``train_head``              ``int64 (n_rollouts,)`` majority head that was updated
        ``dyn_loss``                ``float64 (n_rollouts,)`` mean final segment MSE of the
                                    winning dynamics models
        ``assign_seconds``          ``float64 (n_rollouts,)`` wall-clock of assignment +
                                    dynamics fitting (a component of ``update_seconds``)
        ==========================  ====================================================
    """
    import numpy as _np
    import torch
    from tqdm.auto import tqdm

    seed = int(rep_id)
    seed_everything(seed)

    blocks = _np.asarray(list(n_rollouts_per_block), dtype=int)
    schedule = blocked_schedule(blocks)
    ckpts = checkpoints_from(blocks)
    n_rollouts = int(schedule.size)
    hypers = dict(PPO_KWARGS)
    if ppo_kwargs:
        hypers.update(ppo_kwargs)

    # Built on a CartPole env purely to read the shared padded interface (6-d Box obs,
    # Discrete(3)). Unlike the single-PPO floor this env is kept alive: heads are spawned
    # lazily whenever the CRP opens a new table.
    proto_env = make_task_envs(3, 1, seed=seed, max_episode_steps=max_episode_steps)[0]
    mixture = LLIRLMixture(proto_env, hypers, alpha=alpha, sigma=sigma,
                           dyn_hidden=dyn_hidden, dyn_lr=dyn_lr, dyn_epochs=dyn_epochs,
                           device=hypers.get("device", "cpu"))

    seeds = train_env_seeds(seed, n_rollouts, n_segments)

    A_raw = _np.full((N_TASKS, int(ckpts.size), int(eval_episodes)), _np.nan,
                     dtype=_np.float64)
    train_returns = _np.full(n_rollouts, _np.nan)
    reward_per_step = _np.full(n_rollouts, _np.nan)
    seg_returns = _np.full((n_rollouts, int(n_segments)), _np.nan)
    pol_loss = _np.full(n_rollouts, _np.nan)
    val_loss = _np.full(n_rollouts, _np.nan)
    ent = _np.full(n_rollouts, _np.nan)
    dyn_loss = _np.full(n_rollouts, _np.nan)
    n_clusters = _np.zeros(n_rollouts, dtype=_np.int64)
    assignments = _np.full((n_rollouts, int(n_segments)), -1, dtype=_np.int64)
    acting_head = _np.zeros(n_rollouts, dtype=_np.int64)
    train_head = _np.zeros(n_rollouts, dtype=_np.int64)
    t_collect = _np.zeros(n_rollouts)
    t_assign = _np.zeros(n_rollouts)
    t_update = _np.zeros(n_rollouts)
    t_eval = _np.zeros(int(ckpts.size))

    t_start = time.perf_counter()
    bar = tqdm(range(n_rollouts), desc=f"LLIRL rep {rep_id}", disable=not progress)
    ckpt_i = 0

    for r in bar:
        task = int(schedule[r])
        act_idx = int(mixture.acting_idx)
        acting_head[r] = act_idx
        head = mixture.clusters[act_idx].head
        head.policy.train()

        # ---- collect with the previous rollout's winning head ----
        t0 = time.perf_counter()
        envs = make_task_envs(task, int(n_segments), seed=int(seeds[r]),
                              max_episode_steps=max_episode_steps)
        batch = collect_rollout(head, envs, int(seg_steps), task_id=task)
        close_envs(envs)
        t1 = time.perf_counter()

        # ---- CRP assignment of every segment (+ dynamics fitting of each winner) ----
        # next_obs is the pre-auto-reset observation (fig3_common.collect_segment), so
        # episode-final transitions are genuine dynamics samples, not reset artefacts.
        post_mass: Dict[int, float] = {}
        seg_losses = []
        for s in range(batch.n_segments):
            sl = batch.segment_slice(s)
            feats = dyn_features(batch.obs[sl].to(mixture.device), batch.act[sl],
                                 mixture.act_dim)
            _o = batch.obs[sl].to(mixture.device).float()
            targets = batch.next_obs[sl].to(mixture.device).float() - _o
            idx, post, mse = mixture.assign_segment(feats, targets)
            assignments[r, s] = idx
            # post's last entry is the novel table; a novel win is credited to the cluster
            # it created, so the tie-break mass is indexed consistently.
            mass = float(post[idx]) if idx < post.size - 1 else float(post[-1])
            post_mass[idx] = post_mass.get(idx, 0.0) + mass
            seg_losses.append(mse)
        dyn_loss[r] = float(_np.mean(seg_losses))
        n_clusters[r] = len(mixture.clusters)
        t2 = time.perf_counter()

        # ---- majority winner takes the PPO update on the whole rollout ----
        seg_ids = assignments[r, :batch.n_segments]
        uniq, cnt = _np.unique(seg_ids, return_counts=True)
        best = uniq[cnt == cnt.max()]
        # Ties: highest summed posterior mass, then lowest cluster index.
        win = int(min(best, key=lambda k: (-post_mass.get(int(k), 0.0), int(k))))
        train_head[r] = win
        mixture.acting_idx = win

        train_agent = mixture.clusters[win].head
        train_agent.policy.train()
        # Rollout.val came from the acting head; recompute under the head being trained so
        # its GAE targets are its own. old_logp is deliberately left as the behaviour
        # policy's -- that is what the PPO ratio is meant to correct for.
        with torch.no_grad():
            batch.val = train_agent.value_net(
                batch.obs.to(train_agent.device)).squeeze(-1).cpu().numpy().astype(
                    _np.float64)
        stats = ppo_update(train_agent, batch, int(mini_epochs), int(mb_size))
        t3 = time.perf_counter()

        t_collect[r] = t1 - t0
        t_assign[r] = t2 - t1
        t_update[r] = t3 - t1                 # assignment + dynamics + PPO
        train_returns[r] = stats["mean_episode_return"]
        reward_per_step[r] = stats["mean_reward_per_step"]
        seg_returns[r, :batch.seg_returns.size] = batch.seg_returns
        pol_loss[r], val_loss[r], ent[r] = (
            stats["policy_loss"], stats["value_loss"], stats["entropy"])

        if progress:
            bar.set_postfix(task=TASK_NAMES[task], K=len(mixture.clusters),
                            ret=f"{train_returns[r]:.1f}")

        # ---- checkpoint: frozen eval on all five tasks ----
        if ckpt_i < ckpts.size and (r + 1) == int(ckpts[ckpt_i]):
            te = time.perf_counter()
            cell = evaluate_all_tasks(
                lambda _t: LLIRLEvalPolicy(mixture), int(eval_episodes),
                int(eval_max_steps), seed, max_episode_steps=max_episode_steps)
            for c in mixture.clusters:
                c.head.policy.train()
                c.dyn.train()
            elapsed = time.perf_counter() - te
            # A zero-length block makes two checkpoints coincide; fill both.
            while ckpt_i < ckpts.size and (r + 1) == int(ckpts[ckpt_i]):
                A_raw[:, ckpt_i, :] = cell
                t_eval[ckpt_i] = elapsed
                elapsed = 0.0
                ckpt_i += 1

    total = time.perf_counter() - t_start
    n_params = mixture.n_params()
    counts = mixture.counts
    proto_env.close()

    result = {
        "A_raw": A_raw,
        "train_returns": train_returns,
        "train_reward_per_step": reward_per_step,
        "seg_returns": seg_returns,
        "task_ids": schedule.astype(_np.int64),
        "policy_loss": pol_loss,
        "value_loss": val_loss,
        "entropy": ent,
        "dyn_loss": dyn_loss,
        "n_clusters": n_clusters,
        "n_clusters_final": int(counts.size),
        "assignments": assignments,
        "cluster_counts": counts.astype(_np.int64),
        "acting_head": acting_head,
        "train_head": train_head,
        "collect_seconds": t_collect,
        "assign_seconds": t_assign,
        "update_seconds": t_update,
        "rollout_seconds": t_collect + t_update,
        "eval_seconds": t_eval,
        "total_seconds": float(total),
        "checkpoints": ckpts.astype(_np.int64),
        "block_sizes": blocks.astype(_np.int64),
        "env_steps": (_np.arange(1, n_rollouts + 1, dtype=_np.int64)
                      * int(n_segments) * int(seg_steps)),
        "n_params": int(n_params),
        "task_names": list(TASK_NAMES),
        "method": "llirl",
        "seed": seed,
        "meta": {"seed": seed, "n_segments": int(n_segments), "seg_steps": int(seg_steps),
                 "rollout_steps": int(n_segments) * int(seg_steps),
                 "mini_epochs": int(mini_epochs), "mb_size": int(mb_size),
                 "alpha": float(alpha), "sigma": float(sigma),
                 "dyn_hidden": int(dyn_hidden), "dyn_epochs": int(dyn_epochs),
                 "dyn_lr": float(dyn_lr),
                 "dyn_target": "state delta s' - s",
                 "log_lik": "sum over segment transitions and state dims",
                 "assignment_granularity": "segment (dynamics) / rollout majority (policy)",
                 "acting_head_rule": "winner of the previous rollout; head 0 initially",
                 "new_head_init": "deep copy of the most recently used head, fresh Adam",
                 "max_episode_steps": int(max_episode_steps),
                 "eval_episodes": int(eval_episodes),
                 "eval_max_steps": int(eval_max_steps),
                 "block_sizes": blocks.tolist(), "checkpoints": ckpts.tolist(),
                 **hypers},
    }
    if return_agent:
        return result, mixture
    return result
