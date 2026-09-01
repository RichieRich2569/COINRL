"""
recurrent_ppo.py

Recurrent PPO (RL^2-class) baseline for the **Figure 3 v2** experiment
(`fig3_amortised_plan.md`, Part I section 3.4, Part II section 8).

**Source.** Duan, Schulman, Chen, Bartlett, Sutskever & Abbeel, *"RL^2: Fast Reinforcement
Learning via Slow Reinforcement Learning"* (arXiv:1611.02779, 2016). This is the on-policy
stand-in for the whole latent-task-inference family; PEARL (Rakelly et al., ICML 2019) and
VariBAD (Zintgraf et al., ICLR 2020) are the same idea class with more machinery (off-policy
SAC with an episodic meta-training protocol, and a belief VAE respectively) and are cited in
the paper rather than run.

**Mechanism.** A GRU carries a hidden state that can, in principle, encode "which task am I
in" from the recent history of the interaction itself::

    h_t   = GRU( h_{t-1}, [ s_t , a_{t-1} , r_{t-1} ] )
    a_t  ~ pi( . | h_t ),     V(h_t)

There is no explicit context variable, no instantiation event and no interpretable
structure: identification, if it happens at all, is implicit in ``h``. The recurrent net
replaces both ``rl._MLP`` towers of :class:`rl.PPOAgent` one for one -- an actor tower and
a critic tower, each GRU(64) over the same 10-d input (6-d padded observation + 3-d one-hot
previous action + scalar previous reward) then Linear(64,64)+Tanh then its head -- exactly
as :class:`rl.PPOAgent` keeps ``policy`` and ``value_net`` independent (a single shared
core was measured to break the baseline; see :class:`RecurrentPPOAgent`). Everything else
-- the rollout shape (8 segments x 256 steps from genuine ``env.reset()``s), the blocked
stream, the PPO objective and its hyper-parameters, the frozen 100-episode evaluation --
comes from :mod:`baselines.fig3_common` unchanged, so performance *and* wall-clock stay
comparable.

**GRANTED: reward as input.** The canonical RL^2 formulation feeds ``r_{t-1}`` to the
recurrent core, and it is kept as-is. This is a handicap in the baseline's favour and is
stated in the paper: reward is precisely the context signal we deliberately removed from
our own contingency encoder, which sees ``(s, a, s')`` only. Nothing else is granted --
no block boundaries, no task labels, no cues: it runs the raw unlabelled stream like ours.

**Limitation probed: monolithic representation.** All five tasks share one set of recurrent
weights, so the blocked stream should induce catastrophic forgetting exactly as for the
single-head PPO floor. Beyond that, this class normally assumes i.i.d. meta-training over
the task distribution -- on a continual, blocked stream that assumption is violated, which
is exactly the comparison being drawn.

Conventions chosen here (all documented at their point of use, and all overridable):

* **Hidden state carries across episodes *and* across segments within a rollout**, and is
  zeroed at rollout boundaries (plan section 3.4). Because every segment starts from a
  genuine ``env.reset()`` (plan section 5, decision 2), a segment boundary *is* an episode
  boundary, so "across episodes within a rollout" is read as "across the whole 2048-step
  rollout" -- the reading that gives the baseline the most task evidence to accumulate.
* ``a_{t-1}`` / ``r_{t-1}`` are zeroed at every **episode** start (a zero one-hot and 0.0),
  i.e. at segment starts and after any within-segment ``done`` -- there genuinely is no
  previous action there.
* The PPO update uses **truncated-BPTT chunk minibatches**. Each 256-step segment is cut
  into consecutive chunks of ``bptt_chunk`` (default 64) steps -- 32 chunks per 2048-step
  rollout -- the chunks are shuffled, and each minibatch is ``mb_chunks`` (default 4) of
  them, with the GRU forward pass recomputed with gradients from the chunk's detached
  behaviour-time entry hidden state. That is 80 optimiser steps per rollout against the
  earlier whole-segment version's 40 and the feed-forward methods' 320 on the same data;
  the one-chunk minibatch that would reproduce the feed-forward schedule exactly was tried
  and is worse (:data:`MB_CHUNKS`), because a recurrent minibatch is necessarily
  *consecutive* -- a recurrent log-probability depends on the steps preceding it -- and 64
  consecutive steps of one trajectory carry much less information than 64 shuffled
  transitions. Entry states are the behaviour-time ones (computed once per update,
  detached, allowed to go stale across mini-epochs), the standard stored-state choice; see
  :func:`recurrent_ppo_update`.
* Gradients are **globally norm-clipped at 10.0** (:data:`MAX_GRAD_NORM`). This is the one
  place the Figure-3 hyper-parameter set is not shared, and the pilot forced it: unclipped,
  the baseline solved shaped MountainCar and then collapsed irrecoverably to the floor with
  zero policy entropy. ``None`` restores the unclipped behaviour.
* GAE is computed **per segment** with a bootstrap from the value at that segment's own
  terminal hidden state, never carried between segments -- the repo's no-carry rule
  (plan section 6, item 3).
* Evaluation zeroes ``h`` at the start of every evaluation episode. Episodes within an
  evaluation cell are i.i.d. draws from one task, and :func:`fig3_common.evaluate_on_task`
  calls ``reset()`` per episode; carrying ``h`` across them would let the baseline pool
  evidence over 100 episodes, which is not what the heatmap cell is meant to measure. The
  choice is exposed as ``eval_carry_hidden`` for an appendix check.

Following the Figure-2 convention (see ``baselines/cmdp_q.py``), the pool entry point
:func:`run_single_rep_recurrent_ppo` is a module-level function with its heavy imports
inside, so it pickles cleanly for ``multiprocess``.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from baselines.fig3_common import (
    BLOCK_SIZES, EVAL_EPISODES, EVAL_MAX_STEPS, MAX_EPISODE_STEPS, MINI_EPOCHS,
    N_SEGMENTS, N_TASKS, PPO_KWARGS, SEG_STEPS, TASK_NAMES, EvalPolicy, blocked_schedule,
    checkpoints_from, close_envs, collect_rollout, count_parameters, evaluate_all_tasks,
    make_task_envs, seed_everything, train_env_seeds,
)

#: GRU width (plan section 8).
GRU_HIDDEN = 64

#: Truncated-BPTT chunk length, in steps. Each 256-step segment is split into
#: ``seg_steps // BPTT_CHUNK`` consecutive chunks (must divide ``seg_steps``); a chunk is
#: the unit of both backpropagation and minibatching. 64 gives 4 chunks per segment, i.e.
#: 32 chunks per 2048-step rollout.
BPTT_CHUNK = 64

#: Chunks per PPO minibatch. 4 gives 8 minibatches per mini-epoch, i.e. 80 optimiser steps
#: per rollout on 256 transitions each -- 2x the whole-segment version's 40 and, measured,
#: the fastest *and* the most stable of the schedules tried.
#:
#: One chunk per minibatch would make the schedule numerically identical to the
#: feed-forward methods' (32 minibatches x 64 transitions x 10 mini-epochs = 320 steps,
#: ``fig3_common.MB_SIZE`` = 64), and that was the first choice -- but the parity is
#: superficial and it measurably loses. A feed-forward minibatch of 64 is 64 *shuffled*
#: transitions from anywhere in the rollout; a recurrent minibatch of one chunk is 64
#: *consecutive* steps of a single trajectory, which carries far less independent
#: information. Measured on shaped MountainCar, 150 rollouts, seed 0 (all three on the
#: shared-core net, before the actor/critic split): ``bptt_chunk=64, mb_chunks=1`` ended
#: with gradient norms in the hundreds-to-thousands and entropy thrashing between 1e-3 and
#: its 1.10 ceiling; ``bptt_chunk=32, mb_chunks=2`` (also 320 steps) did the same and
#: finished *below* where it peaked; ``mb_chunks=4`` -- four chunks drawn by the shuffle
#: from different segments and different time offsets, which is where the decorrelation the
#: feed-forward shuffle gets for free has to come from here -- held gradient norms at O(20)
#: and learned fastest. It is also the cheapest of the three (batched forward passes, a
#: quarter as many optimiser steps), which is what keeps the wall-clock column comparable.
MB_CHUNKS = 4

#: Global-norm gradient clip. ``None`` disables it, and that was the original default:
#: the plan pins one set of PPO hypers for every Figure-3 method and none of the others
#: clips (``rl.PPOAgent.train_step`` does not), so clipping looked like a unilateral
#: handicap on the one baseline that is already the hardest to optimise.
#:
#: The pilot overruled that. Backpropagating through a GRU is the one place in Figure 3
#: where gradients genuinely explode, and they do: on shaped MountainCar the unclipped
#: agent broke through to +36 by rollout 75 and then, over five rollouts, collapsed back to
#: the -145 floor with policy entropy at 1e-3 and never recovered -- a dead deterministic
#: policy. Both 10.0 and 0.5 prevented it (+38 and +39 at rollout 155, against the
#: feed-forward floor's +42), and an unclipped second seed survived, so this is a
#: variance-of-outcome problem, exactly what a clip is for. 10.0 is the default because it
#: is the looser of the two that works. ``grad_norm`` is returned per rollout, pre-clip,
#: either way, so the diagnostic that found this stays available.
MAX_GRAD_NORM = 10.0


# ======================================================================================
# 1. The recurrent actor-critic
# ======================================================================================

def _build_net(obs_dim: int, act_dim: int, hidden: int, separate_cores: bool = True):
    """
    Build the recurrent actor-critic module (defined inside a function so this module
    imports without torch, matching the Figure-2 pool convention).

    Each tower mirrors ``rl._MLP(in, out, 64)`` with its first Tanh layer replaced by a
    GRU::

        [s_t, a_{t-1}, r_{t-1}]  ->  GRU(64)  ->  Linear(64,64)+Tanh  ->  logits / value

    so the depth and width of the feed-forward path match the other Figure-3 methods; the
    parameter count does not (a GRU cell is ~3x a Linear of the same width), which is
    itself a line in the compute table.

    Args:
        separate_cores (bool): One GRU tower per head (default), i.e. the literal
            "each ``rl._MLP`` tower becomes a GRU tower" reading of :class:`rl.PPOAgent`,
            which has fully decoupled ``policy`` and ``value_net``. ``False`` gives the
            single shared GRU with two linear heads. See
            :class:`RecurrentPPOAgent` for why the default is the separated one.

    The recurrent state is always a single ``[1, B, state_dim]`` tensor -- the concatenation
    of the two towers' states when they are separate -- so every caller (collector, update,
    evaluation) handles one object either way.
    """
    import torch
    import torch.nn as nn

    class _RecurrentActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_dim = int(obs_dim) + int(act_dim) + 1
            self.hidden = int(hidden)
            self.separate = bool(separate_cores)
            self.state_dim = self.hidden * (2 if self.separate else 1)
            self.gru = nn.GRU(self.in_dim, self.hidden)
            self.trunk = nn.Sequential(nn.Linear(self.hidden, self.hidden), nn.Tanh())
            self.pi = nn.Linear(self.hidden, int(act_dim))
            if self.separate:
                self.gru_v = nn.GRU(self.in_dim, self.hidden)
                self.trunk_v = nn.Sequential(nn.Linear(self.hidden, self.hidden), nn.Tanh())
            self.v = nn.Linear(self.hidden, 1)

        def _run(self, x, h):
            """``(y_pi, y_v, h_new)`` -- per-step GRU outputs of both towers."""
            if not self.separate:
                y, h_new = self.gru(x, h)
                return y, y, h_new
            h_pi, h_v = h[..., :self.hidden].contiguous(), h[..., self.hidden:].contiguous()
            y_pi, h_pi_new = self.gru(x, h_pi)
            y_v, h_v_new = self.gru_v(x, h_v)
            return y_pi, y_v, torch.cat([h_pi_new, h_v_new], dim=-1)

        def forward(self, x, h):
            """
            Args:
                x: ``[T, B, in_dim]`` sequence of inputs.
                h: ``[1, B, state_dim]`` initial recurrent state.

            Returns:
                ``(logits [T, B, act_dim], value [T, B], h_final [1, B, state_dim])``.
            """
            y_pi, y_v, h_new = self._run(x, h)
            value = self.v(self.trunk_v(y_v) if self.separate else self.trunk(y_v))
            return self.pi(self.trunk(y_pi)), value.squeeze(-1), h_new

        def hidden_states(self, x, h):
            """
            ``[T, B, state_dim]`` recurrent state *after* every step of ``x`` -- i.e. the
            state entering step ``t + 1``. Used to recover truncated-BPTT chunk entry
            states in one pass (:func:`recurrent_ppo_update`).
            """
            y_pi, y_v, _ = self._run(x, h)
            return torch.cat([y_pi, y_v], dim=-1) if self.separate else y_pi

    return _RecurrentActorCritic()


class RecurrentPPOAgent:
    """
    Minimal PPO agent with a GRU actor tower and a GRU critic tower.

    Deliberately *not* a :class:`rl.PPOAgent` subclass: the parent would build two ``_MLP``
    towers that this method never uses and that
    :func:`fig3_common.count_parameters` would then charge to the compute table. Instead it
    exposes exactly the surface :mod:`baselines.fig3_common` needs -- ``_flatten_obs`` for
    :func:`fig3_common.collect_rollout` and the GAE hyper-parameters -- plus its own
    sequence-aware update. Discrete actions only (every Figure-3 task is ``Discrete(3)``).

    **Why two cores** (``separate_cores=True``, the default). Every other Figure-3 method
    runs :class:`rl.PPOAgent`, whose ``policy`` and ``value_net`` are two independent
    ``_MLP`` towers: its value loss never touches the policy's parameters. A single shared
    GRU with two linear heads breaks that, and the pilot showed the break is not benign --
    on shaped MountainCar the shared-core agent's value loss (O(100), rising) dominated the
    trunk's gradient while the entropy bonus (0.01 x an entropy already below 0.3) could
    not push back, so its policy entropy decayed monotonically to 0.03 and it locked onto a
    non-solving policy. The feed-forward floor on identical data does the opposite: entropy
    dips to 0.34, *recovers* to 0.68 as it finds the goal, and only then anneals. Giving
    the critic its own GRU tower is both the literal "each ``_MLP`` tower becomes a GRU
    tower" reading of the plan and the only way to keep the pinned ``vf_coef=0.5`` from
    being a different quantity here than everywhere else in the figure. It costs a second
    GRU in the compute table -- which the compute table is there to report.
    ``separate_cores=False`` restores the shared-core variant for an appendix check.

    Attributes:
        net: The recurrent actor-critic (``vars(self)`` exposure is what makes
            :func:`fig3_common.count_parameters` see it).
        optim: Adam over ``net`` at ``lr``.
        hidden: Width of one GRU tower.
        state_dim: Width of the recurrent state passed around (``2 * hidden`` when the
            cores are separate: the two towers' states concatenated).
    """

    def __init__(self, env, hidden: int = GRU_HIDDEN, gamma: float = 0.995,
                 gae_lambda: float = 0.95, clip_eps: float = 0.2, lr: float = 3e-4,
                 ent_coef: float = 0.01, vf_coef: float = 0.5, device: str = "cpu",
                 max_grad_norm: Optional[float] = MAX_GRAD_NORM,
                 separate_cores: bool = True):
        import gymnasium as gym
        import torch
        import torch.optim as optim

        assert isinstance(env.observation_space, gym.spaces.Box), \
            "RecurrentPPOAgent supports only Box observation spaces."
        assert isinstance(env.action_space, gym.spaces.Discrete), \
            "RecurrentPPOAgent supports only Discrete action spaces (Figure 3 is Discrete(3))."

        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.act_dim = int(env.action_space.n)
        self.action_continuous = False
        self.device = device
        self.gamma, self.lam = float(gamma), float(gae_lambda)
        self.clip_eps, self.ent_coef, self.vf_coef = (
            float(clip_eps), float(ent_coef), float(vf_coef))
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        self.hidden = int(hidden)
        self.separate_cores = bool(separate_cores)
        self.state_dim = self.hidden * (2 if self.separate_cores else 1)

        self.net = _build_net(self.obs_dim, self.act_dim, self.hidden,
                              self.separate_cores).to(device)
        self.optim = optim.Adam(self.net.parameters(), lr=float(lr))
        self._torch = torch
        # Kept for API parity with rl.PPOAgent (fig3_common's ppo_update touches it).
        self._weights_version = 0

    # ------------------------------------------------------------------ utilities
    def _flatten_obs(self, obs):
        """Flat float32 observation tensor on ``self.device`` (same contract as
        :meth:`rl.PPOAgent._flatten_obs`, which :func:`fig3_common.collect_segment` calls)."""
        torch = self._torch
        if isinstance(obs, np.ndarray):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs_t = obs.to(self.device, dtype=torch.float32)
        return obs_t.view(-1)

    def zero_hidden(self, batch: int = 1):
        """``[1, batch, state_dim]`` of zeros -- the rollout-boundary reset."""
        return self._torch.zeros(1, int(batch), self.state_dim, device=self.device)

    def make_input(self, obs_t, prev_a: int, prev_r: float):
        """
        Build one RL^2 input ``[s_t, a_{t-1} one-hot, r_{t-1}]`` of shape ``[in_dim]``.

        ``prev_a < 0`` means "no previous action" (episode start) and yields a **zero**
        one-hot, the standard RL^2 encoding of the absent first action; ``prev_r`` is 0.0
        there for the same reason.
        """
        torch = self._torch
        extra = torch.zeros(self.act_dim + 1, dtype=torch.float32, device=self.device)
        if int(prev_a) >= 0:
            extra[int(prev_a)] = 1.0
        extra[-1] = float(prev_r)
        return torch.cat([obs_t.view(-1), extra])

    def batch_inputs(self, obs, prev_a: np.ndarray, prev_r: np.ndarray):
        """
        Vectorised :meth:`make_input` over a whole rollout.

        Args:
            obs: ``[N, obs_dim]`` float tensor (``Rollout.obs``).
            prev_a: ``[N]`` int array, ``-1`` for "no previous action".
            prev_r: ``[N]`` float array.

        Returns:
            torch.Tensor: ``[N, in_dim]`` float32 on ``self.device``.
        """
        torch = self._torch
        n = obs.shape[0]
        extra = torch.zeros(n, self.act_dim + 1, dtype=torch.float32, device=self.device)
        pa = torch.as_tensor(np.asarray(prev_a), dtype=torch.long, device=self.device)
        valid = pa >= 0
        if bool(valid.any()):
            idx = torch.nonzero(valid, as_tuple=False).view(-1)
            extra[idx, pa[idx]] = 1.0
        extra[:, -1] = torch.as_tensor(np.asarray(prev_r, dtype=np.float32),
                                       dtype=torch.float32, device=self.device)
        return torch.cat([obs.to(self.device).float(), extra], dim=-1)

    def _compute_advantages(self, rewards, values, dones, last_value: float):
        """
        GAE(``gamma``, ``lam``) for one segment.

        Delegates to :meth:`rl.PPOAgent._compute_advantages` (it needs only ``gamma``,
        ``lam`` and ``device``, all of which this class provides) so the advantage
        arithmetic is *literally* the same code every other Figure-3 method runs.
        """
        from rl import PPOAgent
        return PPOAgent._compute_advantages(self, rewards, values, dones, last_value)


# ======================================================================================
# 2. Collection: the stateful acting policy
# ======================================================================================

class _RecurrentCollector:
    """
    Stateful acting policy for one rollout, plugged into
    :func:`fig3_common.collect_rollout` via ``policy_fn`` / ``transition_cb``.

    It owns the three pieces of recurrent state -- ``h``, ``a_{t-1}``, ``r_{t-1}`` -- and,
    crucially, records everything the sequence update needs to *recompute* the GRU forward
    pass with gradients:

    * ``seg_h0[s]``: the (detached) hidden state entering segment ``s``. Because ``h``
      persists across segments, this is not zero for ``s > 0``; it is the truncation point
      of BPTT.
    * ``prev_a`` / ``prev_r`` per step, index-aligned with ``Rollout.obs``.
    * ``boot[s]``: the hidden state, previous action and previous reward **after** segment
      ``s``'s final transition, plus that transition's ``next_obs`` -- the ingredients of
      the segment's own bootstrap value (no carry to segment ``s+1``).

    ``policy_fn`` runs once per step and then ``transition_cb`` runs once per step, in that
    order, which is what lets the two of them keep a consistent step counter.
    """

    def __init__(self, agent: RecurrentPPOAgent, seg_steps: int):
        self.agent = agent
        self.seg_steps = int(seg_steps)
        self.h = agent.zero_hidden(1)          # persists across the WHOLE rollout
        self.prev_a = -1                       # zeroed at every episode start
        self.prev_r = 0.0
        self.t = 0
        self.seg_h0: List[Any] = []
        self.prev_a_buf: List[int] = []
        self.prev_r_buf: List[float] = []
        self.boot: List[tuple] = []

    def policy_fn(self, obs_t):
        """``(action, logp, value)`` from the recurrent policy; advances ``h`` in place."""
        torch = self.agent._torch
        if self.t % self.seg_steps == 0:
            # Segment start = fresh env.reset() = episode start: no previous action/reward,
            # but the hidden state is deliberately carried in from the previous segment.
            self.seg_h0.append(self.h.detach().clone())
            self.prev_a, self.prev_r = -1, 0.0

        x = self.agent.make_input(obs_t, self.prev_a, self.prev_r)
        with torch.no_grad():
            logits, value, h_new = self.agent.net(x.view(1, 1, -1), self.h)
        self.h = h_new
        dist = torch.distributions.Categorical(logits=logits.view(-1))
        action = dist.sample()
        logp = dist.log_prob(action)

        self.prev_a_buf.append(int(self.prev_a))
        self.prev_r_buf.append(float(self.prev_r))
        return int(action.item()), logp, float(value.view(-1).item())

    def transition_cb(self, i: int, obs_t, action, reward: float, next_obs_t, done: bool):
        """Fold the just-taken ``(a_t, r_t)`` into the RL^2 input for step ``t + 1``."""
        self.prev_a, self.prev_r = int(action), float(reward)
        if i == self.seg_steps - 1:
            # Bootstrap ingredients for THIS segment, captured before the episode reset
            # below can wipe (prev_a, prev_r). When the segment ended terminally the done
            # mask neutralises the bootstrap anyway.
            self.boot.append((self.h.detach().clone(), self.prev_a, self.prev_r,
                              next_obs_t.detach()))
        if done:
            self.prev_a, self.prev_r = -1, 0.0
        self.t += 1


# ======================================================================================
# 3. The sequence-minibatch PPO update
# ======================================================================================

def _bootstrap_values(agent: RecurrentPPOAgent, collector: _RecurrentCollector) -> List[float]:
    """One value per segment, at that segment's own terminal hidden state."""
    torch = agent._torch
    out = []
    with torch.no_grad():
        for h, prev_a, prev_r, next_obs in collector.boot:
            x = agent.make_input(next_obs, prev_a, prev_r)
            _, value, _ = agent.net(x.view(1, 1, -1), h)
            out.append(float(value.view(-1).item()))
    return out


def recurrent_ppo_update(agent: RecurrentPPOAgent, batch, collector: _RecurrentCollector,
                         mini_epochs: int = MINI_EPOCHS,
                         bptt_chunk: int = BPTT_CHUNK,
                         mb_chunks: int = MB_CHUNKS) -> Dict[str, float]:
    """
    PPO update over truncated-BPTT chunks.

    Term for term the objective of :func:`fig3_common.ppo_update` (clipped surrogate,
    ``vf_coef``, ``ent_coef``, same gamma/lambda/clip/lr). The batching differs only in
    that a recurrent minibatch has to be *consecutive*: a recurrent log-probability at step
    ``t`` depends on every step before it, so transitions cannot be shuffled individually.
    Each 256-step segment is therefore cut into ``seg_steps // bptt_chunk`` consecutive
    chunks (32 chunks per rollout at the default 64), the chunks are shuffled, and each
    minibatch is ``mb_chunks`` of them. The GRU forward pass is recomputed *with gradients*
    over the chunk from the chunk's detached entry hidden state, so gradients flow back
    ``bptt_chunk`` steps -- standard truncated BPTT.

    **Why chunks and not whole segments** (the fix for the pilot's flat learning curve):
    minibatching whole 256-step segments (2 of the 8 per minibatch) gave only 4 minibatches
    per mini-epoch, i.e. 40 optimiser steps per rollout against the feed-forward methods'
    ``2048 / 64 * 10 = 320`` on the very same data. That 8x optimisation-speed handicap has
    nothing to do with the recurrent-vs-monolithic comparison this baseline exists to make,
    and it alone kept shaped MountainCar pinned at its ~-146 floor for 300 rollouts while
    the feed-forward floor reached +42. The defaults here take 80 steps per rollout on 256
    transitions each; ``mb_chunks=1`` would take the feed-forward's exact 320 x 64 but
    measured *worse* and unstable, because 64 consecutive steps of one trajectory are not
    the same minibatch as 64 shuffled transitions -- see :data:`MB_CHUNKS`.

    **Chunk entry hidden states are the behaviour-time ones**, computed once at the top of
    the update (where the network still holds the parameters that collected the rollout)
    with a single no-grad pass over the rollout, and detached. They therefore go stale as
    the mini-epochs progress -- this is the standard truncated-BPTT/PPO choice (it is what
    stored-state recurrent replay does, R2D2's "stored state" included), it keeps the
    entry state consistent with the ``old_logp`` the importance ratio is taken against, and
    the alternative (re-running the rollout every mini-epoch to refresh the states) buys
    nothing at a 64-step truncation where ``h`` is dominated by the chunk's own inputs.
    A one-layer GRU's outputs *are* its per-step hidden states, so the pass costs one
    ``nn.GRU`` call: ``h`` entering step ``t`` is the output at ``t-1``.

    Advantages are per-segment GAE with the segment's own bootstrap
    (:func:`_bootstrap_values`), normalised across the whole rollout -- the same no-carry
    rule and the same normalisation scope as :func:`fig3_common.compute_advantages`.

    Args:
        agent, batch, collector: The agent, its :class:`fig3_common.Rollout` and the
            collector that produced it (for the per-segment entry states and the RL^2
            ``prev_a`` / ``prev_r`` streams).
        mini_epochs (int): Passes over the rollout (10, as everywhere in Figure 3).
        bptt_chunk (int): Truncation length in steps; must divide ``batch.seg_steps``.
        mb_chunks (int): Chunks per minibatch (4 -> 256 transitions per optimiser step).

    Returns:
        dict: ``policy_loss``, ``value_loss``, ``entropy``, ``mean_episode_return``,
        ``mean_reward_per_step``, ``grad_norm`` (last minibatch, pre-clip -- always
        measured, whether or not ``agent.max_grad_norm`` clips).
    """
    import torch

    n_seg, T = batch.n_segments, batch.seg_steps
    L = max(1, min(int(bptt_chunk), int(T)))
    if T % L:
        raise ValueError(f"bptt_chunk={bptt_chunk} must divide seg_steps={T}")
    n_chunks = T // L
    boot_vals = _bootstrap_values(agent, collector)

    # ---- per-segment GAE, rollout-wide normalisation ----
    adv_parts, ret_parts = [], []
    for s in range(n_seg):
        sl = batch.segment_slice(s)
        a, r = agent._compute_advantages(
            list(batch.rew[sl]), list(batch.val[sl]),
            list(batch.done[sl].astype(float)), boot_vals[s])
        adv_parts.append(a)
        ret_parts.append(r)
    adv = torch.cat(adv_parts).to(agent.device).float()
    ret = torch.cat(ret_parts).to(agent.device).float()
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # ---- reshape everything to [T, n_seg, ...] (GRU's seq-first layout) ----
    x_all = agent.batch_inputs(batch.obs, np.asarray(collector.prev_a_buf, dtype=np.int64),
                               np.asarray(collector.prev_r_buf, dtype=np.float64))
    x_seq = x_all.view(n_seg, T, -1).transpose(0, 1).contiguous()          # [T, n_seg, D]
    act_seq = torch.as_tensor(batch.act, dtype=torch.long,
                              device=agent.device).view(n_seg, T).t().contiguous()
    old_logp_seq = batch.logp.to(agent.device).view(n_seg, T).t().contiguous()
    adv_seq = adv.view(n_seg, T).t().contiguous()
    ret_seq = ret.view(n_seg, T).t().contiguous()
    h0_all = torch.cat(collector.seg_h0, dim=1).to(agent.device)           # [1, n_seg, H]

    # ---- chunk everything to [L, n_chunks * n_seg, ...] ----
    # torch.split cuts the time axis into n_chunks pieces of L; concatenating them along
    # the batch axis makes chunk c of segment s column ``c * n_seg + s``, and lets one
    # forward pass cover an arbitrary set of chunks.
    def _chunkify(t):
        return torch.cat(torch.split(t, L, dim=0), dim=1).contiguous()

    x_ch = _chunkify(x_seq)                                                # [L, K, D]
    act_ch = _chunkify(act_seq)                                            # [L, K]
    old_logp_ch = _chunkify(old_logp_seq)
    adv_ch = _chunkify(adv_seq)
    ret_ch = _chunkify(ret_seq)

    # Behaviour-time entry hidden state of every chunk (see the docstring): one no-grad
    # GRU pass, whose per-step outputs are the per-step hidden states, so the state
    # entering step t is the output at t-1 -- and entering chunk 0 it is the segment's own
    # stored entry state.
    with torch.no_grad():
        y_all = agent.net.hidden_states(x_seq, h0_all)                     # [T, n_seg, S]
    h_ch = torch.cat([h0_all] + [y_all[c * L - 1].unsqueeze(0)
                                 for c in range(1, n_chunks)], dim=1).contiguous()

    K = x_ch.shape[1]                                                      # n_chunks*n_seg
    actor_loss = critic_loss = entropy = None
    grad_norm = 0.0

    for _ in range(int(mini_epochs)):
        perm = torch.randperm(K)
        for start in range(0, K, int(mb_chunks)):
            idx = perm[start:start + int(mb_chunks)]
            logits, values, _ = agent.net(x_ch[:, idx, :], h_ch[:, idx, :].contiguous())
            dist = torch.distributions.Categorical(logits=logits)
            new_logp = dist.log_prob(act_ch[:, idx])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logp - old_logp_ch[:, idx])
            surr1 = ratio * adv_ch[:, idx]
            surr2 = torch.clamp(ratio, 1 - agent.clip_eps,
                                1 + agent.clip_eps) * adv_ch[:, idx]
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (ret_ch[:, idx] - values).pow(2).mean()

            loss = actor_loss + agent.vf_coef * critic_loss - agent.ent_coef * entropy
            agent.optim.zero_grad()
            loss.backward()
            # Always measured (the BPTT stability monitor), clipped only if asked:
            # clip_grad_norm_ with an infinite threshold returns the norm and rescales
            # nothing.
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                agent.net.parameters(),
                float("inf") if agent.max_grad_norm is None else agent.max_grad_norm
            ).item())
            agent.optim.step()

    agent._weights_version += 1
    return {"policy_loss": float(actor_loss.item()),
            "value_loss": float(critic_loss.item()),
            "entropy": float(entropy.item()),
            "grad_norm": grad_norm,
            "mean_episode_return": batch.mean_episode_return,
            "mean_reward_per_step": float(np.mean(batch.rew))}


# ======================================================================================
# 4. Frozen evaluation
# ======================================================================================

class RecurrentEvalPolicy(EvalPolicy):
    """
    Frozen deterministic evaluation policy for the recurrent baseline.

    The policy is run *as-is* -- it self-identifies through ``h`` or it does not; there is
    no auxiliary machinery to help it (plan section 8). Actions are ``argmax`` over the
    logits, no learning happens, and ``h`` is zeroed at the start of every evaluation
    episode by default.

    Why per-episode reset: the 100 episodes of one heatmap cell are i.i.d. draws from a
    single task, and :func:`fig3_common.evaluate_on_task` calls ``reset()`` between them.
    Carrying ``h`` across them would let the baseline pool evidence over ~20 000 steps of
    one task -- a different measurement from "can it identify within an episode", which is
    what the cell is for. ``carry_hidden=True`` runs the other convention for an appendix
    check; note that :func:`fig3_common.evaluate_all_tasks` builds a fresh policy per task,
    so carrying never crosses a task boundary either way.
    """

    def __init__(self, agent: RecurrentPPOAgent, carry_hidden: bool = False):
        import torch

        self.agent = agent
        self.torch = torch
        self.carry_hidden = bool(carry_hidden)
        agent.net.eval()
        self.h = agent.zero_hidden(1)
        self.prev_a, self.prev_r = -1, 0.0

    def reset(self) -> None:
        if not self.carry_hidden:
            self.h = self.agent.zero_hidden(1)
        self.prev_a, self.prev_r = -1, 0.0

    def act(self, obs):
        torch = self.torch
        with torch.inference_mode():
            obs_t = self.agent._flatten_obs(np.asarray(obs, dtype=np.float32))
            x = self.agent.make_input(obs_t, self.prev_a, self.prev_r)
            logits, _, h_new = self.agent.net(x.view(1, 1, -1), self.h)
            self.h = h_new.clone()
            return int(torch.argmax(logits.view(-1)).item())

    def observe(self, obs, action, reward: float, next_obs, done: bool) -> None:
        self.prev_a, self.prev_r = int(action), float(reward)
        if done:
            self.prev_a, self.prev_r = -1, 0.0


# ======================================================================================
# 5. Entry point
# ======================================================================================

def run_single_rep_recurrent_ppo(rep_id, n_rollouts_per_block=BLOCK_SIZES,
                                 n_segments=N_SEGMENTS, seg_steps=SEG_STEPS,
                                 mini_epochs=MINI_EPOCHS, bptt_chunk=BPTT_CHUNK,
                                 mb_chunks=MB_CHUNKS,
                                 gru_hidden=GRU_HIDDEN, separate_cores=True,
                                 max_grad_norm=MAX_GRAD_NORM,
                                 eval_episodes=EVAL_EPISODES, eval_max_steps=EVAL_MAX_STEPS,
                                 eval_carry_hidden=False, max_episode_steps=MAX_EPISODE_STEPS,
                                 ppo_kwargs=None, progress=True, return_agent=False):
    """
    One repetition of the **recurrent PPO (RL^2-class)** baseline for Figure 3 v2.

    Mirrors :func:`fig3_common.run_single_rep_single_ppo` step for step -- same blocked
    stream, same 8 x 256 rollout, same checkpoints, same frozen eval -- with the
    feed-forward actor-critic replaced by the GRU core and the PPO update replaced by its
    sequence-minibatch version. Pool-ready in the Figure-2 style: module-level, heavy
    imports inside, plain numpy/dict return.

    Args:
        rep_id (int): Repetition index, used directly as the seed (torch, numpy, env
            streams), exactly as in the single-PPO floor.
        n_rollouts_per_block (sequence of int): Rollouts per task block in
            :data:`fig3_common.TASKS` order; checkpoints are its cumulative sums, so
            shrinking it for a smoke test shrinks the checkpoints with it.
        n_segments (int), seg_steps (int): Rollout shape (default 8 x 256 = 2048).
        mini_epochs (int): Passes over the rollout (10, as everywhere in Figure 3).
        bptt_chunk (int): Truncated-BPTT chunk length in steps (default 64; must divide
            ``seg_steps``). See :data:`BPTT_CHUNK`.
        mb_chunks (int): Chunks per PPO minibatch (default 4 -> 8 minibatches of 256
            transitions per mini-epoch). See :data:`MB_CHUNKS`.
        gru_hidden (int): Width of one GRU tower (default 64).
        separate_cores (bool): Independent GRU towers for actor and critic (default
            ``True``, mirroring :class:`rl.PPOAgent`'s independent ``policy`` /
            ``value_net``); ``False`` shares one GRU between the two heads. See
            :class:`RecurrentPPOAgent`.
        max_grad_norm (float or None): Global-norm gradient clip for BPTT. Default 10.0;
            ``None`` disables clipping (the other Figure-3 methods' convention, but the
            pilot showed the unclipped recurrent agent collapsing after it had solved the
            task). The per-rollout ``grad_norm`` output is pre-clip either way. See
            :data:`MAX_GRAD_NORM`.
        eval_episodes (int), eval_max_steps (int): Frozen-eval budget per heatmap cell.
        eval_carry_hidden (bool): Carry ``h`` across the episodes of one evaluation cell
            (see :class:`RecurrentEvalPolicy`). Default ``False``.
        max_episode_steps (int): Env time limit (200 everywhere in Figure 3).
        ppo_kwargs (dict, optional): Overrides merged onto
            :data:`fig3_common.PPO_KWARGS` (``gamma``/``gae_lambda``/``clip_eps``/``lr``/
            ``ent_coef``/``vf_coef``/``device``).
        progress (bool): Show a ``tqdm`` bar.
        return_agent (bool): Also return the trained agent (debugging).

    Returns:
        dict: The schema of :func:`fig3_common.run_single_rep_single_ppo` (``A_raw``,
        ``train_returns``, ``train_reward_per_step``, ``seg_returns``, ``task_ids``,
        ``policy_loss``, ``value_loss``, ``entropy``, ``collect_seconds``,
        ``update_seconds``, ``rollout_seconds``, ``eval_seconds``, ``total_seconds``,
        ``checkpoints``, ``block_sizes``, ``env_steps``, ``n_params``, ``task_names``,
        ``method`` = ``"recurrent_ppo"``, ``seed``, ``meta``), plus one extra:

        ==========================  ====================================================
        key                         value
        ==========================  ====================================================
        ``grad_norm``               ``float64 (n_rollouts,)`` pre-clip global gradient
                                    norm of the update's last minibatch -- the BPTT
                                    stability monitor
        ==========================  ====================================================
    """
    import numpy as _np
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
    # Discrete(3)) -- every Figure-3 task exposes the same one.
    proto_env = make_task_envs(3, 1, seed=seed, max_episode_steps=max_episode_steps)[0]
    agent = RecurrentPPOAgent(proto_env, hidden=int(gru_hidden),
                              separate_cores=bool(separate_cores),
                              max_grad_norm=max_grad_norm, **hypers)
    proto_env.close()

    seeds = train_env_seeds(seed, n_rollouts, n_segments)

    A_raw = _np.full((N_TASKS, int(ckpts.size), int(eval_episodes)), _np.nan,
                     dtype=_np.float64)
    train_returns = _np.full(n_rollouts, _np.nan)
    reward_per_step = _np.full(n_rollouts, _np.nan)
    seg_returns = _np.full((n_rollouts, int(n_segments)), _np.nan)
    pol_loss = _np.full(n_rollouts, _np.nan)
    val_loss = _np.full(n_rollouts, _np.nan)
    ent = _np.full(n_rollouts, _np.nan)
    gnorm = _np.full(n_rollouts, _np.nan)
    t_collect = _np.zeros(n_rollouts)
    t_update = _np.zeros(n_rollouts)
    t_eval = _np.zeros(int(ckpts.size))

    t_start = time.perf_counter()
    bar = tqdm(range(n_rollouts), desc=f"recurrent-PPO rep {rep_id}", disable=not progress)
    ckpt_i = 0

    for r in bar:
        task = int(schedule[r])

        t0 = time.perf_counter()
        envs = make_task_envs(task, int(n_segments), seed=int(seeds[r]),
                              max_episode_steps=max_episode_steps)
        # Fresh collector => h zeroed at the rollout boundary (plan section 3.4).
        collector = _RecurrentCollector(agent, int(seg_steps))
        agent.net.train()
        batch = collect_rollout(agent, envs, int(seg_steps),
                                policy_fn=collector.policy_fn,
                                transition_cb=collector.transition_cb, task_id=task)
        close_envs(envs)
        t1 = time.perf_counter()

        stats = recurrent_ppo_update(agent, batch, collector, int(mini_epochs),
                                     int(bptt_chunk), int(mb_chunks))
        t2 = time.perf_counter()

        t_collect[r], t_update[r] = t1 - t0, t2 - t1
        train_returns[r] = stats["mean_episode_return"]
        reward_per_step[r] = stats["mean_reward_per_step"]
        seg_returns[r, :batch.seg_returns.size] = batch.seg_returns
        pol_loss[r], val_loss[r], ent[r] = (
            stats["policy_loss"], stats["value_loss"], stats["entropy"])
        gnorm[r] = stats["grad_norm"]

        if progress:
            bar.set_postfix(task=TASK_NAMES[task], ret=f"{train_returns[r]:.1f}")

        # ---- checkpoint: frozen eval on all five tasks ----
        if ckpt_i < ckpts.size and (r + 1) == int(ckpts[ckpt_i]):
            te = time.perf_counter()
            cell = evaluate_all_tasks(
                lambda _t: RecurrentEvalPolicy(agent, carry_hidden=bool(eval_carry_hidden)),
                int(eval_episodes), int(eval_max_steps), seed,
                max_episode_steps=max_episode_steps)
            agent.net.train()
            elapsed = time.perf_counter() - te
            # A zero-length block makes two checkpoints coincide; fill both.
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
        "grad_norm": gnorm,
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
        "task_names": list(TASK_NAMES),
        "method": "recurrent_ppo",
        "seed": seed,
        "meta": {"seed": seed, "n_segments": int(n_segments), "seg_steps": int(seg_steps),
                 "rollout_steps": int(n_segments) * int(seg_steps),
                 "mini_epochs": int(mini_epochs), "bptt_chunk": int(bptt_chunk),
                 "mb_chunks": int(mb_chunks),
                 "updates_per_rollout": int(mini_epochs) * int(
                     np.ceil((int(n_segments) * int(seg_steps) / int(bptt_chunk))
                             / int(mb_chunks))),
                 "gru_hidden": int(gru_hidden),
                 "separate_cores": bool(separate_cores),
                 "max_grad_norm": (None if max_grad_norm is None else float(max_grad_norm)),
                 "eval_carry_hidden": bool(eval_carry_hidden),
                 "hidden_carry": "across episodes and segments within a rollout; "
                                 "zeroed at rollout boundaries",
                 "prev_action_reward_reset": "zeroed at every episode start",
                 "max_episode_steps": int(max_episode_steps),
                 "eval_episodes": int(eval_episodes),
                 "eval_max_steps": int(eval_max_steps),
                 "block_sizes": blocks.tolist(), "checkpoints": ckpts.tolist(),
                 **hypers},
    }
    if return_agent:
        return result, agent
    return result
