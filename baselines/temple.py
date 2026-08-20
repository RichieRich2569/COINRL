"""
temple.py

O-TempLe (Sun, Yin & Huang, *TempLe: Learning Template of Transitions for Sample
Efficient Multi-task RL*, AAAI 2021; arXiv:2002.06659) adapted to the Figure-2
continual MountainCar experiment.

The agent is an R-Max base learner on the *same* 30x30 discretisation used by
``rl.QLearningAgent``, augmented with a persistent store of **transition
templates**.  A template is the transition-probability vector of a
state-action pair, sorted in descending order (i.e. abstracting *which*
successor states receive the probability mass), together with the pair's mean
reward.  Pairs whose templates are within an l2 threshold ``tau_hat`` are
grouped; a group pools the visit counts of its members, and a pair joining a
group has its own counts **augmented** by back-permuting the group's pooled
statistics into the pair's own (rank-ordered) successors.

Following the paper, TempLe is granted **known task boundaries** -- here the
block schedule of the Figure-2 curriculum (a new task every ``block_len``
episodes).  ``start_new_task()`` folds the finished task's statistics into the
template store and resets the per-task model.  This replaces the probe channel
that COIN-Q and the other baselines use (plan section 1.4).

Hyperparameters
---------------
``m = 30``, ``m_s = 10``, ``tau_hat = 0.15``, ``Rmax = 0`` and ``gamma = 0.99``
are the plan's values and were all kept.  One parameter was added:

* ``aug_cap = 30`` -- the paper transfers a group's *full* pooled count mass.
  Over the 18 blocks of the Figure-2 curriculum that mass grows without bound,
  so a pair's successor rank ordering -- estimated from only ``m_s = 10``
  samples -- ends up carrying an arbitrarily large template that swamps every
  subsequent real observation.  Uncapped augmentation scored -167 mean reward
  against -152 for ``aug_cap = 30`` (4 x 600-episode curriculum, 3 seeds).
  Rescaling the transferred vector to at most ``aug_cap`` preserves the
  template's *shape* -- the thing TempLe actually transfers -- while leaving
  fresh observations influential.

Tuning notes (short-curriculum sweeps, then confirmed at the real 2,500-episode
block length over 3 seeds):

* ``m = 20`` looks better on short blocks but at the real block length it
  causes *accumulating negative transfer* on the harder amplitude
  (block means -159 -> -169 -> -175 over the three encounters of a = 1.5),
  because pairs are trusted after 10 real visits and the store fills with
  under-sampled templates.  ``m = 30`` is statistically tied overall
  (-130.5 vs -131.7) but transfers cleanly in both directions
  (a = 0.5 time-to-competence 89 -> 56 -> 54 episodes; a = 1.5 228 -> 200 -> 214),
  so the plan's value is kept.
* Disabling template matching entirely (a pure per-block R-Max control, obtained
  with ``tau_hat = -1`` so nothing ever matches) removes the transfer effect
  outright: time-to-competence is *flat* across the three encounters of each
  amplitude (a = 0.5: 92 -> 80 -> 82 episodes; a = 1.5: 288 -> 284 -> 296),
  where TempLe gives 89 -> 56 -> 54 and 228 -> 200 -> 214.  The template store
  is what buys the sample efficiency on recurrence, which is the property this
  baseline is meant to demonstrate.
* ``m_s = 5`` and ``tau_hat = 0.05`` were both worse (-161.8 and -157.2).
* Replanning periodically rather than only on known-set changes made no
  measurable difference (-151.6 vs -153.3), so the plan's "replan only when the
  known-set changed" rule is used as written.

Known limitation (worth a sentence in the paper): the transfer is reliable on
a = 0.5 but **not** on a = 1.5.  The template store is shared across both
amplitudes, and the a = 0.5 blocks converge fast and therefore contribute far
more pooled count mass, so a = 1.5 pairs sometimes match groups whose shape is
mildly wrong.  Over 3 seeds a = 1.5 time-to-competence goes 228 -> 200 -> 214,
but individual seeds range from 218 -> 174 -> 182 to 178 -> 216 -> 247.  This is
exactly the cross-amplitude pooling risk flagged in plan section 1.4 (iii) --
templates transfer dynamics only where they genuinely match -- and it needs
averaging over many reps to characterise.

Reproducibility note: as with the other Figure-2 baselines, the Gymnasium
environment's start state is not seeded (``env.reset()`` takes no seed), so
``rep_id`` fixes the agent's tie-breaking RNG but not the environment; run-to-run
spread between single reps is appreciable and reps should be averaged.

Public API
----------
``TempLeAgent``            -- the agent (``train_step`` / ``start_new_task``).
``run_single_rep_temple``  -- one repetition over an amplitude stream,
                              mirroring the ``run_single_rep_*`` pattern used
                              in ``figures.ipynb``.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from typing import Dict, List, Optional, Tuple

from rl import discretize_state


# ---------------------------------------------------------------------------
# Template store
# ---------------------------------------------------------------------------

def _align(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-pad the shorter of two 1-D arrays so both have the same length."""
    if a.size == b.size:
        return a, b
    n = max(a.size, b.size)
    aa = np.zeros(n, dtype=np.float64)
    bb = np.zeros(n, dtype=np.float64)
    aa[: a.size] = a
    bb[: b.size] = b
    return aa, bb


class TemplateStore:
    """
    Persistent (cross-task) collection of transition templates.

    Each group holds
      ``p``       : pooled transition probabilities, sorted descending,
      ``r``       : pooled mean reward,
      ``counts``  : pooled *count* mass, sorted descending (same ordering as p),
      ``n``       : total pooled count mass,
      ``r_sum``   : pooled reward sum (so ``r = r_sum / n``).
    """

    def __init__(self, tau_hat: float = 0.15):
        self.tau_hat = float(tau_hat)
        self.groups: List[Dict[str, object]] = []

    def __len__(self) -> int:
        return len(self.groups)

    # -- template generation ------------------------------------------------

    @staticmethod
    def gen_tt(counts: np.ndarray, mean_reward: float) -> Tuple[np.ndarray, float]:
        """
        Transition template of a pair: its transition-probability vector sorted
        descending and truncated to the observed support, plus its mean reward.
        """
        v = np.sort(np.asarray(counts, dtype=np.float64))[::-1]
        v = v[v > 0.0]
        total = v.sum()
        if total <= 0.0:
            return np.zeros(0, dtype=np.float64), float(mean_reward)
        return v / total, float(mean_reward)

    # -- matching -----------------------------------------------------------

    def distance(self, p: np.ndarray, r: float, gi: int) -> float:
        g = self.groups[gi]
        pa, pb = _align(p, g["p"])              # type: ignore[arg-type]
        return float(np.linalg.norm(pa - pb) + abs(r - float(g["r"])))

    def match(self, p: np.ndarray, r: float) -> Optional[int]:
        """Index of the closest group within ``tau_hat``, else ``None``."""
        if not self.groups or p.size == 0:
            return None
        best_i, best_d = -1, np.inf
        for gi in range(len(self.groups)):
            d = self.distance(p, r, gi)
            if d < best_d:
                best_d, best_i = d, gi
        return best_i if best_d < self.tau_hat else None

    def create(self, counts: np.ndarray, r_sum: float) -> int:
        """Create a new group seeded with a pair's raw (descending) counts."""
        v = np.sort(np.asarray(counts, dtype=np.float64))[::-1]
        v = v[v > 0.0]
        n = float(v.sum())
        self.groups.append(
            {
                "p": v / n if n > 0 else v,
                "r": float(r_sum) / n if n > 0 else 0.0,
                "counts": v.copy(),
                "n": n,
                "r_sum": float(r_sum),
            }
        )
        return len(self.groups) - 1

    def pool(self, gi: int, counts: np.ndarray, r_sum: float) -> None:
        """Fold a pair's (descending-sorted) counts into group ``gi``."""
        v = np.sort(np.asarray(counts, dtype=np.float64))[::-1]
        v = v[v > 0.0]
        if v.size == 0:
            return
        g = self.groups[gi]
        gc, vv = _align(np.asarray(g["counts"], dtype=np.float64), v)
        gc = gc + vv
        n = float(gc.sum())
        g["counts"] = gc
        g["n"] = n
        g["r_sum"] = float(g["r_sum"]) + float(r_sum)     # type: ignore[arg-type]
        g["p"] = gc / n if n > 0 else gc
        g["r"] = float(g["r_sum"]) / n if n > 0 else 0.0  # type: ignore[arg-type]

    def pooled_vector(self, gi: int, k: int, cap: Optional[float] = None
                      ) -> Tuple[np.ndarray, float]:
        """
        Group's pooled count mass redistributed onto ``k`` ranked slots.

        Entries beyond ``k`` are folded into the last slot so the total mass is
        preserved; if the group has fewer than ``k`` entries the vector is
        zero-padded.  Optionally rescaled so the total mass is at most ``cap``.
        Returns (vector, group mean reward).
        """
        g = self.groups[gi]
        c = np.asarray(g["counts"], dtype=np.float64)
        if k <= 0 or c.size == 0:
            return np.zeros(max(k, 0), dtype=np.float64), float(g["r"])
        out = np.zeros(k, dtype=np.float64)
        if c.size >= k:
            out[: k - 1] = c[: k - 1]
            out[k - 1] = c[k - 1:].sum()
        else:
            out[: c.size] = c
        if cap is not None:
            tot = out.sum()
            if tot > cap > 0:
                out = out * (cap / tot)
        return out, float(g["r"])


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TempLeAgent:
    """
    O-TempLe with an R-Max base learner on the Figure-2 MountainCar
    discretisation.

    Args:
        env: an initialised environment (used only for its bins/action count).
        num_position_bins / num_velocity_bins: 30x30, matching ``QLearningAgent``.
        gamma: discount factor (0.99, parity rule).
        m: R-Max "known" threshold on *augmented* counts.
        m_s: visits at which a pair's template is identified (m_s << m).
        tau_hat: template matching threshold.
        Rmax: optimistic reward for unknown pairs (0.0 -- real rewards are -1).
        vi_tol / vi_max_iter: value-iteration stopping rule.
        aug_cap: ceiling on the count mass added by one augmentation.  The
            paper transfers the group's *full* pooled mass (``aug_cap=None``);
            here that mass grows without bound across the 18 blocks of the
            Figure-2 curriculum, so a pair's rank ordering -- estimated from
            only ``m_s`` samples -- ends up carrying an arbitrarily large
            template, which measurably hurts (see module notes).  Rescaling the
            transferred vector to at most ``aug_cap`` preserves the template's
            *shape* (the thing being transferred) while keeping fresh
            observations influential.
        rng: optional numpy Generator for reproducibility.
    """

    TERMINAL = None  # set in __init__ (index of the absorbing terminal state)

    def __init__(
        self,
        env: gym.Env,
        num_position_bins: int = 30,
        num_velocity_bins: int = 30,
        gamma: float = 0.99,
        m: int = 30,
        m_s: int = 10,
        tau_hat: float = 0.15,
        Rmax: float = 0.0,
        vi_tol: float = 1e-4,
        vi_max_iter: int = 500,
        aug_cap: Optional[float] = 30.0,
        template_store: Optional[TemplateStore] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.num_position_bins = int(num_position_bins)
        self.num_velocity_bins = int(num_velocity_bins)
        self.gamma = float(gamma)
        self.m = int(m)
        self.m_s = int(m_s)
        self.Rmax = float(Rmax)
        self.vi_tol = float(vi_tol)
        self.vi_max_iter = int(vi_max_iter)
        self.aug_cap = aug_cap

        self.rng = rng or np.random.default_rng()

        # --- bins: built exactly as in rl.QLearningAgent ---
        self.position_min, self.position_max = env.low[0], env.high[0]
        self.velocity_min, self.velocity_max = env.low[1], env.high[1]
        self.position_bins = np.linspace(
            self.position_min, self.position_max, self.num_position_bins
        )
        self.velocity_bins = np.linspace(
            self.velocity_min, self.velocity_max, self.num_velocity_bins
        )

        self.n_actions = int(env.action_space.n)
        self.n_states = self.num_position_bins * self.num_velocity_bins
        self.TERMINAL = self.n_states  # absorbing, V = 0

        self.store = template_store if template_store is not None else TemplateStore(tau_hat)

        self.n_tasks = 0
        self.n_plans = 0          # cumulative value-iteration solves (all tasks)
        self._reset_task_model()

    # -- bookkeeping --------------------------------------------------------

    def _state_index(self, state: Tuple[int, int]) -> int:
        return int(state[0]) * self.num_velocity_bins + int(state[1])

    def _reset_task_model(self) -> None:
        shape = (self.n_states, self.n_actions)
        self.visits = np.zeros(shape, dtype=np.float64)      # real visits n(s,a)
        self.rsum = np.zeros(shape, dtype=np.float64)        # real reward sums
        self.aug_total = np.zeros(shape, dtype=np.float64)   # augmented mass
        self.raug = np.zeros(shape, dtype=np.float64)        # augmented reward mass
        self.known = np.zeros(shape, dtype=bool)

        self.succ_real: Dict[int, Dict[int, float]] = {}     # pair -> {s': count}
        self.succ_aug: Dict[int, Dict[int, float]] = {}      # pair -> {s': mass}
        self.group_of: Dict[int, int] = {}                   # pair -> group index
        self._seed_vec: Dict[int, np.ndarray] = {}           # pair -> counts it seeded a new group with

        self._V = np.zeros(self.n_states + 1, dtype=np.float64)
        self.Q_plan = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        self._dirty = True                                    # known-set changed since last plan

    def start_new_task(self) -> None:
        """
        Task-boundary hook (granted by the curriculum).  Folds the finished
        task's per-pair statistics into the persistent template store, then
        resets the per-task counts, known-set and plan.
        """
        if self.n_tasks > 0:
            for pair, gi in self.group_of.items():
                s, a = divmod(pair, self.n_actions)
                d = self.succ_real.get(pair)
                if not d:
                    continue
                v = np.sort(np.fromiter(d.values(), dtype=np.float64, count=len(d)))[::-1]
                r_sum = float(self.rsum[s, a])
                seed = self._seed_vec.get(pair)
                if seed is not None:
                    # This pair created the group and already seeded it with
                    # its first m_s counts -- pool only the increment.
                    vv, ss = _align(v, seed)
                    v = np.maximum(vv - ss, 0.0)
                    n_seed = float(seed.sum())
                    n_now = float(self.visits[s, a])
                    r_sum *= (max(n_now - n_seed, 0.0) / n_now) if n_now > 0 else 0.0
                self.store.pool(gi, v, r_sum)
        self.n_tasks += 1
        self._reset_task_model()

    # -- template identification / augmentation -----------------------------

    def _identify_template(self, s: int, a: int) -> None:
        """Called once, when a pair reaches ``m_s`` real visits."""
        pair = s * self.n_actions + a
        d = self.succ_real[pair]
        counts = np.fromiter(d.values(), dtype=np.float64, count=len(d))
        mean_r = float(self.rsum[s, a]) / max(float(self.visits[s, a]), 1.0)
        p_vec, r = self.store.gen_tt(counts, mean_r)

        gi = self.store.match(p_vec, r)
        if gi is None:
            gi = self.store.create(counts, float(self.rsum[s, a]))
            self._seed_vec[pair] = np.sort(counts)[::-1]
            self.group_of[pair] = gi
            return  # new group: nothing to transfer

        self.group_of[pair] = gi
        self._augment(s, a, gi)

    def _augment(self, s: int, a: int, gi: int) -> None:
        """
        Back-permute the group's pooled statistics onto this pair's own
        successors: the largest pooled probability goes to this pair's
        most-visited successor, and so on.
        """
        pair = s * self.n_actions + a
        d = self.succ_real[pair]
        keys = np.fromiter(d.keys(), dtype=np.int64, count=len(d))
        vals = np.fromiter(d.values(), dtype=np.float64, count=len(d))
        order = np.argsort(-vals, kind="stable")     # this pair's rank ordering
        ranked_keys = keys[order]

        vec, g_r = self.store.pooled_vector(gi, ranked_keys.size, cap=self.aug_cap)
        added = float(vec.sum())
        if added <= 0.0:
            return

        aug = self.succ_aug.setdefault(pair, {})
        for k, mass in zip(ranked_keys.tolist(), vec.tolist()):
            if mass > 0.0:
                aug[int(k)] = aug.get(int(k), 0.0) + float(mass)
        self.aug_total[s, a] += added
        self.raug[s, a] += g_r * added

    # -- R-Max planning -----------------------------------------------------

    def _build_induced_model(self):
        ks, ka = np.nonzero(self.known)
        n_known = int(ks.size)
        if n_known == 0:
            return None
        succ_chunks: List[np.ndarray] = []
        prob_chunks: List[np.ndarray] = []
        pid_chunks: List[np.ndarray] = []
        rbar = np.empty(n_known, dtype=np.float64)
        for i in range(n_known):
            s = int(ks[i]); a = int(ka[i])
            pair = s * self.n_actions + a
            d = dict(self.succ_real.get(pair, {}))
            for k, v in self.succ_aug.get(pair, {}).items():
                d[k] = d.get(k, 0.0) + v
            keys = np.fromiter(d.keys(), dtype=np.int64, count=len(d))
            vals = np.fromiter(d.values(), dtype=np.float64, count=len(d))
            tot = vals.sum()
            succ_chunks.append(keys)
            prob_chunks.append(vals / tot)
            pid_chunks.append(np.full(len(d), i, dtype=np.int64))
            n_eff = self.visits[s, a] + self.aug_total[s, a]
            rbar[i] = (self.rsum[s, a] + self.raug[s, a]) / n_eff
        return (
            ks, ka,
            np.concatenate(succ_chunks),
            np.concatenate(prob_chunks),
            np.concatenate(pid_chunks),
            rbar,
            n_known,
        )

    def plan(self) -> None:
        """
        Value iteration on the R-Max induced MDP: empirical dynamics for known
        pairs, an absorbing optimistic state with reward ``Rmax`` for unknown
        pairs.  Warm-started from the previous solution.
        """
        self.n_plans += 1
        self._dirty = False
        model = self._build_induced_model()

        # Unknown pairs are worth Rmax + gamma * Rmax / (1 - gamma) ... with an
        # absorbing optimistic state of reward Rmax this is Rmax / (1 - gamma).
        q_unknown = self.Rmax / (1.0 - self.gamma)

        if model is None:
            self.Q_plan = np.full((self.n_states, self.n_actions), q_unknown)
            self._V[: self.n_states] = q_unknown
            self._V[self.TERMINAL] = 0.0
            return

        ks, ka, succ, prob, pid, rbar, n_known = model
        V = self._V
        V[self.TERMINAL] = 0.0
        Q = np.full((self.n_states, self.n_actions), q_unknown, dtype=np.float64)

        for _ in range(self.vi_max_iter):
            sums = np.bincount(pid, weights=prob * V[succ], minlength=n_known)
            Q[:] = q_unknown
            Q[ks, ka] = rbar + self.gamma * sums
            V_new = Q.max(axis=1)
            delta = float(np.max(np.abs(V_new - V[: self.n_states])))
            V[: self.n_states] = V_new
            if delta < self.vi_tol:
                break

        self.Q_plan = Q

    # -- acting / learning --------------------------------------------------

    def choose_action(self, s_idx: int) -> int:
        """Greedy on the planned optimistic Q, with random tie-breaking."""
        q = self.Q_plan[s_idx]
        top = q.max()
        cand = np.flatnonzero(q >= top - 1e-9)
        if cand.size == 1:
            return int(cand[0])
        return int(cand[self.rng.integers(cand.size)])

    def observe(self, s_idx: int, action: int, reward: float,
                next_idx: int) -> None:
        """Record one bin-level transition and update the known-set."""
        pair = s_idx * self.n_actions + action
        d = self.succ_real.get(pair)
        if d is None:
            d = {}
            self.succ_real[pair] = d
        d[next_idx] = d.get(next_idx, 0.0) + 1.0

        self.visits[s_idx, action] += 1.0
        self.rsum[s_idx, action] += reward

        # Template identification at m_s visits (once per pair per task).
        if self.visits[s_idx, action] == self.m_s and pair not in self.group_of:
            self._identify_template(s_idx, action)

        if not self.known[s_idx, action]:
            if self.visits[s_idx, action] + self.aug_total[s_idx, action] >= self.m:
                self.known[s_idx, action] = True
                self._dirty = True

    def train_step(self, env: gym.Env, max_steps_per_episode: int = 200) -> float:
        """
        Run one episode acting greedily w.r.t. the planned optimistic Q
        (R-Max optimism supplies the exploration -- no epsilon).

        Returns:
            float: total episode reward.
        """
        # Replan at most once per episode, and only if the known-set moved.
        if self._dirty:
            self.plan()

        obs, _ = env.reset()
        s_idx = self._state_index(
            discretize_state(obs, self.position_bins, self.velocity_bins)
        )
        episode_reward = 0.0

        for _ in range(max_steps_per_episode):
            action = self.choose_action(s_idx)
            next_obs, reward, done, truncated, _ = env.step(action)
            next_idx = self._state_index(
                discretize_state(next_obs, self.position_bins, self.velocity_bins)
            )

            # A genuine termination is absorbing; a time-limit truncation is not.
            self.observe(s_idx, action, reward, self.TERMINAL if done else next_idx)

            s_idx = next_idx
            episode_reward += reward

            if done or truncated:
                break

        env.close()
        return episode_reward

    # -- diagnostics --------------------------------------------------------

    def stats(self) -> Dict[str, float]:
        return {
            "n_groups": float(len(self.store)),
            "n_pairs_grouped": float(len(self.group_of)),
            "n_pairs_visited": float(np.count_nonzero(self.visits)),
            "n_known": float(int(self.known.sum())),
            "n_plans": float(self.n_plans),
            "n_tasks": float(self.n_tasks),
        }


# ---------------------------------------------------------------------------
# Figure-2 repetition runner
# ---------------------------------------------------------------------------

def run_single_rep_temple(rep_id, true_amplitudes, block_len=2500):
    """
    Runs one repetition of training for TempLe (O-TempLe, granted task
    boundaries).  Returns rewards at each time step.

    TempLe receives no probe: instead of COIN's amplitude estimate it is told
    where the task boundaries are (every ``block_len`` episodes), which is the
    paper's standing assumption.
    """
    from environments import CustomMountainCarEnv
    from temple import TempLeAgent
    from tqdm.auto import tqdm
    import numpy as np

    SEED = rep_id
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    env = CustomMountainCarEnv(amplitude=1.0, render_mode="none")
    agent = TempLeAgent(
        env=env,
        num_position_bins=30,
        num_velocity_bins=30,
        gamma=0.99,
        m=30,
        m_s=10,
        tau_hat=0.15,
        Rmax=0.0,
        aug_cap=30.0,
        rng=rng,
    )

    rewards_for_this_rep = []
    pbar = tqdm(true_amplitudes, desc=f"Rep {rep_id}")
    for i, amplitude in enumerate(pbar):
        # Granted task boundary (replaces the probe channel).
        if i % block_len == 0:
            agent.start_new_task()

        # Create the environment for each amplitude
        env = CustomMountainCarEnv(amplitude=amplitude, render_mode="none")

        training_reward = agent.train_step(env=env, max_steps_per_episode=200)
        rewards_for_this_rep.append(training_reward)

        pbar.set_postfix(
            amplitude=amplitude,
            reward=training_reward,
            known=int(agent.known.sum()),
            groups=len(agent.store),
        )

    return rewards_for_this_rep
