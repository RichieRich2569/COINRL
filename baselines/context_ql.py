"""
context_ql.py

Context-QL baseline for Figure 2 (Padakandla, Prabuchandran & Bhatnagar, *Reinforcement
learning algorithm for non-stationary environments*, Applied Intelligence 50:3590-3606,
2020; arXiv:1905.03970).

The algorithm keeps one tabular Q-table per context, runs Q-learning on the **active**
table only (hard assignment - the defining contrast with COIN-Q's soft, responsibility-
weighted updates), and advances the active context according to a *known pattern of
changes* (the paper's Assumption 3) whenever a change-point detector fires. Our Figure 2
curriculum alternates strictly between two amplitudes, so the assumption holds exactly and
"advance in the known pattern" reduces to toggling between ``K = 2`` tables.

The paper detects changes with ODCP (Online parametric Dirichlet Change Point). No
maintained Python implementation exists, and the authors report ODCP and the Euclidean
E-Divisive (ECP) detector give near-identical change-points on experience tuples (their
Tables 1-2), so this module supplies two lightweight stand-ins:

* :class:`CUSUMDetector` (alias :data:`cusum_detector`) - two-sided CUSUM for the
  univariate probe stream, i.e. the channel COIN-Q sees.
* :func:`energy_change_detector` - a permutation-based energy/E-divisive two-sample test
  over per-episode feature vectors, for the "native" variant that never sees the probe.

Detection logic deliberately lives *outside* the agent (module-level detectors + the run
loops), mirroring how COIN-Q's notebook loop owns ``RealTimeCOIN``.

Hyperparameters follow the Figure 2 parity rules: 30x30 discretisation, ``alpha=0.1``,
``gamma=0.99``, epsilon-greedy from ``1.0`` with ``0.999`` per-episode decay (applied
**per table**, only on the table that was trained) and a ``0.01`` floor.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from typing import Tuple, List, Optional, Union

from rl import (
    discretize_state,
    EpsilonGreedy,
    ExplorationStrategy,
    SoftmaxExploration,
    RandomExploration,
)

__all__ = [
    "ContextQLAgent",
    "CUSUMDetector",
    "cusum_detector",
    "energy_change_detector",
    "run_single_rep_context_ql_probe",
    "run_single_rep_context_ql_native",
]


# ---- Context-QL agent ----

class ContextQLAgent:
    """
    Context-QL: ``K`` tabular Q-tables, Q-learning on the active table only.

    The agent is deliberately **detector-agnostic**: it exposes :meth:`toggle` (advance the
    active context in the known alternating pattern) and leaves the decision of *when* to
    call it to the training loop.
    """

    def __init__(
        self,
        env: gym.Env,
        num_contexts: int = 2,
        num_position_bins: int = 30,
        num_velocity_bins: int = 30,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.999,
        min_epsilon: float = 0.01,
        init_Q_random: bool = True,
        exploration: Union[str, "ExplorationStrategy", None] = "epsilon_greedy",
        softmax_temperature: float = 1.0,
        softmax_decay: float = 0.999,
        softmax_min_temperature: float = 0.05,
        rng: Optional[Union[np.random.Generator, int]] = None,
    ):
        """
        Initialise the Context-QL agent with hyperparameters and bin settings.

        Args:
            env (gym.Env): An initialized Gymnasium environment.
            num_contexts (int): Number of contexts ``K`` in the known change pattern.
            num_position_bins (int, optional): Number of bins for discretizing position.
            num_velocity_bins (int, optional): Number of bins for discretizing velocity.
            alpha (float, optional): Learning rate.
            gamma (float, optional): Discount factor.
            epsilon (float, optional): Initial epsilon for every context's ε-greedy schedule.
            epsilon_decay (float, optional): Per-table epsilon decay, applied once per
                episode to the table that was actually trained.
            min_epsilon (float, optional): Minimum value of epsilon.
            init_Q_random (bool, optional): Random-initialize Q-tables in [-2, 0] if True,
                else zeros.
            exploration (str|ExplorationStrategy|None): "epsilon_greedy" (default),
                "softmax", "random", or a custom strategy instance.
            softmax_*: Parameters for softmax exploration.
            rng: Optional numpy Generator (or integer seed) for reproducibility.
        """
        self.num_contexts = int(num_contexts)
        self.num_position_bins = num_position_bins
        self.num_velocity_bins = num_velocity_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = epsilon
        self.min_epsilon = min_epsilon

        if isinstance(rng, np.random.Generator) or rng is None:
            self.rng = rng or np.random.default_rng()
        else:
            self.rng = np.random.default_rng(rng)

        # Extract state boundaries (assuming a 2D state: [position, velocity])
        self.position_min, self.position_max = env.low[0], env.high[0]
        self.velocity_min, self.velocity_max = env.low[1], env.high[1]

        # Create bins
        self.position_bins = np.linspace(self.position_min, self.position_max, self.num_position_bins)
        self.velocity_bins = np.linspace(self.velocity_min, self.velocity_max, self.num_velocity_bins)

        # Initialize the stack of per-context Q-tables (no novel context: K is given)
        n_actions = env.action_space.n
        if init_Q_random:
            self.Qdat = self.rng.uniform(
                low=-2, high=0,
                size=(self.num_contexts, self.num_position_bins, self.num_velocity_bins, n_actions)
            )
        else:
            self.Qdat = np.zeros(
                (self.num_contexts, self.num_position_bins, self.num_velocity_bins, n_actions)
            )

        # Per-context epsilon (for ε-greedy only; ignored by other strategies)
        self.epsdat = [epsilon for _ in range(self.num_contexts)]

        # Index of the active context in the known change pattern
        self.active = 0

        # Per-episode summary of the last episode, consumed by the native detector
        self.last_episode_features: Optional[np.ndarray] = None

        # Build exploration strategy (global; epsilon is overridden per context at call time)
        self.strategy = self._build_strategy(
            exploration=exploration,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon,
            softmax_temperature=softmax_temperature,
            softmax_decay=softmax_decay,
            softmax_min_temperature=softmax_min_temperature,
        )

    @staticmethod
    def _build_strategy(
        exploration: Union[str, "ExplorationStrategy", None],
        *,
        epsilon: float,
        epsilon_decay: float,
        min_epsilon: float,
        softmax_temperature: float,
        softmax_decay: float,
        softmax_min_temperature: float,
    ) -> "ExplorationStrategy":
        # If a custom strategy instance was passed, accept it
        if isinstance(exploration, ExplorationStrategy):
            return exploration  # type: ignore[return-value]

        # Named strategies
        if exploration is None or exploration == "epsilon_greedy":
            return EpsilonGreedy(epsilon, epsilon_decay, min_epsilon)
        if exploration == "softmax":
            return SoftmaxExploration(
                temperature=softmax_temperature,
                decay=softmax_decay,
                min_temperature=softmax_min_temperature,
            )
        if exploration == "random":
            return RandomExploration()

        raise ValueError(f"Unknown exploration type: {exploration!r}")

    # -------- Context bookkeeping --------

    @property
    def Q(self) -> np.ndarray:
        """The active context's Q-table (a view into ``Qdat``)."""
        return self.Qdat[self.active]

    @property
    def epsilon(self) -> float:
        """The active context's epsilon (kept for logging parity with QLearningAgent)."""
        return self.epsdat[self.active]

    def toggle(self) -> int:
        """
        Advance the active context by one step of the known change pattern.

        Our Figure 2 curriculum alternates ``0.5 <-> 1.5``, so the pattern is a cycle over
        the ``K`` tables; with ``K = 2`` this is exactly the paper's ``c <- 1 - c`` toggle.
        Q-tables are *reused* when a context recurs - no new table is created.
        """
        self.active = (self.active + 1) % self.num_contexts
        return self.active

    def set_context(self, context: int) -> int:
        """Force the active context (used by identification-style variants)."""
        self.active = int(context) % self.num_contexts
        return self.active

    # -------- Policy & Learning --------

    def choose_action(self, env: gym.Env, state: Tuple[int, int], eps: float) -> int:
        """
        Choose an action using the configured exploration strategy on the active table.
        For ε-greedy, 'eps' overrides the strategy's epsilon so each context keeps its own
        schedule. For other strategies, 'eps' is ignored.
        """
        q_row = self.Qdat[self.active][state]
        return self.strategy.select_action(
            q_row,
            env.action_space,
            epsilon_override=eps if isinstance(self.strategy, EpsilonGreedy) else None,
            rng=self.rng,
        )

    def update_q_table(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int]
    ) -> None:
        """Standard tabular Q-learning update, applied to the active table only."""
        Q = self.Qdat[self.active]
        best_next_action = int(np.argmax(Q[next_state]))
        td_target = reward + self.gamma * Q[next_state][best_next_action]
        td_error = td_target - Q[state][action]
        Q[state][action] += self.alpha * td_error

    def _run_episode(
        self,
        env: gym.Env,
        max_steps_per_episode: int = 200,
    ) -> Tuple[float, np.ndarray]:
        """
        Run (and learn from) one episode on the active context, returning the episode
        reward and the episode feature vector ``(return, mean|velocity|, max position)``
        used by the native change-point detector.
        """
        obs, _ = env.reset()
        state = discretize_state(obs, self.position_bins, self.velocity_bins)
        episode_reward = 0.0

        abs_vel_sum = abs(float(obs[1]))
        max_pos = float(obs[0])
        n_obs = 1

        c = self.active
        for _ in range(max_steps_per_episode):
            action = self.choose_action(env, state, self.epsdat[c])
            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = discretize_state(next_obs, self.position_bins, self.velocity_bins)

            self.update_q_table(state, action, reward, next_state)

            state = next_state
            episode_reward += reward

            abs_vel_sum += abs(float(next_obs[1]))
            max_pos = max(max_pos, float(next_obs[0]))
            n_obs += 1

            if done or truncated:
                break

        # Decay only the active table's schedule: a table that is not being trained must
        # not lose its exploration budget (parity rule 3).
        if isinstance(self.strategy, EpsilonGreedy):
            self.epsdat[c] = max(self.min_epsilon, self.epsdat[c] * self.epsilon_decay)
        else:
            self.strategy.on_episode_end()

        features = np.array([episode_reward, abs_vel_sum / n_obs, max_pos], dtype=float)
        self.last_episode_features = features

        env.close()
        return episode_reward, features

    def train_step(
        self,
        env: gym.Env,
        max_steps_per_episode: int = 200
    ) -> float:
        """
        Train the active context's Q-table for one episode.

        The episode summary statistics are also stored on
        :attr:`last_episode_features` (see :meth:`train_step_with_features`).

        Returns:
            float: total episode reward
        """
        episode_reward, _ = self._run_episode(env, max_steps_per_episode)
        return episode_reward

    def train_step_with_features(
        self,
        env: gym.Env,
        max_steps_per_episode: int = 200
    ) -> Tuple[float, np.ndarray]:
        """
        As :meth:`train_step`, but also returns the per-episode feature vector
        ``(return, mean|velocity|, max position)`` fed to :func:`energy_change_detector`.
        """
        return self._run_episode(env, max_steps_per_episode)

    def train(
        self,
        env: gym.Env,
        n_episodes: int = 5000,
        max_steps_per_episode: int = 200,
        verbose: bool = False,
        print_freq: int = 200
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Train on a *stationary* env with the current active context (no detection).
        Mainly a convenience/debug entry point; Figure 2 uses the run functions below.
        """
        all_episode_rewards: List[float] = []

        for episode in range(n_episodes):
            episode_reward = self.train_step(env, max_steps_per_episode)
            all_episode_rewards.append(episode_reward)

            if verbose and (episode + 1) % print_freq == 0:
                avg_reward = float(np.mean(all_episode_rewards[-print_freq:]))
                print(
                    f"Episode: {episode + 1}, "
                    f"Avg Reward (last {print_freq}): {avg_reward:.2f}, "
                    f"Context: {self.active}, Epsilon: {self.epsilon:.3f}"
                )

        env.close()
        return self.Qdat, all_episode_rewards

    def evaluate(
        self,
        env: gym.Env,
        n_episodes: int = 10,
        max_steps_per_episode: int = 200
    ) -> List[float]:
        """
        Execute the greedy policy (argmax over the active context's Q-table).
        This method does not update the Q-tables.
        """
        rewards: List[float] = []
        Q = self.Qdat[self.active]
        for _ in range(n_episodes):
            obs, _ = env.reset()
            state = discretize_state(obs, self.position_bins, self.velocity_bins)
            episode_reward = 0.0

            for _ in range(max_steps_per_episode):
                action = int(np.argmax(Q[state]))
                next_obs, reward, done, truncated, _ = env.step(action)
                state = discretize_state(next_obs, self.position_bins, self.velocity_bins)

                episode_reward += reward

                if done or truncated:
                    break

            rewards.append(episode_reward)

        env.close()
        return rewards


# ---- Change-point detectors (kept outside the agent) ----

class CUSUMDetector:
    """
    Two-sided CUSUM on a univariate stream, used for the probe variant of Context-QL.

    The statistic is accumulated on the deviation from the running mean *since the last
    declared change*, so no prior knowledge of the pre/post levels is needed:

    ``S+ <- max(0, S+ + (x - mean) - drift)``,  ``S- <- max(0, S- - (x - mean) - drift)``

    and a change is declared when either exceeds ``threshold``. On detection the detector
    resets, restarting the running mean from the observation that triggered it.

    Defaults are set for the Figure 2 probe stream, which jumps between ~0.5 and ~1.5: a
    ``drift`` of 0.25 ignores probe noise while a single post-change probe (deviation ~1.0)
    pushes the statistic past ``threshold = 0.5`` immediately.
    """

    def __init__(self, drift: float = 0.25, threshold: float = 0.5, min_samples: int = 1):
        """
        Args:
            drift (float): Slack subtracted at every update (the CUSUM "k"); deviations
                smaller than this never accumulate.
            threshold (float): Decision threshold "h" on either one-sided statistic.
            min_samples (int): Observations required since the last change before a new
                one may be declared.
        """
        self.drift = float(drift)
        self.threshold = float(threshold)
        self.min_samples = int(min_samples)
        self.reset()

    def reset(self, x: Optional[float] = None) -> None:
        """Clear the statistics, optionally seeding the running mean with ``x``."""
        self.n = 0 if x is None else 1
        self.mean = 0.0 if x is None else float(x)
        self.s_hi = 0.0
        self.s_lo = 0.0

    def update(self, x: float) -> bool:
        """
        Feed one observation.

        Returns:
            bool: True if a change is declared (the detector has then reset itself).
        """
        x = float(x)

        # First observation since the last change only defines the reference level
        if self.n == 0:
            self.reset(x)
            return False

        dev = x - self.mean
        self.s_hi = max(0.0, self.s_hi + dev - self.drift)
        self.s_lo = max(0.0, self.s_lo - dev - self.drift)

        if self.n >= self.min_samples and (self.s_hi > self.threshold or self.s_lo > self.threshold):
            self.reset(x)
            return True

        # Fold the observation into the running mean
        self.n += 1
        self.mean += (x - self.mean) / self.n
        return False


# The plan refers to this detector in lower case; keep both names available.
cusum_detector = CUSUMDetector


def _energy_profile(D: np.ndarray, min_seg: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Energy (E-divisive) two-sample statistic for every admissible split of a pre-computed
    pairwise-distance matrix ``D``.

    For a split into ``X`` (m points) and ``Y`` (k points) the statistic is
    ``E = mk/(m+k) * ( 2/(mk) ΣΣ|x-y| - 1/m² ΣΣ|x-x'| - 1/k² ΣΣ|y-y'| )``.
    All three double sums come from a 2-D cumulative sum of ``D``, so the whole profile
    costs one pass regardless of the number of splits.
    """
    n = D.shape[0]
    S = np.zeros((n + 1, n + 1))
    S[1:, 1:] = D.cumsum(axis=0).cumsum(axis=1)
    total = S[n, n]

    taus = np.arange(min_seg, n - min_seg + 1)
    within_x = S[taus, taus]                    # ΣΣ over X x X
    cross = S[taus, n] - within_x               # ΣΣ over X x Y
    within_y = total - within_x - 2.0 * cross   # ΣΣ over Y x Y

    m = taus.astype(float)
    k = (n - taus).astype(float)
    E = (m * k / (m + k)) * (
        2.0 * cross / (m * k) - within_x / m ** 2 - within_y / k ** 2
    )
    return taus, E


def energy_change_detector(
    features_2d: np.ndarray,
    n_perms: int = 99,
    alpha: float = 0.01,
    min_seg: int = 10,
    rng: Optional[Union[np.random.Generator, int]] = None,
) -> Optional[int]:
    """
    Permutation-based energy / E-divisive change-point test over per-episode features.

    This is the ECP-style stand-in for ODCP (see the module docstring). Columns are
    z-scored first so that features on very different scales (episode return ~1e2 vs mean
    speed ~1e-2) contribute comparably to the Euclidean energy distance. The best single
    split is scored against ``n_perms`` random row permutations of the same window.

    Args:
        features_2d (np.ndarray): ``(n_episodes, n_features)`` array of per-episode
            feature vectors, ordered in time.
        n_perms (int): Number of permutations for the p-value.
        alpha (float): Significance level.
        min_seg (int): Minimum number of episodes on each side of a split.
        rng: Optional numpy Generator (or seed) for the permutations.

    Returns:
        Optional[int]: The index of the best split (the first episode of the second
        segment) if significant, else None.
    """
    X = np.asarray(features_2d, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = X.shape[0]
    if n < 2 * min_seg:
        return None

    # Standardise columns so all features weigh comparably in the Euclidean distance
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Z = (X - X.mean(axis=0)) / sd

    # Pairwise Euclidean distances
    diff = Z[:, None, :] - Z[None, :, :]
    D = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))

    taus, E = _energy_profile(D, min_seg)
    if E.size == 0:
        return None
    best = int(np.argmax(E))
    obs = float(E[best])

    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    n_ge = 0
    for _ in range(n_perms):
        p = generator.permutation(n)
        _, E_perm = _energy_profile(D[np.ix_(p, p)], min_seg)
        if float(E_perm.max()) >= obs:
            n_ge += 1

    p_value = (1.0 + n_ge) / (n_perms + 1.0)
    if p_value <= alpha:
        return int(taus[best])
    return None


# ---- Figure 2 run functions (one repetition each, multiprocessing-friendly) ----

def run_single_rep_context_ql_probe(
    rep_id,
    true_amplitudes,
    probe_period: int = 100,
    return_info: bool = False,
):
    """
    Runs one repetition of training for Context-QL, probe variant.

    Detection uses the *same* channel as COIN-Q: a one-step probe every ``probe_period``
    episodes, inverted by ``rl.amplitude_estimator``, never the true amplitude. The
    resulting univariate stream is monitored by :class:`CUSUMDetector`; on detection the
    agent toggles to the next context in the known alternating pattern.

    Returns rewards at each time step (or ``(rewards, info)`` when ``return_info``).
    """
    # Imports for multiprocessing
    from environments import CustomMountainCarEnv
    from rl import probe_amplitude
    from context_ql import ContextQLAgent, CUSUMDetector
    from tqdm.auto import tqdm
    import numpy as np

    SEED = rep_id
    rng = np.random.default_rng(SEED)

    # Create a fresh agent and environment inside each process
    env = CustomMountainCarEnv(amplitude=1.0, render_mode="none")
    agent = ContextQLAgent(
        env=env,
        num_contexts=2,
        num_position_bins=30,
        num_velocity_bins=30,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.999,
        min_epsilon=0.01,
        rng=rng,
    )
    detector = CUSUMDetector()

    rewards_for_this_rep = []
    changepoints = []
    probes = []

    pbar = tqdm(true_amplitudes, desc=f"Rep {rep_id} (Context-QL probe)")
    for i, amplitude in enumerate(pbar):
        # Create the environment for each amplitude
        env = CustomMountainCarEnv(amplitude=amplitude, render_mode="none")
        env.action_space.seed(int(rng.integers(1 << 31)))

        # Obtain a small probe from the environment every probe_period episodes
        if i % probe_period == 0:
            est_a = probe_amplitude(env)
            probes.append((i, est_a))

            # Feed the probe to the change-point detector; toggle on detection
            if detector.update(est_a):
                agent.toggle()
                changepoints.append(i)

        # Train the active context in the current environment
        training_reward = agent.train_step(env=env, max_steps_per_episode=200)
        rewards_for_this_rep.append(training_reward)

        pbar.set_postfix(amplitude=amplitude, reward=training_reward, context=agent.active)

    if return_info:
        return rewards_for_this_rep, {"changepoints": changepoints, "probes": probes}
    return rewards_for_this_rep


def run_single_rep_context_ql_native(
    rep_id,
    true_amplitudes,
    detect_every: int = 10,
    window: int = 200,
    return_info: bool = False,
):
    """
    Runs one repetition of training for Context-QL, native variant (no probe).

    Detection is the paper's own setting - change-points inferred from the agent's own
    experience. We aggregate experience to *episode-level* features
    ``(return, mean|velocity|, max position)`` rather than the paper's raw per-step tuples
    (a permutation test over ~10^5-10^6 tuples is infeasible; the features are the same
    statistics, aggregated). Every ``detect_every`` episodes the features collected since
    the last declared change-point (capped at the most recent ``window`` episodes for
    speed) are passed to :func:`energy_change_detector`; on detection the agent toggles and
    the change-point is advanced to the detected split.

    Returns rewards at each time step (or ``(rewards, info)`` when ``return_info``).
    """
    # Imports for multiprocessing
    from environments import CustomMountainCarEnv
    from context_ql import ContextQLAgent, energy_change_detector
    from tqdm.auto import tqdm
    import numpy as np

    SEED = rep_id
    rng = np.random.default_rng(SEED)

    # Create a fresh agent and environment inside each process
    env = CustomMountainCarEnv(amplitude=1.0, render_mode="none")
    agent = ContextQLAgent(
        env=env,
        num_contexts=2,
        num_position_bins=30,
        num_velocity_bins=30,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.999,
        min_epsilon=0.01,
        rng=rng,
    )

    rewards_for_this_rep = []
    changepoints = []

    buffer: List[np.ndarray] = []   # episode features since the last change-point tau*
    buf_start = 0                   # episode index of buffer[0]

    pbar = tqdm(true_amplitudes, desc=f"Rep {rep_id} (Context-QL native)")
    for i, amplitude in enumerate(pbar):
        # Create the environment for each amplitude
        env = CustomMountainCarEnv(amplitude=amplitude, render_mode="none")
        env.action_space.seed(int(rng.integers(1 << 31)))

        # Train the active context in the current environment, collecting episode features
        training_reward, features = agent.train_step_with_features(
            env=env, max_steps_per_episode=200
        )
        rewards_for_this_rep.append(training_reward)
        buffer.append(features)

        # Periodically test the experience since tau* for a change-point
        if (i + 1) % detect_every == 0:
            offset = max(0, len(buffer) - window)
            F = np.asarray(buffer[offset:])
            tau = energy_change_detector(F, n_perms=99, alpha=0.01, rng=rng)
            if tau is not None:
                tau_global = buf_start + offset + int(tau)
                agent.toggle()
                changepoints.append(tau_global)
                # tau* <- tau: keep only the post-change experience
                buffer = buffer[tau_global - buf_start:]
                buf_start = tau_global

        pbar.set_postfix(amplitude=amplitude, reward=training_reward, context=agent.active)

    if return_info:
        return rewards_for_this_rep, {"changepoints": changepoints}
    return rewards_for_this_rep
