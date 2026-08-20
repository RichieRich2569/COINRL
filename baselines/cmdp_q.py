"""
cmdp_q.py

Contextual-MDP (CMDP) baseline for the Figure 2 continual Q-learning experiment.

A CMDP ``<C, S, A, M>`` indexes one MDP ``M(c) = {R_c, T_c}`` per context ``c``; the agent
simply conditions its value function on the observed context, ``Q(s, a, c)``
(Hallak, Di Castro & Mannor, 2015; the tabular ancestor of UVFA, Schaul et al., 2015).
Here the context signal is the same one-step amplitude probe COIN-Q receives
(``rl.probe_amplitude``), binned onto a fixed grid. There is no change detection and no
inference: on each probe the active Q-table slice is selected and vanilla Q-learning runs
inside that slice. Generalisation between contexts happens only through the binning.

Follows the Figure 2 parity rules: 30x30 state discretisation, ``alpha=0.1``,
``gamma=0.99``, epsilon-greedy with ``epsilon=1.0`` decayed by ``0.999`` per episode
(per context bin, as COIN-Q decays per context), floor ``0.01``, 200-step episodes.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from typing import List, Optional, Tuple, Union

from rl import EpsilonGreedy, ExplorationStrategy, discretize_state


# Bin edges for the probe-based context axis: width 0.25 over [0, 2], i.e. 8 bins.
# Chosen so the two true amplitudes (0.5, 1.5) fall in distinct bins.
DEFAULT_CONTEXT_BIN_EDGES = np.arange(0.0, 2.01, 0.25)

# The Figure 2 amplitudes (0.5, 1.5) land exactly *on* bin edges, and the probe inverts
# float32 environment states, so an estimate of a true 0.5 comes back as e.g. 0.49999362.
# Without a tolerance that round-off alone would scatter one context across two adjacent
# bins. Values within this distance below an edge are treated as sitting on it; the
# tolerance is ~4 orders of magnitude smaller than the bin width, so genuine context
# differences are unaffected.
DEFAULT_EDGE_TOLERANCE = 1e-4


class CMDPQLearningAgent:
    """
    A parameter-augmented (Contextual-MDP) Q-learning agent.

    The Q-table carries an extra leading axis indexed by the binned context signal:
    ``Q[c_bin, pos, vel, a]``. ``set_context(theta_hat)`` bins an amplitude estimate and
    selects the active slice; ``train_step`` then behaves exactly like
    :class:`rl.QLearningAgent` restricted to that slice, with its own epsilon.
    """

    def __init__(
        self,
        env: gym.Env,
        context_bin_edges: Optional[np.ndarray] = None,
        num_position_bins: int = 30,
        num_velocity_bins: int = 30,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.999,
        min_epsilon: float = 0.01,
        init_Q_random: bool = True,
        exploration: Union[str, "ExplorationStrategy", None] = "epsilon_greedy",
        edge_tolerance: float = DEFAULT_EDGE_TOLERANCE,
        rng: Optional[Union[np.random.Generator, int]] = None,
    ):
        """
        Initialize the CMDP Q-learning agent with hyperparameters and bin settings.

        Args:
            env (gym.Env): An initialized Gymnasium environment.
            context_bin_edges (np.ndarray, optional): Monotonic edges of the context axis.
                Defaults to ``np.arange(0.0, 2.01, 0.25)`` (8 bins of width 0.25).
                Estimates are clipped into ``[edges[0], edges[-1]]`` before binning.
            num_position_bins (int, optional): Number of bins for discretizing position.
            num_velocity_bins (int, optional): Number of bins for discretizing velocity.
            alpha (float, optional): Learning rate.
            gamma (float, optional): Discount factor.
            epsilon (float, optional): Initial epsilon for every context bin.
            epsilon_decay (float, optional): Epsilon decay factor, applied to a bin's own
                epsilon after each episode trained in that bin.
            min_epsilon (float, optional): Minimum value of epsilon.
            init_Q_random (bool, optional): Random-initialize Q-table if True, else zeros.
            exploration (str|ExplorationStrategy|None): "epsilon_greedy" (default) or a
                custom strategy instance. Only epsilon-greedy honours the per-bin epsilon.
            edge_tolerance (float, optional): Estimates within this distance below a bin
                edge are treated as sitting on it, absorbing float32 probe round-off.
            rng: Optional numpy Generator (or seed) for reproducibility.
        """
        self.num_position_bins = num_position_bins
        self.num_velocity_bins = num_velocity_bins
        self.alpha = alpha
        self.gamma = gamma

        self.max_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

        # Context axis: binned probe estimate
        edges = DEFAULT_CONTEXT_BIN_EDGES if context_bin_edges is None else context_bin_edges
        self.context_bin_edges = np.asarray(edges, dtype=float)
        self.n_context_bins = len(self.context_bin_edges) - 1
        self.edge_tolerance = float(edge_tolerance)

        # Extract state boundaries (assuming a 2D state: [position, velocity])
        self.position_min, self.position_max = env.low[0], env.high[0]
        self.velocity_min, self.velocity_max = env.low[1], env.high[1]

        # Create bins
        self.position_bins = np.linspace(self.position_min, self.position_max, self.num_position_bins)
        self.velocity_bins = np.linspace(self.velocity_min, self.velocity_max, self.num_velocity_bins)

        # Initialize Q-table, augmented with the context axis
        n_actions = env.action_space.n
        shape = (self.n_context_bins, self.num_position_bins, self.num_velocity_bins, n_actions)
        if init_Q_random:
            self.Qdat = self.rng.uniform(low=-2, high=0, size=shape)
        else:
            self.Qdat = np.zeros(shape)

        # Per-context-bin epsilon (for epsilon-greedy only; ignored by other strategies)
        self.epsdat = [epsilon for _ in range(self.n_context_bins)]

        # Episodes trained in each bin, for diagnostics
        self.context_counts = np.zeros(self.n_context_bins, dtype=int)

        # Active context slice; the probe sets this before the first episode. Default to
        # the bin holding the nominal amplitude 1.0 so the agent is always usable.
        self.context = self.bin_context(1.0)

        # Build exploration strategy
        self.strategy = self._build_strategy(
            exploration=exploration,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon,
        )

    @staticmethod
    def _build_strategy(
        exploration: Union[str, "ExplorationStrategy", None],
        *,
        epsilon: float,
        epsilon_decay: float,
        min_epsilon: float,
    ) -> "ExplorationStrategy":
        # If a custom strategy instance was passed, accept it
        if isinstance(exploration, ExplorationStrategy):
            return exploration  # type: ignore[return-value]

        if exploration is None or exploration == "epsilon_greedy":
            return EpsilonGreedy(epsilon, epsilon_decay, min_epsilon)

        raise ValueError(f"Unknown exploration type: {exploration!r}")

    # -------- Context conditioning --------

    def bin_context(self, theta_hat: float) -> int:
        """
        Bin a scalar context estimate (an amplitude probe) onto the context axis.
        Estimates outside the edge range are clipped into the outermost bins, and
        estimates within ``edge_tolerance`` below an edge are snapped up onto it.
        """
        theta = float(np.clip(theta_hat, self.context_bin_edges[0], self.context_bin_edges[-1]))
        c = int(np.digitize(theta + self.edge_tolerance, self.context_bin_edges) - 1)
        return max(0, min(c, self.n_context_bins - 1))

    def set_context(self, theta_hat: float) -> int:
        """
        Select the active Q-table slice from a probe estimate. Returns the bin index.
        """
        self.context = self.bin_context(theta_hat)
        return self.context

    @property
    def Q(self) -> np.ndarray:
        """The active context's Q-table (a view into ``Qdat``)."""
        return self.Qdat[self.context]

    @property
    def epsilon(self) -> float:
        """The active context's epsilon (kept for logging parity with QLearningAgent)."""
        return self.epsdat[self.context]

    # -------- Policy & Learning --------

    def choose_action(self, env: gym.Env, state: Tuple[int, int], eps: float) -> int:
        """
        Choose an action using the configured exploration strategy, reading only the
        active context slice.
        """
        q_row = self.Qdat[self.context][state]
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
        """Standard Q-learning update, applied only to the active context slice."""
        Q = self.Qdat[self.context]
        best_next_action = int(np.argmax(Q[next_state]))
        td_target = reward + self.gamma * Q[next_state][best_next_action]
        td_error = td_target - Q[state][action]
        Q[state][action] += self.alpha * td_error

    def train_step(
        self,
        env: gym.Env,
        max_steps_per_episode: int = 200
    ) -> float:
        """
        Train the CMDP Q-learning agent for one episode on the active context slice.

        Returns:
            float: total episode reward
        """
        obs, _ = env.reset()
        state = discretize_state(obs, self.position_bins, self.velocity_bins)
        episode_reward = 0.0

        c = self.context
        eps = self.epsdat[c]

        for _ in range(max_steps_per_episode):
            action = self.choose_action(env, state, eps)
            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = discretize_state(next_obs, self.position_bins, self.velocity_bins)

            self.update_q_table(state, action, reward, next_state)

            state = next_state
            episode_reward += reward

            if done or truncated:
                break

        # Decay only the active bin's epsilon (per-table decay, as COIN-Q does), so no
        # context is unfairly stuck exploiting/exploring.
        if isinstance(self.strategy, EpsilonGreedy):
            self.epsdat[c] = max(self.min_epsilon, self.epsdat[c] * self.epsilon_decay)
        else:
            self.strategy.on_episode_end()

        self.context_counts[c] += 1

        env.close()
        return episode_reward

    def train(
        self,
        env: gym.Env,
        n_episodes: int = 5000,
        max_steps_per_episode: int = 200,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Train for ``n_episodes`` on a fixed environment and context (mainly for testing).
        """
        all_episode_rewards: List[float] = []
        for _ in range(n_episodes):
            all_episode_rewards.append(self.train_step(env, max_steps_per_episode))
        env.close()
        return self.Qdat, all_episode_rewards

    def evaluate(
        self,
        env: gym.Env,
        n_episodes: int = 10,
        max_steps_per_episode: int = 200
    ) -> List[float]:
        """
        Execute the greedy policy (argmax over the active slice) without learning.
        """
        rewards: List[float] = []
        Q = self.Qdat[self.context]
        for _ in range(n_episodes):
            obs, _ = env.reset()
            state = discretize_state(obs, self.position_bins, self.velocity_bins)
            episode_reward = 0.0

            for _ in range(max_steps_per_episode):
                action = int(np.argmax(Q[state]))
                next_obs, reward, done, truncated, _ = env.step(action)
                next_state = discretize_state(next_obs, self.position_bins, self.velocity_bins)

                episode_reward += reward
                state = next_state

                if done or truncated:
                    break

            rewards.append(episode_reward)

        env.close()
        return rewards


def run_single_rep_cmdp_q(rep_id, true_amplitudes, probe_period=100, return_agent=False):
    """
    Runs one repetition of training for CMDP-Q.
    Returns rewards at each time step.
    """
    from environments import CustomMountainCarEnv
    from cmdp_q import CMDPQLearningAgent
    from rl import probe_amplitude
    from tqdm.auto import tqdm
    import numpy as np

    SEED = rep_id

    # Create a fresh agent and environment
    env = CustomMountainCarEnv(amplitude=1.0, render_mode="none")
    agent = CMDPQLearningAgent(
        env=env,
        num_position_bins=30,
        num_velocity_bins=30,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.999,
        rng=SEED,
    )

    rewards_for_this_rep = []
    pbar = tqdm(true_amplitudes, desc=f"Rep {rep_id}")
    for i, amplitude in enumerate(pbar):
        # Obtain small experience from environment to update the context estimate
        # every probe_period episodes
        if i % probe_period == 0:
            env = CustomMountainCarEnv(amplitude=amplitude, render_mode="none")
            est_a = probe_amplitude(env)

            # Condition the Q-table on the binned parameter estimate
            agent.set_context(est_a)

        # Train the agent in the current context
        env = CustomMountainCarEnv(amplitude=amplitude, render_mode="none")
        training_reward = agent.train_step(env=env, max_steps_per_episode=200)
        rewards_for_this_rep.append(training_reward)

        pbar.set_postfix(amplitude=amplitude, reward=training_reward)

    if return_agent:
        return rewards_for_this_rep, agent
    return rewards_for_this_rep
