"""
rl.py

This module contains various Reinforcement Learning (RL) algorithms and helper functions,
intended for use with Gymnasium environments. It provides a template for integrating
and organizing different RL methods in one place.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from typing import Tuple, List, Optional, Union, Protocol, Dict, Any, runtime_checkable
import copy
import random


#----- Reproducibility -----

def seed_envs(envs, seed: int) -> None:
    """
    Seed a batch of freshly constructed environments from one base seed.

    Gymnasium seeds an env's episode stream at the FIRST ``reset(seed=...)``; later
    ``reset()`` calls without a seed continue that stream. The harness builds one fresh env
    per segment per rollout (``CustomCartPoleEnv.__init__`` recomputes derived constants,
    so live envs must not be mutated), so this is called once on each fresh batch. Env ``i``
    gets ``seed + i`` for both its reset stream and its ``action_space`` (which
    :meth:`AmortisedCOINPPOAgent.pretrain_encoder` samples from).

    Args:
        envs: One :class:`gymnasium.Env` or an iterable of them.
        seed (int): Base seed; env ``i`` receives ``seed + i``.
    """
    if isinstance(envs, gym.Env):
        envs = [envs]
    for i, env in enumerate(envs):
        env.reset(seed=int(seed) + i)
        env.action_space.seed(int(seed) + i)


def seed_everything(seed: int, envs=None) -> np.random.Generator:
    """
    Drive every stochastic source of one repetition from a single rep seed.

    The full recipe for a reproducible rep is::

        rng  = rl.seed_everything(REP_SEED)          # torch, numpy, python random
        coin = RealTimeCOIN(rng=REP_SEED)            # COIN's own generator
        ...
        for rollout in range(N):
            envs = [make_env(task) for _ in range(S)]         # fresh envs each rollout
            rl.seed_envs(envs, REP_SEED * 100_000 + rollout * 100)

    ``np.random.seed`` is not incidental: :meth:`SegmentReplayBuffer.push` draws its
    reservoir index from the global numpy stream, so the replay pool's contents are part of
    what this fixes. ``torch.manual_seed`` covers network init, PPO action sampling and the
    encoder's minibatch draws. COIN's generator is passed separately (``rng=seed``) because
    ``RealTimeCOIN`` owns its own stream; likewise
    :class:`curriculum.MarkovTaskCurriculum` takes ``rng=seed``.

    Args:
        seed (int): The rep seed.
        envs: Optional env or iterable of envs to seed immediately via :func:`seed_envs`.

    Returns:
        np.random.Generator: An independent generator seeded from ``seed``, for the
        caller's own draws (env seed offsets, task order, ...) so they do not perturb the
        global stream the agent uses.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if envs is not None:
        seed_envs(envs, seed)
    return np.random.default_rng(seed)


def discretize_state(
    observation: np.ndarray,
    position_bins: np.ndarray,
    velocity_bins: np.ndarray
) -> Tuple[int, int]:
    """
    Discretize the continuous state (position, velocity) into integer indices.

    Args:
        observation (np.ndarray): Continuous observation in the form [position, velocity].
        position_bins (np.ndarray): 1D array of boundaries for discretizing position.
        velocity_bins (np.ndarray): 1D array of boundaries for discretizing velocity.

    Returns:
        Tuple[int, int]: Indices representing the discretized state (pos_index, vel_index).
    """
    position, velocity = observation

    # Discretize position
    pos_index = np.digitize(position, position_bins) - 1
    pos_index = max(0, min(pos_index, len(position_bins) - 1))

    # Discretize velocity
    vel_index = np.digitize(velocity, velocity_bins) - 1
    vel_index = max(0, min(vel_index, len(velocity_bins) - 1))

    return (pos_index, vel_index)

#----- Fig. 2 probe helpers (shared by COIN-Q and all baselines) -----

def amplitude_estimator(s1, s2, a, F: float = 0.001, g: float = 0.0025) -> float:
    """
    Estimates the amplitude term of a standard mountain car environment assuming
    accurate observation and dynamics. We assume constant F and g.
    """
    v1 = s1[1]
    v2 = s2[1]
    x1 = s1[0]
    estimated_amplitude = (v1 - v2 + F * (a - 1)) / (g * np.cos(3 * x1))
    return estimated_amplitude


def probe_amplitude(env: gym.Env, action: int = 2) -> float:
    """
    One-step probe of a mountain car environment: resets, takes ``action`` once and
    inverts the known dynamics to estimate the path amplitude. Steps a second time if
    the start position sits on a cos(3x) zero, where the estimator is undefined.
    """
    state0, _ = env.reset()
    state1, _, _, _, _ = env.step(action)
    if np.isclose(np.cos(3 * state0[0]), 0.0):
        # Avoid using a zero point in the amplitude estimator
        state0 = state1
        state1, _, _, _, _ = env.step(action)
    return amplitude_estimator(state0, state1, action, F=env.force)


#----- COIN interface helpers -----

def coin_context_vector(
    vec: np.ndarray,
    k: int,
    width: Optional[int] = None,
    renormalise_novel: bool = True,
) -> np.ndarray:
    """
    Convert one fixed-width realtimecoin context vector into the agent convention.

    ``RealTimeCOIN`` queries return a fixed-width ``(max_contexts + 1,)`` vector in which
    indices ``0 .. k-1`` hold the known (globally aligned) contexts, index ``k`` holds the
    novel context, and every index above ``k`` is a padding zero. The agents in this module
    instead expect ``(width,)`` laid out as **known first, novel last**, with ``np.nan``
    marking a context slot that has not been instantiated yet (as opposed to an
    instantiated context with zero responsibility).

    Args:
        vec (np.ndarray): A fixed-width realtimecoin query vector for a single trial.
        k (int): The number of known contexts for that trial, i.e. ``context_alignment()["K"]``.
        width (Optional[int]): Length of the returned vector. Defaults to ``k + 1``
            (known contexts plus novel, no uninstantiated padding).
        renormalise_novel (bool): If True (default), overwrite the novel entry with
            ``1 - sum(known)`` — the notebooks' novel-column fix, which makes each row sum
            to one after the padding slots are treated as zero. Pass False for raw
            responsibility traces, which are deliberately left unnormalised.

    Returns:
        np.ndarray: A ``(width,)`` float array ``[known..., nan..., novel]``.
    """
    k = int(k)
    width = (k + 1) if width is None else int(width)
    out = np.full(width, np.nan)
    out[:k] = vec[:k]          # known (globally aligned) contexts
    out[-1] = vec[k]           # novel
    if renormalise_novel:
        out[-1] = 1.0 - np.sum(np.nan_to_num(out[:-1]))
    return out


def coin_context_trace(
    vecs: np.ndarray,
    ks: np.ndarray,
    width: Optional[int] = None,
    renormalise_novel: bool = True,
) -> np.ndarray:
    """
    Convert a whole trial-by-trial trace of realtimecoin vectors into the agent convention.

    Applies :func:`coin_context_vector` to each trial, so every row is laid out as
    **known contexts first, novel last**, with ``np.nan`` marking context slots that had
    not been instantiated on that trial. All rows share one common width so the result is
    a rectangular array.

    Args:
        vecs (np.ndarray): Fixed-width realtimecoin query vectors, one row per trial.
        ks (np.ndarray): Per-trial number of known contexts, i.e. ``context_alignment()["K"]``.
        width (Optional[int]): Common row length. Defaults to ``max(ks) + 1``.
        renormalise_novel (bool): If True (default), set each row's novel entry to
            ``1 - sum(known)`` — the notebooks' novel-column fix. Pass False for raw
            responsibility traces, which are deliberately left unnormalised.

    Returns:
        np.ndarray: A ``(len(ks), width)`` float array.
    """
    ks = np.asarray(ks, dtype=int)
    width = (int(ks.max()) + 1) if width is None else int(width)
    return np.stack([coin_context_vector(vecs[t], ks[t], width, renormalise_novel)
                     for t in range(len(ks))])


def coin_predicted_pi(coin, cue=None) -> Tuple[np.ndarray, int]:
    """
    One-step-ahead predicted context probabilities of a ``RealTimeCOIN`` model.

    ``predicted_context_probabilities_vector()`` is NOT the query you want here: it reports
    ``D.predicted_probabilities``, which ``observe_y`` writes as its first step, so it is one
    trial STALE and ignores the cue just staged by ``observe_q``. This helper instead
    propagates the current particle state one trial forward, relabels it into the aligned
    global frame and averages over the modal particles -- exactly what COIN's own aligned
    query does, but for the trial about to happen.

    **Call timing is a contract:** call immediately after ``observe_q``, before anything else
    touches the model. ``realtimecoin`` is imported here, not at module scope, so ``rl.py``
    stays importable without it.

    Args:
        coin: A ``RealTimeCOIN`` model.
        cue: Raw cue value for the upcoming trial, or None to marginalise the cue out.

    Returns:
        Tuple[np.ndarray, int]: A ``(max_contexts + 1,)`` probability vector in the global
        frame (known contexts ``0 .. K-1``, novel at ``K``, padding zeros above) and ``K``.
    """
    from realtimecoin.alignment import global_context_weights
    from realtimecoin.context import next_trial_context_weights
    from realtimecoin.numerics import renormalize_global_weights
    from realtimecoin.state import peek_cue_label

    q = peek_cue_label(coin, cue)
    if q is not None and q >= len(coin.cue_values):
        # An unregistered cue: next_trial_context_weights would CLAMP the label to the last
        # existing column, whereas observe_y registers the cue and grows the matrix first.
        # Marginalising is the only honest option before the fact.
        q = None

    align = coin.context_alignment()
    w = next_trial_context_weights(coin, q)              # (P, C), local labels
    g = global_context_weights(coin, w, align)           # (P_modal, C), global labels
    return renormalize_global_weights(np.mean(g, axis=0)), int(align["K"])


def _as_context_probs_fn(context_probs):
    """Accept either a callable episode -> probs, or a constant (N,) array."""
    if callable(context_probs):
        return context_probs
    snapshot = np.array(context_probs, dtype=float)   # eager copy, float64
    return lambda _ep: snapshot

#----- Exploration Strategies -----

@runtime_checkable
class ExplorationStrategy(Protocol):
    """Interface for action selection during exploration."""
    name: str

    def select_action(
        self,
        q_values: np.ndarray,
        action_space: gym.Space,
        *,
        epsilon_override: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        ...

    def on_episode_end(self) -> None:
        """Hook to update internal schedules (e.g., decay)."""
        ...


class EpsilonGreedy(ExplorationStrategy):
    """Classic epsilon-greedy with decay."""
    def __init__(self, epsilon: float, decay: float, min_epsilon: float):
        self.name = "epsilon_greedy"
        self.epsilon = float(epsilon)
        self.decay = float(decay)
        self.min_epsilon = float(min_epsilon)

    def select_action(
        self,
        q_values: np.ndarray,
        action_space: gym.Space,
        *,
        epsilon_override: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        rng = rng or np.random.default_rng()
        eps = self.epsilon if epsilon_override is None else float(epsilon_override)
        if rng.random() < eps:
            return action_space.sample()
        return int(np.argmax(q_values))

    def on_episode_end(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)


class SoftmaxExploration(ExplorationStrategy):
    """Boltzmann/Softmax over Q with temperature decay."""
    def __init__(self, temperature: float = 1.0, decay: float = 0.999, min_temperature: float = 0.05):
        self.name = "softmax"
        self.temperature = float(temperature)
        self.decay = float(decay)
        self.min_temperature = float(min_temperature)

    @staticmethod
    def _softmax_logits(q_values: np.ndarray, temperature: float) -> np.ndarray:
        # numerically-stable softmax
        z = (q_values - np.max(q_values)) / max(temperature, 1e-8)
        e = np.exp(z)
        p = e / np.sum(e)
        return p

    def select_action(
        self,
        q_values: np.ndarray,
        action_space: gym.Space,
        *,
        epsilon_override: Optional[float] = None,   # ignored for softmax
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        rng = rng or np.random.default_rng()
        probs = self._softmax_logits(q_values, self.temperature)
        return int(rng.choice(len(q_values), p=probs))

    def on_episode_end(self) -> None:
        self.temperature = max(self.min_temperature, self.temperature * self.decay)


class RandomExploration(ExplorationStrategy):
    """Pure random actions (mainly for debugging)."""
    def __init__(self):
        self.name = "random"

    def select_action(
        self,
        q_values: np.ndarray,
        action_space: gym.Space,
        *,
        epsilon_override: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        return action_space.sample()

    def on_episode_end(self) -> None:
        pass


# ---- Q-learning ----

class QLearningAgent:
    """
    A Q-learning agent that discretizes the state space and updates a tabular Q-table.
    """

    def __init__(
        self,
        env: gym.Env,
        num_position_bins: int = 30,
        num_velocity_bins: int = 30,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.999,
        min_epsilon: float = 0.01,
        init_Q_random: bool = True,
        exploration: Union[str, ExplorationStrategy, None] = "epsilon_greedy",
        softmax_temperature: float = 1.0,
        softmax_decay: float = 0.999,
        softmax_min_temperature: float = 0.05,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize the Q-learning agent with hyperparameters and bin settings.

        Args:
            env (gym.Env): An initialized Gymnasium environment.
            num_position_bins (int, optional): Number of bins for discretizing position.
            num_velocity_bins (int, optional): Number of bins for discretizing velocity.
            alpha (float, optional): Learning rate.
            gamma (float, optional): Discount factor.
            epsilon (float, optional): Initial epsilon for ε-greedy strategy.
            epsilon_decay (float, optional): Epsilon decay factor after each episode.
            min_epsilon (float, optional): Minimum value of epsilon.
            init_Q_random (bool, optional): Random-initialize Q-table if True, else zeros.
            exploration (str|ExplorationStrategy|None): Which exploration strategy to use.
                - "epsilon_greedy" (default, backward compatible)
                - "softmax"
                - "random"
                - Or pass a custom ExplorationStrategy instance.
            softmax_temperature/decay/min_temperature: params for softmax when selected.
            rng: Optional numpy Generator for reproducibility.
        """
        self.num_position_bins = num_position_bins
        self.num_velocity_bins = num_velocity_bins
        self.alpha = alpha
        self.gamma = gamma

        # Keep epsilon-related fields for backward compatibility & logging
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.rng = rng or np.random.default_rng()

        # Extract state boundaries (assuming a 2D state: [position, velocity])
        # Note: Adjust to your env if needed (e.g., env.observation_space.low/high)
        self.position_min, self.position_max = env.low[0], env.high[0]
        self.velocity_min, self.velocity_max = env.low[1], env.high[1]

        # Create bins
        self.position_bins = np.linspace(self.position_min, self.position_max, self.num_position_bins)
        self.velocity_bins = np.linspace(self.velocity_min, self.velocity_max, self.num_velocity_bins)

        # Initialize Q-table
        n_actions = env.action_space.n
        if init_Q_random:
            self.Q = np.random.uniform(
                low=-2, high=0, size=(self.num_position_bins, self.num_velocity_bins, n_actions)
            )
        else:
            self.Q = np.zeros((self.num_position_bins, self.num_velocity_bins, n_actions))

        # Build exploration strategy (default is epsilon-greedy for compatibility)
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

    # -------- Policy & Learning --------

    def choose_action(self, env: gym.Env, state: Tuple[int, int], eps: float) -> int:
        """
        Choose an action using the configured exploration strategy.
        - If using epsilon-greedy, we respect this 'eps' value at call-time.
          For other strategies, 'eps' is safely ignored.
        """
        q_row = self.Q[state]
        # For epsilon-greedy, pass epsilon_override=eps, else ignored by strategy
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
        best_next_action = int(np.argmax(self.Q[next_state]))
        td_target = reward + self.gamma * self.Q[next_state][best_next_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    def train_step(
        self,
        env: gym.Env,
        max_steps_per_episode: int = 200
    ) -> float:
        """
        Train the Q-learning agent for one episode using the configured exploration strategy.

        Returns:
            float: total episode reward
        """
        obs, _ = env.reset()
        state = discretize_state(obs, self.position_bins, self.velocity_bins)
        episode_reward = 0.0

        for _ in range(max_steps_per_episode):
            action = self.choose_action(env, state, self.epsilon)  # 'self.epsilon' maintained for compatibility
            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = discretize_state(next_obs, self.position_bins, self.velocity_bins)

            self.update_q_table(state, action, reward, next_state)

            state = next_state
            episode_reward += reward

            if done or truncated:
                break

        # Keep legacy epsilon fields in sync when using epsilon-greedy,
        # so your existing logs/prints remain meaningful.
        if isinstance(self.strategy, EpsilonGreedy):
            self.strategy.on_episode_end()
            self.epsilon = self.strategy.epsilon
        else:
            self.strategy.on_episode_end()

        env.close()
        return episode_reward

    def train(
        self,
        env: gym.Env,
        n_episodes: int = 5000,
        max_steps_per_episode: int = 200,
        verbose: bool = False,
        print_freq: int = 200
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Train the Q-learning agent (default epsilon-greedy; configurable).
        """
        all_episode_rewards: List[float] = []

        for episode in range(n_episodes):
            episode_reward = self.train_step(env, max_steps_per_episode)
            all_episode_rewards.append(episode_reward)

            if verbose and (episode + 1) % print_freq == 0:
                avg_reward = float(np.mean(all_episode_rewards[-print_freq:]))
                # Backward-compatible progress line
                if isinstance(self.strategy, EpsilonGreedy):
                    expline = f"Epsilon: {self.epsilon:.3f}"
                elif isinstance(self.strategy, SoftmaxExploration):
                    expline = f"Temp: {self.strategy.temperature:.3f}"
                else:
                    expline = f"Exploration: {self.strategy.name}"
                print(
                    f"Episode: {episode + 1}, "
                    f"Avg Reward (last {print_freq}): {avg_reward:.2f}, "
                    f"{expline}"
                )

        env.close()
        return self.Q, all_episode_rewards

    def evaluate(
        self,
        env: gym.Env,
        n_episodes: int = 10,
        max_steps_per_episode: int = 200
    ) -> List[float]:
        """
        Execute the greedy policy (argmax over Q) to evaluate performance.
        This method does not update the Q-table.
        """
        rewards: List[float] = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            state = discretize_state(obs, self.position_bins, self.velocity_bins)
            episode_reward = 0.0

            for _ in range(max_steps_per_episode):
                # Force greedy evaluation independent of exploration strategy
                action = int(np.argmax(self.Q[state]))
                next_obs, reward, done, truncated, _ = env.step(action)
                next_state = discretize_state(next_obs, self.position_bins, self.velocity_bins)

                episode_reward += reward
                state = next_state

                if done or truncated:
                    break

            rewards.append(episode_reward)

        env.close()
        return rewards

class COINQLearningAgent:
    """
    A Contextual Q-learning agent that uses Contextual Inference to update a database of tabular Q-tables.
    """

    def __init__(
        self,
        env: gym.Env,
        max_contexts: int,
        num_position_bins: int = 30,
        num_velocity_bins: int = 30,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.999,
        min_epsilon: float = 0.0,
        init_Q_random: bool = True,
        instantiate_from_average: bool = False,
        avoid_novel: bool = False,
        exploration: Union[str, "ExplorationStrategy", None] = "epsilon_greedy",
        softmax_temperature: float = 1.0,
        softmax_decay: float = 0.999,
        softmax_min_temperature: float = 0.05,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize the COIN Q-learning agent with hyperparameters and bin settings.

        Args:
            env (gym.Env): An initialized Gymnasium environment.
            max_contexts (int): Number of known (non-novel) contexts.
            num_position_bins (int, optional): Number of bins for discretizing position.
            num_velocity_bins (int, optional): Number of bins for discretizing velocity.
            alpha (float, optional): Learning rate.
            gamma (float, optional): Discount factor.
            epsilon (float, optional): Initial epsilon for all ε-greedy strategies.
            epsilon_decay (float, optional): Epsilon decay factor after each episode.
            min_epsilon (float, optional): Minimum value of epsilon.
            init_Q_random (bool, optional): When True, initialise Q-table randomly, otherwise to zeros.
            instantiate_from_average (bool, optional): Initialise new context from weighted average Q.
            avoid_novel (bool, optional): Ignore novel context for action selection if True.
            exploration (str|ExplorationStrategy|None): "epsilon_greedy" (default), "softmax", "random", or a custom strategy instance.
            softmax_*: Parameters for softmax exploration.
            rng: Optional numpy Generator for reproducibility.
        """
        self.num_position_bins = num_position_bins
        self.num_velocity_bins = num_velocity_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.instantiate_from_average = instantiate_from_average
        self.avoid_novel = avoid_novel
        self.rng = rng or np.random.default_rng()

        # Extract state boundaries (assuming a 2D state: [position, velocity])
        self.position_min, self.position_max = env.low[0], env.high[0]
        self.velocity_min, self.velocity_max = env.low[1], env.high[1]

        # Create bins
        self.position_bins = np.linspace(self.position_min, self.position_max, self.num_position_bins)
        self.velocity_bins = np.linspace(self.velocity_min, self.velocity_max, self.num_velocity_bins)

        # Initialize Q-table database as one stacked array (append one extra for the novel context)
        n_actions = env.action_space.n
        if init_Q_random:
            self.Qdat = np.random.uniform(
                low=-2, high=0,
                size=(max_contexts + 1, self.num_position_bins, self.num_velocity_bins, n_actions)
            )
        else:
            self.Qdat = np.zeros((max_contexts + 1, self.num_position_bins, self.num_velocity_bins, n_actions))

        # Track which contexts have been initialised - only novel initialised initially
        self.context_init = np.zeros((max_contexts + 1,))
        self.context_init[-1] = 1  # novel

        # Per-context epsilon (for ε-greedy only; ignored by other strategies)
        self.epsdat = [epsilon for _ in range(max_contexts)]  # excludes novel; novel uses max_epsilon implicitly if needed

        # Build exploration strategy (global)
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
        # Accept a custom strategy instance
        try:
            from typing import runtime_checkable, Protocol  # noqa: F401
            from typing import Any
            # If ExplorationStrategy is in scope and runtime-checkable:
            if isinstance(exploration, ExplorationStrategy):  # type: ignore[name-defined]
                return exploration  # type: ignore[return-value]
        except Exception:
            # If typing Protocol check isn't available, fall through to string handling
            pass

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

    # -------- Policy & Learning --------

    def choose_action(self, env: gym.Env, q_row: np.ndarray, eps: float) -> int:
        """
        Choose an action using the configured exploration strategy.
        'q_row' holds the averaged Q-values at the current state.
        For ε-greedy, 'eps' overrides the strategy's epsilon to support per-context averaging.
        For other strategies, 'eps' is ignored.
        """
        epsilon_override = eps if isinstance(self.strategy, EpsilonGreedy) else None
        return self.strategy.select_action(
            q_row,
            env.action_space,
            epsilon_override=epsilon_override,
            rng=self.rng,
        )

    def update_q_table(
        self,
        td_error: float,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
        p_context: np.ndarray,
        Z: float = None,
        idx: np.ndarray = None,
    ) -> None:
        """
        Update the Q-tables using the COIN Q-learning update rule.
        'td_error' is computed by the caller from the averaged Q-values; 'Z' and
        'idx' may be precomputed once per episode since p_context is fixed.
        """
        if Z is None:
            Z = np.nansum(p_context ** 2)  # normalizing constant for learning rates
        if idx is None:
            # Zero-probability contexts receive a zero update, so only touch the rest
            # (NaN entries compare False here, matching the previous per-context skip)
            idx = np.flatnonzero((self.context_init > 0) & (p_context > 0))
        self.Qdat[idx, state[0], state[1], action] += p_context[idx] * self.alpha * td_error / max(Z, 1e-8)

    def _pad_p_context(self, p_context: np.ndarray) -> np.ndarray:
        """Pad a shorter (C+1,) vector to (max_contexts+1,), keeping novel last."""
        if p_context.shape[0] < len(self.context_init):
            pad = np.full(len(self.context_init) - p_context.shape[0], np.nan)
            p_context = np.concatenate([p_context[:-1], pad, p_context[-1:]])
        return p_context

    def _averaged_q(
        self,
        state: Tuple[int, int],
        probs: np.ndarray,
        idx: np.ndarray,
        average_bias: np.ndarray = None,
    ) -> np.ndarray:
        """Averaged Q-values at a single state, weighted over the contexts in 'idx'."""
        q_row = probs[idx] @ self.Qdat[idx, state[0], state[1], :]
        if average_bias is not None:
            q_row = q_row + average_bias[state]
        return q_row

    def instantiate_context_Q(
        self,
        new_context: int,
        probs: np.ndarray = None,
    ):
        """When a novel context is instantiated, copy current Q novel table (or average) to that new context value."""
        if self.instantiate_from_average and probs is not None:
            Qavg = np.zeros_like(self.Qdat[0])
            for i in range(len(self.Qdat)):
                if self.context_init[i] and not np.isnan(probs[i]):
                    Qavg += probs[i] * self.Qdat[i]
            self.Qdat[new_context] = Qavg
        else:
            # Copy the last Q-table (novel) to the new context
            self.Qdat[new_context] = self.Qdat[-1].copy()
        # Reset epsilon high for the new context (ε-greedy only)
        if new_context < len(self.epsdat):
            self.epsdat[new_context] = self.max_epsilon

    def train_step(
        self,
        env: gym.Env,
        p_context: np.ndarray,
        max_steps_per_episode: int = 200,
        average_bias: np.ndarray = None, # bias to be added for average
    ) -> float:
        """
        Train the COIN Q-learning agent for one episode using the configured exploration.
        """
        obs, _ = env.reset()
        state = discretize_state(obs, self.position_bins, self.velocity_bins)
        episode_reward = 0.0

        # p_context may not be the same size as max_contexts+1
        # rearrange it to extend it, moving novel to the end
        p_context = self._pad_p_context(p_context)

        # Instantiate any new contexts that became active
        for i, init in enumerate(self.context_init):
            if init == 0 and not np.isnan(p_context[i]):
                instant_probs = p_context.copy()
                # Remove current context from averaging weights
                instant_probs[i] = 0.0
                if np.nansum(instant_probs[:-1]) > 0:
                    instant_probs[:-1] = instant_probs[:-1] / np.nansum(instant_probs[:-1])
                    instant_probs[-1] = 0.0
                else:
                    instant_probs[-1] = 1.0  # ensure novel is 1.0
                self.instantiate_context_Q(i, probs=instant_probs)
                self.context_init[i] = 1

        # If "avoid_novel" is True, attempt to ignore novel context for action selection
        action_probs = p_context.copy()
        if self.avoid_novel and np.nansum(action_probs[:-1]) > 0:
            action_probs[:-1] = action_probs[:-1] / np.nansum(action_probs[:-1])
            action_probs[-1] = 0.0

        # Contexts that contribute to the average / receive updates, and the
        # learning-rate normaliser; all fixed within the episode
        avg_idx = np.flatnonzero((self.context_init > 0) & (action_probs > 0))
        upd_idx = np.flatnonzero((self.context_init > 0) & (p_context > 0))
        Z = np.nansum(p_context ** 2)

        # Compute averaged epsilon (for ε-greedy only); fixed within the episode
        epsavg = 0.0
        for i in range(len(self.Qdat)):
            if self.context_init[i] and not np.isnan(action_probs[i]):
                # context epsilon (ε-greedy only); for novel (index == last), use max_epsilon
                ctx_eps = self.epsdat[i] if i < len(self.epsdat) else self.max_epsilon
                epsavg += action_probs[i] * ctx_eps

        for _ in range(max_steps_per_episode):
            # Averaged Q-values at the current state (the full averaged table is never needed)
            q_row = self._averaged_q(state, action_probs, avg_idx, average_bias)

            # Choose action via the pluggable strategy
            action = self.choose_action(env, q_row, epsavg)

            # Step and update
            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = discretize_state(next_obs, self.position_bins, self.velocity_bins)
            next_q_row = self._averaged_q(next_state, action_probs, avg_idx, average_bias)
            td_error = reward + self.gamma * np.max(next_q_row) - q_row[action]
            self.update_q_table(td_error, state, action, reward, next_state, p_context, Z=Z, idx=upd_idx)

            state = next_state
            episode_reward += reward

            if done or truncated:
                break

        # Decay schedules:
        # - For ε-greedy we keep your per-context decay rule.
        # - For other strategies, we call their episode hook (e.g., softmax temperature decay).
        if isinstance(self.strategy, EpsilonGreedy):
            for i in range(len(self.epsdat)):
                if self.context_init[i] and not np.isnan(p_context[i]):
                    self.epsdat[i] = max(
                        self.min_epsilon,
                        self.epsdat[i] * (self.epsilon_decay ** (p_context[i]))
                    )
        else:
            self.strategy.on_episode_end()

        env.close()
        return episode_reward

    def evaluate(
        self,
        env: gym.Env,
        p_context: np.ndarray,
        n_episodes: int = 10,
        max_steps_per_episode: int = 500,
        ignore_novel: bool = False,
        average_bias: np.ndarray = None, # bias to be added for average
    ) -> List[float]:
        """
        Execute the learned policy (greedy w.r.t. the averaged Q-table) to evaluate performance.
        """
        rewards: List[float] = []

        p_context = self._pad_p_context(p_context)

        if ignore_novel and np.nansum(p_context[:-1]) > 0:
            p_context = p_context.copy()
            p_context[:-1] = p_context[:-1] / (np.nansum(p_context[:-1]) + 1e-4)
            p_context[-1] = 0.0

        # Contexts that actually contribute to the average
        idx = np.flatnonzero((self.context_init > 0) & (p_context > 0))

        for _ in range(n_episodes):
            obs, _ = env.reset()
            state = discretize_state(obs, self.position_bins, self.velocity_bins)
            episode_reward = 0.0

            for _ in range(max_steps_per_episode):
                # Greedy evaluation independent of exploration strategy
                q_row = self._averaged_q(state, p_context, idx, average_bias)
                action = int(np.argmax(q_row))
                next_obs, reward, done, truncated, _ = env.step(action)
                next_state = discretize_state(next_obs, self.position_bins, self.velocity_bins)

                episode_reward += reward
                state = next_state

                if done or truncated:
                    break

            rewards.append(episode_reward)

        env.close()
        return rewards
    
class EmbodiedCOINQLearningAgent(COINQLearningAgent):
    """
    An Embodied Contextual Q-Learning agent that splits Q into a body and a contextual head.
    The body has no contextual information, and aims to learn the average Q given the stationary contextual distribution.
    The head learns the variations about this average in the same way as COINQLearningAgent.
    """
    def __init__(
        self,
        env: gym.Env,
        max_contexts: int,
        num_position_bins: int = 30,
        num_velocity_bins: int = 30,
        alpha: float = 0.1,
        alpha_body: float = 0.01,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.999,
        min_epsilon: float = 0.0,
        init_Q_random: bool = True,
        instantiate_from_average: bool = False,
        avoid_novel: bool = False,
        exploration: Union[str, "ExplorationStrategy", None] = "epsilon_greedy",
        softmax_temperature: float = 1.0,
        softmax_decay: float = 0.999,
        softmax_min_temperature: float = 0.05,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize the Embodied COIN Q-learning agent with hyperparameters and bin settings.

        Args:
            env (gym.Env): An initialized Gymnasium environment.
            max_contexts (int): Number of known (non-novel) contexts.
            num_position_bins (int, optional): Number of bins for discretizing position.
            num_velocity_bins (int, optional): Number of bins for discretizing velocity.
            alpha (float, optional): Learning rate.
            alpha_body (float, optional): Body learning rate.
            gamma (float, optional): Discount factor.
            epsilon (float, optional): Initial epsilon for all ε-greedy strategies.
            epsilon_decay (float, optional): Epsilon decay factor after each episode.
            min_epsilon (float, optional): Minimum value of epsilon.
            init_Q_random (bool, optional): When True, initialise Q-table randomly, otherwise to zeros.
            instantiate_from_average (bool, optional): Initialise new context from weighted average Q.
            avoid_novel (bool, optional): Ignore novel context for action selection if True.
            exploration (str|ExplorationStrategy|None): "epsilon_greedy" (default), "softmax", "random", or a custom strategy instance.
            softmax_*: Parameters for softmax exploration.
            rng: Optional numpy Generator for reproducibility.
        """
        super().__init__(
            env,
            max_contexts,
            num_position_bins,
            num_velocity_bins,
            alpha,
            gamma,
            epsilon,
            epsilon_decay,
            min_epsilon,
            init_Q_random,
            instantiate_from_average,
            avoid_novel,
            exploration,
            softmax_temperature,
            softmax_decay,
            softmax_min_temperature,
            rng
        )
        self.alpha_body = alpha_body

        # Initialize body Q-table
        n_actions = env.action_space.n
        if init_Q_random:
            self.Qbody = np.random.uniform(
                low=-2, high=0, size=(self.num_position_bins, self.num_velocity_bins, n_actions)
            )
        else:
            self.Qbody = np.zeros((self.num_position_bins, self.num_velocity_bins, n_actions))

    def update_q_table(
        self,
        td_error: float,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
        p_context: np.ndarray,
        Z: float = None,
        idx: np.ndarray = None,
    ) -> None:
        """
        Update the Q-tables using the Embodied COIN Q-learning update rules.
        """
        best_next_action_body = np.argmax(self.Qbody[next_state])

        # Update body table
        td_target_body = reward + self.gamma * self.Qbody[next_state][best_next_action_body]
        td_error_body = td_target_body - self.Qbody[state][action]
        self.Qbody[state][action] += self.alpha_body * td_error_body

        # Head updates
        super().update_q_table(td_error, state, action, reward, next_state, p_context, Z=Z, idx=idx)

        # Body inhibition on head values (applies to every instantiated context,
        # not only those with nonzero probability)
        idx = np.flatnonzero((self.context_init > 0) & ~np.isnan(p_context))
        self.Qdat[idx, state[0], state[1], action] -= self.alpha_body * td_error_body

    def train_step(
        self,
        env: gym.Env,
        p_context: np.ndarray,
        max_steps_per_episode: int = 200
    ) -> float:
        return super().train_step(env, p_context, max_steps_per_episode, average_bias=self.Qbody)
    
    def evaluate(
        self,
        env: gym.Env,
        p_context: np.ndarray,
        n_episodes: int = 10,
        max_steps_per_episode: int = 500,
        ignore_novel: bool = False
    ) -> List[float]:
        return super().evaluate(env, p_context, n_episodes, max_steps_per_episode, ignore_novel, average_bias=self.Qbody)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PPOAgent:
    """
    Vanilla PPO (clip) agent.
    """
    def __init__(
        self,
        env: gym.Env,           # PPO Agent form depends on the environment it is being applied to
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        lr: float = 3e-4,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        device: str = "cpu",
    ):
        self.gamma, self.lam = gamma, gae_lambda
        self.clip_eps, self.ent_coef, self.vf_coef = clip_eps, ent_coef, vf_coef
        self.device = device

        # ----- Infer observation / action spaces from the env -----
        obs_space = env.observation_space
        act_space = env.action_space

        # This version assumes continuous (Box) observations.
        # If you have Dict / Discrete obs etc., encoder is needed.
        assert isinstance(obs_space, gym.spaces.Box), \
            "This PPOAgent currently supports only Box observation spaces."

        self.obs_dim = int(np.prod(obs_space.shape))

        # Determine whether the action space is discrete or continuous
        if isinstance(act_space, gym.spaces.Discrete):
            self.action_continuous = False
            self.act_dim = act_space.n
            self.act_low = None
            self.act_high = None
        elif isinstance(act_space, gym.spaces.Box):
            self.action_continuous = True
            assert len(act_space.shape) == 1, \
                "Only 1D Box action spaces are supported (shape = (act_dim,))."
            self.act_dim = act_space.shape[0]
            # Store bounds as tensors
            self.act_low = torch.as_tensor(
                act_space.low, device=self.device, dtype=torch.float32
            )
            self.act_high = torch.as_tensor(
                act_space.high, device=self.device, dtype=torch.float32
            )
        else:
            raise NotImplementedError(
                f"Action space type {type(act_space)} not supported. "
                "Only Discrete and 1D Box are supported."
            )

        # ----- Networks -----
        self.policy = _MLP(self.obs_dim, self.act_dim).to(device)
        self.value_net = _MLP(self.obs_dim, 1).to(device)

        # For continuous actions, keep a fixed log_std
        if self.action_continuous:
            self.log_std = torch.ones(self.act_dim, device=device) * np.log(0.5)
        else:
            self.log_std = None

        # Optimizer
        params = list(self.policy.parameters()) + list(self.value_net.parameters())
        self.optim = optim.Adam(params, lr=lr)

        # --- evaluation shadow (CPU) ---
        self._eval_policy_cpu = None
        self._eval_value_cpu = None
        self._weights_version = 0
        self._eval_sync_version = -1

        # keep numpy bounds for fast clipping on CPU
        if self.action_continuous:
            self.act_low_np = act_space.low.astype(np.float32)
            self.act_high_np = act_space.high.astype(np.float32)
        else:
            self.act_low_np = None
            self.act_high_np = None

    # --------------- utilities -----------------
    def _flatten_obs(self, obs) -> torch.Tensor:
        """
        Convert an observation (np array or tensor) to a flat float32 tensor on self.device.
        """
        if isinstance(obs, np.ndarray):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs_t = obs.to(self.device, dtype=torch.float32)
        return obs_t.view(-1)  # flatten to [obs_dim]

    def _act(self, obs: torch.Tensor):
        """
        Given a single *flat* observation tensor of shape [obs_dim],
        sample an action and return (action_np, logp, entropy, raw_output).
        """
        if self.action_continuous:
            # Continuous: Gaussian policy
            mu = self.policy(obs)                     # [act_dim]
            std = self.log_std.exp().expand_as(mu)    # [act_dim]
            dist = torch.distributions.Normal(mu, std)

            raw_action = dist.sample()                # [act_dim]
            # (Simple version) clamp to bounds for env step
            action = raw_action
            if self.act_low is not None and self.act_high is not None:
                action = torch.max(torch.min(action, self.act_high), self.act_low)

            logp = dist.log_prob(action).sum(-1)  # scalar
            entropy = dist.entropy().sum(-1)          # scalar

            action_np = action.detach().cpu().numpy().astype(np.float32)
            return action_np, logp, entropy, mu

        else:
            # Discrete: Categorical policy
            logits = self.policy(obs)                 # [act_dim]
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()                    # scalar
            logp = dist.log_prob(action)
            entropy = dist.entropy()

            action_np = int(action.detach().cpu().item())
            return action_np, logp, entropy, logits

    def _compute_advantages(self, rewards, values, dones, last_value: float):
        # Make 'values' a simple Python list of floats
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy().tolist()
        else:
            values = list(values)

        # Append bootstrap value as V_{T}
        values = values + [last_value]

        adv = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1]
            delta = rewards[t] + self.gamma * (1 - dones[t]) * next_value - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            adv.insert(0, gae)

        returns = [a + v for a, v in zip(adv, values[:-1])]
        adv = torch.tensor(adv, device=self.device, dtype=torch.float32)
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)
        return adv, returns


    # --------------- main public API -----------------
    def train_step(self, env, rollout_steps: int = 2048, mini_epochs: int = 10, mb_size: int = 64):
        # Reset env and flatten observation
        obs = env.reset()[0]
        obs_t = self._flatten_obs(obs)

        ep_returns = []     # collect episodic returns for logging
        ep_len = ep_ret = 0

        # Storage
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf, ent_buf = [], [], [], [], [], [], []

        # ---------- rollout ----------
        for _ in range(rollout_steps):
            with torch.no_grad():
                value = self.value_net(obs_t).squeeze().item()
                action, logp, ent, _ = self._act(obs_t)

            next_obs, reward, done, trunc, _ = env.step(action)

            obs_buf.append(obs_t.cpu())
            act_buf.append(action)
            logp_buf.append(logp.cpu())
            rew_buf.append(reward)
            val_buf.append(value)
            done_buf.append(done or trunc)
            ent_buf.append(ent.cpu())

            ep_ret += reward
            ep_len += 1

            if done or trunc:
                next_obs, _ = env.reset()
                ep_returns.append(ep_ret)
                ep_len = ep_ret = 0

            obs_t = self._flatten_obs(next_obs)

        # ---------- advantages ----------
        with torch.no_grad():
            last_val = self.value_net(obs_t).squeeze().item()
        adv, ret = self._compute_advantages(rew_buf, val_buf, done_buf, last_val)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # normalise

        # ---------- prepare tensors ----------
        dataset_size = rollout_steps
        idxs = torch.randperm(dataset_size)

        obs_tensor = torch.stack(obs_buf).to(self.device)

        if self.action_continuous:
            act_tensor = torch.as_tensor(np.array(act_buf), dtype=torch.float32, device=self.device)
        else:
            act_tensor = torch.as_tensor(act_buf, dtype=torch.long, device=self.device)

        old_logp_tensor = torch.stack(logp_buf).to(self.device)

        # ---------- optimisation ----------
        for _ in range(mini_epochs):
            for start in range(0, dataset_size, mb_size):
                end = start + mb_size
                mb_idx = idxs[start:end]

                # Slice minibatch
                batch_obs = obs_tensor[mb_idx]
                batch_act = act_tensor[mb_idx]
                batch_adv = adv[mb_idx]
                batch_ret = ret[mb_idx]
                batch_old_logp = old_logp_tensor[mb_idx]

                # New logprobs & value
                if self.action_continuous:
                    mu = self.policy(batch_obs)                          # [B, act_dim]
                    std = self.log_std.exp().expand_as(mu)               # [B, act_dim]
                    dist = torch.distributions.Normal(mu, std)

                    new_logp = dist.log_prob(batch_act).sum(-1)          # [B]
                    entropy = dist.entropy().sum(-1).mean()
                else:
                    logits = self.policy(batch_obs)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logp = dist.log_prob(batch_act)                  # [B]
                    entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - batch_old_logp)

                # Clipped surrogate
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                value_pred = self.value_net(batch_obs).squeeze(-1)
                critic_loss = (batch_ret - value_pred).pow(2).mean()

                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

        mean_ep_return = float(np.mean(ep_returns)) if ep_returns else 0.0

        self._weights_version += 1

        return {
            "mean_episode_return": mean_ep_return,
            "mean_reward_per_step": float(np.mean(rew_buf)),
            "value_loss": critic_loss.item(),
            "policy_loss": actor_loss.item(),
        }
    
    def _get_eval_nets_cpu(self):
        if self._eval_policy_cpu is None:
            self._eval_policy_cpu = _MLP(self.obs_dim, self.act_dim).cpu()
            self._eval_value_cpu = _MLP(self.obs_dim, 1).cpu()

        if self._eval_sync_version != self._weights_version:
            # one-time copy from current device to CPU
            self._eval_policy_cpu.load_state_dict(self.policy.state_dict())
            self._eval_value_cpu.load_state_dict(self.value_net.state_dict())
            self._eval_sync_version = self._weights_version

        return self._eval_policy_cpu, self._eval_value_cpu


    def evaluate(
        self,
        env: gym.Env,
        n_episodes: int = 2,
        max_steps_per_episode: int = 200,
        deterministic: bool = True,
        eval_on_cpu: bool = True,
    ):
        rewards = []

        if eval_on_cpu and str(self.device).startswith("cuda"):
            policy_net, _ = self._get_eval_nets_cpu()
            device = "cpu"
        else:
            policy_net = self.policy
            device = self.device

        policy_net.eval()

        with torch.inference_mode():
            for _ in range(n_episodes):
                obs = env.reset()[0]
                ep_ret = 0.0

                for _ in range(max_steps_per_episode):
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).view(-1)

                    if self.action_continuous:
                        mu = policy_net(obs_t)
                        if deterministic:
                            a = mu
                        else:
                            std = self.log_std.exp().expand_as(mu) if device != "cpu" else torch.exp(
                                torch.as_tensor(self.log_std.detach().cpu())
                            ).expand_as(mu)
                            a = torch.distributions.Normal(mu, std).sample()

                        if device == "cpu":
                            action = a.numpy().astype(np.float32)
                            action = np.clip(action, self.act_low_np, self.act_high_np)
                        else:
                            action = a.detach().cpu().numpy().astype(np.float32)
                            action = np.clip(action, self.act_low_np, self.act_high_np)

                    else:
                        logits = policy_net(obs_t)
                        action = int(torch.argmax(logits).item()) if deterministic else int(
                            torch.distributions.Categorical(logits=logits).sample().item()
                        )

                    obs, reward, done, trunc, _ = env.step(action)
                    ep_ret += float(reward)
                    if done or trunc:
                        break

                rewards.append(ep_ret)

        return rewards

    
class EmbodiedCOINPPOAgent(PPOAgent):
    """
    A PPO agent that splits policy and value into a body and a contextual head.
    Our proposed embodied COIN technique that adapts the PPO algorithm.

    ctx_probs convention:
      - We fix an ordering of contexts: self.context_keys = list(ctx_ids) + ["novel"]
      - Column j of ctx_probs corresponds to self.context_keys[j]
    """

    def __init__(
        self,
        env: gym.Env,
        ctx_ids: dict,
        body_LR: float = 1e-3,  # learning ratio for body networks relative to head
        **kwargs,
    ):
        super().__init__(env, **kwargs)

        # ---- context bookkeeping ----
        # Ordered list of contextual heads (excluding the body)
        self.context_keys = list(ctx_ids)            # user-specified context IDs
        if "novel" not in self.context_keys:
            self.context_keys.append("novel")        # always have a 'novel' context at the end
        self.cid_to_index = {cid: i for i, cid in enumerate(self.context_keys)}
        self.num_contexts = len(self.context_keys)

        # context_init[cid] = 0/1 -> whether that context head is instantiated
        self.context_init = {cid: 0 for cid in self.context_keys}
        self.context_init["novel"] = 1  # novel context always initialised

        # ---- learning rates ----
        self.lr = kwargs.get("lr", 3e-4)
        self.body_lr = self.lr * body_LR

        # ---- Body networks ----
        self.body_policy = _MLP(self.obs_dim, self.act_dim).to(self.device)
        self.body_value_net = _MLP(self.obs_dim, 1).to(self.device)

        if self.action_continuous:
            # Fixed log_std - no learning for simplicity
            self.body_log_std = torch.ones(self.act_dim, device=self.device) * np.log(0.5)
        else:
            self.body_log_std = None

        body_params = list(self.body_policy.parameters()) + list(self.body_value_net.parameters())
        self.body_optim = optim.Adam(body_params, lr=self.body_lr)

        # self.nets holds per-context networks (including 'novel')
        #   self.nets[cid] = (optimizer, policy_net, value_net, log_std_param)
        self.nets: Dict[Any, Tuple[optim.Optimizer, nn.Module, nn.Module, Optional[nn.Parameter]]] = {}

        # ---- Create initial 'novel' context networks ----
        policy = _MLP(self.obs_dim, self.act_dim).to(self.device)
        value_net = _MLP(self.obs_dim, 1).to(self.device)
        if self.action_continuous:
            # Fixed log_std - no learning for simplicity
            log_std = torch.ones(self.act_dim, device=self.device) * np.log(0.5)
        else:
            log_std = None

        params = list(policy.parameters()) + list(value_net.parameters())
        opt = optim.Adam(params, lr=self.lr)
        self.nets["novel"] = (opt, policy, value_net, log_std)

        # Evaluation shadows (CPU)
        self._eval_body_policy_cpu = None
        self._eval_body_value_cpu = None
        self._eval_context_policies_cpu: Dict[Any, nn.Module] = {}
        self._eval_context_values_cpu: Dict[Any, nn.Module] = {}

    # ------------------------------------------------------------------
    # Context instantiation
    # ------------------------------------------------------------------
    def _instantiate_context_net(self, new_cid):
        """When a new context is instantiated, copy 'novel' networks to the new context."""
        if new_cid in self.nets:
            return  # already instantiated

        _, pnovel, vn_novel, log_std_novel = self.nets["novel"]
        policy = copy.deepcopy(pnovel).to(self.device)
        value_net = copy.deepcopy(vn_novel).to(self.device)
        log_std = copy.deepcopy(log_std_novel).to(self.device) if log_std_novel is not None else None

        params = list(policy.parameters()) + list(value_net.parameters())
        opt = optim.Adam(params, lr=self.lr)
        self.nets[new_cid] = (opt, policy, value_net, log_std)
        self.context_init[new_cid] = 1

    # ------------------------------------------------------------------
    # Mixed logits (discrete)
    # ------------------------------------------------------------------
    def _mixed_logits(self, obs_t: torch.Tensor, ctx_probs: torch.Tensor):
        """
        Return context-weighted logits + body logits, batched.

        obs_t:     [B, obs_dim]
        ctx_probs: [B, N], N == self.num_contexts
        Returns:   [B, act_dim]
        """
        B = obs_t.shape[0]
        device = obs_t.device

        if ctx_probs.dim() == 1:
            ctx_probs = ctx_probs.unsqueeze(0)  # [1, N]
        ctx_probs = ctx_probs.to(device)

        if ctx_probs.size(1) != self.num_contexts:
            raise ValueError(f"ctx_probs second dim {ctx_probs.size(1)} != num_contexts {self.num_contexts}")

        # Body logits with fixed weight 1.0
        body_logits = self.body_policy(obs_t)  # [B, act_dim]
        mixed_logits = body_logits.clone()

        # Add contextual contributions
        for j, cid in enumerate(self.context_keys):
            if self.context_init.get(cid, 0) == 0:
                continue

            _, policy, _, _ = self.nets[cid]
            logits_c = policy(obs_t)                           # [B, act_dim]
            w_c = ctx_probs[:, j].view(B, 1)                   # [B, 1]
            mixed_logits = mixed_logits + w_c * logits_c       # [B, act_dim]

        return mixed_logits

    # ------------------------------------------------------------------
    # Mixed value (critic)
    # ------------------------------------------------------------------
    def _mixed_value(self, obs_t: torch.Tensor, ctx_probs: torch.Tensor):
        """
        Batched contextual value:
        obs_t:     [B, obs_dim]
        ctx_probs: [B, N], N == self.num_contexts
        Returns:   [B] (one scalar value per sample)
        """
        B = obs_t.shape[0]
        device = obs_t.device

        if ctx_probs.dim() == 1:
            ctx_probs = ctx_probs.unsqueeze(0)  # [1, N]
        ctx_probs = ctx_probs.to(device)

        if ctx_probs.size(1) != self.num_contexts:
            raise ValueError(f"ctx_probs second dim {ctx_probs.size(1)} != num_contexts {self.num_contexts}")

        # Body value with fixed weight 1.0
        body_v = self.body_value_net(obs_t).squeeze(-1)  # [B]
        mixed_value = body_v.clone()

        # Contextual contributions
        for j, cid in enumerate(self.context_keys):
            if self.context_init.get(cid, 0) == 0:
                continue

            _, _, value_net, _ = self.nets[cid]
            v_c = value_net(obs_t).squeeze(-1)                 # [B]
            w_c = ctx_probs[:, j]                              # [B]
            mixed_value = mixed_value + w_c * v_c              # [B]

        return mixed_value   # [B]

    # ------------------------------------------------------------------
    # Mixed Gaussian (continuous)
    # ------------------------------------------------------------------
    def _mixed_gaussian(self, obs_t: torch.Tensor, ctx_probs: torch.Tensor):
        """
        Precision-weighted combination of contextual and body Gaussians (batched).

        For each component i (contexts + body) and each sample b, dim k:
          mu_i(b,k), std_i(b,k), alpha_i(b)

        Let S_i(b,k) = 1 / std_i(b,k)^2  (precision).
        Then:
            S_bar(b,k) = sum_i alpha_i(b) * S_i(b,k)
            mu_bar(b,k) = [sum_i alpha_i(b) * S_i(b,k) * mu_i(b,k)] / S_bar(b,k)
            sigma_bar(b,k)^2 = 1 / S_bar(b,k)

        obs_t:     [B, obs_dim]
        ctx_probs: [B, N], N == self.num_contexts
        Returns:
            mixed_mu:  [B, act_dim]
            mixed_std: [B, act_dim]
        """
        B = obs_t.shape[0]
        device = obs_t.device

        if ctx_probs.dim() == 1:
            ctx_probs = ctx_probs.unsqueeze(0)  # [1, N]
        ctx_probs = ctx_probs.to(device)

        if ctx_probs.size(1) != self.num_contexts:
            raise ValueError(f"ctx_probs second dim {ctx_probs.size(1)} != num_contexts {self.num_contexts}")

        mus, stds, weights = [], [], []

        # Context-specific Gaussians
        for j, cid in enumerate(self.context_keys):
            if self.context_init.get(cid, 0) == 0:
                continue

            _, policy, _, log_std = self.nets[cid]
            if log_std is None:
                raise RuntimeError("Continuous actions require log_std for each context.")

            mu = policy(obs_t)                                              # [B, act_dim]
            std = log_std.exp().view(1, -1).expand_as(mu)                   # [B, act_dim]
            alpha = ctx_probs[:, j].view(B, 1)                              # [B, 1]

            mus.append(mu)
            stds.append(std)
            weights.append(alpha)

        # Body Gaussian with fixed weight 1.0
        if self.body_log_std is None:
            raise RuntimeError("Continuous actions require body_log_std on body as well.")
        mu_body = self.body_policy(obs_t)                                   # [B, act_dim]
        std_body = self.body_log_std.exp().view(1, -1).expand_as(mu_body)   # [B, act_dim]
        alpha_body = torch.ones(B, 1, device=device)                        # [B, 1]

        mus.append(mu_body)
        stds.append(std_body)
        weights.append(alpha_body)

        if not mus:
            raise RuntimeError("No Gaussian components collected (all weights zero / no contexts).")

        mus = torch.stack(mus, dim=0)        # [C, B, act_dim]
        stds = torch.stack(stds, dim=0)      # [C, B, act_dim]
        weights = torch.stack(weights, dim=0)  # [C, B, 1]

        # Normalise weights per sample to avoid degenerate scaling
        weight_sum = weights.sum(dim=0, keepdim=True).clamp_min(1e-8)  # [1, B, 1]
        alphas = weights / weight_sum                                  # [C, B, 1]

        # Per-component precisions
        precisions = 1.0 / (stds ** 2)                                 # [C, B, act_dim]

        # S_bar = sum_i alpha_i * precision_i
        S_bar = (alphas * precisions).sum(dim=0)                       # [B, act_dim]

        # Numerator for mean
        num = (alphas * precisions * mus).sum(dim=0)                   # [B, act_dim]

        mixed_mu = num / S_bar                                         # [B, act_dim]
        mixed_std = torch.sqrt(1.0 / S_bar)                            # [B, act_dim]

        return mixed_mu, mixed_std
    
    def _all_optimizers(self):
        """Helper to extract all optimizers (body + contexts)."""
        opts = []
        # body optimizer (if you actually want it to learn)
        if self.body_optim is not None:
            opts.append(self.body_optim)
        # each context optimizer
        for cid, (opt, _, _, _) in self.nets.items():
            if opt is not None:
                opts.append(opt)
        return opts

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def act(self, obs: torch.Tensor, ctx_probs: torch.Tensor):
        """
        Generate action given observation and context probabilities.

        obs:        [obs_dim] or [B, obs_dim]
        ctx_probs:  [N] or [B, N] with N == self.num_contexts
        """
        if obs.dim() == 1:
            obs_t = obs.unsqueeze(0)        # [1, obs_dim]
        else:
            obs_t = obs

        if ctx_probs.dim() == 1:
            ctx_t = ctx_probs.unsqueeze(0)  # [1, N]
        else:
            ctx_t = ctx_probs

        if self.action_continuous:
            # Continuous: diagonal Gaussian policy
            mixed_mu, mixed_std = self._mixed_gaussian(obs_t, ctx_t)  # [B, act_dim]
            dist = torch.distributions.Normal(mixed_mu, mixed_std)
            raw_action = dist.sample()                                # [B, act_dim]

            action = raw_action
            if self.act_low is not None and self.act_high is not None:
                action = torch.max(torch.min(action, self.act_high), self.act_low)

            logp = dist.log_prob(action).sum(-1)   # [B]
            entropy = dist.entropy().sum(-1)          # [B]

            # Assuming we call this with B=1 during rollout
            action_np = action.detach().cpu().numpy()
            return action_np.squeeze(0), logp.squeeze(0), entropy.squeeze(0), mixed_mu.squeeze(0)

        else:
            # Discrete: Categorical policy
            logits = self._mixed_logits(obs_t, ctx_t)          # [B, act_dim]
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()                             # [B]
            logp = dist.log_prob(action)                       # [B]
            entropy = dist.entropy()                           # [B]

            action_np = action.detach().cpu().numpy()
            return action_np.squeeze(0), logp.squeeze(0), entropy.squeeze(0), logits.squeeze(0)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def train_step(
        self,
        env,
        context_probs_fn,
        rollout_steps: int = 2048,
        mini_epochs: int = 10,
        mb_size: int = 64,
    ):
        """
        context_probs_fn: either a function that, given an episode index (or step index),
                          returns an array-like [N] of context probabilities, or a constant
                          array-like [N] used for every episode, where index j corresponds
                          to self.context_keys[j].

        Otherwise same interface as PPOAgent.
        """
        context_probs_fn = _as_context_probs_fn(context_probs_fn)
        obs = env.reset()[0]
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        ep_returns = []
        ep_len = ep_ret = ep_num = 0

        storage: Dict[str, List[Any]] = defaultdict(list)

        # ---------- rollout ----------
        for step in range(rollout_steps):
            # 1D array-like [N]
            ctx_probs_vec = np.asarray(context_probs_fn(ep_num), dtype=np.float32)  # [N]

            # Possibly instantiate new context heads
            for j, cid in enumerate(self.context_keys):
                if cid == "novel":
                    continue  # novel already initialised
                if self.context_init[cid] == 0:
                    p_c = ctx_probs_vec[j]
                    if not np.isnan(p_c) and p_c != 0.0:
                        self._instantiate_context_net(cid)

            ctx_probs_t = torch.as_tensor(ctx_probs_vec, device=self.device, dtype=torch.float32)  # [N]

            with torch.no_grad():
                # value for this state
                value = self._mixed_value(obs_t.unsqueeze(0), ctx_probs_t.unsqueeze(0))[0].item()
                # action
                action_np, logp, ent, _ = self.act(obs_t, ctx_probs_t)

            next_obs, reward, done, trunc, _ = env.step(action_np)

            # store (we also keep ctx_probs row to weight backprop later)
            storage["obs"].append(obs_t.detach().cpu())                  # [obs_dim]
            storage["act"].append(torch.as_tensor(action_np))            # [act_dim] or scalar
            storage["logp"].append(logp.detach().cpu())                  # []
            storage["rew"].append(reward)
            storage["val"].append(torch.tensor(value, dtype=torch.float32))
            storage["done"].append(done or trunc)
            storage["ctx_probs"].append(torch.as_tensor(ctx_probs_vec, dtype=torch.float32))  # [N]
            storage["ent"].append(ent.detach().cpu())

            ep_ret += reward
            ep_len += 1

            if done or trunc:
                next_obs, _ = env.reset()
                ep_returns.append(ep_ret)
                ep_len = ep_ret = 0
                ep_num += 1

            obs_t = self._flatten_obs(next_obs)

        # ---------- advantages / returns ----------
        with torch.no_grad():
            last_ctx_probs_vec = np.asarray(context_probs_fn(ep_num), dtype=np.float32)  # [N]
            last_ctx_probs_t = torch.as_tensor(last_ctx_probs_vec, device=self.device, dtype=torch.float32)
            last_val = self._mixed_value(obs_t.unsqueeze(0), last_ctx_probs_t.unsqueeze(0))[0].item()

        val_tensor = torch.stack(storage["val"])  # [T]
        adv, ret = self._compute_advantages(storage["rew"], val_tensor, storage["done"], last_val)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ---------- build tensors for optimisation ----------
        dataset_size = rollout_steps
        idxs = torch.randperm(dataset_size)

        obs_tensor = torch.stack(storage["obs"]).to(self.device)          # [T, obs_dim]
        ctx_probs_tensor = torch.stack(storage["ctx_probs"]).to(self.device)  # [T, N]

        if self.action_continuous:
            act_tensor = torch.stack(storage["act"]).to(self.device).float()   # [T, act_dim]
        else:
            act_tensor = torch.stack(storage["act"]).to(self.device).long()    # [T]

        old_logp_tensor = torch.stack(storage["logp"]).to(self.device).float()  # [T]
        ret_tensor = ret.to(self.device).float()                                # [T]
        adv_tensor = adv.to(self.device).float()                                # [T]

        # ---------- optimisation ----------
        for _ in range(mini_epochs):
            for start in range(0, dataset_size, mb_size):
                end = start + mb_size
                mb_idx = idxs[start:end]

                batch_obs = obs_tensor[mb_idx]          # [B, obs_dim]
                batch_act = act_tensor[mb_idx]          # [B, act_dim] or [B]
                batch_adv = adv_tensor[mb_idx]          # [B]
                batch_ret = ret_tensor[mb_idx]          # [B]
                batch_old_logp = old_logp_tensor[mb_idx]  # [B]
                batch_ctx_probs = ctx_probs_tensor[mb_idx]  # [B, N]

                if self.action_continuous:
                    mixed_mu, mixed_std = self._mixed_gaussian(batch_obs, batch_ctx_probs)  # [B, act_dim]
                    dist = torch.distributions.Normal(mixed_mu, mixed_std)
                    new_logp = dist.log_prob(batch_act).sum(-1)         # [B]
                    entropy = dist.entropy().sum(-1).mean()
                else:
                    logits = self._mixed_logits(batch_obs, batch_ctx_probs)  # [B, act_dim]
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logp = dist.log_prob(batch_act)                      # [B]
                    entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - batch_old_logp)  # [B]

                # Clipped surrogate
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                value_pred = self._mixed_value(batch_obs, batch_ctx_probs)   # [B]
                critic_loss = (batch_ret - value_pred).pow(2).mean()

                optimizers = self._all_optimizers()

                for opt in optimizers:
                    opt.zero_grad()

                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy
                loss.backward()

                for opt in optimizers:
                    opt.step()


        rew_buf = np.array(storage["rew"])
        mean_ep_return = float(np.mean(ep_returns)) if ep_returns else 0.0

        self._weights_version += 1

        return {
            "mean_episode_return": mean_ep_return,
            "mean_reward_per_step": float(rew_buf.mean()) if len(rew_buf) > 0 else 0.0,
            "value_loss": critic_loss.item(),
            "policy_loss": actor_loss.item(),
        }
    
    def _get_eval_nets_cpu(self):
        if self._eval_body_policy_cpu is None:
            self._eval_body_policy_cpu = _MLP(self.obs_dim, self.act_dim).cpu()
            self._eval_body_value_cpu = _MLP(self.obs_dim, 1).cpu()

        # Check all context nets too
        for cid in self.context_keys:
            if self.context_init.get(cid, 0) == 0:
                continue
            if cid not in self._eval_context_policies_cpu:
                self._eval_context_policies_cpu[cid] = _MLP(self.obs_dim, self.act_dim).cpu()
                self._eval_context_values_cpu[cid] = _MLP(self.obs_dim, 1).cpu()

        if self._eval_sync_version != self._weights_version:
            # one-time copy from current device to CPU
            self._eval_body_policy_cpu.load_state_dict(self.body_policy.state_dict())
            self._eval_body_value_cpu.load_state_dict(self.body_value_net.state_dict())
            for cid in self.context_keys:
                if self.context_init.get(cid, 0) == 0:
                    continue
                _, policy, value_net, _ = self.nets[cid]
                self._eval_context_policies_cpu[cid].load_state_dict(policy.state_dict())
                self._eval_context_values_cpu[cid].load_state_dict(value_net.state_dict())
            self._eval_sync_version = self._weights_version

        return self._eval_body_policy_cpu, self._eval_body_value_cpu, self._eval_context_policies_cpu, self._eval_context_values_cpu
    
    def evaluate(
        self,
        env: gym.Env,
        context_probs_fn,
        n_episodes: int = 2,
        max_steps_per_episode: int = 200,
        deterministic: bool = True,
        eval_on_cpu: bool = True,
    ):
        """
        Execute the learned policies to evaluate performance.
        This method does not train the model.

        context_probs_fn: either a function that, given an episode index (or step index),
                          returns an array-like [N] of context probabilities, or a constant
                          array-like [N] used for every episode, where index j corresponds
                          to self.context_keys[j].
        """
        context_probs_fn = _as_context_probs_fn(context_probs_fn)
        rewards = []

        if eval_on_cpu and str(self.device).startswith("cuda"):
            body_policy_cpu, _, context_policies_cpu, _ = self._get_eval_nets_cpu()
            device = "cpu"
        else:
            body_policy_cpu = self.body_policy
            context_policies_cpu = {cid: self.nets[cid][1] for cid in self.context_keys if self.context_init.get(cid, 0) == 1}
            device = self.device

        body_policy_cpu.eval()
        for cid in context_policies_cpu:
            context_policies_cpu[cid].eval()

        with torch.inference_mode():
            for epnum in range(n_episodes):
                obs = env.reset()[0]
                ep_ret = 0.0

                ctx_probs_vec = np.asarray(context_probs_fn(epnum), dtype=np.float32)  # [N]

                # All given contexts must be initialised
                for j, cid in enumerate(self.context_keys):
                    p_c = ctx_probs_vec[j]
                    if p_c != 0.0 and self.context_init.get(cid, 0) == 0:
                        raise RuntimeError(f"Context ID {cid} required by context_probs_fn but not initialised.")

                ctx_probs = torch.as_tensor(ctx_probs_vec, device=device, dtype=torch.float32).unsqueeze(0)  # [1, N]

                for _ in range(max_steps_per_episode):
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).view(-1)
                    B = obs_t.shape[0]

                    # Mixed Means/Logits
                    mus = [context_policies_cpu[cid](obs_t) for cid in context_policies_cpu]
                    mus.append(body_policy_cpu(obs_t))
                    mus = torch.stack(mus, dim=0)

                    # Context Weights
                    weight_sum = ctx_probs.nansum(dim=1, keepdim=True).clamp_min(1e-8)
                    alphas = (ctx_probs / weight_sum).view(-1, 1)
                    # Add 1 to the dimension of alphas for the body
                    alphas = torch.cat([alphas, torch.ones(1, 1, device=device)], dim=0)

                    if self.action_continuous:
                        stds = [torch.exp(torch.as_tensor(self.nets[cid][3].detach().cpu())).view(1, -1).expand_as(mus[0]) for cid in context_policies_cpu]
                        stds.append(self.body_log_std.exp().view(1, -1).expand_as(mus[0]))

                        stds = torch.stack(stds, dim=0)
                        precisions = 1.0 / (stds ** 2)
                        S_bar = (alphas * precisions).sum(dim=0)
                        num = (alphas * precisions * mus).sum(dim=0)
                        mixed_mu = num / S_bar
                        mixed_std = torch.sqrt(1.0 / S_bar)
                        if deterministic:
                            a = mixed_mu
                        else:
                            a = torch.distributions.Normal(mixed_mu, mixed_std).sample()

                        if device == "cpu":
                            action = a.numpy().astype(np.float32)
                            action = np.clip(action, self.act_low_np, self.act_high_np)
                        else:
                            action = a.detach().cpu().numpy().astype(np.float32)
                            action = np.clip(action, self.act_low_np, self.act_high_np)

                    else:
                        mixed_logits = (alphas * mus).sum(dim=0)  # [1, act_dim]
                        action = int(torch.argmax(mixed_logits).item()) if deterministic else int(
                            torch.distributions.Categorical(logits=mixed_logits).sample().item()
                        )

                    obs, reward, done, trunc, _ = env.step(action)
                    ep_ret += float(reward)
                    if done or trunc:
                        break

                rewards.append(ep_ret)

        return rewards
    
    def reset_body_from(
        self,
        src_policy: nn.Module,
        src_value_net: nn.Module,
        src_log_std: Optional[nn.Parameter] = None,
    ) -> None:
        """
        Reset the body policy / value networks (and optional log_std) to copies of
        the given source networks.

        Parameters
        ----------
        src_policy : nn.Module
            Source policy network to copy into `self.body_policy`.
        src_value_net : nn.Module
            Source value network to copy into `self.body_value_net`.
        src_log_std : Optional[nn.Parameter], default=None
            Source log-std parameter to copy into `self.body_log_std` (for
            continuous actions). If None and `self.action_continuous` is True,
            the body log-std is reset to zeros.
        """

        # Copy weights into body networks
        self.body_policy.load_state_dict(src_policy.state_dict())
        self.body_value_net.load_state_dict(src_value_net.state_dict())

        # Handle log-std for continuous actions
        if self.action_continuous:
            if src_log_std is not None:
                with torch.no_grad():
                    self.body_log_std.copy_(src_log_std.data.to(self.body_log_std.device))
            else:
                # If no source log_std provided, reset to zeros
                with torch.no_grad():
                    self.body_log_std.zero_()

        # Rebuild body optimizer so it tracks the (possibly new) Parameter objects
        body_params = list(self.body_policy.parameters()) + list(self.body_value_net.parameters())
        if self.body_log_std is not None:
            body_params.append(self.body_log_std)

        self.body_optim = optim.Adam(body_params, lr=self.body_lr)



class COINPPOAgent(PPOAgent):
    """
    COIN-style contextual PPO (NO BODY).

    Parallel to EmbodiedCOINPPOAgent, but assumes the body contribution is identically zero:
      - no body networks are created
      - action/value are obtained purely by mixing contextual heads using ctx_probs

    ctx_probs convention:
      - We fix an ordering of contexts: self.context_keys = list(ctx_ids) + ["novel"]
      - Column j of ctx_probs corresponds to self.context_keys[j]
    """

    def __init__(
        self,
        env: gym.Env,
        ctx_ids: dict,
        **kwargs,
    ):
        super().__init__(env, **kwargs)

        # ---- context bookkeeping ----
        self.context_keys = list(ctx_ids)
        if "novel" not in self.context_keys:
            self.context_keys.append("novel")
        self.cid_to_index = {cid: i for i, cid in enumerate(self.context_keys)}
        self.num_contexts = len(self.context_keys)

        # context_init[cid] = 0/1 -> whether that context head is instantiated
        self.context_init = {cid: 0 for cid in self.context_keys}
        self.context_init["novel"] = 1

        # ---- learning rate ----
        self.lr = kwargs.get("lr", 3e-4)

        # ---- Per-context networks ----
        #   self.nets[cid] = (optimizer, policy_net, value_net, log_std_param_or_tensor)
        self.nets: Dict[Any, Tuple[optim.Optimizer, nn.Module, nn.Module, Optional[torch.Tensor]]] = {}

        # Create initial 'novel' context networks
        policy = _MLP(self.obs_dim, self.act_dim).to(self.device)
        value_net = _MLP(self.obs_dim, 1).to(self.device)
        if self.action_continuous:
            # Fixed log_std (tensor) - no learning for simplicity
            log_std = torch.ones(self.act_dim, device=self.device) * np.log(0.5)
        else:
            log_std = None

        params = list(policy.parameters()) + list(value_net.parameters())
        opt = optim.Adam(params, lr=self.lr)
        self.nets["novel"] = (opt, policy, value_net, log_std)

        # ---- evaluation shadows (CPU) ----
        self._eval_context_policies_cpu: Dict[Any, nn.Module] = {}
        self._eval_context_values_cpu: Dict[Any, nn.Module] = {}

    # ------------------------------------------------------------------
    # Context instantiation
    # ------------------------------------------------------------------
    def _instantiate_context_net(self, new_cid):
        """When a new context is instantiated, copy 'novel' networks to the new context."""
        if new_cid in self.nets:
            return

        _, pnovel, vn_novel, log_std_novel = self.nets["novel"]
        policy = copy.deepcopy(pnovel).to(self.device)
        value_net = copy.deepcopy(vn_novel).to(self.device)
        log_std = log_std_novel.clone().to(self.device) if log_std_novel is not None else None

        params = list(policy.parameters()) + list(value_net.parameters())
        opt = optim.Adam(params, lr=self.lr)
        self.nets[new_cid] = (opt, policy, value_net, log_std)
        self.context_init[new_cid] = 1

    # ------------------------------------------------------------------
    # Mixed logits (discrete)
    # ------------------------------------------------------------------
    def _mixed_logits(self, obs_t: torch.Tensor, ctx_probs: torch.Tensor):
        """
        Context-weighted logits, batched.

        obs_t:     [B, obs_dim]
        ctx_probs: [B, N], N == self.num_contexts
        Returns:   [B, act_dim]
        """
        B = obs_t.shape[0]
        device = obs_t.device

        if ctx_probs.dim() == 1:
            ctx_probs = ctx_probs.unsqueeze(0)
        ctx_probs = ctx_probs.to(device)

        if ctx_probs.size(1) != self.num_contexts:
            raise ValueError(
                f"ctx_probs second dim {ctx_probs.size(1)} != num_contexts {self.num_contexts}"
            )

        mixed_logits = torch.zeros(B, self.act_dim, device=device)

        for j, cid in enumerate(self.context_keys):
            if self.context_init.get(cid, 0) == 0:
                continue
            _, policy, _, _ = self.nets[cid]
            logits_c = policy(obs_t)                     # [B, act_dim]
            w_c = ctx_probs[:, j].view(B, 1)            # [B, 1]
            mixed_logits = mixed_logits + w_c * logits_c

        return mixed_logits

    # ------------------------------------------------------------------
    # Mixed value (critic)
    # ------------------------------------------------------------------
    def _mixed_value(self, obs_t: torch.Tensor, ctx_probs: torch.Tensor):
        """
        Context-weighted value, batched.

        obs_t:     [B, obs_dim]
        ctx_probs: [B, N], N == self.num_contexts
        Returns:   [B]
        """
        B = obs_t.shape[0]
        device = obs_t.device

        if ctx_probs.dim() == 1:
            ctx_probs = ctx_probs.unsqueeze(0)
        ctx_probs = ctx_probs.to(device)

        if ctx_probs.size(1) != self.num_contexts:
            raise ValueError(
                f"ctx_probs second dim {ctx_probs.size(1)} != num_contexts {self.num_contexts}"
            )

        mixed_value = torch.zeros(B, device=device)

        for j, cid in enumerate(self.context_keys):
            if self.context_init.get(cid, 0) == 0:
                continue
            _, _, value_net, _ = self.nets[cid]
            v_c = value_net(obs_t).squeeze(-1)          # [B]
            w_c = ctx_probs[:, j]                       # [B]
            mixed_value = mixed_value + w_c * v_c

        return mixed_value

    # ------------------------------------------------------------------
    # Mixed Gaussian (continuous)
    # ------------------------------------------------------------------
    def _mixed_gaussian(self, obs_t: torch.Tensor, ctx_probs: torch.Tensor):
        """
        Precision-weighted combination of context Gaussians (batched), no body component.

        obs_t:     [B, obs_dim]
        ctx_probs: [B, N]
        Returns:
            mixed_mu:  [B, act_dim]
            mixed_std: [B, act_dim]
        """
        B = obs_t.shape[0]
        device = obs_t.device

        if ctx_probs.dim() == 1:
            ctx_probs = ctx_probs.unsqueeze(0)
        ctx_probs = ctx_probs.to(device)

        if ctx_probs.size(1) != self.num_contexts:
            raise ValueError(
                f"ctx_probs second dim {ctx_probs.size(1)} != num_contexts {self.num_contexts}"
            )

        mus, stds, weights = [], [], []

        for j, cid in enumerate(self.context_keys):
            if self.context_init.get(cid, 0) == 0:
                continue

            _, policy, _, log_std = self.nets[cid]
            if log_std is None:
                raise RuntimeError("Continuous actions require log_std for each context.")

            mu = policy(obs_t)                                           # [B, act_dim]
            std = log_std.exp().view(1, -1).expand_as(mu)                # [B, act_dim]
            alpha = ctx_probs[:, j].view(B, 1)                           # [B, 1]

            mus.append(mu)
            stds.append(std)
            weights.append(alpha)

        if not mus:
            raise RuntimeError("No Gaussian components collected (all contexts inactive).")

        mus = torch.stack(mus, dim=0)        # [C, B, act_dim]
        stds = torch.stack(stds, dim=0)      # [C, B, act_dim]
        weights = torch.stack(weights, dim=0)  # [C, B, 1]

        # Normalise weights per sample
        weight_sum = weights.sum(dim=0, keepdim=True).clamp_min(1e-8)  # [1, B, 1]
        alphas = weights / weight_sum                                   # [C, B, 1]

        precisions = 1.0 / (stds ** 2)                                  # [C, B, act_dim]
        S_bar = (alphas * precisions).sum(dim=0)                        # [B, act_dim]
        num = (alphas * precisions * mus).sum(dim=0)                    # [B, act_dim]

        mixed_mu = num / S_bar
        mixed_std = torch.sqrt(1.0 / S_bar)

        return mixed_mu, mixed_std

    def _all_optimizers(self):
        """All context optimizers."""
        return [opt for (opt, _, _, _) in self.nets.values() if opt is not None]

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def act(self, obs: torch.Tensor, ctx_probs: torch.Tensor):
        """
        Generate action given observation and context probabilities.

        obs:        [obs_dim] or [B, obs_dim]
        ctx_probs:  [N] or [B, N]
        """
        if obs.dim() == 1:
            obs_t = obs.unsqueeze(0)
        else:
            obs_t = obs

        if ctx_probs.dim() == 1:
            ctx_t = ctx_probs.unsqueeze(0)
        else:
            ctx_t = ctx_probs

        if self.action_continuous:
            mixed_mu, mixed_std = self._mixed_gaussian(obs_t, ctx_t)
            dist = torch.distributions.Normal(mixed_mu, mixed_std)
            raw_action = dist.sample()

            action = raw_action
            if self.act_low is not None and self.act_high is not None:
                action = torch.max(torch.min(action, self.act_high), self.act_low)

            logp = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)

            action_np = action.detach().cpu().numpy()
            return action_np.squeeze(0), logp.squeeze(0), entropy.squeeze(0), mixed_mu.squeeze(0)

        logits = self._mixed_logits(obs_t, ctx_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        entropy = dist.entropy()

        action_np = action.detach().cpu().numpy()
        return action_np.squeeze(0), logp.squeeze(0), entropy.squeeze(0), logits.squeeze(0)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def train_step(
        self,
        env,
        context_probs_fn,
        rollout_steps: int = 2048,
        mini_epochs: int = 10,
        mb_size: int = 64,
    ):
        """
        context_probs_fn: either a function that takes an episode index and returns [N] ctx
                          probs, or a constant array-like [N] used for every episode; in
                          both cases aligned with self.context_keys.
        """
        context_probs_fn = _as_context_probs_fn(context_probs_fn)
        obs = env.reset()[0]
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        ep_returns = []
        ep_ret = ep_len = ep_num = 0

        storage: Dict[str, List[Any]] = defaultdict(list)

        # ---------- rollout ----------
        for _ in range(rollout_steps):
            ctx_probs_vec = np.asarray(context_probs_fn(ep_num), dtype=np.float32)  # [N]

            # instantiate any newly-active known contexts
            for j, cid in enumerate(self.context_keys):
                if cid == "novel":
                    continue
                if self.context_init[cid] == 0:
                    p_c = ctx_probs_vec[j]
                    if not np.isnan(p_c) and p_c != 0.0:
                        self._instantiate_context_net(cid)

            ctx_probs_t = torch.as_tensor(ctx_probs_vec, device=self.device, dtype=torch.float32)

            with torch.no_grad():
                value = self._mixed_value(obs_t.unsqueeze(0), ctx_probs_t.unsqueeze(0))[0].item()
                action_np, logp, ent, _ = self.act(obs_t, ctx_probs_t)

            next_obs, reward, done, trunc, _ = env.step(action_np)

            storage["obs"].append(obs_t.detach().cpu())
            storage["act"].append(torch.as_tensor(action_np))
            storage["logp"].append(logp.detach().cpu())
            storage["rew"].append(reward)
            storage["val"].append(torch.tensor(value, dtype=torch.float32))
            storage["done"].append(done or trunc)
            storage["ctx_probs"].append(torch.as_tensor(ctx_probs_vec, dtype=torch.float32))
            storage["ent"].append(ent.detach().cpu())

            ep_ret += reward
            ep_len += 1

            if done or trunc:
                next_obs, _ = env.reset()
                ep_returns.append(ep_ret)
                ep_ret = ep_len = 0
                ep_num += 1

            obs_t = self._flatten_obs(next_obs)

        # ---------- advantages / returns ----------
        with torch.no_grad():
            last_ctx_probs_vec = np.asarray(context_probs_fn(ep_num), dtype=np.float32)
            last_ctx_probs_t = torch.as_tensor(last_ctx_probs_vec, device=self.device, dtype=torch.float32)
            last_val = self._mixed_value(obs_t.unsqueeze(0), last_ctx_probs_t.unsqueeze(0))[0].item()

        val_tensor = torch.stack(storage["val"])
        adv, ret = self._compute_advantages(storage["rew"], val_tensor, storage["done"], last_val)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ---------- tensors ----------
        dataset_size = rollout_steps
        idxs = torch.randperm(dataset_size)

        obs_tensor = torch.stack(storage["obs"]).to(self.device)
        ctx_probs_tensor = torch.stack(storage["ctx_probs"]).to(self.device)

        if self.action_continuous:
            act_tensor = torch.stack(storage["act"]).to(self.device).float()
        else:
            act_tensor = torch.stack(storage["act"]).to(self.device).long()

        old_logp_tensor = torch.stack(storage["logp"]).to(self.device).float()
        ret_tensor = ret.to(self.device).float()
        adv_tensor = adv.to(self.device).float()

        # ---------- optimisation ----------
        for _ in range(mini_epochs):
            for start in range(0, dataset_size, mb_size):
                end = start + mb_size
                mb_idx = idxs[start:end]

                batch_obs = obs_tensor[mb_idx]
                batch_act = act_tensor[mb_idx]
                batch_adv = adv_tensor[mb_idx]
                batch_ret = ret_tensor[mb_idx]
                batch_old_logp = old_logp_tensor[mb_idx]
                batch_ctx_probs = ctx_probs_tensor[mb_idx]

                if self.action_continuous:
                    mixed_mu, mixed_std = self._mixed_gaussian(batch_obs, batch_ctx_probs)
                    dist = torch.distributions.Normal(mixed_mu, mixed_std)
                    new_logp = dist.log_prob(batch_act).sum(-1)
                    entropy = dist.entropy().sum(-1).mean()
                else:
                    logits = self._mixed_logits(batch_obs, batch_ctx_probs)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logp = dist.log_prob(batch_act)
                    entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - batch_old_logp)

                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                value_pred = self._mixed_value(batch_obs, batch_ctx_probs)
                critic_loss = (batch_ret - value_pred).pow(2).mean()

                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy

                for opt in self._all_optimizers():
                    opt.zero_grad()
                loss.backward()
                for opt in self._all_optimizers():
                    opt.step()

        self._weights_version += 1
        rew_buf = np.asarray(storage["rew"], dtype=float)

        return {
            "mean_episode_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "mean_reward_per_step": float(rew_buf.mean()) if rew_buf.size else 0.0,
            "value_loss": float(critic_loss.item()),
            "policy_loss": float(actor_loss.item()),
        }

    # ------------------------------------------------------------------
    # Evaluation (optional CPU shadow)
    # ------------------------------------------------------------------
    def _get_eval_nets_cpu(self):
        # Create missing CPU shadows for instantiated contexts
        for cid in self.context_keys:
            if self.context_init.get(cid, 0) == 0:
                continue
            if cid not in self._eval_context_policies_cpu:
                self._eval_context_policies_cpu[cid] = _MLP(self.obs_dim, self.act_dim).cpu()
                self._eval_context_values_cpu[cid] = _MLP(self.obs_dim, 1).cpu()

        if self._eval_sync_version != self._weights_version:
            for cid in self.context_keys:
                if self.context_init.get(cid, 0) == 0:
                    continue
                _, policy, value_net, _ = self.nets[cid]
                self._eval_context_policies_cpu[cid].load_state_dict(policy.state_dict())
                self._eval_context_values_cpu[cid].load_state_dict(value_net.state_dict())
            self._eval_sync_version = self._weights_version

        return self._eval_context_policies_cpu, self._eval_context_values_cpu

    def evaluate(
        self,
        env: gym.Env,
        context_probs_fn,
        n_episodes: int = 2,
        max_steps_per_episode: int = 200,
        deterministic: bool = True,
        eval_on_cpu: bool = True,
    ):
        """
        Execute the learned policies to evaluate performance.
        This method does not train the model.

        context_probs_fn: either a function that takes an episode index and returns [N] ctx
                          probs, or a constant array-like [N] used for every episode; in
                          both cases aligned with self.context_keys.
        """
        context_probs_fn = _as_context_probs_fn(context_probs_fn)
        rewards = []

        if eval_on_cpu and str(self.device).startswith("cuda"):
            context_policies_cpu, _ = self._get_eval_nets_cpu()
            device = "cpu"
        else:
            context_policies_cpu = {
                cid: self.nets[cid][1]
                for cid in self.context_keys
                if self.context_init.get(cid, 0) == 1
            }
            device = self.device

        for cid in context_policies_cpu:
            context_policies_cpu[cid].eval()

        with torch.inference_mode():
            for epnum in range(n_episodes):
                obs = env.reset()[0]
                ep_ret = 0.0

                ctx_probs_vec = np.asarray(context_probs_fn(epnum), dtype=np.float32)

                # All required contexts must be initialised (except novel, which always is)
                for j, cid in enumerate(self.context_keys):
                    p_c = ctx_probs_vec[j]
                    if p_c != 0.0 and self.context_init.get(cid, 0) == 0:
                        raise RuntimeError(
                            f"Context ID {cid} required by context_probs_fn but not initialised."
                        )

                ctx_probs = torch.as_tensor(ctx_probs_vec, device=device, dtype=torch.float32).unsqueeze(0)  # [1, N]

                for _ in range(max_steps_per_episode):
                    # Ensure batched observation: [1, obs_dim]
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, -1)

                    # Collect mus/logits for all active contexts in fixed order (batched)
                    # ctx_probs: [1, N]
                    alphas = []
                    mus = []

                    for j, cid in enumerate(self.context_keys):
                        if self.context_init.get(cid, 0) == 0:
                            continue

                        mus.append(context_policies_cpu[cid](obs_t))      # obs_t: [1, obs_dim] -> [1, act_dim]
                        alphas.append(ctx_probs[:, j].view(1, 1))         # [1, 1]

                    mus = torch.stack(mus, dim=0)                         # [C, 1, act_dim]
                    alphas = torch.stack(alphas, dim=0)                   # [C, 1, 1]

                    # Normalise weights safely (handle NaNs)
                    alphas = torch.nan_to_num(alphas, nan=0.0)
                    wsum = alphas.sum(dim=0, keepdim=True).clamp_min(1e-8)    # [1, 1, 1]
                    alphas = alphas / wsum                                    # [C, 1, 1]

                    if self.action_continuous:
                        stds = []
                        for cid in [c for c in self.context_keys if self.context_init.get(c, 0) == 1]:
                            log_std = self.nets[cid][3]
                            if log_std is None:
                                raise RuntimeError("Continuous actions require log_std for each context.")
                            stds.append(log_std.exp().view(1, -1).expand_as(mus[0]))  # [1, act_dim]
                        stds = torch.stack(stds, dim=0)                               # [C, 1, act_dim]

                        precisions = 1.0 / (stds ** 2)                                # [C, 1, act_dim]
                        S_bar = (alphas * precisions).sum(dim=0)                      # [1, act_dim]
                        num = (alphas * precisions * mus).sum(dim=0)                  # [1, act_dim]
                        mixed_mu = num / S_bar                                        # [1, act_dim]
                        mixed_std = torch.sqrt(1.0 / S_bar)                           # [1, act_dim]

                        a = mixed_mu if deterministic else torch.distributions.Normal(mixed_mu, mixed_std).sample()
                        action = a.detach().cpu().numpy().astype(np.float32)          # [1, act_dim]
                        action = np.clip(action, self.act_low_np, self.act_high_np).squeeze(0)  # [act_dim]

                    else:
                        mixed_logits = (alphas * mus).sum(dim=0)                      # [1, act_dim]
                        if deterministic:
                            action = int(torch.argmax(mixed_logits, dim=-1).item())
                        else:
                            action = int(torch.distributions.Categorical(logits=mixed_logits).sample().item())

                    obs, reward, done, trunc, _ = env.step(action)
                    ep_ret += float(reward)
                    if done or trunc:
                        break

                rewards.append(ep_ret)

        return rewards


#----- Amortised COIN PPO -----

class ContingencyEncoder(nn.Module):
    """
    PEARL-style amortised posterior ``q_phi(z | h)`` over a scalar latent contingency.

    A shared MLP maps every transition ``(s, a, s')`` to its own Gaussian factor; the
    posterior is their product, held in natural parameters so every *prefix* posterior is a
    running sum -- one ``cumsum`` per segment. Means are ``z_scale * tanh(.)``, so the sample
    cannot leave COIN's dynamic range. ``prior_sd`` is both the ``k = 0`` posterior and the
    KL reference.

    **Reward is absent from the features by default.** The contingency is a property of
    the DYNAMICS, and reward is the PPO learning signal: feeding it here would let the
    encoder identify a task from its reward scale rather than from how the world responds
    to actions, which is exactly the shortcut the experiment was first designed to exclude
    (the recurrent-PPO baseline is the one granted reward-as-context). ``use_reward=True``
    (decision reversed 2026-08-31: the on-policy pilot showed the dynamics-only signal
    under-constrains ``z`` wherever state or policy already identify the task) inserts the
    step reward between the action and ``s'``: feature order ``(s, a_repr, r, s')``, so
    ``in_dim = 2 * obs_dim + act_dim + 1``, and ``s'`` stays the LAST ``obs_dim`` columns
    (the slice :meth:`AmortisedCOINPPOAgent._dyn_loss` and the replay pipeline rely on).
    """

    def __init__(self, obs_dim: int, act_dim: int, action_continuous: bool, hidden: int = 64,
                 z_scale: float = 0.15, prior_sd: float = 1.0, sigma_min: float = 0.05,
                 sigma_max: float = 20.0, use_reward: bool = False):
        super().__init__()
        self.obs_dim, self.act_dim = int(obs_dim), int(act_dim)
        self.action_continuous = bool(action_continuous)
        self.use_reward = bool(use_reward)
        self.z_scale, self.prior_sd = float(z_scale), float(prior_sd)
        self.log_sigma_min, self.log_sigma_max = float(np.log(sigma_min)), float(np.log(sigma_max))
        self.in_dim = 2 * self.obs_dim + self.act_dim + (1 if self.use_reward else 0)
        self.net = _MLP(self.in_dim, 2, hidden)

        # A zero final layer makes every mu_t exactly zero at initialisation, so an untrained
        # encoder emits a constant z and buys COIN no spurious contexts. The dynamics
        # gradient survives it, because tanh'(0) = 1.
        last = self.net.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        with torch.no_grad():
            last.bias[1] = 1.0    # sigma ~ 2.72: uninformative, well inside the clamps

    def transition_features(self, obs, act, next_obs, rew=None) -> torch.Tensor:
        """``[T, in_dim]`` inputs ``concat(s, a_repr, s')`` -- or ``(s, a_repr, r, s')``
        with ``use_reward`` -- where ``next_obs`` is the post-step observation, captured
        before any episode reset. See the class docstring for the reward decision."""
        dev = next(self.parameters()).device
        obs = obs.to(device=dev, dtype=torch.float32).view(-1, self.obs_dim)
        next_obs = next_obs.to(device=dev, dtype=torch.float32).view(-1, self.obs_dim)
        if self.action_continuous:
            act_repr = act.to(device=dev, dtype=torch.float32).view(-1, self.act_dim)
        else:
            idx = act.to(device=dev).long().view(-1)
            act_repr = torch.nn.functional.one_hot(idx, self.act_dim).float()
        if not self.use_reward:
            return torch.cat([obs, act_repr, next_obs], dim=-1)
        if rew is None:
            raise ValueError("use_reward encoder needs the step rewards")
        r = torch.as_tensor(rew, device=dev, dtype=torch.float32).view(-1, 1)
        return torch.cat([obs, act_repr, r, next_obs], dim=-1)

    def factors(self, feats: torch.Tensor):
        """Per-transition Gaussian factors ``mu[T]``, ``sigma[T]``."""
        out = self.net(feats)
        log_sigma = out[..., 1].clamp(self.log_sigma_min, self.log_sigma_max)
        return self.z_scale * torch.tanh(out[..., 0]), torch.exp(log_sigma)

    def prefix_posterior(self, feats: torch.Tensor, seg_len: int):
        """
        Prefix posteriors over ``z`` for segment-major ``feats`` ``[S * seg_len, in_dim]``.

        Accumulating the natural parameters with a ``cumsum`` along an ``[S, seg_len]`` view
        makes cross-segment leakage structurally impossible: the posterior RESTARTS at every
        boundary. The prior precision is added after the cumsum, never inside it, so column 0
        of every row is exactly the prior. Returns ``mean``/``sd`` ``[S, seg_len + 1]``,
        column ``k`` being the posterior after ``k`` transitions; segments must be equal
        length.
        """
        seg_len = int(seg_len)
        if feats.shape[0] % seg_len:
            raise ValueError(f"{feats.shape[0]} transitions is not a multiple of {seg_len}.")
        mu, sigma = self.factors(feats)
        inv_var = 1.0 / (sigma * sigma)
        eta2, eta1 = inv_var.view(-1, seg_len), (mu * inv_var).view(-1, seg_len)
        zero = eta2.new_zeros(eta2.shape[0], 1)
        sum2 = torch.cat([zero, eta2.cumsum(dim=1)], dim=1)
        sum1 = torch.cat([zero, eta1.cumsum(dim=1)], dim=1)
        var = 1.0 / (sum2 + 1.0 / (self.prior_sd ** 2))
        return sum1 * var, torch.sqrt(var)


class FiLMDecoder(nn.Module):
    """
    Linearly modulated dynamics decoder, ``s' = f(s, a) + z * g(s, a)``.

    The concat decoder ``h([s, a, z])`` can represent the latent's effect through any
    smooth warping of ``z``, which is exactly the gauge freedom that lets the encoder
    relabel its whole code map at no cost to the dynamics loss. Restricting the latent to a
    LINEAR modulation of a state-dependent direction (FiLM; Perez et al., AAAI 2018, and
    the same trick CoDA-style contextual dynamics models use) removes most of that freedom:
    ``z`` can now only scale ``g``, so the map is pinned up to one affine reparameterisation
    instead of an arbitrary one.

    That residual affine gauge is real -- ``z -> a*z`` with ``g -> g/a`` is exactly
    equivalent -- and ``normalise=True`` closes it by renormalising ``g``'s output layer to
    a fixed norm on every forward pass, so scaling ``g`` down is no longer available and the
    scale of ``z`` becomes observable.

    Args:
        in_dim (int): Width of the ``(s, a_repr)`` input.
        out_dim (int): Observation dimension being predicted.
        hidden (int): Hidden width of both branches.
        normalise (bool): Fix the norm of ``g``'s output layer (kills the affine gauge).
        g_norm (float): The norm to fix it to.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64,
                 normalise: bool = False, g_norm: float = 1.0):
        super().__init__()
        self.f = _MLP(in_dim, out_dim, hidden)
        self.g = _MLP(in_dim, out_dim, hidden)
        self.normalise, self.g_norm = bool(normalise), float(g_norm)

    def _g_out(self, sa: torch.Tensor) -> torch.Tensor:
        if not self.normalise:
            return self.g(sa)
        # Renormalise the final layer's weight (and bias with it) to a fixed norm. Done
        # functionally so the parameter itself stays free and the gradient still flows.
        last = self.g.net[-1]
        scale = self.g_norm / last.weight.detach().norm().clamp_min(1e-8)
        body = sa
        for layer in self.g.net[:-1]:
            body = layer(body)
        return torch.nn.functional.linear(body, last.weight * scale, last.bias * scale)

    def forward(self, sa: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.f(sa) + z.unsqueeze(-1) * self._g_out(sa)


class SegmentReplayBuffer:
    """
    Reservoir pool of per-segment encoder features ``[L, in_dim]``, held on CPU, each with
    a stored **code anchor** and the reservoir counter it was pushed at.

    The dynamics objective is off-policy, so old segments stay valid targets. Replaying
    them decouples the encoder's training data from whatever tasks the latest rollout
    happened to contain, which is what stops the latent from drifting with the curriculum.
    Capacity is counted in SEGMENTS, since a prefix posterior is a whole-segment quantity.

    Retention is Vitter's **Algorithm R**, not FIFO: the first ``capacity`` segments are
    kept, and segment ``n`` (0-based, ``n >= capacity``) replaces a uniformly drawn slot
    ``j = randint(0, n)`` only when ``j < capacity``. Every segment ever pushed is therefore
    held with the same probability ``capacity / n_seen``, so the pool stays a uniform sample
    of the WHOLE stream. That property is the point on a blocked curriculum: a FIFO pool of
    512 segments is flushed by ~64 rollouts of the current task, after which the encoder has
    no evidence of the earlier tasks left and its latent drifts off them; a reservoir keeps
    a thinning-but-never-vanishing trace of every block.

    Replay alone keeps the DYNAMICS of old tasks predictable; it does not keep their
    *code* -- the encoder is free to relabel every segment at once, and a rotation of the
    whole ``z`` map costs the dynamics objective nothing (the decoder rotates with it).
    That is why each segment also carries ``anchors[i]``: the segment-final posterior mean
    the encoder assigned it, used by
    :meth:`AmortisedCOINPPOAgent._update_encoder` as a self-supervised pin. ``push_ids[i]``
    is the value of :attr:`n_seen` at push time, which is how
    :meth:`AmortisedCOINPPOAgent._settle_anchors` tells a still-arriving task's segments
    (free to move) from settled ones (pinned).
    """

    def __init__(self, capacity: int = 128):
        self.capacity = int(capacity)
        self.buffer: List[torch.Tensor] = []
        self.anchors: List[float] = []   # segment-final z when it settled; nan while free
        self.push_ids: List[int] = []    # n_seen at push, i.e. the segment's arrival order
        self.group_ids: List[int] = []   # segments known to share a task; see push()
        self.n_seen = 0                  # segments pushed since the last length change

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        """Empty the pool and restart the reservoir count."""
        self.buffer.clear()
        self.anchors.clear()
        self.push_ids.clear()
        self.group_ids.clear()
        self.n_seen = 0

    def push(self, feats: torch.Tensor, anchor: Optional[float] = None,
             group: Optional[int] = None) -> None:
        """Store one segment's features, its code anchor and its group under Algorithm R.

        ``anchor`` is the encoder's CURRENT segment-final posterior mean for this segment,
        computed by the caller under ``no_grad``; ``None`` stores ``nan``, which
        :meth:`AmortisedCOINPPOAgent._update_encoder` reads as "not anchored".

        ``group`` marks segments the CALLER knows came from the same task, without saying
        which task -- the label-free "two views of one thing" that
        :meth:`AmortisedCOINPPOAgent._dispersion_loss` needs to tell within-task spread from
        between-task spread. ``None`` gives the segment a group of its own, which makes
        every group a singleton and the invariance term identically zero, so a caller that
        cannot make the guarantee simply gets the old behaviour.

        A change of segment length empties the pool (and restarts the reservoir count):
        :meth:`ContingencyEncoder.prefix_posterior` needs one common length.
        """
        feats = feats.detach().cpu()
        if self.buffer and self.buffer[-1].shape[0] != feats.shape[0]:
            self.clear()
        a = float("nan") if anchor is None else float(anchor)
        g = int(self.n_seen) if group is None else int(group)
        if len(self.buffer) < self.capacity:
            self.buffer.append(feats)
            self.anchors.append(a)
            self.push_ids.append(self.n_seen)
            self.group_ids.append(g)
        else:
            # j uniform on [0, n_seen]; the reservoir keeps the new segment with
            # probability capacity / (n_seen + 1), which is Algorithm R's invariant.
            j = int(np.random.randint(0, self.n_seen + 1))
            if j < self.capacity:
                self.buffer[j] = feats
                self.anchors[j] = a
                self.push_ids[j] = self.n_seen
                self.group_ids[j] = g
        self.n_seen += 1

    def sample_balanced(self, n_segments: int, recent_window: int):
        """
        ``(feats, anchors)`` with HALF the draw taken from the newest ``recent_window``
        segments and half uniformly from the whole reservoir.

        A uniform draw from a reservoir gives a task a share of the gradient equal to its
        share of the STREAM SO FAR. On the Figure-3 blocked curriculum that is ruinous for
        whoever arrives last: by the CartPole block the pool is ~90% earlier tasks, so the
        gradient that has to carve out a CartPole-vs-Inverted-CartPole distinction arrives
        at roughly a tenth of the strength it would have under joint training -- where the
        same encoder separates that pair perfectly well.

        Reserving half the minibatch for the newest segments guarantees every task a fixed
        gradient share for as long as it is arriving, whatever the pool composition. It
        needs no task labels and no boundaries: "newest" is a property of the reservoir's
        own arrival counter, which is why it is safe here. The uniform half still covers the
        whole stream, so the anchor keeps seeing (and holding) the old codes.

        Falls back to a plain uniform draw when the pool has too few recent segments.
        """
        n = len(self.buffer)
        if n == 0:
            raise RuntimeError("empty replay pool")
        want = int(n_segments)
        cutoff = self.n_seen - int(recent_window)
        recent = [i for i in range(n) if self.push_ids[i] >= cutoff]

        n_recent = min(len(recent), want // 2)
        idx: List[int] = []
        if n_recent:
            pick = torch.randperm(len(recent))[:n_recent]
            idx += [recent[int(i)] for i in pick]
        rest = [i for i in range(n) if i not in set(idx)]
        need = min(want - len(idx), len(rest))
        if need > 0:
            pick = torch.randperm(len(rest))[:need]
            idx += [rest[int(i)] for i in pick]

        feats = torch.cat([self.buffer[i] for i in idx])
        anchors = torch.tensor([self.anchors[i] for i in idx], dtype=torch.float32)
        return feats, anchors

    def code_clusters(self, gap: float):
        """
        Single-linkage clusters of the SETTLED segments by stamped code, as a list of index
        lists. One dimension makes this exact and free: sort the stamped values and cut
        wherever consecutive ones differ by more than ``gap``.

        Label-free by construction -- it reads only the codes the encoder itself stamped,
        never a task id. Unsettled (``nan``) segments are excluded; they are not anchored.
        """
        idx = [i for i in range(len(self.buffer)) if np.isfinite(self.anchors[i])]
        if not idx:
            return []
        idx.sort(key=lambda i: self.anchors[i])
        out, cur = [], [idx[0]]
        for prev, i in zip(idx, idx[1:]):
            if self.anchors[i] - self.anchors[prev] > float(gap):
                out.append(cur)
                cur = []
            cur.append(i)
        out.append(cur)
        return out

    def sample_code_groups(self, n_segments: int, gap: float, per_group: int = 2):
        """
        ``(feats, anchors)`` drawn CLUSTER-STRATIFIED over stamped codes.

        The centroid anchor needs several members of one code cluster in the same minibatch
        to have a centroid to speak of; a uniform draw of four from a pool of hundreds
        almost never supplies them, and the penalty then silently degenerates to the
        per-segment form it is meant to replace. Draws ``per_group`` segments from each of
        ``n_segments // per_group`` randomly chosen clusters, and tops up uniformly when
        there are too few settled segments.
        """
        clusters = self.code_clusters(gap)
        want = int(n_segments)
        idx: List[int] = []
        if clusters:
            n_g = max(1, want // int(per_group))
            pick = torch.randperm(len(clusters))[:n_g]
            for c in pick:
                members = clusters[int(c)]
                take = torch.randperm(len(members))[:int(per_group)]
                idx += [members[int(t)] for t in take]
        if len(idx) < want:
            rest = [i for i in range(len(self.buffer)) if i not in set(idx)]
            need = min(want - len(idx), len(rest))
            if need > 0:
                take = torch.randperm(len(rest))[:need]
                idx += [rest[int(t)] for t in take]
        if not idx:
            raise RuntimeError("empty replay pool")
        feats = torch.cat([self.buffer[i] for i in idx])
        anchors = torch.tensor([self.anchors[i] for i in idx], dtype=torch.float32)
        return feats, anchors

    def sample_groups(self, n_groups: int, per_group: int = 2):
        """
        ``(feats, anchors, group_index)`` drawn GROUP-WISE: ``n_groups`` distinct groups,
        up to ``per_group`` segments from each.

        The structured draw the VICReg terms need -- a plain uniform sample of four
        segments from a pool of hundreds almost never contains two members of one group, so
        a within-group statistic computed on it would be undefined nearly every step.
        ``group_index`` is 0-based and contiguous over the groups actually drawn, so it can
        index a scatter directly. Groups with a single member still contribute to the
        between-group statistic; they just carry no within-group variance.
        """
        by_group: Dict[int, List[int]] = defaultdict(list)
        for i, g in enumerate(self.group_ids):
            by_group[g].append(i)
        keys = list(by_group)
        if not keys:
            raise RuntimeError("empty replay pool")
        chosen = [keys[int(i)] for i in torch.randperm(len(keys))[:int(n_groups)]]

        idx, gidx = [], []
        for k, g in enumerate(chosen):
            members = by_group[g]
            take = torch.randperm(len(members))[:int(per_group)]
            for t in take:
                idx.append(members[int(t)])
                gidx.append(k)
        feats = torch.cat([self.buffer[i] for i in idx])
        anchors = torch.tensor([self.anchors[i] for i in idx], dtype=torch.float32)
        return feats, anchors, torch.tensor(gidx, dtype=torch.long)

    def sample(self, n_segments: int):
        """``(feats, anchors)``: ``[n * L, in_dim]`` segment-major features sampled
        uniformly without replacement, and the matching ``[n]`` float32 anchors."""
        if not self.buffer:
            raise RuntimeError("sample from an empty SegmentReplayBuffer")
        idx = torch.randperm(len(self.buffer))[:int(n_segments)]
        feats = torch.cat([self.buffer[int(i)] for i in idx])
        anchors = torch.tensor([self.anchors[int(i)] for i in idx], dtype=torch.float32)
        return feats, anchors


class AmortisedCOINPPOAgent(COINPPOAgent):
    """
    COIN-PPO whose context observation is inferred, not supplied.

    A :class:`ContingencyEncoder` compresses a segment's transitions into a posterior over a
    scalar latent contingency ``z``; a decoder ``f_psi(s, a, z) -> s'`` gives that latent
    something to mean. Encoder and decoder are trained together on
    ``L_dyn (MSE) + kl_coef * KL(q || N(0, prior_sd^2))`` -- a self-supervised objective that
    has a gradient from the first step, unlike the value-based training it replaces (there
    the PPO loss reached ``phi`` only through a differentiable responsibility, whose
    derivative is identically zero whenever the context heads agree, which is exactly the
    case at onset and at every context instantiation, since new heads are deep copies).

    ``z`` is therefore stop-gradiented out of PPO entirely. COIN sits in the loop but is
    never differentiated through: the agent acts on COIN's one-step-ahead predicted context
    probabilities (constant for a segment), and COIN observes the segment-final posterior
    mean of ``z``. Responsibilities are diagnostics only.

    **One rollout is S contiguous fixed-length segments, each a different task and its own
    COIN trial** (see :meth:`train_step`). Interleaving the tasks is what forces ``z`` to be
    informative: with one task per rollout a constant ``z`` predicts the dynamics just as
    well, whereas across interleaved tasks a single decoder cannot fit contradictory
    dynamics unless ``z`` splits them. The task switches at a boundary; the EPISODE does
    not -- it is carried into the next segment, so a switch is a perturbation of an ongoing
    episode rather than a fresh start.

    Args:
        encoder_hidden (int): Hidden width of both the encoder and the decoder MLP.
        z_scale (float): Bound on ``|mu_t|``, i.e. COIN's dynamic range for the latent.
        prior_sd (float): Prior sd, used both as the empty-prefix posterior and as the KL
            reference.
        avoid_novel (bool): Drop the novel column from the acting weights whenever the known
            contexts carry mass; see :meth:`_policy_weights`.
        kl_coef (float): Weight of the KL term in the encoder objective. Keep it above zero:
            :meth:`train_step` hands the posterior sd to COIN as sensory noise, and an
            unpenalised sd is not calibrated.
        replay_capacity (int): Size of the encoder's :class:`SegmentReplayBuffer`, in
            segments. The pool is a reservoir, so it stays a uniform sample of the whole
            stream rather than a window on the newest block.
        anchor_mode (str): Which stability penalty the stamped anchor applies --
            ``"value"`` (default, pin each code), ``"centroid"`` (pin each code cluster's
            mean, leaving symmetric local splits free) or ``"distance"`` (one-sided hinge
            against contraction plus a translation term). See :meth:`_anchor_penalty`.
        anchor_group_gap (float): Single-linkage gap that separates two code clusters, used
            by the ``"centroid"`` mode. A few times the motor-noise floor: codes closer
            than this are ones COIN could not tell apart anyway.
        decoder_arch (str): ``"concat"`` (default) feeds ``[s, a, z]`` to one MLP;
            ``"film"`` uses :class:`FiLMDecoder`, ``f(s, a) + z * g(s, a)``, which removes
            most of the gauge freedom the code anchors were fighting -- see that class.
        film_normalise (bool): For ``decoder_arch="film"``, fix the norm of ``g``'s output
            layer, closing the residual affine gauge ``z -> a*z, g -> g/a``.
        decoder_lr_ratio (float): Decoder learning rate is ``encoder_lr / this``. Above one
            it slows the decoder relative to the encoder, so the decoder can no longer
            chase a relabelled latent as fast as the encoder can relabel it -- gauge
            control at the source rather than a penalty on the codes. ``1.0`` reproduces
            the single-rate behaviour exactly.
        balanced_replay (bool): Draw half of every encoder minibatch from the newest
            ``anchor_window`` segments and half uniformly from the reservoir
            (:meth:`SegmentReplayBuffer.sample_balanced`). Off by default. Addresses
            reservoir dilution: under a uniform draw a late-arriving task's gradient share
            equals its share of the stream so far, which on the blocked curriculum leaves
            the CartPole pair a fraction of the signal that separates it under joint
            training. Task-agnostic -- "newest" comes from the reservoir's arrival counter,
            not from any label or boundary.
        anchor_ema_tau (Optional[float]): Set this to run the anchor in **EMA-teacher**
            mode, the standard continual-self-supervised form: a trailing copy of the
            encoder is kept as ``theta_ema <- tau * theta_ema + (1 - tau) * theta`` after
            every optimiser step, and the anchor target for a replayed segment is that
            teacher's code for the same segment, recomputed each step under ``no_grad``.

            This is the mode to prefer, and not only on results. The stamping alternative
            needs ``anchor_warmup`` to outlast the first task's learning, and the only
            value that worked was one calibrated to the length of the MountainCar block --
            task-boundary information smuggled into a method whose headline claim is that
            it is given none. ``tau`` is a generic timescale with no task content: the
            teacher trails from step 0, so no early code is ever frozen, and drift is
            RATE-LIMITED rather than forbidden. In this mode ``anchor_window``,
            ``anchor_warmup`` and the stored per-segment stamps are all inert.
        anchor_coef (float): Weight of the **code-stability anchor** on replayed segments
            (see :meth:`_update_encoder`). Zero reproduces the unanchored objective. The
            dynamics MSE runs at ``1e-5`` to ``1e-3``, so the term is doing real work well
            below one; the default is the encoder-only study's pick at
            ``anchor_window = 64``, where ``0.3`` drifts a little more (0.025) with a wider
            map and ``3.0`` starts to flatten it.
        anchor_window (int): A pooled segment carries NO anchor while it is among the last
            ``anchor_window`` segments pushed; beyond that its code is stamped once and
            pinned. Counted in segments, so ``8 * n_rollouts``. Pin late, pin right: on the
            shrunken stream, widening it from 32 to 64 segments both cut the drift (0.23 ->
            0.06) and widened the separation (0.57 -> 0.93), because a code stamped before
            it has settled leaves a residual force that then drags it anyway.
        value_coef (float): Weight of the **value-gradient encoder term**. Revives the
            original pre-dynamics training signal (see the b3dfced docstring for why it
            was once dropped): per segment, responsibilities are recomputed
            DIFFERENTIABLY from the encoder's posterior and COIN's per-context Gaussians
            (all COIN quantities constants), the context heads' value estimates are
            mixed under them (head outputs detached), and the mismatch against the PPO
            return targets is backpropagated into the ENCODER alone. Its gradient is
            zero while the heads agree -- the cold-start failure that killed it as a
            sole objective -- but that is exactly where ``L_dyn`` is strong; and it is
            NONZERO wherever different heads value the same states differently, which is
            exactly where ``L_dyn`` goes blind (state-identifiable task pairs, and
            on-policy action-support divergence: the pilot's rail collapse). The two
            terms are complements, not alternatives. Zero (default) disables it.
        observe_value (bool): COIN observes ``(z, mean episodic return)`` as a 2-D
            vector instead of ``z`` alone, via realtimecoin's MD pipeline. Performance
            becomes part of what a context IS: an arriving task that parks at an
            established code still collapses the return dimension many sigma below
            that context's history, so COIN births a new context BEFORE the old head
            is overwritten -- detect-before-adapt inside the inference model. The
            return is read only from episodes that END inside the segment (their mean,
            divided by ``value_obs_scale``); a segment that finishes none observes
            ``(z, nan)`` and the MD pipeline conditions on ``z`` alone. The same nan
            masking makes the within-segment step weights and the self-identifying
            eval use the ``z`` marginal automatically (a return does not exist
            mid-episode). The notebook must construct the matching model:
            ``RealTimeCOIN(state_dim=2,
            process_noise_covariance=np.diag([sigma_process_noise**2,
            value_process_noise**2]), ...)`` with ``prior_mean_retention`` as
            currently used -- the value dim's process noise is its learning-curve
            drift budget, so within-block improvement tracks as drift while a
            block-switch collapse triggers a birth. The MD pipeline computes
            likelihoods in log space, so it is also immune to the scalar path's
            extreme-surprise underflow.
        value_obs_scale (float): Return normaliser for the value dimension (200 maps
            the classic-control range onto ~[-1, 1]).
        value_obs_noise_floor (float): Minimum observation noise on the value
            dimension, folded in quadrature with the across-episode standard error --
            the value-dim counterpart of ``sigma_motor_noise``.
        value_process_noise (float): Documented per-trial drift budget for the value
            dimension; read by the notebook when building the 2-D COIN (the agent
            itself never constructs COIN models).
        repel_coef (float): Weight of the **value-surprise repulsion** -- the objective
            fix for head-level parking. The mixture-form value term
            (``value_coef``) is a boundary refiner: its gradient carries a
            ``w(1 - w)`` factor, so it vanishes whenever routing is CONFIDENT --
            including confidently wrong, which is exactly what parking produces (an
            arriving task at an established context's centre). This term uses the
            same value evidence through a path with no saturation: when the routed
            context is ESTABLISHED and its head's value error on a segment explodes
            past ``repel_gate_ratio`` times that context's running baseline
            ("massive now, was not before"), the segment's code is pushed a full
            ``repel_margin`` away from that context's centre via
            ``relu(margin - |z - mu_c|)^2`` inside :meth:`_update_encoder` -- direct
            distance geometry, full per-rollout step budget, no alternative head
            required. A newborn context has no baseline, so the gate structurally
            cannot fire on it: a fresh head's large error is expected learning, not
            surprise. Zero (default) disables everything.
        repel_margin (float | None): Push-to distance for the repulsion hinge; None
            (default) uses ``2 * z_channel_noise`` -- one channel-enforced slot -- or
            ``0.4 * z_scale`` when the channel is off.
        repel_gate_ratio (float): Value-error blow-up factor over the context's
            baseline that flags a segment as mis-parked.
        vloss_ema_tau (float): EMA rate of the per-context value-error baseline;
            updated only on NON-flagged segments so the anomaly cannot poison the
            baseline it is measured against.
        rail_coef (float): Weight of the **rail hinge** keeping codes out of the tanh
            saturation zone: ``relu(|mean| / z_scale - 0.8)^2`` on segment-final means.
            The raw-z pilot showed the rails are a code graveyard -- tanh' vanishes
            there, so neither the dynamics loss nor the value-gradient term can ever
            move a saturated code again, and arriving tasks park on it (CP and Mirror
            both froze at +2). The hinge is zero on 80% of the range, so it costs
            legitimate spread nothing. Zero (default) disables it.
        encoder_reward (bool): Insert the RAW step reward into the encoder features
            (layout ``(s, a, r, s')``) and add a reward-prediction head to the decoder
            -- reward is a decoder TARGET, never a decoder input (an input would revive
            the identify-from-anything-but-z shortcut). Reverses the original
            reward-free decision: on-policy, dynamics prediction alone under-constrains
            ``z`` wherever state or policy already identify the task, while reward
            structure is task-specific there. Callers must supply raw rewards
            (``info['raw_reward']`` on shaped envs) so a task's code cannot differ
            between its shaped training stream and its raw evaluation. Note for the
            paper: this grants the encoder reward-as-context, the concession the
            recurrent-PPO baseline already enjoys.
        decoder_residual (bool): The decoder predicts the state CHANGE and the model
            adds it back: ``s' = s + f_psi(s, a, z)`` (dt folded into ``f``). Standard
            stabiliser for learned dynamics -- the identity map is the zero function, so
            an untrained decoder starts as "nothing changes" instead of an arbitrary
            state, and the network models the (small, contingency-carrying) per-step
            delta rather than reproducing the (large) state.
        quantile_readout (bool): COIN observes ``probit(rank(z))`` -- the code's rank
            within a re-encoded replay-buffer subsample, mapped through the inverse
            normal CDF -- instead of raw ``z``. The rank is the maximal invariant of the
            scalar gauge group (any invertible relabelling of a scalar latent is
            monotone, and monotone maps preserve order), so a context's observed address
            survives every warp of the code axis that the decoder can absorb; the
            encoder-only study measured a 30-300x drift reduction over raw ``z``, with
            the z-score readout (affine-only invariance) refuted in between. The probit
            makes the observed population approximately standard normal and keeps the
            observation unbounded, which is what COIN's Gaussian observation model
            expects. Posterior sd is pushed through the same map (half the transformed
            ``z +- sd`` interval). Until the buffer holds 8 segments the transform is
            the identity -- the handful of trials COIN sees in raw space sit at the very
            start of the first block and are re-contextualised within a few rollouts.
        quantile_pop_size (int): Buffer segments re-encoded per refresh to form the rank
            population. Subsampling adds ~``0.1`` within-task-sd of observation jitter
            at 64; it shrinks as ``1/sqrt(n)``.
        z_channel_noise (float): Extra noise sd added in quadrature to the posterior sd
            when sampling ``z`` for the decoder during :meth:`_update_encoder` -- a noisy
            channel between encoder and decoder. With it, two contexts whose codes sit
            within the noise radius are indistinguishable to the decoder, so where their
            dynamics actually differ the dynamics loss itself pushes the codes at least
            one noise radius apart; contexts with identical dynamics may still merge,
            which costs nothing downstream. Set it to the resolution COIN needs (its
            sensory-noise floor) so the encoder is trained against the same metric COIN
            classifies with. Zero (default) reproduces the plain objective.
        disp_coef (float): Weight of the **dispersion** term (see :meth:`_update_encoder`).
            Zero disables it. The anchor makes codes stationary but cannot make them
            distinct: measured across four configurations, once the map stops drifting a
            new task simply parks on an old code, because parking is what the dynamics loss
            prefers. This is the term that asks for separation.
        disp_target_sd (float): Standard deviation of the per-group code means that the
            dispersion hinge aims for. Five well-spread codes in ``[-z_scale, z_scale]``
            have ``sd ~ 0.63 * z_scale``, so ``1.0`` is a modest target at ``z_scale = 2``.
        inv_coef (float): Weight of the within-group invariance term. Without it the
            variance hinge is bought with within-task code noise rather than between-task
            separation -- measured, not hypothesised (see :meth:`_dispersion_loss`).
        disp_per_group (int): Segments drawn per group by
            :meth:`SegmentReplayBuffer.sample_groups`. Two is enough for a within-group
            variance and keeps the encoder minibatch at ``2 * mb_segments``.
        same_task_rollout (bool): The caller guarantees that all segments of one
            :meth:`train_step` call come from the SAME task, so they may share a replay
            group. True for the Figure-3 blocked harness (eight envs of one task per call);
            it MUST stay False for the interleaved mode, where every segment is a different
            task.
        anchor_warmup (int): Segments to push before pinning starts at all. Below it every
            anchor is cleared each update, so the objective is exactly the unanchored one.

            **The default is 0, and deliberately so.** A non-zero warm-up has to be long
            enough to outlast the first task's own learning, and the only value that worked
            on the blocked stream was one calibrated to the MountainCar block's length --
            task-boundary information smuggled into a method whose headline claim is that
            it is given none. The encoder-only study showed the warm-up is not needed:
            ``anchor_window`` alone (a generic "segments of new data before a code counts as
            settled" timescale, carrying no task information) reaches a post-block drift of
            0.008 with settled-code velocity 0.004, inside the 0.01 motor-noise floor, while
            still letting the map separate.
        **kwargs: Forwarded to :class:`COINPPOAgent`.
    """

    def __init__(self, env: gym.Env, ctx_ids: dict, encoder_hidden: int = 64,
                 encoder_lr: float = 3e-4, enc_grad_clip: Optional[float] = 1.0,
                 z_scale: float = 0.15, prior_sd: float = 1.0, avoid_novel: bool = True,
                 kl_coef: float = 1e-3, replay_capacity: int = 128,
                 dyn_obs_norm: bool = False, anchor_coef: float = 1.0,
                 anchor_window: int = 64, anchor_warmup: int = 0,
                 disp_coef: float = 0.0, disp_target_sd: float = 1.0,
                 inv_coef: float = 0.0, disp_per_group: int = 2,
                 same_task_rollout: bool = False,
                 anchor_ema_tau: Optional[float] = None,
                 balanced_replay: bool = False, anchor_mode: str = "value",
                 anchor_group_gap: float = 0.05,
                 anchor_strat_sampling: Optional[bool] = None,
                 decoder_arch: str = "concat", film_normalise: bool = False,
                 decoder_lr_ratio: float = 1.0, z_channel_noise: float = 0.0,
                 quantile_readout: bool = False, quantile_pop_size: int = 64,
                 value_coef: float = 0.0, decoder_residual: bool = False,
                 encoder_reward: bool = False, rail_coef: float = 0.0,
                 repel_coef: float = 0.0, repel_margin: Optional[float] = None,
                 repel_gate_ratio: float = 4.0, vloss_ema_tau: float = 0.9,
                 observe_value: bool = False, value_obs_scale: float = 200.0,
                 value_obs_noise_floor: float = 0.05,
                 value_process_noise: float = 0.01,
                 **kwargs):
        super().__init__(env, ctx_ids, **kwargs)
        self.z_scale, self.prior_sd = float(z_scale), float(prior_sd)
        self.avoid_novel, self.kl_coef = bool(avoid_novel), float(kl_coef)
        self.dyn_obs_norm = bool(dyn_obs_norm)
        self.enc_grad_clip = enc_grad_clip
        self.anchor_coef = float(anchor_coef)
        self.anchor_window = int(anchor_window)
        self.anchor_warmup = int(anchor_warmup)
        self.anchor_ema_tau = None if anchor_ema_tau is None else float(anchor_ema_tau)
        self.balanced_replay = bool(balanced_replay)
        if anchor_mode not in ("value", "centroid", "distance"):
            raise ValueError(f"anchor_mode must be value/centroid/distance, got {anchor_mode!r}")
        self.anchor_mode = str(anchor_mode)
        self.anchor_group_gap = float(anchor_group_gap)
        # Cluster-stratified draws are ON by default only for the centroid form, which
        # needs them. Set explicitly to control for the sampling change on its own.
        self.anchor_per_group = 2      # members drawn per code cluster; see _update_encoder
        self.anchor_strat_sampling = (self.anchor_mode == "centroid"
                                      if anchor_strat_sampling is None
                                      else bool(anchor_strat_sampling))
        self.z_channel_noise = float(z_channel_noise)
        self.quantile_readout = bool(quantile_readout)
        self.quantile_pop_size = int(quantile_pop_size)
        self.value_coef = float(value_coef)
        self.decoder_residual = bool(decoder_residual)
        self.encoder_reward = bool(encoder_reward)
        self.rail_coef = float(rail_coef)
        self.repel_coef = float(repel_coef)
        self.repel_margin = None if repel_margin is None else float(repel_margin)
        self.repel_gate_ratio = float(repel_gate_ratio)
        self.vloss_ema_tau = float(vloss_ema_tau)
        self.observe_value = bool(observe_value)
        self.value_obs_scale = float(value_obs_scale)
        self.value_obs_noise_floor = float(value_obs_noise_floor)
        self.value_process_noise = float(value_process_noise)
        self._vloss_ema: Dict[Any, float] = {}   # per-context value-error baseline
        self._repel_batch: List[Tuple[torch.Tensor, float]] = []
        self._code_pop: Optional[np.ndarray] = None   # sorted; see _refresh_code_population
        self.disp_coef = float(disp_coef)
        self.disp_target_sd = float(disp_target_sd)
        self.inv_coef = float(inv_coef)
        self.disp_per_group = int(disp_per_group)
        self.same_task_rollout = bool(same_task_rollout)
        self._group_counter = 0
        self.encoder = ContingencyEncoder(
            self.obs_dim, self.act_dim, self.action_continuous, hidden=encoder_hidden,
            z_scale=z_scale, prior_sd=prior_sd,
            use_reward=self.encoder_reward).to(self.device)
        if decoder_arch not in ("concat", "film"):
            raise ValueError(f"decoder_arch must be concat/film, got {decoder_arch!r}")
        self.decoder_arch = str(decoder_arch)
        sa_dim = self.obs_dim + self.act_dim
        # With encoder_reward the decoder PREDICTS the reward as one extra output --
        # never receives it as input, which would hand back the identify-from-anything-
        # but-z shortcut the on-policy pilot exposed. Predicting it makes reward
        # differences one more thing only a well-placed z can explain.
        dec_out = self.obs_dim + (1 if self.encoder_reward else 0)
        if self.decoder_arch == "film":
            if self.encoder_reward:
                raise ValueError("encoder_reward supports decoder_arch='concat' only")
            self.decoder = FiLMDecoder(sa_dim, self.obs_dim, encoder_hidden,
                                       normalise=bool(film_normalise)).to(self.device)
        else:
            self.decoder = _MLP(sa_dim + 1, dec_out, encoder_hidden).to(self.device)

        # Decoder-side gauge control (C1-C3). The dynamics loss is invariant to any smooth
        # relabelling of z that the decoder can absorb, so slowing, anchoring or freezing
        # the DECODER removes the freedom at its source instead of penalising the codes --
        # which is what made the code anchors oppose pair separation.
        self.decoder_lr_ratio = float(decoder_lr_ratio)

        # Deliberately NOT in self._all_optimizers(): the encoder is trained by the dynamics
        # objective alone, and the PPO step must not touch it. Two param groups so the
        # decoder can be slowed (C1) or stopped (C3) without touching the encoder.
        self.enc_optim = optim.Adam(
            [{"params": list(self.encoder.parameters()), "lr": float(encoder_lr)},
             {"params": list(self.decoder.parameters()),
              "lr": float(encoder_lr) / max(self.decoder_lr_ratio, 1e-12)}])
        # EMA teacher: a trailing copy of the encoder that never receives gradients and is
        # never optimised. Built only in EMA mode, so the stamp path costs nothing.
        self.encoder_ema: Optional[ContingencyEncoder] = None
        if self.anchor_ema_tau is not None:
            self.encoder_ema = copy.deepcopy(self.encoder).to(self.device)
            for prm in self.encoder_ema.parameters():
                prm.requires_grad_(False)
            self.encoder_ema.eval()

        self.replay = SegmentReplayBuffer(replay_capacity)
        # The episode a segment stopped in, kept on the agent so a rollout boundary is no
        # different from any other segment boundary. See :meth:`_start_segment`.
        self._carry: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Small pieces
    # ------------------------------------------------------------------
    def ensure_contexts(self, K: int) -> None:
        """
        Instantiate a head for every aligned global context COIN holds. Keyed on the
        alignment ``K``, not on ``pi > 0``: a context with negligible predicted probability
        can still take responsibility later, and ``_mixed_logits``/``_mixed_value`` silently
        drop the mass of an uninstantiated head.
        """
        for j in range(min(int(K), self.num_contexts - 1)):
            cid = self.context_keys[j]
            if self.context_init.get(cid, 0) == 0:
                self._instantiate_context_net(cid)

    def _policy_weights(self, pi_agent) -> torch.Tensor:
        """
        Turn one agent-layout context vector into acting weights ``[num_contexts]``.

        Not differentiable and not meant to be -- these are COIN's own probabilities. NaN
        padding (uninstantiated slots) becomes zero before it can reach the ``_mixed_*``
        helpers, which do no sanitising of their own. With ``avoid_novel`` the novel column
        is dropped and the known contexts renormalised, EXCEPT when the known mass is
        negligible (trial 0, where ``K = 0`` puts everything on novel) -- dropping it there
        would leave an all-zero weight vector.
        """
        w = np.nan_to_num(np.asarray(pi_agent, dtype=np.float64), nan=0.0)
        if self.avoid_novel and w[:-1].sum() > 1e-6:
            w = np.concatenate([w[:-1] / w[:-1].sum(), [0.0]])
        return torch.as_tensor(w, dtype=torch.float32, device=self.device)

    def _logp_entropy(self, obs: torch.Tensor, act: torch.Tensor, w: torch.Tensor):
        """``log pi(act | obs)`` and mean entropy under the context-mixed heads."""
        if self.action_continuous:
            dist = torch.distributions.Normal(*self._mixed_gaussian(obs, w))
            return dist.log_prob(act).sum(-1), dist.entropy().sum(-1).mean()
        dist = torch.distributions.Categorical(logits=self._mixed_logits(obs, w))
        return dist.log_prob(act), dist.entropy().mean()

    def _kl_to_prior(self, mean: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
        """``KL(N(mean, sd^2) || N(0, prior_sd^2))``, elementwise."""
        p2 = self.prior_sd ** 2
        return np.log(self.prior_sd) - torch.log(sd) + (sd * sd + mean * mean) / (2.0 * p2) - 0.5

    def _decode_next_obs(self, feats: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Predicted ``s'`` from ``(s, a_repr, z)``. The reward column is deliberately left
        out: only the state transition carries the contingency."""
        return self._decode_full(feats, z)[0]

    def _decode_full(self, feats: torch.Tensor, z: torch.Tensor):
        """``(predicted s', predicted reward or None)`` from ``(s, a_repr, z)``.

        The reward column of ``feats`` (encoder_reward layout ``(s, a, r, s')``) is an
        encoder input and a decoder TARGET, never a decoder input; the residual form
        applies to the state head only."""
        sa = feats[:, :self.obs_dim + self.act_dim]
        if self.decoder_arch == "film":
            out = self.decoder(sa, z)
        else:
            out = self.decoder(torch.cat([sa, z.unsqueeze(-1)], dim=-1))
        r_hat = out[:, -1] if self.encoder_reward else None
        s_hat = out[:, :self.obs_dim]
        if self.decoder_residual:
            s_hat = feats[:, :self.obs_dim] + s_hat
        return s_hat, r_hat

    def _dyn_loss(self, feats: torch.Tensor, z: torch.Tensor,
                  seg_len: Optional[int] = None) -> torch.Tensor:
        """Dynamics MSE, per-SEGMENT per-dimension normalised when :attr:`dyn_obs_norm`.

        Raw-unit MSE lets large-scale dimensions (positions, O(1)) drown the small-scale
        ones (velocities) that carry most of the contingency signal -- MountainCar's
        amplitude term is O(1e-3) in raw velocity units, invisible next to an O(1)
        position residual. And because the padded interface makes tasks SHARE dimensions
        (MountainCar's velocity is CartPole's cart velocity at ~100x the scale), no single
        global per-dimension weighting can serve every task at once. Each residual is
        therefore scaled by the std of its dimension's per-step change ``s' - s`` computed
        WITHIN ITS OWN SEGMENT -- segments are single-task, so every task's dynamics get
        O(1) salience in their own frame. Dimensions that do not move within a segment
        (interface padding) carry no dynamics information there and are masked out.
        """
        obs_dim = self.obs_dim
        target = feats[:, -obs_dim:]
        s_hat, r_hat = self._decode_full(feats, z)
        err = s_hat - target
        r_term = 0.0
        if r_hat is not None:
            r_term = (r_hat - feats[:, self.obs_dim + self.act_dim]).pow(2).mean()
        if not self.dyn_obs_norm:
            return err.pow(2).mean() + r_term
        L = int(seg_len) if seg_len else int(feats.shape[0])
        S = feats.shape[0] // L
        delta = (target - feats[:, :obs_dim]).view(S, L, obs_dim)
        sd = delta.std(dim=1, unbiased=False).detach()             # [S, obs_dim]
        active = (sd > 1e-8).float()
        e = err.view(S, L, obs_dim) / sd.clamp_min(1e-3).unsqueeze(1) * active.unsqueeze(1)
        return e.pow(2).sum() / (active.sum() * L).clamp_min(1.0) + r_term

    def _segment_features(self, obs, act, next_obs, rew=None) -> torch.Tensor:
        """Stack one segment's stored transitions into encoder features ``[L, in_dim]``.
        ``rew`` is required exactly when the encoder was built with ``encoder_reward``;
        callers pass the RAW step reward (``info['raw_reward']`` on shaped envs), so the
        code cannot shift between shaped training streams and raw evaluation ones."""
        act_t = torch.stack([torch.as_tensor(a) for a in act])
        act_t = act_t.float() if self.action_continuous else act_t.long()
        return self.encoder.transition_features(torch.stack(obs), act_t,
                                                torch.stack(next_obs), rew=rew)

    def reset_carry(self) -> None:
        """Forget the carried episode, so the next segment starts from a plain reset."""
        self._carry = None

    def _segment_context_gaussians(self, coin, k: int):
        """
        Feedback-space Gaussians for the within-segment responsibility update, queried ONCE
        per segment: ``(mean, var)`` for each of the ``k`` known contexts plus the novel
        column last. Known contexts use COIN's aligned per-context estimates (feedback mean
        = state + bias); novel uses the same stationary moments COIN re-seeds a fresh
        context from, ``d/(1-a)`` and ``Q/(1-a^2)`` under the prior means. Observation
        noise is NOT added here -- the caller supplies the encoder's sd per step.
        """
        proto = coin.context_alignment()["global_contexts"]
        a0 = float(coin.prior_mean_retention)
        if int(getattr(coin, "state_dim", 1)) > 1:
            # MD model (observe_value): per-dim means and the DIAGONAL of each
            # context's covariance -- the step weights treat dims independently.
            n = int(coin.state_dim)
            mu = (np.asarray(proto["state_mean"][:k], dtype=float).reshape(k, n)
                  + np.asarray(proto["bias_mean"][:k], dtype=float).reshape(k, n))
            cov = np.asarray(proto["state_cov"][:k], dtype=float).reshape(k, n, n)
            var = np.einsum("kii->ki", cov)
            q = getattr(coin, "process_noise_covariance", None)
            q_diag = (np.diag(np.asarray(q, dtype=float)) if q is not None
                      else np.full(n, float(coin.sigma_process_noise) ** 2))
            nov_mu = np.full(n, float(coin.prior_mean_drift) / (1.0 - a0))
            nov_var = q_diag / (1.0 - a0 ** 2)
            return np.vstack([mu, nov_mu[None]]), np.vstack([var, nov_var[None]])
        mu = (np.asarray(proto["state_mean"][:k], dtype=float)
              + np.asarray(proto["bias_mean"][:k], dtype=float))
        var = np.asarray(proto["state_var"][:k], dtype=float)
        nov_mu = float(coin.prior_mean_drift) / (1.0 - a0)
        nov_var = float(coin.sigma_process_noise) ** 2 / (1.0 - a0 ** 2)
        return np.append(mu, nov_mu), np.append(var, nov_var)

    def _step_context_weights(self, pi_agent, z: float, sd: float,
                              ctx_mu: np.ndarray, ctx_var: np.ndarray, floor2: float):
        """
        Within-segment responsibility update:
        ``w(c) propto pi_pred(c) * N(z; mu_c, var_c + sd^2 + floor^2)``.

        COIN is never touched here -- the predicted pi is reweighted by how well each
        context's feedback Gaussian explains the encoder's CURRENT latent estimate. Early
        in a segment ``sd`` is large, the likelihood is flat and the weights sit at the
        prior; as evidence accumulates they sharpen toward the responsible context. Layout
        in and out is the agent frame ``[known..., nan pad, novel last]``; ``ctx_mu`` /
        ``ctx_var`` are ``[known..., novel]`` from :meth:`_segment_context_gaussians`.
        """
        w = np.asarray(pi_agent, dtype=float).copy()
        k = len(ctx_mu) - 1
        mu_arr = np.asarray(ctx_mu, dtype=float)
        if mu_arr.ndim > 1:
            # MD contexts (observe_value): sum the log-likelihood over the OBSERVED
            # dims only. A scalar z is nan-padded, so the step loop and the eval --
            # where no return exists yet -- use the z marginal with no special case.
            n = mu_arr.shape[1]
            z_arr = np.atleast_1d(np.asarray(z, dtype=float))
            sd_arr = np.atleast_1d(np.asarray(sd, dtype=float))
            if z_arr.size < n:
                z_arr = np.concatenate([z_arr, np.full(n - z_arr.size, np.nan)])
            if sd_arr.size < n:
                sd_arr = np.concatenate([sd_arr, np.zeros(n - sd_arr.size)])
            seen = np.isfinite(z_arr)
            if not seen.any():
                return pi_agent
            var = (np.asarray(ctx_var, dtype=float)[:, seen]
                   + sd_arr[seen] ** 2 + floor2)
            ll = -0.5 * np.sum((z_arr[seen] - mu_arr[:, seen]) ** 2 / var
                               + np.log(var), axis=1)
        else:
            var = ctx_var + sd * sd + floor2
            ll = -0.5 * ((z - ctx_mu) ** 2 / var + np.log(var))
        like = np.exp(ll - ll.max())
        w[:k] *= like[:k]
        w[-1] *= like[-1]
        total = np.nansum(w)
        if not np.isfinite(total) or total <= 0.0:
            return pi_agent          # degenerate likelihoods: fall back to the prior
        return w / total

    def _start_segment(self, env, carry_state: bool):
        """
        Start a segment in ``env``, continuing the previous segment's episode if there is
        one. The task changes at a boundary, the episode does not: state and the time-limit
        counter are copied into the fresh env after its own reset, which keeps the step
        limit per EPISODE rather than per segment. Returns the starting observation and the
        episode return so far.
        """
        obs, _ = env.reset()
        carry = self._carry if carry_state else None
        if carry is None:
            return self._flatten_obs(obs), 0.0

        env.state = np.array(carry["state"])
        env._elapsed_steps = carry["elapsed"]
        # CartPole's observation IS its state; envs that transform it expose a getter.
        get_obs = getattr(env, "_get_obs", None) or getattr(env, "_get_ob", None)
        obs = get_obs() if get_obs is not None else np.asarray(env.state, dtype=np.float32)
        return self._flatten_obs(obs), carry["ep_ret"]

    # ------------------------------------------------------------------
    # Encoder / decoder training
    # ------------------------------------------------------------------
    def _anchor_penalty(self, z_new: torch.Tensor,
                        stamped: torch.Tensor) -> torch.Tensor:
        """
        The stability penalty for a minibatch's SETTLED codes, in one of three forms.

        ``z_new`` are the differentiable segment-final codes under the current encoder;
        ``stamped`` the values the encoder assigned when those segments settled. Both are
        already filtered to the settled subset.

        * ``"value"`` -- ``mean (z_new - stamped)^2``. Pins every code where it was. It is
          what made the map stationary, and it is also what made the CartPole pair
          impossible: separating CP from Inverted CP requires exactly the local movement
          this forbids. Measured under joint training, switching it on collapses the pair
          from 14.9 to 0.0 pooled sd.
        * ``"centroid"`` -- clusters the settled codes by stamped proximity
          (:attr:`anchor_group_gap`, single-linkage in one dimension) and pins each
          CLUSTER'S MEAN. A symmetric split of a cluster moves its members apart while its
          centroid stays, so the penalty is zero; a sweep moves centroids and is charged in
          full. Structure, not values.
        * ``"distance"`` -- a ONE-SIDED hinge on pairwise distances,
          ``mean relu(d_stamped - d_new)^2``, plus ``(mean z_new - mean stamped)^2`` to
          forbid bodily translation. Contraction of any pair is charged; EXPANSION is free,
          because a split is an expansion. The translation term is what stops the map
          escaping the hinge by drifting as a rigid whole.

          It is STRICTER than the centroid form on the very move we want. Splitting a
          cluster inside a wider set pushes one member TOWARD the neighbouring cluster, so
          although the split itself expands, the cross-pair contracts and is charged.
          Splitting ``[-1, -1]`` of ``[-1, -1, 1, 1]`` by +-0.09 costs one charged pair in
          six, where the centroid form costs exactly nothing.

        All three read only the encoder's own stamps -- no labels, no boundaries.
        """
        if z_new.numel() == 0:
            return z_new.new_zeros(())
        if self.anchor_mode == "value":
            return (z_new - stamped).pow(2).mean()

        if self.anchor_mode == "centroid":
            order = torch.argsort(stamped)
            zs, an = z_new[order], stamped[order]
            if zs.numel() == 1:
                return (zs - an).pow(2).mean()
            cuts = ((an[1:] - an[:-1]) > self.anchor_group_gap).long()
            gid = torch.cat([cuts.new_zeros(1), torch.cumsum(cuts, 0)])
            new_mu, _ = self._group_stats(zs, gid)
            old_mu, _ = self._group_stats(an, gid)
            return (new_mu - old_mu).pow(2).mean()

        # distance
        trans = (z_new.mean() - stamped.mean()).pow(2)
        if z_new.numel() < 2:
            return trans
        iu = torch.triu_indices(z_new.numel(), z_new.numel(), offset=1)
        d_new = (z_new[:, None] - z_new[None, :]).abs()[iu[0], iu[1]]
        d_old = (stamped[:, None] - stamped[None, :]).abs()[iu[0], iu[1]]
        return torch.relu(d_old - d_new).pow(2).mean() + trans

    def _update_ema_teacher(self) -> None:
        """
        ``theta_ema <- tau * theta_ema + (1 - tau) * theta``, over parameters AND buffers.

        Called after every optimiser step. Runs under ``no_grad`` and writes in place, so
        the teacher is a pure function of the student's history and can never contribute a
        gradient. Buffers are copied rather than mixed only if they are non-floating
        (integer counters); the encoder has none today, but a future normalisation layer
        would.
        """
        if self.encoder_ema is None:
            return
        tau = self.anchor_ema_tau
        with torch.no_grad():
            for p_ema, p in zip(self.encoder_ema.parameters(), self.encoder.parameters()):
                p_ema.mul_(tau).add_(p.detach(), alpha=1.0 - tau)
            for b_ema, b in zip(self.encoder_ema.buffers(), self.encoder.buffers()):
                if b_ema.dtype.is_floating_point:
                    b_ema.mul_(tau).add_(b.detach(), alpha=1.0 - tau)
                else:
                    b_ema.copy_(b)

    def _settle_anchors(self, seg_len: int, chunk: int = 32) -> int:
        """
        Stamp the code of every pooled segment that has just stopped being new, clear the
        anchor of every segment that is still new, and return how many are still free.

        A segment is **free** -- ``anchors[i] = nan``, contributing nothing to the anchor
        term -- in two cases, both for the same reason: an anchor is worth keeping only
        once it means something.

        * **Warm-up** (``replay.n_seen <= anchor_warmup``): every segment. During the first
          block there is one task and the map is still organising, so the codes assigned
          then are arbitrary; pinning them would fix an accident, and worse, would pin
          early segments collected under an untrained policy, which for MountainCar are
          nearly indistinguishable from Flat MountainCar and would drag the two together.
          With every anchor nan the objective is exactly the unanchored one.
        * **Still arriving** (``push_id >= n_seen - anchor_window``): the newest segments.
          A task that has just appeared must be free to claim its own region of ``z``
          without paying for the move, and a policy that is still improving keeps changing
          what its own task's segments look like.

        Freeing rather than re-stamping is load-bearing. Re-stamping a fresh segment every
        call turns the anchor into a slew-rate limit on the WHOLE map, and a new task can
        then migrate no faster than an old one may drift -- measured on the shrunken
        stream, that is a coefficient with no good setting (stable enough to hold
        MountainCar means Acrobot never leaves it). With the newest segments carrying no
        anchor at all, the only thing resisting a new code is the requirement that the
        SETTLED ones do not move with it, which is exactly the constraint wanted.

        Once a segment falls out of the window it is stamped with the code it has then --
        which is also what stops a mid-block sweep of the CURRENT task, not only drift of
        the tasks whose blocks are over. Stamping is a forward pass only (``no_grad``, no
        RNG), so it cannot change the parameters or the seeded random stream, and it
        touches each segment once.
        """
        if self.encoder_ema is not None:
            return 0            # EMA mode: the teacher IS the target; nothing is stamped
        replay = self.replay
        n = len(replay)
        if n == 0:
            return 0
        if replay.n_seen <= self.anchor_warmup:
            free = set(range(n))
        elif self.anchor_window > 0:
            cutoff = replay.n_seen - self.anchor_window
            free = {i for i in range(n) if replay.push_ids[i] >= cutoff}
        else:
            free = set()

        for i in free:
            replay.anchors[i] = float("nan")
        todo = [i for i in range(n)
                if i not in free and not np.isfinite(replay.anchors[i])]

        with torch.no_grad():
            for start in range(0, len(todo), int(chunk)):
                part = todo[start:start + int(chunk)]
                feats = torch.cat([replay.buffer[i] for i in part]).to(self.device)
                mean, _ = self.encoder.prefix_posterior(feats, seg_len)
                for j, i in enumerate(part):
                    replay.anchors[i] = float(mean[j, -1])
        return len(free)

    def _group_stats(self, z_final: torch.Tensor, gidx: torch.Tensor):
        """``(group_means, within_group_var)`` for codes labelled by contiguous ``gidx``."""
        n_g = int(gidx.max().item()) + 1
        counts = torch.zeros(n_g, device=z_final.device).index_add_(
            0, gidx, torch.ones_like(z_final))
        sums = torch.zeros(n_g, device=z_final.device).index_add_(0, gidx, z_final)
        means = sums / counts.clamp_min(1.0)
        dev2 = (z_final - means[gidx]).pow(2)
        within = torch.zeros(n_g, device=z_final.device).index_add_(0, gidx, dev2)
        return means, within / counts.clamp_min(1.0)

    def _dispersion_loss(self, z_final: torch.Tensor,
                         gidx: Optional[torch.Tensor] = None):
        """
        VICReg's variance and invariance terms on segment codes, returned as
        ``(dispersion, invariance)``.

        The variance term (Bardes, Ponce & LeCun, ICLR 2022) is a HINGE,
        ``relu(disp_target_sd - std(.))^2``: once the codes reach the stated spread its
        gradient is exactly zero, so it buys separation up to a scale and then stops
        pulling -- it never fights the anchor for more. The invariance term is the mean
        within-group variance, which is what makes the hinge mean the right thing.

        **The hinge alone is exploitable, and measured on the random-action proxy it is
        exploited immediately.** Asked for spread with no notion of which segments belong
        together, the encoder buys variance the cheapest way available: it makes the code
        NOISY on the tasks whose dynamics it cannot tell apart (MountainCar and Flat
        MountainCar reached a within-task code spread of 1.0 and 1.7 respectively, against
        0.007 unpenalised), because a code the decoder cannot use anyway is free to
        randomise. Between-task separation in units of the pooled sd got worse, not better.

        So the spread is asked of the GROUP MEANS -- segments the caller has marked as one
        task (:meth:`SegmentReplayBuffer.push`) -- while the within-group variance is
        penalised. Between-task dispersion is then the only route to satisfying the hinge.
        With no grouping every group is a singleton: the invariance term is identically
        zero and the hinge falls back to the plain, exploitable form.

        Fewer than two groups carries no variance information and contributes nothing.
        """
        if gidx is None:
            means, within = z_final, z_final.new_zeros(z_final.shape)
        else:
            means, within = self._group_stats(z_final, gidx)
        if means.numel() < 2:
            return z_final.new_zeros(()), within.mean()
        disp = torch.relu(self.disp_target_sd - means.std()).pow(2)
        return disp, within.mean()

    def _update_encoder(self, seg_len: int, enc_steps: int,
                        mb_segments: int) -> Dict[str, float]:
        """
        Fit encoder and decoder on
        ``L_dyn + kl_coef * KL + anchor_coef * L_anchor + disp_coef * L_disp``,
        minibatched over SEGMENTS.

        Each gradient step draws ``mb_segments`` segments from :attr:`replay` and uses all
        of their transitions -- a prefix posterior is a whole-segment quantity, so a
        minibatch of loose transitions would have to encode their segments in full anyway.
        The posterior is recomputed from scratch every step (it depends on all parameters,
        so a cached one would go stale immediately) and resampled with fresh noise. Prefix
        column ``t`` is the posterior BEFORE transition ``t``, so the latent the decoder
        sees never contains the transition it has to predict.

        **The anchor is what makes the code stationary.** Nothing in ``L_dyn`` prefers one
        labelling of the latent over another: the decoder can absorb any smooth
        reparameterisation of ``z``, so each block's on-policy data is free to rotate the
        whole map, and it does -- faster than COIN's drift tracking can follow, which
        spawns spurious contexts and misroutes the tasks whose blocks are over. The anchor
        term

            ``L_anchor = mean_over_pinned_segments( (z_final(seg) - anchor[seg])^2 )``

        makes the encoder its own teacher. There are two ways to supply ``anchor[seg]``:

        * **EMA teacher** (``anchor_ema_tau`` set, preferred). The target is a trailing
          copy of the encoder evaluated on the same segments,
          ``theta_ema <- tau * theta_ema + (1 - tau) * theta``. Nothing is stored and
          nothing is frozen: the student may move the map as fast as the teacher follows,
          which converts an unbounded rotation into a bounded VELOCITY. Being a pure
          timescale, ``tau`` carries no task information -- the property the stamping mode
          could not honour.
        * **Stamped** (default). ``anchor[seg]`` is the value the encoder itself assigned
          when the segment settled (:meth:`_settle_anchors`); segments still inside
          ``anchor_window`` carry none. This is the mode that WORKS, and the reason is that
          it is SELECTIVE: it conditions on a segment's age, so it can hold an old code
          still while leaving a newly arrived task completely free. The EMA teacher cannot
          make that distinction -- it rate-limits the whole map at once -- and measured on
          the encoder-only study it trades drift against spread along a single curve with
          no usable point on it.

        Either way: no labels, no cues, and the supervision comes entirely from the model's
        own past.

        **The anchor alone is not enough.** Stationary is not the same as distinct, and
        across four measured configurations the tasks that a stable map merges stay
        merged: parking a new task on an old code costs the dynamics loss nothing, so it
        parks. ``L_disp`` (:meth:`_dispersion_loss`) is the missing half -- a hinge that
        asks the batch of codes to reach a stated spread and then stops. Anchor and
        dispersion are complementary, not competing: one forbids old codes from moving,
        the other requires the population to spread, and the only way to satisfy both is
        to move the NEW code away.
        """
        clip = float("inf") if self.enc_grad_clip is None else float(self.enc_grad_clip)
        dyn_val = kl_val = grad_norm = anchor_val = float("nan")

        n_free = self._settle_anchors(seg_len)

        # The variance hinge is meaningless while the pool holds a single task: asking a
        # one-task population to spread IS a demand for within-task spread, and on the
        # random-action proxy the hinge was satisfied that way by rollout 4-13, deep inside
        # the first block, locking the damage in before anything was pinned. The warm-up
        # boundary is exactly "the first block is over", so it gates both.
        structured = (self.disp_coef > 0.0 or self.inv_coef > 0.0)
        dispersing = structured and self.replay.n_seen > self.anchor_warmup

        for _ in range(int(enc_steps)):
            if structured:
                feats, anchors, gidx = self.replay.sample_groups(mb_segments,
                                                                 self.disp_per_group)
                gidx = gidx.to(self.device)
            elif self.anchor_strat_sampling:
                # The centroid form needs cluster-mates in the same batch to have a
                # centroid at all; see SegmentReplayBuffer.sample_code_groups.
                feats, anchors = self.replay.sample_code_groups(
                    mb_segments, self.anchor_group_gap, self.anchor_per_group)
                gidx = None
            elif self.balanced_replay:
                feats, anchors = self.replay.sample_balanced(mb_segments,
                                                             self.anchor_window)
                gidx = None
            else:
                feats, anchors = self.replay.sample(mb_segments)
                gidx = None
            feats = feats.to(self.device)
            anchors = anchors.to(self.device)
            mean, sd = self.encoder.prefix_posterior(feats, seg_len)
            sd_z = sd[:, :-1]
            if self.z_channel_noise > 0.0:
                # Noisy channel: the decoder must work at COIN's resolution, so codes
                # closer than the noise radius are confusable and differing dynamics
                # push them apart through the dynamics loss itself.
                sd_z = torch.sqrt(sd_z ** 2 + self.z_channel_noise ** 2)
            z = (mean[:, :-1] + torch.randn_like(sd_z) * sd_z).reshape(-1)

            dyn = self._dyn_loss(feats, z, seg_len)
            kl = self._kl_to_prior(mean, sd).mean()
            # nan anchors mark segments pushed without one (pre-anchor pools, or a caller
            # that passed None); they are simply not pinned.
            z_final = mean[:, -1]
            if self.encoder_ema is not None:
                # EMA mode: the target is the trailing teacher's code for these very
                # segments, recomputed every step under no_grad. Nothing is stored, so no
                # code is ever frozen and no boundary information is consulted.
                with torch.no_grad():
                    t_mean, _ = self.encoder_ema.prefix_posterior(feats, seg_len)
                anchor = (z_final - t_mean[:, -1]).pow(2).mean()
            else:
                ok = torch.isfinite(anchors)
                anchor = self._anchor_penalty(z_final[ok], anchors[ok])
            disp, inv = self._dispersion_loss(z_final, gidx)
            rail = torch.relu(z_final.abs() / self.z_scale - 0.8).pow(2).mean()
            loss = (dyn + self.kl_coef * kl + self.anchor_coef * anchor
                    + (self.disp_coef * disp if dispersing else 0.0)
                    + self.inv_coef * inv + self.rail_coef * rail)
            if self._repel_batch:
                loss = loss + self.repel_coef * self._repel_hinge(seg_len)

            self.enc_optim.zero_grad()
            loss.backward()
            grad_norm = float(nn.utils.clip_grad_norm_(self.encoder.parameters(), clip))
            self.enc_optim.step()
            self._update_ema_teacher()
            dyn_val, kl_val = float(dyn.item()), float(kl.item())
            anchor_val, disp_val = float(anchor.item()), float(disp.item())
            inv_val = float(inv.item())

        return {"dyn_loss": dyn_val, "encoder_kl": kl_val, "enc_grad_norm": grad_norm,
                "anchor_loss": anchor_val, "anchor_free": float(n_free),
                "disp_loss": disp_val, "inv_loss": inv_val}

    def pretrain_encoder(self, envs, seg_steps: int = 512, n_iters: int = 50,
                         enc_steps: int = 32, mb_segments: int = 4) -> Dict[str, np.ndarray]:
        """
        Train encoder and decoder on uniform-random rollouts, with no COIN and no PPO.

        The pretrain-then-RL regime: a latent learned before any policy exists, then frozen
        while the RL model trains online (``train_step(..., update_encoder=False)``). Random
        actions are the standard choice for dynamics pretraining -- they excite the
        transition model broadly, and the contingency signature is in the transitions, not
        in the returns. Pass one FRESH env per segment, as for :meth:`train_step`. Segments
        go into the same :attr:`replay` pool, so later iterations still see earlier ones.

        Returns:
            Dict[str, np.ndarray]: Per-iteration ``dyn_loss``, ``encoder_kl``,
            ``enc_grad_norm``, ``anchor_loss``, ``anchor_free``, ``disp_loss`` and
            ``inv_loss``.
        """
        L = int(seg_steps)
        history: Dict[str, List[float]] = defaultdict(list)

        for _ in range(n_iters):
            for env in envs:
                obs_t = self._flatten_obs(env.reset()[0])
                obs, act, nxt = [], [], []
                for _ in range(L):
                    action = env.action_space.sample()
                    next_obs, _reward, done, trunc, _ = env.step(action)
                    obs.append(obs_t.cpu())
                    act.append(action)
                    nxt.append(self._flatten_obs(next_obs).cpu())
                    if done or trunc:
                        next_obs, _ = env.reset()
                    obs_t = self._flatten_obs(next_obs)
                feats = self._segment_features(obs, act, nxt)
                with torch.no_grad():
                    mean_p, _ = self.encoder.prefix_posterior(feats, L)
                # No group: pretrain passes one env per TASK, so this loop's segments are
                # deliberately NOT the same task.
                self.replay.push(feats, float(mean_p[0, -1]))

            for key, val in self._update_encoder(L, enc_steps, mb_segments).items():
                history[key].append(val)

        return {k: np.asarray(v, dtype=float) for k, v in history.items()}

    # ------------------------------------------------------------------
    # Quantile readout (gauge-invariant observation space for COIN)
    # ------------------------------------------------------------------
    def _refresh_code_population(self, seg_len: Optional[int] = None) -> None:
        """Re-encode a replay subsample and cache its sorted segment-final codes.

        Called once per :meth:`train_step` and once per :meth:`evaluate_identifying`, so
        every observation within one trial batch shares one population -- and the
        population always reflects the CURRENT encoder, which is what makes the rank
        cancel gauge motion (stale push-time codes would not move with the map).
        No-op unless :attr:`quantile_readout`; population stays ``None`` below 8 segments.
        """
        if not self.quantile_readout:
            return
        n = len(self.replay)
        if n < 8:
            self._code_pop = None
            return
        if seg_len is None:
            seg_len = int(self.replay.buffer[0].shape[0])
        idx = np.random.choice(n, size=min(self.quantile_pop_size, n), replace=False)
        feats = torch.cat([self.replay.buffer[i] for i in idx]).to(self.device)
        with torch.no_grad():
            mean, _ = self.encoder.prefix_posterior(feats, int(seg_len))
        self._code_pop = np.sort(mean[:, -1].cpu().numpy())

    def _obs_transform(self, z: float, sd: float) -> Tuple[float, float]:
        """Map an ``(z, sd)`` code posterior into COIN's observation space.

        Identity without :attr:`quantile_readout` (or before a population exists).
        Otherwise ``z -> probit(midrank(z))`` with quantiles clamped to
        ``[1/2n, 1 - 1/2n]`` (the resolution an ``n``-point population supports), and the
        sd mapped as half the transformed ``z +- sd`` interval, floored at ``1e-3`` so a
        code beyond the population's edge never claims certainty.
        """
        pop = self._code_pop
        if pop is None:
            return float(z), float(sd)
        from statistics import NormalDist
        n = pop.size
        inv = NormalDist().inv_cdf

        def probit(x):
            lo = float(np.searchsorted(pop, x, side="left"))
            hi = float(np.searchsorted(pop, x, side="right"))
            q = np.clip((lo + hi) / (2.0 * n), 0.5 / n, 1.0 - 0.5 / n)
            return inv(q)

        z_obs = probit(z)
        sd_obs = max((probit(z + sd) - probit(z - sd)) / 2.0, 1e-3)
        return z_obs, sd_obs

    def _soft_obs_transform(self, z: torch.Tensor, sd: torch.Tensor):
        """Differentiable twin of :meth:`_obs_transform` for the value-gradient term.

        The hard rank is piecewise constant (zero gradient a.e.), so the encoder could
        never feel the value loss through it. A smooth ECDF -- mean of sigmoids with a
        bandwidth tied to the population spread -- keeps the same map at scale while
        letting the gradient through; COIN itself still receives the exact transform.
        """
        pop = self._code_pop
        if pop is None:
            return z, sd
        pop_t = torch.as_tensor(pop, dtype=z.dtype, device=z.device)
        n = pop_t.numel()
        h = max(float(pop.std()), 1e-3) * 0.1

        def probit(x):
            q = torch.sigmoid((x - pop_t) / h).mean()
            q = q.clamp(0.5 / n, 1.0 - 0.5 / n)
            return torch.special.ndtri(q)

        z_obs = probit(z)
        sd_obs = ((probit(z + sd) - probit(z - sd)) / 2.0).clamp_min(1e-3)
        return z_obs, sd_obs

    def _flag_value_surprise(self, seg_feats, seg_rho, seg_ctx, obs_tensor,
                             ret_tensor, seg_len: int):
        """Fill :attr:`_repel_batch` with segments whose routed head mis-values them.

        Per segment: take COIN's dominant post-observation context; if it is a known,
        instantiated, ESTABLISHED context (has a value-error baseline from previous
        segments), compare its head's value MSE on this segment against
        ``repel_gate_ratio *`` that baseline. A blow-up flags the segment for
        repulsion away from the context's centre ``mu_c`` (stored with its features
        for :meth:`_update_encoder`); an ordinary error updates the baseline EMA.
        The novel column and contexts born this very rollout are never flagged.
        """
        self._repel_batch = []
        if self.repel_coef <= 0.0:
            return 0
        L = int(seg_len)
        flagged = 0
        with torch.no_grad():
            for s, feats_s in enumerate(seg_feats):
                rho = np.asarray(seg_rho[s], dtype=float)
                if not np.isfinite(rho).any():
                    continue
                j = int(np.nanargmax(rho))
                ctx_mu, _ctx_var, _f2 = seg_ctx[s]
                k = len(ctx_mu) - 1
                if j >= k or j >= self.num_contexts - 1:
                    continue                      # novel, or born after the ctx query
                cid = self.context_keys[j]
                if self.context_init.get(cid, 0) == 0:
                    continue
                obs_s = obs_tensor[s * L:(s + 1) * L]
                ret_s = ret_tensor[s * L:(s + 1) * L]
                err = float(((self.nets[cid][2](obs_s).squeeze(-1) - ret_s) ** 2
                             ).mean())
                base = self._vloss_ema.get(cid)
                if base is None:
                    self._vloss_ema[cid] = err     # first sight: baseline, no gate
                elif err > self.repel_gate_ratio * max(base, 1e-8):
                    # MD contexts carry (z, value) centres; repulsion acts on z.
                    mu_j = float(np.asarray(ctx_mu[j]).reshape(-1)[0])
                    self._repel_batch.append((feats_s.detach().cpu(), mu_j))
                    flagged += 1
                else:
                    tau = self.vloss_ema_tau
                    self._vloss_ema[cid] = tau * base + (1.0 - tau) * err
        return flagged

    def _repel_hinge(self, seg_len: int) -> torch.Tensor:
        """Repulsion loss for the flagged segments, summed; zero tensor when none."""
        margin = self.repel_margin
        if margin is None:
            margin = (2.0 * self.z_channel_noise if self.z_channel_noise > 0.0
                      else 0.4 * self.z_scale)
        total = torch.zeros((), device=self.device)
        for feats, mu_c in self._repel_batch:
            mean, _ = self.encoder.prefix_posterior(feats.to(self.device),
                                                    int(seg_len))
            d = (mean[0, -1] - mu_c).abs()
            total = total + torch.relu(margin - d).pow(2)
        return total

    def _encoder_value_loss(self, seg_feats, seg_pi, seg_ctx, obs_tensor,
                            ret_tensor, seg_len: int):
        """The value-gradient encoder term (see ``value_coef`` in the class docstring).

        Per segment: re-encode its features (grad ON), rebuild the segment-final
        responsibilities exactly as :meth:`_step_context_weights` does but in torch --
        COIN's per-context Gaussians, the predicted pi and the noise floor all enter as
        constants -- then mix the heads' DETACHED value estimates under those weights
        and score the mix against the PPO return targets. The only gradient path runs
        value-mismatch -> responsibilities -> posterior -> encoder: the term teaches the
        encoder to place ``z`` where the value function routes correctly, which is
        information the dynamics loss cannot see when state or policy already identify
        the task. Returns ``None`` when no segment yields usable weights.
        """
        L = int(seg_len)
        losses = []
        for s, feats_s in enumerate(seg_feats):
            pi_agent = np.asarray(seg_pi[s], dtype=float)
            ctx_mu, ctx_var, floor2 = seg_ctx[s]
            ctx_mu = np.asarray(ctx_mu, dtype=float)
            ctx_var = np.asarray(ctx_var, dtype=float)
            if ctx_mu.ndim > 1:
                # MD contexts: the differentiable path runs through z, so the
                # routing comparison here uses the z components of the centres.
                ctx_mu, ctx_var = ctx_mu[:, 0], ctx_var[:, 0]
            k = len(ctx_mu) - 1
            mean, sd = self.encoder.prefix_posterior(feats_s.to(self.device), L)
            z_t, sd_t = self._soft_obs_transform(mean[0, -1], sd[0, -1])

            mu_t = torch.as_tensor(ctx_mu, dtype=z_t.dtype, device=z_t.device)
            var_t = (torch.as_tensor(ctx_var, dtype=z_t.dtype, device=z_t.device)
                     + sd_t * sd_t + floor2)
            ll = -0.5 * ((z_t - mu_t) ** 2 / var_t + torch.log(var_t))
            like = torch.exp(ll - ll.max().detach())

            pi_t = torch.as_tensor(np.nan_to_num(pi_agent, nan=0.0),
                                   dtype=z_t.dtype, device=z_t.device)
            w = torch.zeros(self.num_contexts, dtype=z_t.dtype, device=z_t.device)
            w[:k] = pi_t[:k] * like[:k]
            w[-1] = pi_t[-1] * like[-1]
            total = w.sum()
            if not torch.isfinite(total) or float(total) <= 0.0:
                continue
            w = w / total

            obs_s = obs_tensor[s * L:(s + 1) * L]
            ret_s = ret_tensor[s * L:(s + 1) * L]
            with torch.no_grad():
                v_heads = torch.zeros(self.num_contexts, obs_s.shape[0],
                                      device=obs_s.device)
                for j, cid in enumerate(self.context_keys):
                    if self.context_init.get(cid, 0) == 0:
                        continue
                    v_heads[j] = self.nets[cid][2](obs_s).squeeze(-1)
            v_mixed = (w.unsqueeze(1) * v_heads).sum(dim=0)
            losses.append((v_mixed - ret_s).pow(2).mean())
        if not losses:
            return None
        return torch.stack(losses).mean()

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def train_step(self, envs, coin, seg_steps: int = 512, mini_epochs: int = 10,
                   mb_size: int = 64, cues=None, update_encoder: bool = True,
                   enc_steps: int = 32, mb_segments: int = 4, carry_state: bool = True):
        """
        Collect one segment-interleaved rollout and update on it.

        One call is ``S = len(envs)`` contiguous segments of ``seg_steps`` transitions, each
        a different task and its own COIN trial, so this method drives ``coin`` itself
        (``observe_q`` -> predicted pi -> segment -> ``observe_y``, advancing ``S`` trials).
        ``S = 1`` reproduces one task per rollout. Pass one FRESH env per segment --
        ``CustomCartPoleEnv.__init__`` computes its derived constants once, so mutating a
        live env leaves them stale -- and optionally one ``cues`` entry per segment.

        The episode is carried across boundaries (:meth:`_start_segment`), including the
        boundary between two calls, so a rollout edge is not a special event for the task
        sequence. COIN's sensory noise for each trial is set to the encoder's own posterior
        sd; the noise FLOOR belongs on the model (``sigma_motor_noise``), which the pipeline
        adds in quadrature.

        Acting weights are updated WITHIN the segment: step 0 acts on COIN's predicted pi,
        and every transition thereafter folds its encoder factor into an online prefix
        posterior whose current ``(z, sd)`` reweights the prediction via
        :meth:`_step_context_weights`. COIN's own trial-end update is untouched -- it still
        observes only the segment-final posterior mean.

        Args:
            update_encoder (bool): False freezes encoder and decoder and reports their
                losses under ``no_grad`` only. With a pretrained encoder that reduces this
                to the plain COIN-PPO routine with COIN stepped inside.
            enc_steps (int): Encoder gradient steps per call, each on ``mb_segments``
                segments drawn from :attr:`replay`. PPO still trains on the fresh rollout
                alone.
            carry_state (bool): False resets every segment to a fresh episode -- which also
                changes how GAE bootstraps: see the comment on ``w_next`` below.

        Returns:
            dict: Per-segment arrays ``z``, ``z_sd``, ``K`` (pre-observation), ``pi``
            ``[S, W]``, ``rho`` ``[S, W]`` (post-observation diagnostics), ``w_mean``
            ``[S, W]`` (step-averaged acting weights), ``sharpen_step`` (first step whose
            max acting weight exceeds 0.9; ``nan`` if never),
            ``mean_episode_return`` (``nan`` where no episode ENDED in that segment), plus
            scalar ``mean_reward_per_step``, ``value_loss``, ``policy_loss``, ``dyn_loss``,
            ``encoder_kl``, ``enc_grad_norm``, ``anchor_loss`` (the code-stability term,
            see :meth:`_update_encoder`), ``anchor_free`` (pooled segments still allowed
            to move their code), ``disp_loss`` (the separation hinge) and ``inv_loss``
            (the within-task consistency term).
        """
        S, L = len(envs), int(seg_steps)
        enc, W = self.encoder, self.num_contexts
        # One group per CALL, but only when the caller has guaranteed that this call's
        # segments are all the same task (see :attr:`same_task_rollout`). The interleaved
        # mode of this method gives every segment a different task, where sharing a group
        # would assert the exact opposite of the truth.
        self._group_counter += 1
        group = self._group_counter if self.same_task_rollout else None
        # One rank population per rollout: every COIN observation and step-level
        # comparison in this call shares the same observation space.
        self._refresh_code_population(L)
        store: Dict[str, List[Any]] = defaultdict(list)
        seg_feats, seg_w, seg_w_final, seg_last_obs, seg_returns = [], [], [], [], []
        seg_z, seg_z_sd, seg_K, seg_pi, seg_rho, seg_ctx = [], [], [], [], [], []

        # ---------- 1. interleaved rollout ----------
        for s, env in enumerate(envs):
            cue = None if cues is None else cues[s]
            coin.observe_q(cue)
            # Never predicted_context_probabilities_vector() here: it is one trial stale and
            # ignores the cue just staged. Query strictly before anything else touches coin.
            pi_vec, K = coin_predicted_pi(coin, cue)
            self.ensure_contexts(K)
            # min(K, W - 1) keeps coin_context_vector in range; renormalise_novel folds any
            # overflow mass (more COIN contexts than heads) into the novel column.
            pi_agent = coin_context_vector(pi_vec, min(K, W - 1), width=W)
            w_s = self._policy_weights(pi_agent)      # w_0: the predicted pi, pre-evidence
            seg_K.append(K)
            seg_pi.append(pi_agent)
            seg_w.append(w_s)

            # Solution 2 machinery: per-context Gaussians queried once, then an online
            # prefix posterior (running natural parameters, the streaming twin of
            # :meth:`ContingencyEncoder.prefix_posterior`) reweights pi every step.
            ctx_mu, ctx_var = self._segment_context_gaussians(coin, min(K, W - 1))
            floor2 = float(getattr(coin, "sigma_motor_noise", 0.0)) ** 2
            seg_ctx.append((ctx_mu, ctx_var, floor2))
            prior_prec = 1.0 / (self.prior_sd ** 2)
            eta1 = eta2 = 0.0
            w_t = w_s

            obs_t, ep_ret = self._start_segment(env, carry_state)
            obs, act, rew, nxt, frew = [], [], [], [], []
            ep_returns, done, trunc = [], False, False

            for _ in range(L):
                with torch.no_grad():
                    value = self._mixed_value(obs_t.unsqueeze(0), w_t)[0].item()
                    action_np, logp, _, _ = self.act(obs_t, w_t)
                next_obs, reward, done, trunc, info = env.step(action_np)
                # Encoder features carry the RAW reward: a shaped training env and a
                # raw eval env must give the same task the same code.
                frew.append(float(info.get("raw_reward", reward))
                            if isinstance(info, dict) else float(reward))

                # next_obs is stored BEFORE any reset: letting the reset overwrite it
                # corrupts every episode-final transition with no visible symptom.
                obs.append(obs_t.cpu())
                act.append(action_np)
                nxt.append(self._flatten_obs(next_obs).cpu())
                rew.append(float(reward))
                store["logp"].append(logp.cpu())
                store["val"].append(value)
                store["done"].append(bool(done or trunc))
                store["w"].append(w_t.cpu())

                # Fold this transition's factor into the online posterior and update the
                # acting weights for the NEXT step. COIN itself is not consulted.
                with torch.no_grad():
                    mu_f, sig_f = enc.factors(
                        self._segment_features(obs[-1:], act[-1:], nxt[-1:],
                                               rew=frew[-1:]))
                iv = float(1.0 / (sig_f * sig_f))
                eta2 += iv
                eta1 += float(mu_f) * iv
                var_t = 1.0 / (eta2 + prior_prec)
                # The posterior lives in z-space; the comparison against COIN's context
                # Gaussians happens in observation space (quantile-probit when enabled).
                z_t, sd_t = self._obs_transform(eta1 * var_t, float(np.sqrt(var_t)))
                w_t = self._policy_weights(self._step_context_weights(
                    pi_agent, z_t, sd_t, ctx_mu, ctx_var, floor2))

                ep_ret += reward
                if done or trunc:
                    next_obs, _ = env.reset()
                    ep_returns.append(ep_ret)
                    ep_ret = 0.0
                obs_t = self._flatten_obs(next_obs)

            seg_w_final.append(w_t)

            # An episode that outlives the segment is handed to the next one; only a
            # terminal ending forces the next segment to reset.
            if carry_state:
                self._carry = None if (done or trunc) else {
                    "state": np.array(env.state), "elapsed": int(env._elapsed_steps),
                    "ep_ret": ep_ret}

            store["obs"] += obs
            store["act"] += act
            store["rew"] += rew
            feats_s = self._segment_features(obs, act, nxt, rew=frew)
            seg_feats.append(feats_s)
            seg_last_obs.append(obs_t)
            # A return belongs to the segment the episode ENDED in; nan, not zero, where no
            # episode ended, so the diagnostic is not diluted by the carried ones.
            seg_returns.append(float(np.mean(ep_returns)) if ep_returns else np.nan)

            with torch.no_grad():
                mean_s, sd_s = enc.prefix_posterior(feats_s, L)
            # The segment goes into the pool WITH the code the encoder just gave it: that
            # value is its anchor from the moment it settles (:meth:`_settle_anchors`).
            self.replay.push(feats_s, float(mean_s[0, -1]), group=group)
            seg_z.append(float(mean_s[0, -1]))
            seg_z_sd.append(float(sd_s[0, -1]))
            # COIN's sensory noise for this trial IS the encoder's uncertainty; the pipeline
            # reads it fresh at every use and adds sigma_motor_noise (the floor) in quadrature.
            z_obs, sd_obs = self._obs_transform(float(mean_s[0, -1]), float(sd_s[0, -1]))
            if self.observe_value:
                # Second observation dim: mean return of the episodes that ENDED in
                # this segment (nan when none did -- the MD pipeline masks that dim),
                # with its standard error; floors enter in quadrature because the
                # explicit R override bypasses the isotropic sigma defaults.
                n_eps = len(ep_returns)
                r_bar = (float(np.mean(ep_returns)) / self.value_obs_scale
                         if n_eps else float("nan"))
                sd_r = (float(np.std(ep_returns)) / np.sqrt(n_eps)
                        / self.value_obs_scale if n_eps else 0.0)
                floor = float(getattr(coin, "sigma_motor_noise", 0.0))
                coin.observation_noise_covariance = np.diag(
                    [sd_obs ** 2 + floor ** 2,
                     sd_r ** 2 + self.value_obs_noise_floor ** 2])
                coin.observe_y(np.array([z_obs, r_bar]))
            else:
                coin.sigma_sensory_noise = sd_obs
                coin.observe_y(z_obs)                  # plain float: COIN works in float64

            # Post-observation diagnostics; K may have grown, so re-query the alignment.
            K_post = int(coin.context_alignment()["K"])
            seg_rho.append(coin_context_vector(coin.responsibilities_vector(),
                                               min(K_post, W - 1), width=W,
                                               renormalise_novel=False))

        # ---------- 2. tensors and advantages ----------
        obs_tensor = torch.stack(store["obs"]).to(self.device)
        act_tensor = torch.stack([torch.as_tensor(a) for a in store["act"]]).to(self.device)
        act_tensor = act_tensor.float() if self.action_continuous else act_tensor.long()
        old_logp = torch.stack(store["logp"]).to(self.device).float()
        feats = torch.cat(seg_feats)
        w_all = torch.stack(store["w"]).to(self.device)                 # [S * L, W]

        adv_parts, ret_parts = [], []
        with torch.no_grad():
            for s in range(S):
                sl = slice(s * L, (s + 1) * L)
                # GAE never crosses a boundary, but the episode may: bootstrap under the
                # weights the episode will actually continue under.
                #   carry_state=True  -> the episode continues INTO segment s+1, so its
                #     value is the next segment's w_0 (the last segment has no successor
                #     yet and uses its own final, sharpest within-segment weights).
                #   carry_state=False -> nothing continues. Segment s+1 is a fresh
                #     episode, possibly of a DIFFERENT task, so its w_0 is the wrong
                #     mixture entirely; every segment must bootstrap under its own final
                #     weights, which are also the sharpest belief it ever held.
                w_next = (seg_w[s + 1] if (carry_state and s + 1 < S)
                          else seg_w_final[s])
                last_val = self._mixed_value(seg_last_obs[s].unsqueeze(0), w_next)[0].item()
                a, r = self._compute_advantages(
                    store["rew"][sl], store["val"][sl], store["done"][sl], last_val)
                adv_parts.append(a)
                ret_parts.append(r)

        # Normalise across the WHOLE rollout, not per segment. Per-segment normalisation was
        # measured and is a REGRESSION: equalising the segments' advantage scales removes the
        # asymmetry that lets one task break away, and without a leader neither separates.
        adv = torch.cat(adv_parts)
        adv_tensor = ((adv - adv.mean()) / (adv.std() + 1e-8)).to(self.device).float()
        ret_tensor = torch.cat(ret_parts).to(self.device).float()

        # Value-surprise gating runs BEFORE the PPO update: the routed head's error is
        # sharpest before this rollout's minibatches co-adapt it to the new task.
        n_repel = self._flag_value_surprise(seg_feats, seg_rho, seg_ctx,
                                            obs_tensor, ret_tensor, L)

        # ---------- 3. PPO update (context heads only) ----------
        n = S * L
        actor_loss = critic_loss = None
        for _ in range(mini_epochs):
            idxs = torch.randperm(n)
            for start in range(0, n, mb_size):
                mb = idxs[start:start + mb_size]
                b_obs, b_act, b_w = obs_tensor[mb], act_tensor[mb], w_all[mb]

                new_logp, entropy = self._logp_entropy(b_obs, b_act, b_w)
                ratio = torch.exp(new_logp - old_logp[mb])
                surr = torch.min(
                    ratio * adv_tensor[mb],
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_tensor[mb])
                actor_loss = -surr.mean()
                critic_loss = (ret_tensor[mb] - self._mixed_value(b_obs, b_w)).pow(2).mean()
                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy

                for opt in self._all_optimizers():
                    opt.zero_grad()
                loss.backward()
                for opt in self._all_optimizers():
                    opt.step()

        # ---------- 4. encoder + decoder update ----------
        if update_encoder:
            enc_stats = self._update_encoder(L, enc_steps, mb_segments)
            enc_stats["enc_value_loss"] = float("nan")
            if self.value_coef > 0.0:
                # Value-gradient term: one extra encoder step on THIS rollout's
                # segments (returns are on-policy quantities; replay has none).
                vloss = self._encoder_value_loss(seg_feats, seg_pi, seg_ctx,
                                                 obs_tensor, ret_tensor, L)
                if vloss is not None:
                    self.enc_optim.zero_grad()
                    (self.value_coef * vloss).backward()
                    clip = (float("inf") if self.enc_grad_clip is None
                            else float(self.enc_grad_clip))
                    nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
                    self.enc_optim.step()
                    enc_stats["enc_value_loss"] = float(vloss.item())
        else:
            with torch.no_grad():
                mean, sd = enc.prefix_posterior(feats, L)
                z = (mean[:, :-1] + torch.randn_like(sd[:, :-1]) * sd[:, :-1]).reshape(-1)
                dyn = self._dyn_loss(feats, z, L)
                enc_stats = {"dyn_loss": float(dyn.item()),
                             "encoder_kl": float(self._kl_to_prior(mean, sd).mean().item()),
                             "enc_grad_norm": 0.0, "anchor_loss": 0.0,
                             "anchor_free": 0.0, "disp_loss": 0.0, "inv_loss": 0.0,
                             "enc_value_loss": float("nan")}

        # ---------- 5. diagnostics ----------
        self._weights_version += 1
        w_steps = w_all.detach().cpu().numpy().reshape(S, L, W)
        sharp = np.argmax(w_steps.max(axis=2) > 0.9, axis=1).astype(float)
        sharp[~(w_steps.max(axis=2) > 0.9).any(axis=1)] = np.nan
        return {
            "z": np.asarray(seg_z, dtype=float), "z_sd": np.asarray(seg_z_sd, dtype=float),
            "K": np.asarray(seg_K, dtype=int),
            "pi": np.asarray(seg_pi, dtype=float), "rho": np.asarray(seg_rho, dtype=float),
            "w_mean": w_steps.mean(axis=1), "sharpen_step": sharp,
            "mean_episode_return": np.asarray(seg_returns, dtype=float),
            "mean_reward_per_step": float(np.mean(store["rew"])),
            "value_loss": float(critic_loss.item()), "policy_loss": float(actor_loss.item()),
            "repel_flagged": float(n_repel),
            **enc_stats,
        }

    # ------------------------------------------------------------------
    # Self-identifying evaluation
    # ------------------------------------------------------------------
    def _eval_prior_pi(self, coin, k: int) -> np.ndarray:
        """
        COIN's stationary context distribution in the agent layout ``[known..., nan, novel]``.

        The episode-0 belief of a self-identifying evaluation: not the predicted pi (that is
        conditioned on whichever context the training stream happened to leave COIN in, which
        would leak the task identity into the test) but the chain's long-run marginal, which
        knows nothing about which task is about to be presented. Slots whose PPO head was
        never instantiated are marked NaN so :meth:`_policy_weights` renormalises over the
        heads that actually exist -- ``_mixed_logits`` silently drops the rest.
        """
        W = self.num_contexts
        pi_agent = np.full(W, np.nan)
        stat = np.asarray(coin.stationary_context_probabilities(), dtype=float)
        if k > 0:
            head = stat[:k].copy()
            total = float(np.sum(head))
            pi_agent[:k] = head / total if total > 0.0 else np.full(k, 1.0 / k)
        pi_agent[-1] = 0.0
        for j, cid in enumerate(self.context_keys[:-1]):
            if self.context_init.get(cid, 0) == 0:
                pi_agent[j] = np.nan
        if not np.nansum(pi_agent[:-1]) > 0.0:
            pi_agent[-1] = 1.0          # nothing known yet: novel is all there is
        return pi_agent

    def _deterministic_action(self, obs_t: torch.Tensor, w: torch.Tensor):
        """Greedy action under the ``w``-mixed heads: argmax logit, or the Gaussian mode."""
        if self.action_continuous:
            mu, _ = self._mixed_gaussian(obs_t.unsqueeze(0), w)
            if self.act_low is not None and self.act_high is not None:
                mu = torch.max(torch.min(mu, self.act_high), self.act_low)
            return mu.squeeze(0).cpu().numpy().astype(np.float32)
        logits = self._mixed_logits(obs_t.unsqueeze(0), w)
        return int(torch.argmax(logits, dim=-1).item())

    def evaluate_identifying(self, env, coin, n_episodes: int = 100, max_steps: int = 200,
                             w_threshold: float = 0.9, deterministic: bool = True,
                             seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Self-identifying evaluation: the agent must find its own head, from scratch, per
        episode.

        Everything is frozen -- no gradient, no optimiser step, and COIN is only ever
        QUERIED (no ``observe_y``, no ``observe_q``, no ``sigma_sensory_noise`` write), so a
        checkpoint can be evaluated on every task without the evaluation itself changing
        what the next checkpoint sees. The three COIN queries
        (``stationary_context_probabilities``, ``context_alignment`` via
        :meth:`_segment_context_gaussians`, and ``sigma_motor_noise``) are read ONCE per
        call: they are functions of the model's particle state, which nothing here advances.

        Each episode starts from the stationary prior (:meth:`_eval_prior_pi`) with an empty
        natural-parameter posterior ``eta1 = eta2 = 0``, then folds one encoder factor per
        observed transition and re-derives the acting weights through
        :meth:`_step_context_weights` -> :meth:`_policy_weights`. The step-0 action is
        therefore taken under the prior alone and the belief sharpens from the episode's own
        first transitions -- which makes every cell of the Figure-3 heatmap a joint
        identification and performance test.

        Args:
            env: One environment for the task being evaluated. Reset once per episode.
            coin: The ``RealTimeCOIN`` model, queried read-only.
            n_episodes (int): Episodes to run.
            max_steps (int): Hard cap per episode, on top of the env's own time limit.
            w_threshold (float): Sharpening criterion -- the first step whose max acting
                weight exceeds it.
            deterministic (bool): Greedy actions (default). False samples the mixed policy,
                which consumes torch RNG.
            seed (Optional[int]): If given, episode ``i`` resets with ``seed + i``, making
                the evaluation reproducible from the rep seed (see :func:`seed_everything`).

        Returns:
            dict: ``returns`` ``[n_episodes]``, ``sharpen_step`` ``[n_episodes]`` (NaN where
            the belief never crossed ``w_threshold``), ``steps`` ``[n_episodes]`` and
            ``w_mean`` ``[n_episodes, num_contexts]`` (the step-averaged acting weights, i.e.
            which head actually drove the episode).
        """
        W = self.num_contexts
        # Same observation space as training: rank population from the current encoder.
        # Re-encoding replay segments reads the buffer and the encoder; it advances
        # neither, so the evaluation stays frozen.
        self._refresh_code_population()
        # ---- read-only COIN queries, once per call ----
        k = min(int(coin.context_alignment()["K"]), W - 1)
        ctx_mu, ctx_var = self._segment_context_gaussians(coin, k)
        floor2 = float(getattr(coin, "sigma_motor_noise", 0.0)) ** 2
        pi_agent = self._eval_prior_pi(coin, k)
        prior_prec = 1.0 / (self.prior_sd ** 2)

        returns = np.zeros(int(n_episodes), dtype=float)
        steps = np.zeros(int(n_episodes), dtype=float)
        sharpen = np.full(int(n_episodes), np.nan)
        w_mean = np.zeros((int(n_episodes), W), dtype=float)

        with torch.no_grad():
            for ep in range(int(n_episodes)):
                obs, _ = env.reset() if seed is None else env.reset(seed=int(seed) + ep)
                obs_t = self._flatten_obs(obs)
                eta1 = eta2 = 0.0
                w_t = self._policy_weights(pi_agent)
                ep_ret, w_sum, n_steps = 0.0, np.zeros(W), 0

                for t in range(int(max_steps)):
                    w_np = w_t.cpu().numpy()
                    w_sum += w_np
                    n_steps += 1
                    if np.isnan(sharpen[ep]) and float(w_np.max()) > w_threshold:
                        sharpen[ep] = float(t)

                    if deterministic:
                        action = self._deterministic_action(obs_t, w_t)
                    else:
                        action, _, _, _ = self.act(obs_t, w_t)
                    next_obs, reward, done, trunc, _ = env.step(action)
                    next_t = self._flatten_obs(next_obs)

                    # Fold this transition's factor in; the weights it yields act NEXT step,
                    # exactly as in train_step.
                    mu_f, sig_f = self.encoder.factors(
                        self._segment_features([obs_t.cpu()], [action], [next_t.cpu()],
                                               rew=[float(reward)]))
                    iv = float(1.0 / (sig_f * sig_f))
                    eta2 += iv
                    eta1 += float(mu_f) * iv
                    var_t = 1.0 / (eta2 + prior_prec)
                    z_t, sd_t = self._obs_transform(eta1 * var_t, float(np.sqrt(var_t)))
                    w_t = self._policy_weights(self._step_context_weights(
                        pi_agent, z_t, sd_t, ctx_mu, ctx_var, floor2))

                    ep_ret += float(reward)
                    obs_t = next_t
                    if done or trunc:
                        break

                returns[ep] = ep_ret
                steps[ep] = float(n_steps)
                w_mean[ep] = w_sum / max(n_steps, 1)

        return {"returns": returns, "sharpen_step": sharpen, "steps": steps,
                "w_mean": w_mean}
