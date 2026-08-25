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

    A shared MLP maps every transition ``(s, a, r, s')`` to its own Gaussian factor; the
    posterior is their product, held in natural parameters so every *prefix* posterior is a
    running sum -- one ``cumsum`` per segment. Means are ``z_scale * tanh(.)``, so the sample
    cannot leave COIN's dynamic range. ``prior_sd`` is both the ``k = 0`` posterior and the
    KL reference.
    """

    def __init__(self, obs_dim: int, act_dim: int, action_continuous: bool, hidden: int = 64,
                 z_scale: float = 0.15, prior_sd: float = 1.0, sigma_min: float = 0.05,
                 sigma_max: float = 20.0):
        super().__init__()
        self.obs_dim, self.act_dim = int(obs_dim), int(act_dim)
        self.action_continuous = bool(action_continuous)
        self.z_scale, self.prior_sd = float(z_scale), float(prior_sd)
        self.log_sigma_min, self.log_sigma_max = float(np.log(sigma_min)), float(np.log(sigma_max))
        self.in_dim = 2 * self.obs_dim + self.act_dim + 1          # (s, a_repr, r, s')
        self.net = _MLP(self.in_dim, 2, hidden)

        # A zero final layer makes every mu_t exactly zero at initialisation, so an untrained
        # encoder emits a constant z and buys COIN no spurious contexts. The dynamics
        # gradient survives it, because tanh'(0) = 1.
        last = self.net.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        with torch.no_grad():
            last.bias[1] = 1.0    # sigma ~ 2.72: uninformative, well inside the clamps

    def transition_features(self, obs, act, rew, next_obs) -> torch.Tensor:
        """``[T, in_dim]`` inputs ``concat(s, a_repr, r, s')``; ``next_obs`` is the post-step
        observation, captured before any episode reset."""
        dev = next(self.parameters()).device
        obs = obs.to(device=dev, dtype=torch.float32).view(-1, self.obs_dim)
        next_obs = next_obs.to(device=dev, dtype=torch.float32).view(-1, self.obs_dim)
        rew = rew.to(device=dev, dtype=torch.float32).view(-1, 1)
        if self.action_continuous:
            act_repr = act.to(device=dev, dtype=torch.float32).view(-1, self.act_dim)
        else:
            idx = act.to(device=dev).long().view(-1)
            act_repr = torch.nn.functional.one_hot(idx, self.act_dim).float()
        return torch.cat([obs, act_repr, rew, next_obs], dim=-1)

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
    dynamics unless ``z`` splits them.

    Args:
        encoder_hidden (int): Hidden width of both the encoder and the decoder MLP.
        z_scale (float): Bound on ``|mu_t|``, i.e. COIN's dynamic range for the latent.
        prior_sd (float): Prior sd, used both as the empty-prefix posterior and as the KL
            reference.
        avoid_novel (bool): Drop the novel column from the acting weights whenever the known
            contexts carry mass; see :meth:`_policy_weights`.
        kl_coef (float): Weight of the KL term in the encoder objective.
        **kwargs: Forwarded to :class:`COINPPOAgent`.
    """

    def __init__(self, env: gym.Env, ctx_ids: dict, encoder_hidden: int = 64,
                 encoder_lr: float = 3e-4, enc_grad_clip: Optional[float] = 1.0,
                 z_scale: float = 0.15, prior_sd: float = 1.0, avoid_novel: bool = True,
                 kl_coef: float = 1e-3, **kwargs):
        super().__init__(env, ctx_ids, **kwargs)
        self.z_scale, self.prior_sd = float(z_scale), float(prior_sd)
        self.avoid_novel, self.kl_coef = bool(avoid_novel), float(kl_coef)
        self.enc_grad_clip = enc_grad_clip
        self.encoder = ContingencyEncoder(
            self.obs_dim, self.act_dim, self.action_continuous, hidden=encoder_hidden,
            z_scale=z_scale, prior_sd=prior_sd).to(self.device)
        self.decoder = _MLP(self.obs_dim + self.act_dim + 1, self.obs_dim,
                            encoder_hidden).to(self.device)
        # Deliberately NOT in self._all_optimizers(): the encoder is trained by the dynamics
        # objective alone, and the PPO step must not touch it.
        self.enc_optim = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=encoder_lr)

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
        return self.decoder(torch.cat([feats[:, :self.obs_dim + self.act_dim],
                                       z.unsqueeze(-1)], dim=-1))

    def _segment_features(self, obs, act, rew, next_obs) -> torch.Tensor:
        """Stack one segment's stored transitions into encoder features ``[L, in_dim]``."""
        act_t = torch.stack([torch.as_tensor(a) for a in act])
        act_t = act_t.float() if self.action_continuous else act_t.long()
        return self.encoder.transition_features(
            torch.stack(obs), act_t, torch.as_tensor(rew, dtype=torch.float32),
            torch.stack(next_obs))

    # ------------------------------------------------------------------
    # Encoder / decoder training
    # ------------------------------------------------------------------
    def _update_encoder(self, feats: torch.Tensor, seg_len: int, mini_epochs: int,
                        mb_size: int) -> Dict[str, float]:
        """
        Fit encoder and decoder on ``L_dyn + kl_coef * KL``, minibatched over transitions.

        The posterior is recomputed from scratch every optimizer step (it depends on all
        parameters, so a cached one would go stale immediately) and resampled with fresh
        noise. Prefix column ``t`` is the posterior BEFORE transition ``t``, so the latent
        the decoder sees never contains the transition it has to predict.
        """
        n = feats.shape[0]
        obs_dim = self.obs_dim
        clip = float("inf") if self.enc_grad_clip is None else float(self.enc_grad_clip)
        dyn_val = kl_val = grad_norm = float("nan")

        for _ in range(mini_epochs):
            idxs = torch.randperm(n)
            for start in range(0, n, mb_size):
                mb = idxs[start:start + mb_size]
                mean, sd = self.encoder.prefix_posterior(feats, seg_len)
                z = (mean[:, :-1] + torch.randn_like(sd[:, :-1]) * sd[:, :-1]).reshape(-1)

                dyn = (self._decode_next_obs(feats[mb], z[mb])
                       - feats[mb, -obs_dim:]).pow(2).mean()
                kl = self._kl_to_prior(mean, sd).mean()
                loss = dyn + self.kl_coef * kl

                self.enc_optim.zero_grad()
                loss.backward()
                grad_norm = float(nn.utils.clip_grad_norm_(self.encoder.parameters(), clip))
                self.enc_optim.step()
                dyn_val, kl_val = float(dyn.item()), float(kl.item())

        return {"dyn_loss": dyn_val, "encoder_kl": kl_val, "enc_grad_norm": grad_norm}

    def pretrain_encoder(self, envs, seg_steps: int = 512, n_iters: int = 50,
                         mini_epochs: int = 10, mb_size: int = 64) -> Dict[str, np.ndarray]:
        """
        Train encoder and decoder on uniform-random rollouts, with no COIN and no PPO.

        The pretrain-then-RL regime: a latent learned before any policy exists, then frozen
        while the RL model trains online (``train_step(..., update_encoder=False)``). Random
        actions are the standard choice for dynamics pretraining -- they excite the
        transition model broadly, and the contingency signature is in the transitions, not
        in the returns. Pass one FRESH env per segment, as for :meth:`train_step`.

        Returns:
            Dict[str, np.ndarray]: Per-iteration ``dyn_loss``, ``encoder_kl`` and
            ``enc_grad_norm``.
        """
        L = int(seg_steps)
        history: Dict[str, List[float]] = defaultdict(list)

        for _ in range(n_iters):
            seg_feats = []
            for env in envs:
                obs_t = self._flatten_obs(env.reset()[0])
                obs, act, rew, nxt = [], [], [], []
                for _ in range(L):
                    action = env.action_space.sample()
                    next_obs, reward, done, trunc, _ = env.step(action)
                    obs.append(obs_t.cpu())
                    act.append(action)
                    nxt.append(self._flatten_obs(next_obs).cpu())
                    rew.append(float(reward))
                    if done or trunc:
                        next_obs, _ = env.reset()
                    obs_t = self._flatten_obs(next_obs)
                seg_feats.append(self._segment_features(obs, act, rew, nxt))

            for key, val in self._update_encoder(
                    torch.cat(seg_feats), L, mini_epochs, mb_size).items():
                history[key].append(val)

        return {k: np.asarray(v, dtype=float) for k, v in history.items()}

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def train_step(self, envs, coin, seg_steps: int = 512, mini_epochs: int = 10,
                   mb_size: int = 64, cues=None, update_encoder: bool = True):
        """
        Collect one segment-interleaved rollout and update on it.

        One call is ``S = len(envs)`` contiguous segments of ``seg_steps`` transitions, each
        a different task and its own COIN trial, so this method drives ``coin`` itself
        (``observe_q`` -> predicted pi -> segment -> ``observe_y``, advancing ``S`` trials).
        ``S = 1`` reproduces one task per rollout. Pass one FRESH env per segment --
        ``CustomCartPoleEnv.__init__`` computes its derived constants once, so mutating a
        live env leaves them stale -- and optionally one ``cues`` entry per segment.

        Args:
            update_encoder (bool): False freezes encoder and decoder and reports their
                losses under ``no_grad`` only. With a pretrained encoder that reduces this
                to the plain COIN-PPO routine with COIN stepped inside.

        Returns:
            dict: Per-segment arrays ``z``, ``z_sd``, ``K`` (pre-observation), ``pi``
            ``[S, W]``, ``rho`` ``[S, W]`` (post-observation diagnostics),
            ``mean_episode_return``, plus scalar ``mean_reward_per_step``, ``value_loss``,
            ``policy_loss``, ``dyn_loss``, ``encoder_kl``, ``enc_grad_norm``.
        """
        S, L = len(envs), int(seg_steps)
        enc, W = self.encoder, self.num_contexts
        store: Dict[str, List[Any]] = defaultdict(list)
        seg_feats, seg_w, seg_last_obs, seg_returns = [], [], [], []
        seg_z, seg_z_sd, seg_K, seg_pi, seg_rho = [], [], [], [], []

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
            w_s = self._policy_weights(pi_agent)      # CONSTANT for the whole segment
            seg_K.append(K)
            seg_pi.append(pi_agent)
            seg_w.append(w_s)

            obs_t = self._flatten_obs(env.reset()[0])
            obs, act, rew, nxt = [], [], [], []
            ep_returns, ep_ret = [], 0.0

            for _ in range(L):
                with torch.no_grad():
                    value = self._mixed_value(obs_t.unsqueeze(0), w_s)[0].item()
                    action_np, logp, _, _ = self.act(obs_t, w_s)
                next_obs, reward, done, trunc, _ = env.step(action_np)

                # next_obs is stored BEFORE any reset: letting the reset overwrite it
                # corrupts every episode-final transition with no visible symptom.
                obs.append(obs_t.cpu())
                act.append(action_np)
                nxt.append(self._flatten_obs(next_obs).cpu())
                rew.append(float(reward))
                store["logp"].append(logp.cpu())
                store["val"].append(value)
                store["done"].append(bool(done or trunc))

                ep_ret += reward
                if done or trunc:
                    next_obs, _ = env.reset()
                    ep_returns.append(ep_ret)
                    ep_ret = 0.0
                obs_t = self._flatten_obs(next_obs)

            store["obs"] += obs
            store["act"] += act
            store["rew"] += rew
            feats_s = self._segment_features(obs, act, rew, nxt)
            seg_feats.append(feats_s)
            seg_last_obs.append(obs_t)
            seg_returns.append(float(np.mean(ep_returns)) if ep_returns else 0.0)

            with torch.no_grad():
                mean_s, sd_s = enc.prefix_posterior(feats_s, L)
            seg_z.append(float(mean_s[0, -1]))
            seg_z_sd.append(float(sd_s[0, -1]))
            coin.observe_y(float(mean_s[0, -1]))       # plain float: COIN works in float64

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
        w_all = torch.stack(seg_w).repeat_interleave(L, dim=0)          # [S * L, W]

        adv_parts, ret_parts = [], []
        with torch.no_grad():
            for s in range(S):
                sl = slice(s * L, (s + 1) * L)
                last_val = self._mixed_value(seg_last_obs[s].unsqueeze(0), seg_w[s])[0].item()
                # GAE never crosses a task boundary: each segment bootstraps on its own.
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
            enc_stats = self._update_encoder(feats, L, mini_epochs, mb_size)
        else:
            with torch.no_grad():
                mean, sd = enc.prefix_posterior(feats, L)
                z = (mean[:, :-1] + torch.randn_like(sd[:, :-1]) * sd[:, :-1]).reshape(-1)
                dyn = (self._decode_next_obs(feats, z) - feats[:, -self.obs_dim:]).pow(2).mean()
                enc_stats = {"dyn_loss": float(dyn.item()),
                             "encoder_kl": float(self._kl_to_prior(mean, sd).mean().item()),
                             "enc_grad_norm": 0.0}

        # ---------- 5. diagnostics ----------
        self._weights_version += 1
        return {
            "z": np.asarray(seg_z, dtype=float), "z_sd": np.asarray(seg_z_sd, dtype=float),
            "K": np.asarray(seg_K, dtype=int),
            "pi": np.asarray(seg_pi, dtype=float), "rho": np.asarray(seg_rho, dtype=float),
            "mean_episode_return": np.asarray(seg_returns, dtype=float),
            "mean_reward_per_step": float(np.mean(store["rew"])),
            "value_loss": float(critic_loss.item()), "policy_loss": float(actor_loss.item()),
            **enc_stats,
        }
