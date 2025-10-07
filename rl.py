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

        # Initialize Q-table database (append one extra for the novel context)
        n_actions = env.action_space.n
        if init_Q_random:
            self.Qdat = [np.random.uniform(
                low=-2, high=0, size=(self.num_position_bins, self.num_velocity_bins, n_actions)
            ) for _ in range(max_contexts + 1)]
        else:
            self.Qdat = [np.zeros((self.num_position_bins, self.num_velocity_bins, n_actions))
                         for _ in range(max_contexts + 1)]

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

    def choose_action(self, env: gym.Env, Q: np.ndarray, state: Tuple[int, int], eps: float) -> int:
        """
        Choose an action using the configured exploration strategy.
        For ε-greedy, 'eps' overrides the strategy's epsilon to support per-context averaging.
        For other strategies, 'eps' is ignored.
        """
        q_row = Q[state]
        epsilon_override = eps if isinstance(self.strategy, EpsilonGreedy) else None
        return self.strategy.select_action(
            q_row,
            env.action_space,
            epsilon_override=epsilon_override,
            rng=self.rng,
        )

    def update_q_table(
        self,
        Qavg: np.ndarray,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
        p_context: np.ndarray
    ) -> None:
        """
        Update the Q-tables using the COIN Q-learning update rule.
        """
        best_next_action = np.argmax(Qavg[next_state])
        Z = np.nansum(p_context ** 2)  # normalizing constant for learning rates
        for i in range(len(self.Qdat)):
            if self.context_init[i] and not np.isnan(p_context[i]):
                td_target = reward + self.gamma * Qavg[next_state][best_next_action]
                td_error = td_target - Qavg[state][action]
                p = float(p_context[i])
                self.Qdat[i][state][action] += p * self.alpha * td_error / max(Z, 1e-8)

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
        if p_context.shape[0] < len(self.context_init):
            B = np.array([np.nan] * (len(self.context_init) - p_context.shape[0]))
            B = np.expand_dims(B, axis=-1)
            p_context = np.vstack([p_context[:-1], B, p_context[-1:]])

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

        for _ in range(max_steps_per_episode):
            # If "avoid_novel" is True, attempt to ignore novel context for action selection
            action_probs = p_context.copy()
            if self.avoid_novel and np.nansum(action_probs[:-1]) > 0:
                action_probs[:-1] = action_probs[:-1] / np.nansum(action_probs[:-1])
                action_probs[-1] = 0.0

            # Compute averaged Q and averaged epsilon (for ε-greedy only)
            if average_bias is None:
                Qavg = np.zeros_like(self.Qdat[0])
            else:
                Qavg = average_bias.copy()
            epsavg = 0.0
            for i in range(len(self.Qdat)):
                if self.context_init[i] and not np.isnan(action_probs[i]):
                    Qavg += action_probs[i] * self.Qdat[i]
                    # context epsilon (ε-greedy only); for novel (index == last), use max_epsilon
                    ctx_eps = self.epsdat[i] if i < len(self.epsdat) else self.max_epsilon
                    epsavg += action_probs[i] * ctx_eps

            # Choose action via the pluggable strategy
            action = self.choose_action(env, Qavg, state, epsavg)

            # Step and update
            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = discretize_state(next_obs, self.position_bins, self.velocity_bins)
            self.update_q_table(Qavg, state, action, reward, next_state, p_context)

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

        if ignore_novel and np.nansum(p_context[:-1]) > 0:
            p_context = p_context.copy()
            p_context[:-1] = p_context[:-1] / (np.nansum(p_context[:-1]) + 1e-4)
            p_context[-1] = 0.0

        for _ in range(n_episodes):
            obs, _ = env.reset()
            state = discretize_state(obs, self.position_bins, self.velocity_bins)
            episode_reward = 0.0

            for _ in range(max_steps_per_episode):
                # Build averaged Q over contexts
                if average_bias is None:
                    Qavg = np.zeros_like(self.Qdat[0])
                else:
                    Qavg = average_bias.copy()
                for i in range(len(self.Qdat)):
                    if self.context_init[i] and not np.isnan(p_context[i]):
                        Qavg += p_context[i] * self.Qdat[i]

                # Greedy evaluation independent of exploration strategy
                action = int(np.argmax(Qavg[state]))
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
        Qavg: np.ndarray,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
        p_context: np.ndarray
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
        super().update_q_table(Qavg, state, action, reward, next_state, p_context)

        # Body inhibition on head values
        for i in range(len(self.Qdat)):
            if self.context_init[i] and not np.isnan(p_context[i]):
                self.Qdat[i][state][action] -= self.alpha_body * td_error_body

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
    call train_step(env, n_steps) to collect a rollout and do one update.
    """
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        lr: float = 3e-4,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        device: str = "cpu",
        action_continuous: bool = False
    ):
        self.gamma, self.lam = gamma, gae_lambda
        self.clip_eps, self.ent_coef, self.vf_coef = clip_eps, ent_coef, vf_coef
        self.device = device
        
        self.action_continuous = action_continuous

        self.policy = _MLP(obs_dim, act_dim).to(device)
        self.value_net = _MLP(obs_dim, 1).to(device)
        self.optim = optim.Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()),
            lr=lr
        )

    # --------------- utilities -----------------
    def _act(self, obs: torch.Tensor):
        logits = self.policy(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        action_np = action.detach().cpu().numpy()
        logp = dist.log_prob(action)
        if self.action_continuous and action_np.shape == ():  # scalar case
            action_np = np.array([action_np])  # wrap in 1D array
        else:
            action_np = action_np.item()

        return action_np, logp, dist.entropy(), logits

    def _compute_advantages(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        last_value: float
    ):
        adv, gae = [], 0.0
        # GAE backwards
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * (1 - dones[t]) * last_value - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            adv.insert(0, gae)
            last_value = values[t]
        returns = [a + v for a, v in zip(adv, values)]
        return torch.tensor(adv, device=self.device), torch.tensor(returns, device=self.device)

    # --------------- main public API -----------------
    def train_step(self, env, rollout_steps: int = 2048, mini_epochs: int = 10, mb_size: int = 64):
        obs = env.reset()[0]  # gymnasium returns (obs, info)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

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
                next_obs, _ = env.reset()
            obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)

        # ---------- advantages ----------
        with torch.no_grad():
            last_val = self.value_net(obs_t).squeeze().item()
        adv, ret = self._compute_advantages(rew_buf, val_buf, done_buf, last_val)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # normalise
        adv = torch.tensor(adv, device=self.device)
        ret = torch.tensor(ret, device=self.device)

        # ---------- optimisation ----------
        dataset_size = rollout_steps
        idxs = torch.randperm(dataset_size)
        obs_tensor = torch.stack(obs_buf).to(self.device)
        act_tensor = torch.tensor(act_buf, dtype=torch.long, device=self.device)
        old_logp_tensor = torch.stack(logp_buf).to(self.device)

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
                logits = self.policy(batch_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(batch_act)
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_logp - batch_old_logp)

                # Clipped surrogate
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                value_pred = self.value_net(batch_obs).squeeze()
                critic_loss = (batch_ret - value_pred).pow(2).mean()

                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

        mean_ep_return = float(np.mean(ep_returns)) if ep_returns else 0.0

        return {
                    "mean_episode_return": mean_ep_return,
                    "mean_reward_per_step": np.mean(rew_buf),
                    "value_loss": critic_loss.item(),
                    "policy_loss": actor_loss.item(),
               }  #  for logging
    
    def evaluate(
        self,
        env: gym.Env,
        n_episodes: int = 2,
        max_steps_per_episode: int = 200,
    ) -> List[float]:
        """
        Execute the learned policies to evaluate performance.
        This method does not train the model.

        Args:
            n_episodes (int, optional): Number of episodes to run for evaluation.
            max_steps_per_episode (int, optional): Maximum steps to run in each episode.

        Returns:
            List[float]: Total rewards for each of the evaluation episodes.
        """
        rewards = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            episode_reward = 0.0
            done = False
            truncated = False

            for _ in range(max_steps_per_episode):
                # Always pick the best action - greedy
                action = self._act(obs_t)[0] # Obtain only action
                next_obs, reward, done, truncated, _ = env.step(action)
                obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)

                episode_reward += reward

                if done or truncated:
                    break

            rewards.append(episode_reward)

        env.close()
        return rewards




class COINPPOAgent(PPOAgent):
    """
    COINstyle contextual PPO.
    Keeps one policy+value pair *per context* and mixes them using p(c|s).
    context_probs_fn(s) must return a Dict[int, float] mapping context id -> prob. In general this
    takes in a vector of observations s. For simplicity here, we split COIN from the model, and allow them to 
    connect via s being the episode number of the current epoch.
    """
    def __init__(self, base_obs_dim: int, act_dim: int, ctx_ids: dict, action_continuous: bool = False, **kwargs):
        super().__init__(base_obs_dim, act_dim, **kwargs)  # create *dummy* nets
        # override: keep dicts of networks
        self.context_nets: Dict[int, Tuple[nn.Module, nn.Module, optim.Optimizer]] = {}
        self.act_dim = act_dim
        self.base_obs_dim = base_obs_dim
        self.lr = kwargs.get("lr", 3e-4)

        self.action_continuous = action_continuous

        # track which contexts have been initialised - only novel initialised initially
        self.context_init = {}
        for ctx in ctx_ids:
            self.context_init[ctx] = 0
        self.context_init["novel"] = 1  # always have a 'novel' context
        
        # Create initial 'novel' context networks
        policy = _MLP(base_obs_dim, act_dim).to(self.device)
        value_net = _MLP(base_obs_dim, 1).to(self.device)
        opt = optim.Adam(
            list(policy.parameters()) + list(value_net.parameters()),
            lr=self.lr
        )
        self.context_nets["novel"] = (policy, value_net, opt)

    # -------- helper to mix outputs ---------
    def _mixed_outputs(self, obs_t: torch.Tensor, ctx_probs: Dict[int, float]):
        """
        Return context weighted logits and value.
        """
        logits_list, value_list, weight_list = [], [], []

        for cid, p_c in ctx_probs.items():
            if self.context_init[cid] == 0 or np.isnan(p_c) or p_c == 0.0:
                continue
            policy, value_net, _ = self.context_nets[cid]
            logits_list.append(policy(obs_t))        # [A]  requires_grad = True
            value_list.append(value_net(obs_t))      # [1]  requires_grad = True
            weight_list.append(p_c)

        if not logits_list:
            raise RuntimeError("All context probabilities were NaN or zero.")

        # Stack and weight
        logits   = torch.stack(logits_list)                    # [C, A]
        values   = torch.stack(value_list).squeeze(-1)         # [C]
        weights  = torch.tensor(weight_list, device=self.device)  # [C]

        mixed_logits = (weights[:, None] * logits).sum(dim=0)  # [A]
        mixed_value  = (weights * values).sum()                # scalar

        return mixed_logits, mixed_value
    
    def instantiate_context_net(
            self,
            new_cid,
    ):
        """When a novel context is instantiated, copy novel network to the new context value"""
        pnovel, vn_novel, _ = self.context_nets["novel"]
        policy = copy.deepcopy(pnovel).to(self.device)
        value_net = copy.deepcopy(vn_novel).to(self.device)
        opt = optim.Adam(
            list(policy.parameters()) + list(value_net.parameters()),
            lr=self.lr
        )
        self.context_nets[new_cid] = (policy, value_net, opt)

    # ------------- public API -------------
    def act(self, obs: torch.Tensor, ctx_probs: Dict[int, float]):
        logits, _ = self._mixed_outputs(obs, ctx_probs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        action_np = action.detach().cpu().numpy()

        if action_np.shape == ():  # scalar case
            action_np = np.array([action_np])  # wrap in 1D array

        return action_np

    def train_step(
        self,
        env,
        context_probs_fn,
        rollout_steps: int = 2048,
        mini_epochs: int = 10,
        mb_size: int = 64,
    ):
        """
        context_probs_fn: lambda eps_num -> {context_id: prob}
        Otherwise same interface as PPOAgent.
        """
        obs = env.reset()[0]
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        ep_returns = []     # collect episodic returns for logging
        ep_len = ep_ret = ep_num = 0

        storage: Dict[str, List[Any]] = defaultdict(list)

        # ---------- rollout ----------
        for _ in range(rollout_steps):
            ctx_probs = context_probs_fn(ep_num)
            # Check if new context initialised
            for ctx, init in self.context_init.items():
                if init == 0 and not np.isnan(ctx_probs[ctx]):
                    # Context initialised
                    self.instantiate_context_net(ctx)
                    # Update tracking
                    self.context_init[ctx] = 1

            logits, value_est = self._mixed_outputs(obs_t, ctx_probs)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            action_np = action.detach().cpu().numpy()

            if self.action_continuous and action_np.shape == ():  # scalar case
                action_np = np.array([action_np])  # wrap in 1D array
            else:
                action_np = action_np.item()

            logp = dist.log_prob(action)
            entropy = dist.entropy()

            next_obs, reward, done, trunc, _ = env.step(action_np)

            # store (we also keep ctx_probs to weight backprop)
            storage["obs"].append(obs_t.detach().cpu())
            storage["act"].append(action.detach().cpu())
            storage["logp"].append(logp.detach().cpu())
            storage["rew"].append(reward)
            storage["val"].append(value_est.detach().cpu())
            storage["done"].append(done or trunc)
            storage["ctx_probs"].append(ctx_probs)

            ep_ret += reward
            ep_len += 1

            if done or trunc:
                next_obs, _ = env.reset()
                ep_returns.append(ep_ret)
                ep_len = ep_ret = 0
                next_obs, _ = env.reset()
                ep_num += 1

            obs, obs_t = next_obs, torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)

        # ---------- advantages ----------
        with torch.no_grad():
            last_ctx_probs = context_probs_fn(obs)
            _, last_val = self._mixed_outputs(obs_t, last_ctx_probs)
            last_val = last_val.item()

        adv, ret = self._compute_advantages(storage["rew"], storage["val"], storage["done"], last_val)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ---------- optimisation per context ----------
        # For simplicity: compute global indices and then distribute grads by importance sampling
        dataset_size = rollout_steps
        idxs = torch.randperm(dataset_size)
        for _ in range(mini_epochs):
            for start in range(0, dataset_size, mb_size):
                end = start + mb_size
                mb_idx = idxs[start:end]

                # For each initialised context, gather the subset where p(c|s) > 0 and not NaN
                ctx_grad_accum = {cid: [] for cid in self.context_nets}
                for j in mb_idx:
                    ctx_probs = storage["ctx_probs"][j]
                    for cid, p_c in ctx_probs.items():
                        if self.context_init == 0 or np.isnan(p_c):
                            continue
                        ctx_grad_accum[cid].append((j, p_c))

                # Iterate contexts and do local PPO update (weighted by p_c)
                for cid, items in ctx_grad_accum.items():
                    if not items:  # no samples for this context in this minibatch
                        continue
                    j_idx = torch.tensor([j for j, _ in items], device=self.device)
                    weights = torch.tensor([w for _, w in items], device=self.device)

                    if not weights.sum():
                        # all weights are zero, skip this context
                        continue

                    policy, value_net, opt = self.context_nets[cid]

                    batch_obs = torch.stack([storage["obs"][k] for k in j_idx]).to(self.device)
                    batch_act = torch.stack([storage["act"][k] for k in j_idx]).to(self.device)
                    batch_old_logp = torch.stack([storage["logp"][k] for k in j_idx]).to(self.device)
                    batch_adv = adv[j_idx]
                    batch_ret = ret[j_idx]

                    # We scale by the weights to decrease the learning rate proportional to the context probabilities
                    logits = policy(batch_obs)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logp = dist.log_prob(batch_act.squeeze())
                    entropy = (dist.entropy()*weights).mean()
                    ratio = torch.exp(new_logp - batch_old_logp)

                    surr1 = ratio * batch_adv
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                    actor_loss = -(torch.min(surr1, surr2)*weights).mean()
                    value_pred = value_net(batch_obs).squeeze()
                    critic_loss = ((batch_ret - value_pred).pow(2)*weights).mean()

                    loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

        mean_ep_return = float(np.mean(ep_returns)) if ep_returns else 0.0

        return {
                    "mean_episode_return": mean_ep_return,
                    "mean_reward_per_step": np.mean(storage["rew"]),
                    "value_loss": critic_loss.item(),
                    "policy_loss": actor_loss.item(),
               }  #  for logging
    
    def evaluate(
        self,
        env: gym.Env,
        context_probs_fn,
        n_episodes: int = 2,
        ignore_novel: bool = False,
    ) -> List[float]:
        """
        Execute the learned policies to evaluate performance.
        This method does not train the model.

        Args:
            n_episodes (int, optional): Number of episodes to run for evaluation.
            context_probs_fn: lambda obs -> {context_id: prob}
            max_steps_per_episode (int, optional): Maximum steps to run in each episode.

        Returns:
            List[float]: Total rewards for each of the evaluation episodes.
        """
        rewards = []
        ctx_probs = context_probs_fn(0) # dummy call to get context probabilities

        ctx_array = np.array(list(ctx_probs.values()))

        if ignore_novel and np.nansum(ctx_array[:-1]) > 0:
            ctx_sum = np.nansum(ctx_array[:-1]) + 1e-4
            for ctx in ctx_probs:
                if ctx != 'novel':
                    ctx_probs[ctx] = ctx_probs[ctx] / ctx_sum
                else:
                    ctx_probs[ctx] = 0.0

        for _ in range(n_episodes):
            obs, _ = env.reset()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            episode_reward = 0.0
            done = False
            trunc = False

            while not done and not trunc:
                logits, value_est = self._mixed_outputs(obs_t, ctx_probs)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                action_np = action.detach().cpu().numpy()

                if self.action_continuous and action_np.shape == ():  # scalar case
                    action_np = np.array([action_np])  # wrap in 1D array
                else:
                    action_np = action_np.item()

                next_obs, reward, done, trunc, _ = env.step(action_np)
                obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)

                episode_reward += reward

            rewards.append(episode_reward)

        env.close()
        return rewards
