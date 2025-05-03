"""
rl.py

This module contains various Reinforcement Learning (RL) algorithms and helper functions,
intended for use with Gymnasium environments. It provides a template for integrating
and organizing different RL methods in one place.
"""

import numpy as np
import gymnasium as gym
from typing import Tuple, List
from environments import CustomMountainCarEnv


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
            init_Q_random (bool, optional): When True, initialise Q-table randomly, otherwise to zeros.
        """
        self.num_position_bins = num_position_bins
        self.num_velocity_bins = num_velocity_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Extract state boundaries (assuming a 2D state: [position, velocity])
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

    def choose_action(self, env: gym.Env, state: Tuple[int, int], eps: float) -> int:
        """
        Choose an action using an ε-greedy policy.

        Args:
            state (Tuple[int, int]): Discretized state indices.
            eps (float): Epsilon value for exploration.

        Returns:
            int: Action chosen by the policy.
        """
        if np.random.random() < eps:
            return env.action_space.sample()
        return np.argmax(self.Q[state])

    def update_q_table(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int]
    ) -> None:
        """
        Update the Q-table using the Q-learning update rule.

        Args:
            state (Tuple[int, int]): Current discretized state indices.
            action (int): Action taken.
            reward (float): Reward received from the environment.
            next_state (Tuple[int, int]): Next discretized state indices.
        """
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state][best_next_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    def train_step(
            self,
            env: gym.Env,
            max_steps_per_episode: int = 200   
    ) -> float:
        """
        Train the Q-learning agent in its environment using an ε-greedy policy through one episode.

        Args:
            env (gym.Env): An initialized Gymnasium environment.
            max_steps_per_episode (int, optional): Maximum steps to run in each episode.

        Returns:
            float: 
                - The total episode reward.
        """
        obs, _ = env.reset()
        state = discretize_state(obs, self.position_bins, self.velocity_bins)
        episode_reward = 0.0

        for _ in range(max_steps_per_episode):
            # Choose action
            action = self.choose_action(env, state, self.epsilon)

            # Step in the environment
            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = discretize_state(next_obs, self.position_bins, self.velocity_bins)

            # Update Q-table
            self.update_q_table(state, action, reward, next_state)

            state = next_state
            episode_reward += reward

            if done or truncated:
                break

        # Epsilon decay
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        # End of training
        env.close()
        return episode_reward

    def train(
        self,
        env : gym.Env,
        n_episodes: int = 5000,
        max_steps_per_episode: int = 200,
        verbose: bool = False,
        print_freq: int = 200
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Train the Q-learning agent in its environment using an ε-greedy policy.

        Args:
            n_episodes (int, optional): Number of training episodes.
            max_steps_per_episode (int, optional): Maximum steps to run in each episode.
            verbose (bool, optional): Whether to print intermediate training info.
            print_freq (int, optional): Frequency for printing training progress.

        Returns:
            Tuple[np.ndarray, List[float]]:
                - The learned Q-table.
                - A list of accumulated rewards per episode.
        """
        all_episode_rewards = []

        for episode in range(n_episodes):
            # Train for the initial episode
            episode_reward = self.train_step(env, max_steps_per_episode)

            all_episode_rewards.append(episode_reward)

            # Optional debug output
            if verbose and (episode + 1) % print_freq == 0:
                avg_reward = np.mean(all_episode_rewards[-print_freq:])
                print(
                    f"Episode: {episode + 1}, "
                    f"Avg Reward (last {print_freq}): {avg_reward:.2f}, "
                    f"Epsilon: {self.epsilon:.3f}"
                )

        # End of training
        env.close()
        return self.Q, all_episode_rewards

    def evaluate(
        self,
        env: gym.Env,
        n_episodes: int = 10,
        max_steps_per_episode: int = 200
    ) -> List[float]:
        """
        Execute the learned policy (greedy w.r.t. the Q-table) to evaluate performance.
        This method does not update the Q-table.

        Args:
            n_episodes (int, optional): Number of episodes to run for evaluation.
            max_steps_per_episode (int, optional): Maximum steps to run in each episode.

        Returns:
            List[float]: Total rewards for each of the evaluation episodes.
        """
        rewards = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            state = discretize_state(obs, self.position_bins, self.velocity_bins)
            episode_reward = 0.0

            for _ in range(max_steps_per_episode):
                # Always pick the best action - greedy
                action = self.choose_action(env, state, 0.0)
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
    ):
        """
        Initialize the COIN Q-learning agent with hyperparameters and bin settings.

        Args:
            env (gym.Env): An initialized Gymnasium environment.
            num_position_bins (int, optional): Number of bins for discretizing position.
            num_velocity_bins (int, optional): Number of bins for discretizing velocity.
            alpha (float, optional): Learning rate.
            gamma (float, optional): Discount factor.
            epsilon (float, optional): Initial epsilon for all ε-greedy strategies.
            epsilon_decay (float, optional): Epsilon decay factor after each episode.
            min_epsilon (float, optional): Minimum value of epsilon.
            init_Q_random (bool, optional): When True, initialise Q-table randomly, otherwise to zeros.
        """
        self.num_position_bins = num_position_bins
        self.num_velocity_bins = num_velocity_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Extract state boundaries (assuming a 2D state: [position, velocity])
        self.position_min, self.position_max = env.low[0], env.high[0]
        self.velocity_min, self.velocity_max = env.low[1], env.high[1]

        # Create bins
        self.position_bins = np.linspace(self.position_min, self.position_max, self.num_position_bins)
        self.velocity_bins = np.linspace(self.velocity_min, self.velocity_max, self.num_velocity_bins)

        # Initialize Q-table and exploration database
        n_actions = env.action_space.n
        if init_Q_random:
            self.Qdat = [np.random.uniform(
                low=-2, high=0, size=(self.num_position_bins, self.num_velocity_bins, n_actions)
            ) for _ in range(max_contexts+1)] # Also set one up for the novel context
        else:
            self.Qdat = [np.zeros((self.num_position_bins, self.num_velocity_bins, n_actions)) for _ in range(max_contexts+1)]
        
        # Track which contexts have been initialised - only novel initialised initially
        self.context_init = np.zeros((max_contexts+1,))
        self.context_init[-1] = 1

        self.epsdat = [epsilon for _ in range(max_contexts)] # Epsilon for each context (not novel)
        

    def choose_action(self, env: gym.Env, Q: np.ndarray, state: Tuple[int, int], eps: float) -> int:
        """
        Choose an action using an ε-greedy policy.

        Args:
            state (Tuple[int, int]): Discretized state indices.
            eps (float): Epsilon value for exploration.

        Returns:
            int: Action chosen by the policy.
        """
        rand = np.random.random()
        if rand < eps:
            return env.action_space.sample()
        return np.argmax(Q[state])

    def update_q_table(
        self,
        Qavg,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
        p_context: np.ndarray
    ) -> None:
        """
        Update the Q-tables using the COIN Q-learning update rule.

        Args:
            state (Tuple[int, int]): Current discretized state indices.
            action (int): Action taken.
            reward (float): Reward received from the environment.
            next_state (Tuple[int, int]): Next discretized state indices.
            p_context (np.ndarray): Probability of each context in the model.
        """
        best_next_action = np.argmax(Qavg[next_state])
        for i in range(len(self.Qdat)):
            if self.context_init[i] and not np.isnan(p_context[i]):
                td_target = reward + self.gamma * self.Qdat[i][next_state][best_next_action]
                td_error = td_target - self.Qdat[i][state][action]
                self.Qdat[i][state][action] += p_context[i] * self.alpha * td_error

    def instantiate_context_Q(
            self,
            new_context,
    ):
        """When a novel context is instantiated, copy current Q novel table to that new context value."""
        self.Qdat[new_context] = self.Qdat[-1].copy()
        
    def train_step(
        self,
        env : gym.Env,
        p_context: np.ndarray,
        max_steps_per_episode: int = 200
    ) -> float:
        """
        Train the COIN Q-learning agent in its environment using an ε-greedy policy through one episode.

        Args:
            env (gym.Env): An initialized Gymnasium environment.
            p_context (np.ndarray): Probability of each context in the model.
            max_steps_per_episode (int, optional): Maximum steps to run in each episode.

        Returns:
            float: 
                - The total episode reward.
        """

        obs, _ = env.reset()
        state = discretize_state(obs, self.position_bins, self.velocity_bins)
        episode_reward = 0.0

        # Check if new context initialised
        for i, init in enumerate(self.context_init):
            if init == 0 and not np.isnan(p_context[i]):
                # Context initialised
                self.instantiate_context_Q(i)
                # Update tracking
                self.context_init[i] = 1

        for _ in range(max_steps_per_episode):
            # Find average Q and eps
            Qavg = np.zeros_like(self.Qdat[0])
            epsavg = 0.0
            for i in range(len(self.Qdat)):
                ctx_exp = self.epsdat[i] if i < len(self.epsdat) else 1.0
                if self.context_init[i] and not np.isnan(p_context[i]):
                    Qavg += p_context[i] * self.Qdat[i]
                    epsavg += p_context[i] * ctx_exp

            # Choose action
            action = self.choose_action(env, Qavg, state, epsavg)

            # Step in the environment
            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = discretize_state(next_obs, self.position_bins, self.velocity_bins)

            # Update Q-table
            self.update_q_table(Qavg, state, action, reward, next_state, p_context)

            state = next_state
            episode_reward += reward

            if done or truncated:
                break

        # Epsilon decay - for every context
        for i in range(len(self.epsdat)):
            if self.context_init[i] and not np.isnan(p_context[i]):
                self.epsdat[i] = max(self.min_epsilon, self.epsdat[i] * self.epsilon_decay**(p_context[i]))

        # End of training
        env.close()
        return episode_reward

    def evaluate(
        self,
        env: gym.Env,
        p_context: np.ndarray,
        n_episodes: int = 10,
        max_steps_per_episode: int = 200
    ) -> List[float]:
        """
        Execute the learned policy (greedy w.r.t. the Q-table) to evaluate performance.
        This method does not update the Q-tables.

        Args:
            env (gym.Env): An initialized Gymnasium environment.
            p_context (np.ndarray): Probability of each context in the model.
            n_episodes (int, optional): Number of episodes to repeat for evaluation - display.
            max_steps_per_episode (int, optional): Maximum steps to run in each episode.

        Returns:
            List[float]: Total rewards for each of the evaluation episodes.
        """
        rewards = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            state = discretize_state(obs, self.position_bins, self.velocity_bins)
            episode_reward = 0.0

            for _ in range(max_steps_per_episode):
                # Find average Q and eps
                Qavg = np.zeros(self.Qdat[0].shape)
                for i in range(len(self.Qdat)):
                    Qavg += p_context[i] * self.Qdat[i]

                # Always pick the best action - greedy
                action = self.choose_action(env, Qavg, state, 0.0)
                next_obs, reward, done, truncated, _ = env.step(action)
                next_state = discretize_state(next_obs, self.position_bins, self.velocity_bins)

                episode_reward += reward
                state = next_state

                if done or truncated:
                    break

            rewards.append(episode_reward)

        env.close()
        return rewards
