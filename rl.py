"""
rl.py

This module contains various Reinforcement Learning (RL) algorithms and helper functions,
intended for use with Gymnasium environments. It provides a template for integrating
and organizing different RL methods in one place.
"""

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from typing import Dict, List, Tuple, Any
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
            epsilon (float, optional): Initial epsilon for -greedy strategy.
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
        Choose an action using an -greedy policy.

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
        Train the Q-learning agent in its environment using an -greedy policy through one episode.

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
        Train the Q-learning agent in its environment using an -greedy policy.

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
        instantiate_from_average: bool = False,
        avoid_novel: bool = False,
    ):
        """
        Initialize the COIN Q-learning agent with hyperparameters and bin settings.

        Args:
            env (gym.Env): An initialized Gymnasium environment.
            num_position_bins (int, optional): Number of bins for discretizing position.
            num_velocity_bins (int, optional): Number of bins for discretizing velocity.
            alpha (float, optional): Learning rate.
            gamma (float, optional): Discount factor.
            epsilon (float, optional): Initial epsilon for all -greedy strategies.
            epsilon_decay (float, optional): Epsilon decay factor after each episode.
            min_epsilon (float, optional): Minimum value of epsilon.
            init_Q_random (bool, optional): When True, initialise Q-table randomly, otherwise to zeros.
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
        Choose an action using an -greedy policy.

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
                if i==len(self.Qdat)-1:
                    # Fully update the novel context at any step
                    p = 1
                else:
                    p = p_context[i]
                self.Qdat[i][state][action] += p * self.alpha * td_error

    def instantiate_context_Q(
            self,
            new_context,
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
        self.epsdat[new_context] = self.max_epsilon # Set epsilon to max for new context
        
    def train_step(
        self,
        env : gym.Env,
        p_context: np.ndarray,
        max_steps_per_episode: int = 200
    ) -> float:
        """
        Train the COIN Q-learning agent in its environment using an -greedy policy through one episode.

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
                # Remove novel from initialisation - only relevant if self.instantiate_from_average is True
                instant_probs = p_context.copy()
                # Remove current context
                instant_probs[i] = 0.0
                if np.nansum(instant_probs[:-1]) > 0:
                    instant_probs[:-1] = instant_probs[:-1]/np.nansum(instant_probs[:-1])
                    instant_probs[-1] = 0.0
                else:
                    instant_probs[-1] = 1.0 # Ensure novel is 1.0
                # Context initialised
                self.instantiate_context_Q(i, probs=instant_probs)
                # Update tracking
                self.context_init[i] = 1

        for _ in range(max_steps_per_episode):
            # If "self.avoid_novel" is True, attempt to ignore novel context
            action_probs = p_context.copy()
            if self.avoid_novel and np.nansum(action_probs[:-1]) > 0:
                # Find probabilities scaled without novel for action selection - avoids instabilities
                action_probs[:-1] = action_probs[:-1]/np.nansum(action_probs[:-1])
                action_probs[-1] = 0.0
            # Find average Q and eps
            Qavg = np.zeros_like(self.Qdat[0])
            epsavg = 0.0
            for i in range(len(self.Qdat)):
                ctx_exp = self.epsdat[i] if i < len(self.epsdat) else self.max_epsilon # Novel context has eps=0.3
                if self.context_init[i] and not np.isnan(action_probs[i]):
                    Qavg += action_probs[i] * self.Qdat[i]
                    epsavg += action_probs[i] * ctx_exp

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
        max_steps_per_episode: int = 500,
        ignore_novel: bool = False,
    ) -> List[float]:
        """
        Execute the learned policy (greedy w.r.t. the Q-table) to evaluate performance.
        This method does not update the Q-tables.

        Args:
            env (gym.Env): An initialized Gymnasium environment.
            p_context (np.ndarray): Probability of each context in the model.
            n_episodes (int, optional): Number of episodes to repeat for evaluation - display.
            max_steps_per_episode (int, optional): Maximum steps to run in each episode.
            ignore_novel (bool, optional): Ignore novel context in evaluation and used trained Q-tables.

        Returns:
            List[float]: Total rewards for each of the evaluation episodes.
        """
        rewards = []

        if ignore_novel and np.nansum(p_context[:-1]) > 0:
            p_context[:-1] = p_context[:-1]/(np.nansum(p_context[:-1]) + 1e-4)
            p_context[-1] = 0.0

        for _ in range(n_episodes):
            obs, _ = env.reset()
            state = discretize_state(obs, self.position_bins, self.velocity_bins)
            episode_reward = 0.0

            for _ in range(max_steps_per_episode):
                # Find average Q and eps
                Qavg = np.zeros_like(self.Qdat[0])
                for i in range(len(self.Qdat)):
                    if self.context_init[i] and not np.isnan(p_context[i]):
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
        device: str = "cpu"
    ):
        self.gamma, self.lam = gamma, gae_lambda
        self.clip_eps, self.ent_coef, self.vf_coef = clip_eps, ent_coef, vf_coef
        self.device = device

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
        logp = dist.log_prob(action)
        return action.item(), logp, dist.entropy(), logits

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
