"""
cece.py
---------
**Fixed‑context variant** — assumes you already know which latent context each
environment belongs to.  No online clustering is performed; instead the agent
receives a pair `(env, ctx_id)` and keeps one learner per context.

Usage
*****
```python
from cece import FixedContextCECE, train_fixed_cece

# envs must be supplied in the execution order you want
envs = [env1, env2, env3, ...]
contexts = [0,    1,    0,   ...]  # same length as envs

a gent = FixedContextCECE(n_contexts=max(contexts)+1)
train_fixed_cece(agent, envs, contexts, episodes_per_env=10)
```

The helper `train_fixed_cece` iterates over the list in order, repeating each
environment `episodes_per_env` times before moving to the next. 
"""

from __future__ import annotations

from typing import List, Tuple
import gymnasium as gym
from tqdm import trange

from rl import QLearningAgent

__all__ = ["FixedContextCECE", "train_fixed_cece"]


class FixedContextCECE:
    """Basic contextual learning agent when context labels are pre‑assigned."""

    def __init__(
        self,
        n_contexts: int,
        explore_steps: int = 100,
        bins: Tuple[int, int] = (18, 14),
        **q_kwargs,
    ) -> None:
        self.n_contexts = n_contexts
        self.explore_steps = explore_steps
        self.q_kwargs = q_kwargs
        self._agents: List[QLearningAgent] = []

        self.eps = q_kwargs.get("epsilon", 1.0)
        self.eps_end = q_kwargs.get("min_epsilon", 0.05)
        self.eps_decay = q_kwargs.get("epsilon_decay", 0.995)

    # ------------------------------------------------------------
    def run_episode(self, env: gym.Env, ctx_id: int) -> float:
        """Perform one episode in *env* knowing its context index."""
        self._ensure_agent(ctx_id, env)
        # exploitation phase
        agent = self._agents[ctx_id]
        old_eps = agent.epsilon
        agent.epsilon = self.eps
        ret = agent.train_step(env)
        agent.epsilon = old_eps
        # decay global epsilon
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        return ret

    def _ensure_agent(self, idx: int, env: gym.Env) -> None:
        while len(self._agents) <= idx:
            self._agents.append(QLearningAgent(env, **self.q_kwargs))


# ------------------------------------------------------------------
# Helper training loop
# ------------------------------------------------------------------

def train_fixed_cece(
    agent: FixedContextCECE,
    envs: List[gym.Env],
    contexts: List[int],
    episodes_per_env: int = 10,
) -> List[float]:
    """Iterate through envs in order, training `episodes_per_env` on each."""
    returns: List[float] = []
    for env, ctx in zip(envs, contexts):
        for _ in trange(episodes_per_env, desc=f"ctx {ctx}"):
            ret = agent.run_episode(env, ctx)
            returns.append(ret)
    return returns


# --------------------------- demo --------------------------------
if __name__ == "__main__":
    from environments import CustomMountainCarEnv
    import matplotlib.pyplot as plt

    # Define environments & their contexts explicitly
    amps = [0.35, 0.45, 0.60, 0.35, 0.60]
    envs = [CustomMountainCarEnv(a, render_mode=None) for a in amps]
    contexts = [0, 1, 2, 0, 2]  # user‑defined mapping

    agent = FixedContextCECE(n_contexts=3, epsilon=1.0)
    rets = train_fixed_cece(agent, envs, contexts, episodes_per_env=1000)

    plt.plot(rets)
    plt.title("Episode returns (fixed‑context CECE)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.show()
