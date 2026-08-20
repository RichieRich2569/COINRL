"""
hmmdp.py

Hidden-Mode MDP (HM-MDP) baseline for the Figure 2 continual Q-learning experiment.

Following Choi, Yeung & Zhang (Hidden-Mode Markov Decision Processes for Nonstationary
Sequential Decision Making, LNCS 1828, 2000; Solving Hidden-Mode Markov Decision Problems,
AISTATS 2001), the environment is assumed to be governed by a small *fixed* number K of
hidden modes, each a stationary MDP sharing S and A, with the mode evolving as a sticky
Markov chain. The agent maintains a belief over modes by HMM filtering on its observations
and acts on that belief.

Here the observation is the same amplitude probe COIN receives (``rl.probe_amplitude``),
the emission model is a per-mode Gaussian with online-estimated mean, and the belief is fed
straight into the existing ``rl.COINQLearningAgent`` as its ``p_context`` vector. This makes
the baseline exactly "COIN minus the Bayesian-nonparametric context creation": K is given,
the novel-context column is held at zero, and the sticky matrix Pi replaces COIN's learned
transition structure.
"""
from __future__ import annotations

import numpy as np
from typing import List, Optional, Sequence


class HMModeFilter:
    """
    Fixed-K HMM belief filter over hidden modes with online-estimated Gaussian emissions.

    The filter step is the standard forward recursion

        b'_k  ∝  N(theta_hat; mu_k, sigma^2) * sum_j Pi[j, k] b_j        (then normalise)

    The mode means ``mu_k`` are not known in advance, so they are estimated online:

    * the first probe initialises ``mu_0``;
    * a later probe further than ``separation * sigma`` from every mean set so far
      initialises the next unset mean (a genuinely new mode has been observed);
    * otherwise the probe is assigned to the belief-argmax mode and folded into that
      mode's mean by a running average.

    Modes whose mean is not yet set are given zero emission likelihood, so belief mass sits
    only on the initialised mode(s) until all K means exist.
    """

    def __init__(
        self,
        n_modes: int = 2,
        Pi: Optional[np.ndarray] = None,
        sigma: float = 0.1,
        separation: float = 3.0,
    ):
        """
        Args:
            n_modes (int): Number of hidden modes K (given, as in HM-MDP).
            Pi (np.ndarray, optional): (K, K) mode transition matrix, row-stochastic, where
                ``Pi[j, k] = P(mode_t = k | mode_{t-1} = j)``. Defaults to the sticky
                matrix [[0.95, 0.05], [0.05, 0.95]] for K = 2 (0.95 on the diagonal in general).
            sigma (float): Standard deviation of the per-mode Gaussian emission on the probe.
            separation (float): A probe more than ``separation * sigma`` from every mean set
                so far is treated as evidence of an as-yet-unseen mode.
        """
        self.n_modes = int(n_modes)
        self.sigma = float(sigma)
        self.separation = float(separation)

        if Pi is None:
            # Sticky chain: stay in the current mode with probability 0.95
            stay = 0.95
            off = (1.0 - stay) / max(self.n_modes - 1, 1)
            Pi = np.full((self.n_modes, self.n_modes), off)
            np.fill_diagonal(Pi, stay)
        Pi = np.asarray(Pi, dtype=float)
        if Pi.shape != (self.n_modes, self.n_modes):
            raise ValueError(f"Pi must have shape ({self.n_modes}, {self.n_modes}), got {Pi.shape}")
        self.Pi = Pi

        # Uniform prior over modes before any probe has been seen
        self.belief = np.full(self.n_modes, 1.0 / self.n_modes)

        # Emission means, NaN until initialised, plus the count of probes assigned to each
        # mode (used for the running average).
        self.mu = np.full(self.n_modes, np.nan)
        self.counts = np.zeros(self.n_modes)

        # Every probe seen so far, in order (useful for diagnostics/plots)
        self.observations: List[float] = []

    # -------- Internals --------

    @property
    def _is_set(self) -> np.ndarray:
        """Boolean mask of modes whose emission mean has been initialised."""
        return ~np.isnan(self.mu)

    def _one_hot(self, k: int) -> np.ndarray:
        """Put all belief mass on mode k (used when a mode is first initialised)."""
        b = np.zeros(self.n_modes)
        b[k] = 1.0
        return b

    def _log_likelihoods(self, theta_hat: float) -> np.ndarray:
        """Log N(theta_hat; mu_k, sigma^2) per mode, -inf for modes with no mean yet."""
        logL = np.full(self.n_modes, -np.inf)
        set_idx = np.flatnonzero(self._is_set)
        d = (theta_hat - self.mu[set_idx]) / self.sigma
        # Normalising constant is identical across modes and cancels, so it is dropped
        logL[set_idx] = -0.5 * d ** 2
        return logL

    # -------- Public API --------

    def update(self, theta_hat: float) -> np.ndarray:
        """
        Fold one probe observation into the mode belief and return the updated belief.

        Args:
            theta_hat (float): Probe estimate of the current path amplitude.

        Returns:
            np.ndarray: The posterior belief over modes, shape (K,), summing to 1.
        """
        theta_hat = float(theta_hat)
        self.observations.append(theta_hat)

        # --- Mean bookkeeping: does this probe reveal a mode we have not seen yet? ---
        set_idx = np.flatnonzero(self._is_set)
        if set_idx.size == 0:
            # First probe ever: it defines mode 0, and it is certainly the current mode.
            self.mu[0] = theta_hat
            self.counts[0] = 1.0
            self.belief = self._one_hot(0)
            return self.belief.copy()

        nearest_dist = np.min(np.abs(theta_hat - self.mu[set_idx]))
        unset_idx = np.flatnonzero(~self._is_set)
        if nearest_dist > self.separation * self.sigma and unset_idx.size > 0:
            # Far from every known mode and a mode slot is free: initialise the next mode.
            k_new = int(unset_idx[0])
            self.mu[k_new] = theta_hat
            self.counts[k_new] = 1.0
            self.belief = self._one_hot(k_new)
            return self.belief.copy()

        # --- HMM filter step: predict through Pi, then weight by the emission likelihood ---
        predicted = self.belief @ self.Pi                      # sum_j Pi[j, k] b_j
        logL = self._log_likelihoods(theta_hat)

        with np.errstate(divide="ignore"):
            log_post = logL + np.log(predicted)
        finite = np.isfinite(log_post)
        if not np.any(finite):
            # Numerically hopeless (e.g. the predicted mass sits entirely on unset modes):
            # fall back to the nearest known mean.
            k = int(set_idx[np.argmin(np.abs(theta_hat - self.mu[set_idx]))])
            self.belief = self._one_hot(k)
        else:
            log_post = log_post - np.max(log_post[finite])      # stabilise before exponentiating
            post = np.where(finite, np.exp(log_post), 0.0)
            self.belief = post / np.sum(post)

        # --- Assign the probe to the most likely mode and update that mode's mean ---
        k = int(np.argmax(self.belief))
        if self._is_set[k]:
            self.counts[k] += 1.0
            self.mu[k] += (theta_hat - self.mu[k]) / self.counts[k]   # running average

        return self.belief.copy()

    def most_likely_mode(self) -> int:
        """Index of the mode with the highest current belief."""
        return int(np.argmax(self.belief))

    def p_context(self, n_known: Optional[int] = None) -> np.ndarray:
        """
        Belief in ``COINQLearningAgent.p_context`` form: the K mode probabilities followed by
        a novel-context entry that is always exactly 0.0 (HM-MDP has no novel context).

        Args:
            n_known (int, optional): Number of known-context slots in the agent. Defaults to K.
        """
        n_known = self.n_modes if n_known is None else int(n_known)
        if n_known < self.n_modes:
            raise ValueError(f"n_known ({n_known}) must be at least the number of modes ({self.n_modes})")
        vec = np.zeros(n_known + 1)
        vec[: self.n_modes] = self.belief
        return vec


def run_single_rep_hmmdp_q(
    rep_id: int,
    true_amplitudes: Sequence[float],
    probe_period: int = 100,
) -> List[float]:
    """
    Runs one repetition of training for HM-MDP-Q (fixed-K Bayesian mode inference).

    Identical to the COIN-Q loop in figures.ipynb except that the context responsibilities
    come from a K = 2 HMM mode filter instead of RealTimeCOIN. Returns rewards at each
    time step.

    Args:
        rep_id (int): Repetition index; also the random seed.
        true_amplitudes (Sequence[float]): Per-episode amplitude schedule (hidden from the agent).
        probe_period (int): Probe the environment every this many episodes (Figure 2 uses 100).

    Returns:
        List[float]: Per-episode training reward.
    """
    from environments import CustomMountainCarEnv
    from rl import COINQLearningAgent, probe_amplitude
    from tqdm.auto import tqdm
    import numpy as np

    from hmmdp import HMModeFilter

    SEED = rep_id
    np.random.seed(SEED)                        # Q-table init draws from the global numpy state
    rng = np.random.default_rng(SEED)           # action selection

    # Fixed number of hidden modes, sticky mode chain, Gaussian probe emissions
    K = 2
    mode_filter = HMModeFilter(
        n_modes=K,
        Pi=np.array([[0.95, 0.05], [0.05, 0.95]]),
        sigma=0.1,
    )

    # Create a fresh agent and environment (same hyperparameters as the COIN-Q run)
    env = CustomMountainCarEnv(amplitude=1.0, render_mode="none")
    agent = COINQLearningAgent(
        env=env,
        max_contexts=K,
        num_position_bins=30,
        num_velocity_bins=30,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.999,
        rng=rng,
    )

    # Belief before the first probe: uniform over modes, novel entry pinned to 0.0
    p_context = mode_filter.p_context(n_known=K)

    rewards_for_this_rep = []
    pbar = tqdm(true_amplitudes, desc=f"Rep {rep_id}")
    for i, amplitude in enumerate(pbar):
        # Create the environment for each amplitude
        env = CustomMountainCarEnv(amplitude=amplitude, render_mode="none")

        # Obtain small experience from environment to update the mode filter every probe_period episodes
        if i % probe_period == 0:
            est_a = probe_amplitude(env)

            # HMM filter step; p_context = [b_0, b_1, 0.0] with the novel slot exactly zero
            mode_filter.update(est_a)
            p_context = mode_filter.p_context(n_known=K)

        # Train the agent in the current context
        training_reward = agent.train_step(env=env, p_context=p_context, max_steps_per_episode=200)
        rewards_for_this_rep.append(training_reward)

        pbar.set_postfix(amplitude=amplitude, reward=training_reward)

    return rewards_for_this_rep
