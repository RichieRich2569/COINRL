"""
This code extends on the Python implementation of COIN by
Changmin Yu (https://github.com/changmin-yu/COIN_Python). 
It provides a realtime version of COIN that can be called trial by trial.

COIN Generative Model (Core Equations)

At each time step (trial) t = 1,...,T:

1. Context evolution (Markov process):
    c_t | c_{t-1}, Pi ~ Discrete(pi_{c_{t-1}})
    where Pi = (pi_j)_j is the transition matrix,
    pi_j = (pi_jk)_k is row j (transition probabilities from context j to k).

2. State dynamics (linear Gaussian per context):
    x_t^(j) = a^(j) * x_{t-1}^(j) + d^(j) + w_t^(j)
    w_t^(j) ~ Normal(0, sigma_q^2)
    Prior (stationary):
    x_t^(j) ~ Normal( d^(j)/(1 - a^(j)), sigma_q^2 / (1 - (a^(j))^2) )

3. Observation model:
    y_t = x_t^(c_t) + b^(c_t) + v_t
    v_t ~ Normal(0, sigma_r^2)
    where b^(j) is a bias term for context j, which can be set to be estimated or not (set to zero otherwise).
    If estimated, it has a prior:
    b^(j) | mu_b, sigma_b^2 ~ Normal(mu_b, sigma_b^2) - mu_b set to zero in this implementation.

4. Cue emission:
    q_t | c_t, Phi ~ Discrete(phi_{c_t})
    where Phi = (phi_j)_j is the cue probability matrix,
    phi_j = (phi_jk)_k is row j (probabilities of cues in context j).

5. Transition probability priors:
    beta | gamma ~ GEM(gamma)
    pi_j | alpha, beta, kappa ~ DP( alpha + kappa,
                                    (alpha * beta + kappa * delta_j) / (alpha + kappa) )
    where delta_j is a one-hot vector for self-transition.

6. Cue probability priors:
    beta_e | gamma_e ~ GEM(gamma_e)
    phi_j | alpha_e, beta_e ~ DP(alpha_e, beta_e)

7. State parameter priors:
    omega^(j) = [a^(j), d^(j)]^T
    omega^(j) | mu, Sigma ~ TruncatedNormal(mu, Sigma)
    where mu = [mu_a, 0]^T, Sigma = diag(sigma_a^2, sigma_d^2)

8. Adaptation. In measuring adaptation, we assume a motor output with noise:
    a_t ~ Normal(u_t, sigma_m^2)
    where u_t is the COIN model's motor output (predicted state in this implementation).
    sigma_m is motor noise standard deviation.
    Our implementation currently removes motor noise processing.

Key Variables:
    c_t      : Discrete latent context at time t
    x_t^(j)  : Continuous latent state for context j
    y_t      : Continuous observation (state feedback)
    q_t      : Discrete observation (cue)
    a^(j)    : State retention factor for context j
    d^(j)    : Drift term for context j
    sigma_q  : Std. dev. of process noise
    sigma_r  : Std. dev. of observation noise
    Pi, pi_j : Transition probability matrix and rows
    Phi, phi_j : Cue probability matrix and rows
    beta, beta_e : Global transition and cue probabilities
    alpha, alpha_e : DP concentration parameters
    kappa    : Self-transition bias ("stickiness")
    mu, Sigma : Prior mean and covariance for state parameters

---------------------------------------------------------------------
Copyright (c) 2025 Richard Marques Monteiro

This file is part of COINRL.

COINRL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

COINRL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with COINRL. If not, see <https://www.gnu.org/licenses/>.
"""

from typing import Optional, List, Dict, Any

import numpy as np

import coin

from math import factorial
from itertools import permutations

from utils.general_utils import (
    sample_num_tables_CRF, 
    per_slice_invert, 
    per_slice_multiply, 
    log_sum_exp, 
    systematic_resampling, 
)
from utils.distribution_utils import (
    random_dirichlet, 
    stationary_distribution, 
    random_truncated_bivariate_normal, 
    random_univariate_normal, 
)

class COIN_RT(coin.COIN):
        """Realtime version of the COIN model.
        params:
        - sigma_process_noise: float
            Standard deviation of the process noise sigma_q.
        - sigma_sensory_noise: float
            Standard deviation of the sensory noise sigma_r.
        - prior_mean_retention: float
            Prior mean of the state retention factor a^(j).
        - prior_precision_retention: float
            Prior precision (1/sigma_a^2) of the state retention factor a^(j).
        - prior_precision_drift: float
            Prior precision (1/sigma_d^2) of the state drift term d^(j).
        - gamma_context: float
            Hyperparameter for the GEM distribution of the global context transition probabilities beta.
        - alpha_context: float
            Concentration parameter for the DP prior over rows of the context transition matrix Pi.
        - rho_context: float
            Normalised self-transition bias parameter in context transition, kappa / (alpha + kappa).
        - gamma_cue: float
            Hyperparameter for the GEM distribution of the global cue emission probabilities beta_e.
        - alpha_cue: float
            Concentration parameter for the DP prior over rows of the cue emission matrix Phi.
        - infer_bias: bool
            Whether to infer the context-specific bias terms b^(j).
        - prior_precision_bias: float
            Prior precision (1/sigma_b^2) of the context-specific bias terms b^(j).
        - runs: int
            Number of independent runs of the model to perform.
        - max_cores: int
            Maximum number of CPU cores to use for parallel processing.
        - particles: int
            Number of particles to use in the particle filter.
        - max_contexts: int
            Maximum number of contexts to consider.
        - retention_values: Optional[np.ndarray]
            Optional array of ground truth retention values for evaluation.
        - drift_values: Optional[np.ndarray]
            Optional array of ground truth drift values for evaluation.
        - state_values: Optional[np.ndarray]
            Optional array of ground truth state values for evaluation.
        - bias_values: Optional[np.ndarray]
            Optional array of ground truth bias values for evaluation.
        - state_feedback_values: Optional[np.ndarray]
            Optional array of ground truth state feedback values for evaluation.
        -----------
        """
        def __init__(
            self, 
            sigma_process_noise: float = 0.0089, 
            sigma_sensory_noise: float = 0.03, 
            prior_mean_retention: float = 0.9425, 
            prior_precision_retention: float = 837.1 ** 2, 
            prior_precision_drift: float = 1.2227e3 ** 2, 
            gamma_context: float = 0.1, 
            alpha_context: float = 8.955, 
            rho_context: float = 0.2501, 
            gamma_cue: float=  0.1, 
            alpha_cue: float = 25, 
            infer_bias: bool = False, 
            prior_precision_bias: float = 70 ** 2, 
            # model implementation
            particles: int = 100, 
            max_contexts: int = 10,
            max_cues: int = 10,
            # evaluation
            retention_values: Optional[np.ndarray] = None, 
            drift_values: Optional[np.ndarray] = None, 
            state_values: Optional[np.ndarray] = None, 
            bias_values: Optional[np.ndarray] = None, 
            state_feedback_values: Optional[np.ndarray] = None, 
            # Initial trial
            trial: int = 0,
            cues_exist: bool = False,
        ):
            super().__init__(
                sigma_process_noise=sigma_process_noise, 
                sigma_sensory_noise=sigma_sensory_noise, 
                prior_mean_retention=prior_mean_retention, 
                prior_precision_retention=prior_precision_retention, 
                prior_precision_drift=prior_precision_drift, 
                gamma_context=gamma_context, 
                alpha_context=alpha_context, 
                rho_context=rho_context, 
                gamma_cue=gamma_cue, 
                alpha_cue=alpha_cue, 
                infer_bias=infer_bias, 
                prior_precision_bias=prior_precision_bias, 
                # runs - only one run for RT version
                runs=1, 
                # parallel processing - disabled
                max_cores=0, 
                # model implementation
                particles=particles, 
                max_contexts=max_contexts, 
                # evaluation
                retention_values=retention_values, 
                drift_values=drift_values, 
                state_values=state_values, 
                bias_values=bias_values, 
                state_feedback_values=state_feedback_values, 
            )
            # Keep track of current trial
            self.trial = trial

            # Max cues
            self.max_cues = max_cues

            # Initialise coin
            self.coin_state = self.initialise_coin(cues_exist=cues_exist)

            # Perturbations
            self.perturbations = np.array([]) # This array is not used in COIN-RT, but its size is incremented for compatibility with parent.

            # For context tracking and probabilities
            # Context sequence
            self.context_seq = {} # Dictionary of form {trial: context_sequence} for context_sequence shape (particles, trial)
            # Maximum number of instantiated contexts across particles
            self.C = np.zeros((self.particles, 1), dtype=int) # shape (particles, 1)
            # Posterior over number of contexts
            self.posterior = np.zeros((self.max_contexts+1, 1))
            self.posterior_mean = np.zeros((1, ))
            self.posterior_mode = np.zeros((1, ), dtype=int)

            # Label-alignment cache
            self._ocl_cache = dict(
                last_computed_trial=-1,           # last i for which we computed results
                optimal_assignment=dict(),        # {i: np.ndarray shape (1, K_i, n_sequences)}
                from_unique=dict(),               # {i: reverse_inds from np.unique}
                to_unique=dict(),                 # {i: inds from np.unique}
                typical_index_by_trial=dict(),    # {i: int index into unique seqs}
            )

            self.n_particles_used = np.zeros((0,))




        def initialise_coin(self, cues_exist: bool):
            """Initialise the COIN model state."""
            self.perturbations = []
            self.cues = None
            coin_state = super().initialise_coin()

            if cues_exist:
                coin_state["cues_exist"] = 1
                
                # number of contextual cues observed so far
                coin_state["Q"] = 0
                
                # cue emission counts
                coin_state["n_cue"] = np.zeros((self.max_contexts + 1, self.max_cues + 1, self.particles))

                # use self.cues to actually store the maximum cue observed so far - different from original COIN
                self.cues = [self.max_cues]
            return coin_state

        def step(
            self, 
            state_feedback: float, 
            cue: Optional[int] = None, 
        ) -> List[Dict[str, Any]]:
            """Process a single trial of data through the COIN model.

            Parameters:
            - state_feedback: float
                The observed state feedback (y_t) for the current trial.
            - cue: Optional[int]
                The observed cue (q_t) for the current trial. If no cue is present, set to None.

            Returns:
            - results: List[Dict[str, Any]]
                A list of dictionaries containing results for each run.
            """
            # TODO: handle multiple runs
            # Increment trial and num_trials as well as size of perturbations
            self.coin_state["trial"] += 1
            self.coin_state["num_trials"] += 1
            self.trial = self.coin_state["trial"]
            # Ensure perturbations is a 1-D numpy array and extend its length by 1
            p = np.asarray(self.perturbations)
            if p.ndim > 1:
                p = p.reshape(-1)
            else:
                p = p.copy()
            p = p.astype(float) if p.size else p
            self.perturbations = np.append(p, np.nan)

            # Check if cues are provided when cues_exist is True
            if self.coin_state["cues_exist"] and cue is None:
                raise ValueError("Cue must be provided when cues_exist is True.")
            if not self.coin_state["cues_exist"] and cue is not None:
                print("Warning: Cue provided but cues_exist is False. Ignoring cue.")

            # If cue is provided, ensure cue checks
            cue_exists = self.coin_state["cues_exist"]
            if cue is not None and cue_exists:
                max_cue = self.coin_state.get("Q", 0)
                if cue < 1 or cue > self.max_cues:
                    raise ValueError(f"Cue must be between 1 and {self.max_cues} (inclusive).")
                if cue > max_cue + 1:
                    raise ValueError(f"Can only introduce one new cue, in ascending order, at a time. Current max cue is {max_cue}.")
                self.cues.insert(self.trial-1, cue)
            
            # Feedback observed or not
            temp = np.ones((self.trial, ))
            temp[:self.trial-1] = self.coin_state["feedback_observed"]
            if np.isnan(state_feedback):
                temp[-1] = 0
            self.coin_state["feedback_observed"] = temp

            # Particle learning step
            cs = self.coin_state # shorthand, changes will be reflected in self.coin_state
            cs = self.predict_context(cs, cue)
            cs = self.predict_states(cs)
            cs = self.predict_state_feedback(cs, state_feedback) # RT-specific, state feedback now external input
            cs = self.resample_particles(cs)
            cs = self.sample_context(cs, cue)
            cs = self.update_belief_about_states(cs)
            cs = self.sample_states(cs)
            cs = self.update_sufficient_statistics_for_parameters(cs, cue)
            cs = self.sample_parameters(cs)
            cs = self.store_variables(cs)

            S = self._build_output()
            
            return S
        
        def get_predicted_probabilities(self):
            # Get predicted probabilities p(c_t | y_{1:t-1}, ...) from model
            #TODO: very inefficient, runs for all iterations. Optimise later.
            S = self._build_output()
            prob = super().get_predicted_probabilities(S)
            return super().get_predicted_probabilities(S)
        
        def get_responsibilities(self, separate_novel: bool = True):
            # Get responsibilities p(c_t | y_{1:t}, ...) from model
            #TODO: very inefficient, runs for all iterations. Optimise later.
            S = self._build_output()
            known, novel = super().get_responsibilities(S)
            if self.coin_state["trial"] > 0:
                known = known[-1,:] # Get last trial only
                novel = novel[-1,:] # Get last trial only
            if separate_novel:
                return known, novel
            prob = np.concatenate([known, novel.reshape(1,1)], axis=1)
            return prob
        
        def get_predicted_responsibilities(self, y):
            # Get predicted responsibilities p(c_t | y_{1:t}, ...) for a given observation y
            #TODO: very inefficient, runs for all iterations. Optimise later.
            S = self._build_output()
            return super().get_predicted_responsibilities(S, y)
        
        def _build_output(self):
            # Build "output" structure for compatibility with original COIN
            S = {}
            S["runs"] = {}
            S["runs"][0] = self.coin_state["stored"]
            S["properties"] = self
            return S
        
        def _contingency_matrix(self, seq_a: np.ndarray, seq_b: np.ndarray) -> np.ndarray:
            """
            Build a K×K contingency matrix of co-occurrences between labels in seq_a and seq_b.
            Labels are 1-based; matrix is indexed 0..K-1.
            K is chosen as max label seen in either sequence.
            """
            assert seq_a.ndim == 1 and seq_b.ndim == 1
            assert seq_a.shape[0] == seq_b.shape[0]
            K = int(max(seq_a.max(initial=1), seq_b.max(initial=1)))
            M = np.zeros((K, K), dtype=int)
            # 1-based -> 0-based
            for a, b in zip(seq_a, seq_b):
                M[a - 1, b - 1] += 1
            return M

        @staticmethod
        def _hungarian_min(cost: np.ndarray):
            """
            Minimal Hungarian algorithm (square cost, dense) for small K (<= ~10).
            Returns (row_ind, col_ind) minimizing total cost.
            No SciPy dependency.
            """
            cost = np.asarray(cost, dtype=float).copy()
            n = cost.shape[0]
            assert cost.shape == (n, n)

            # Step 1: subtract row mins, then col mins
            cost -= cost.min(axis=1, keepdims=True)
            cost -= cost.min(axis=0, keepdims=True)

            # Masks and covers
            starred = np.zeros((n, n), dtype=bool)
            primed = np.zeros((n, n), dtype=bool)
            row_cov = np.zeros(n, dtype=bool)
            col_cov = np.zeros(n, dtype=bool)

            # Step 2: star independent zeros
            for r in range(n):
                for c in range(n):
                    if (cost[r, c] == 0) and not row_cov[r] and not col_cov[c]:
                        starred[r, c] = True
                        row_cov[r] = True
                        col_cov[c] = True
                        break
            row_cov[:] = False
            col_cov[:] = np.any(starred, axis=0)

            def find_zero():
                for r in range(n):
                    if row_cov[r]:
                        continue
                    for c in range(n):
                        if (not col_cov[c]) and cost[r, c] == 0 and not starred[r, c] and not primed[r, c]:
                            return r, c
                return None

            def find_star_in_row(r):
                cs = np.where(starred[r])[0]
                return cs[0] if cs.size else None

            def find_star_in_col(c):
                rs = np.where(starred[:, c])[0]
                return rs[0] if rs.size else None

            def find_prime_in_row(r):
                cs = np.where(primed[r])[0]
                return cs[0] if cs.size else None

            def augment_path(start_r, start_c):
                # Build alternating path starting from primed zero at (start_r, start_c)
                path = [(start_r, start_c)]
                done = False
                while not done:
                    r = np.where(starred[:, path[-1][1]])[0]
                    if r.size:
                        r = r[0]
                        path.append((r, path[-1][1]))
                    else:
                        done = True
                        break
                    c = np.where(primed[r])[0]
                    c = c[0]
                    path.append((r, c))
                # Flip stars along path
                for (r, c) in path:
                    starred[r, c] = not starred[r, c]
                # Clear primes and covers
                primed[:, :] = False

            # Main loop
            while True:
                if np.sum(col_cov) == n:
                    break
                z = find_zero()
                while z is None:
                    # Step: adjust matrix
                    uncovered = cost[~row_cov][:, ~col_cov]
                    m = uncovered.min() if uncovered.size else 0.0
                    cost[ row_cov, : ] += m
                    cost[:, ~col_cov] -= m
                    z = find_zero()

                r, c = z
                primed[r, c] = True
                star_c = find_star_in_row(r)
                if star_c is not None:
                    # Cover row, uncover the star's column, continue
                    row_cov[r] = True
                    col_cov[star_c] = False
                else:
                    # Augmenting path starts
                    augment_path(r, c)
                    row_cov[:] = False
                    col_cov[:] = np.any(starred, axis=0)

            # Extract assignment from starred zeros
            row_ind = np.arange(n)
            col_ind = np.argmax(starred, axis=1)
            return row_ind, col_ind

        def _optimal_hamming_matrix(self, unique_seqs: np.ndarray) -> np.ndarray:
            """
            Compute pairwise *optimally relabeled* Hamming distances between unique sequences.
            Uses Hungarian on contingency counts (max assignment ↔ max label agreements).
            Sequences contain 1-based labels. Returns H_opt of shape (S, S).
            """
            S, L = unique_seqs.shape
            H = np.zeros((S, S), dtype=float)
            for a in range(S):
                H[a, a] = 0.0
                for b in range(a + 1, S):
                    M = self._contingency_matrix(unique_seqs[a], unique_seqs[b])  # K×K
                    # Maximize matches ⇒ minimize (max - M)
                    maxv = M.max(initial=0)
                    cost = (maxv - M).astype(float)
                    # Pad to square if needed (safety; should already be square K×K)
                    if cost.shape[0] != cost.shape[1]:
                        K = max(cost.shape)
                        pad = np.zeros((K, K), dtype=float)
                        pad[:cost.shape[0], :cost.shape[1]] = cost
                        cost = pad
                    r, c = self._hungarian_min(cost)
                    matches = int(M[r, c].sum())
                    Hval = L - matches
                    H[a, b] = Hval
                    H[b, a] = Hval
            return H

        def _perm_from_assignment(self, typical_seq: np.ndarray, other_seq: np.ndarray, K_target: int) -> np.ndarray:
            """
            Compute the 0-based permutation π of length K_target that maps labels of the
            typical sequence to labels of the other sequence by maximizing agreements.
            """
            M = self._contingency_matrix(typical_seq, other_seq)  # K×K for observed labels
            maxv = M.max(initial=0)
            cost = (maxv - M).astype(float)
            if cost.shape[0] != cost.shape[1]:
                K = max(cost.shape)
                pad = np.zeros((K, K), dtype=float)
                pad[:cost.shape[0], :cost.shape[1]] = cost
                cost = pad
            r, c = self._hungarian_min(cost)  # row r (typical label index) → column c (other label index)
            mapping = dict(zip(r.tolist(), c.tolist()))
            # Build 0-based permutation of length K_target (labels 1..K_target → 0..K_target-1)
            perm = np.arange(K_target, dtype=int)
            for lab in range(K_target):       # lab is 0-based label index for typical
                perm[lab] = mapping.get(lab, lab)
                # Clamp in case 'other' has labels beyond K_target (won't matter for valid-K sequences)
                if perm[lab] >= K_target:
                    perm[lab] = lab
            return perm

        
        def find_optimal_context_labels(self):
            """
            Incremental, assignment-based (Hungarian) optimal label alignment.

            Returns:
            P: dict with P["mode_number_of_contexts"] = array[T] (modal K per trial)
            optimal_assignment: dict[i] -> np.ndarray of shape (1, K_i, n_sequences_i)
                Each column is a 0-based permutation vector π aligning the "typical"
                sequence's labels (1..K_i) to that sequence at trial i.
            from_unique: dict[i] -> reverse indices from np.unique for particles kept at i
            context_sequence: dict[trial] -> (P, trial+1) array (1-based labels)
            C: (R*P, T) array: instantiated #contexts per particle/trial
            """
            # Rebuild or extend context sequences for the *current* trial
            context_sequence = self.context_sequence()  # dict {trial: (P, trial+1)}
            # Compute posterior #contexts and modal counts across trials
            C, _, _, mode_number_of_contexts = self.posterior_number_of_contexts(context_sequence)

            P = {"mode_number_of_contexts": mode_number_of_contexts}
            optimal_assignment = self._ocl_cache["optimal_assignment"]
            from_unique = self._ocl_cache["from_unique"]
            to_unique = self._ocl_cache["to_unique"]
            typical_index_by_trial = self._ocl_cache["typical_index_by_trial"]

            # Compute only for *new* trials since last call
            start_i = int(self._ocl_cache["last_computed_trial"]) + 1
            end_i = int(self.trial)  # original code loops i in range(self.trial)
            if start_i < 0:
                start_i = 0

            # Constant used in original filter: allow sequences with contexts up to the global max modal
            max_mode_all = int(np.max(mode_number_of_contexts).astype(int)) if mode_number_of_contexts.size else 0

            for i in range(start_i, end_i):
                # Filter particles whose C <= global max modal (exclude never-to-be-analysed descendants)
                f_i = np.where(C[:, i] <= max_mode_all)[0]

                # Unique sequences among the filtered particles (for trial i, using prefix 0..i)
                uniq, inds, rev = np.unique(context_sequence[i][f_i], axis=0, return_index=True, return_inverse=True)
                to_unique[i] = inds
                from_unique[i] = rev
                n_sequences = uniq.shape[0]

                # Keep only particles with modal K at trial i
                K_i = int(mode_number_of_contexts[i])
                valid_particle_mask = (C[f_i, i] == K_i)

                # Count how many times each unique sequence occurs among valid-K particles
                seq_ids = from_unique[i][valid_particle_mask]
                sequence_count = np.bincount(seq_ids, minlength=n_sequences).astype(int)

                # Pairwise optimally-relabeled Hamming distances among unique sequences
                H_opt = self._optimal_hamming_matrix(uniq)

                # Weighted mean distance of each sequence to all others
                #   weight(j) = sequence_count[j], but reduce self-weight by 1
                W = np.repeat(sequence_count[None, :], n_sequences, axis=0)
                diag = np.arange(n_sequences)
                W[diag, diag] = np.maximum(W[diag, diag] - 1, 0)
                denom = W.sum(axis=1).astype(float)
                denom[denom == 0] = 1.0  # avoid /0; these will be invalidated below
                H_mean = (H_opt * W).sum(axis=1) / denom

                # Invalidate sequences that never occur among valid-K particles
                H_mean[sequence_count == 0] = np.inf

                # Typical sequence: minimal weighted mean distance (medoid under optimal relabeling)
                min_ind = int(np.argmin(H_mean))
                typical_index_by_trial[i] = min_ind
                typical_seq = uniq[min_ind]

                # Build per-sequence optimal permutation that aligns typical -> sequence
                # Output shape: (1, K_i, n_sequences)
                perms = np.zeros((K_i, n_sequences), dtype=int)
                for s in range(n_sequences):
                    perm = self._perm_from_assignment(typical_seq, uniq[s], K_target=K_i)  # 0-based π
                    perms[:, s] = perm

                optimal_assignment[i] = perms[None, :, :]  # (1, K_i, n_sequences)

                # Book-keeping: mark last computed trial
                self._ocl_cache["last_computed_trial"] = i

            return P, optimal_assignment, from_unique, context_sequence, C

        
        def context_sequence(self):
            """Reconstruct the context sequence for each particle in each run, after resampling. Rebuild for RT version."""
            
            # Update our context sequence dictionary
            i = self.trial
            self.context_seq[i] = np.zeros((self.particles, i+1), dtype=int)
            if i > 0:
                # Set the context sequence up to trial i-1 to be the same as that of previous trial sequence
                self.context_seq[i][:, :i] = self.context_seq[i-1][:, :]
                # Appropriately resample all context values according to inds_resampled
                self.context_seq[i][:, :] = self.context_seq[i][self.coin_state["inds_resampled"], :]
            # Set the context at trial i to be the context sampled at trial i
            self.context_seq[i][:, i] = self.coin_state["context"]

            return self.context_seq
        
        def posterior_number_of_contexts(self, context_sequence: Dict[int, Any]): 
            """Compute the posterior over the number of contexts instantiated by the model at each trial, given the context sequences."""           
            # extend context tracking variables if number of trials greater than current size of C
            trial = self.trial

            # Ensure shape of all context tracking variables is identical in trial dimension
            if not (np.shape(self.C)[1]
                    == np.shape(self.posterior_mean)[0] 
                    == np.shape(self.posterior_mode)[0] 
                    == np.shape(self.posterior)[1]):
                raise ValueError("Context tracking variables have inconsistent trial dimensions.")
            
            existing_trials = np.shape(self.C)[1] - 1
            if existing_trials <  trial:
                self.C = np.hstack((self.C, np.zeros((self.particles * self.runs, trial - existing_trials), dtype=int)))
                self.posterior = np.hstack((self.posterior, np.zeros((self.max_contexts+1, trial - existing_trials))))
                self.posterior_mean = np.hstack((self.posterior_mean, np.zeros((trial - existing_trials, ))))
                self.posterior_mode = np.hstack((self.posterior_mode, np.zeros((trial - existing_trials, ), dtype=int)))

            for i in range(existing_trials+1, trial+1):
                self.C[:, i] = np.max(context_sequence[i], axis=1)

            particle_weight = np.repeat(np.array([1.0]), self.particles) / self.particles
            
            
            for i in range(existing_trials+1, trial+1):
                for context in range(np.max(self.C[:, i])):
                    self.posterior[context, i] = np.sum((self.C[:, i] == (context+1)) * particle_weight)

                self.posterior_mean[i] = np.sum(np.arange(1,self.max_contexts+2) * self.posterior[:, i])
                self.posterior_mode[i] = np.argmax(self.posterior[:, i])+1 # contexts seen as 1 and not 0

            return self.C, self.posterior, self.posterior_mean, self.posterior_mode
        
        def compute_variables_for_plotting(
                self, 
                P: Dict[str, Any], 
                S: Dict[str, Any], 
                optimal_assignment: Dict[int, Any], 
                from_unique: Dict[int, Any], 
                context_sequence: Dict[int, Any], 
                C: np.ndarray, 
            ):         
            P = self.preallocate_memory(P)
            
            # cumulative number of particles for which C <= np.max(P["mode_number_of_contexts"])
            N = 0
            i = self.trial

            # expand n_particles_used if size is less than trial
            if np.shape(self.n_particles_used)[0] < self.trial + 1:
                n_particles_used = np.zeros((self.trial + 1, 1), dtype=int)
                if np.shape(self.n_particles_used)[0] > 0:
                    n_particles_used[:np.shape(self.n_particles_used)[0], :] = self.n_particles_used
                self.n_particles_used = n_particles_used

            # inds of particles that are either valid now or could be valid in the future
            # C <= np.max(P["mode_number_of_contexts"])
            valid_future = np.where(C[:, i] <= np.max(P["mode_number_of_contexts"]))[0]
            
            # inds of particles that are valid now
            # C == np.max(P["mode_number_of_contexts"])
            valid_now = np.where(C[:, i] == P["mode_number_of_contexts"][i])[0]
            n_particles_used[i, 0] = len(valid_now)

            if len(valid_now) > 0:
                for particle in valid_now:
                    # index of the optimal label permutations of the current particle
                    ind = N + np.where(particle == valid_future)[0]
                    
                    # is the latest context a novel context
                    # this is needed to store novel context probabilities
                    context_trajectory = context_sequence[i][particle, :]
                    
                    if i > 0:
                        novel_context = context_trajectory[i] > np.max(context_trajectory[:i])
                    else:
                        novel_context = False

                    S = self.relabel_context_variables(S, optimal_assignment[i][0, :, from_unique[i][ind][0]].astype(int), novel_context, particle, i, 0)
                P = self.integrate_over_particles(S, P, valid_now, i, 0)
            
            N += len(valid_future)
            
            P = self.integrate_over_runs(P, S)
            P = self.normalise_relabelled_variables(P, n_particles_used, S)
            
            if self.plot_state_given_context:
                P["state_given_novel_context"] = np.tile(
                    np.nanmean(P["state_given_context"][:, :, -1], axis=1, keepdims=True)[:,:,None], 
                    [1, self.trial, 1], 
                )
                P["state_given_context"] = P["state_given_context"][:, :, :-1]
            
            return P, S

        def predict_context(self, coin_state: Dict[str, Any], cue: Optional[int] = None) -> Dict[str, Any]:
            # Check if cues are provided when cues_exist is True
            if coin_state["cues_exist"] and cue is None:
                raise ValueError("Cue must be provided when cues_exist is True.")
            
            prior_probabilities = np.zeros((self.max_contexts+1, self.particles))
            
            inds_1 = np.tile(coin_state["context"][None], (self.max_contexts+1, 1)) - 1
            inds_2 = np.tile(np.arange(self.max_contexts+1)[None], (self.particles, 1)).T
            inds_3 = np.tile(np.arange(self.particles)[None], (self.max_contexts+1, 1))
            for i in range(self.max_contexts+1):
                for j in range(self.particles):
                    prior_probabilities[i, j] = coin_state["local_transition_matrix"][
                        inds_1[i, j], inds_2[i, j], inds_3[i, j], 
                    ]
            
            coin_state["prior_probabilities"] = prior_probabilities

            if coin_state["cues_exist"] and cue is not None:
                cue_probabilities = np.zeros((self.max_contexts+1, self.particles))
                inds_1 = np.tile(np.arange(self.max_contexts+1)[None], (self.particles, 1)).T
                inds_2 = np.ones((self.max_contexts+1, self.particles), dtype=int) * cue
                inds_3 = np.tile(np.arange(self.particles)[None], (self.max_contexts+1, 1))
                
                for i in range(self.max_contexts+1):
                    for j in range(self.particles):
                        cue_probabilities[i, j] = coin_state["local_cue_matrix"][
                            inds_1[i, j], inds_2[i, j], inds_3[i, j], 
                        ]
                coin_state["cue_probabilities"] = cue_probabilities
                
                coin_state["predicted_probabilities"] = coin_state["prior_probabilities"] * coin_state["cue_probabilities"]
                coin_state["predicted_probabilities"] = coin_state["predicted_probabilities"] / np.sum(coin_state["predicted_probabilities"], axis=0, keepdims=True)
            
            else:
                coin_state["predicted_probabilities"] = coin_state["prior_probabilities"]
                
            if "Kalman_gain_given_cstar2" in self.store:
                if coin_state["trial"] > 1:
                    max_inds = np.argmax(coin_state["predicted_probabilities"], axis=0)
                    inds = np.arange(self.particles)
                    
                    assert len(max_inds) == self.particles
                    
                    kalman_gain = np.zeros((self.particles, ))
                    for i in range(self.particles):
                        kalman_gain[i] = np.mean(coin_state["Kalman_gains"][max_inds[i], inds[i]])
                    coin_state["Kalman_gain_given_cstar2"] = np.mean(kalman_gain)
                        
            if "state_given_cstar2" in self.store:
                if coin_state["trial"] > 1:
                    max_inds = np.argmax(coin_state["predicted_probabilities"], axis=0)
                    inds = np.arange(self.particles)
                    
                    state = np.zeros((self.particles, ))
                    
                    for i in range(len(max_inds)):
                        state[i] = coin_state["state_mean"][max_inds[i], inds[i]]
                    coin_state["state_given_cstar2"] = np.mean(state)
            
            if "predicted_probability_cstar3" in self.store:
                coin_state["predicted_probability_cstar3"] = np.mean(np.max(coin_state["predicted_probabilities"], axis=0))
            
            return coin_state
        
        def predict_state_feedback(self, coin_state: Dict[str, Any], state_feedback: float) -> Dict[str, Any]:
            # predict state feedback for each context (potential non-trivial context-specific bias term in visuo-motor tasks)
            coin_state["state_feedback_mean"] = coin_state["state_mean"] + coin_state["bias"]
            
            # variance of state feedback prediction for each context
            coin_state["state_feedback_var"] = coin_state["state_var"] + np.square(coin_state["sigma_observation_noise"])
            
            coin_state = self.compute_marginal_distribution(coin_state)
            
            # predict marginalised state feedback (marginalise over contexts and particles)
            # mean of the distribution
            coin_state["motor_output"] = np.sum(coin_state["predicted_probabilities"] * coin_state["state_feedback_mean"]) / self.particles
            
            if "implicit" in self.store:
                coin_state["implicit"] = coin_state["motor_output"] - coin_state["average_state"]
            
            # state feedback - consume the same number of randomisations as in the original COIN (for testing purposes)
            coin_state["sensory_noise"] = self.sigma_sensory_noise * np.random.randn()
            coin_state["motor_noise"]   = self.sigma_motor_noise   * np.random.randn()

            coin_state["state_feedback"] = state_feedback # only this is necessary
            
            # state feedback prediction error
            coin_state["prediction_error"] = coin_state["state_feedback"] - coin_state["state_feedback_mean"]
            
            return coin_state
        
        def sample_context(self, coin_state: Dict[str, Any], cue: Optional[int] = None) -> Dict[str, Any]:
            # Check if cues are provided when cues_exist is True
            if coin_state["cues_exist"] and cue is None:
                raise ValueError("Cue must be provided when cues_exist is True.")
            
            coin_state["context"] = np.sum(np.random.rand(self.particles) > np.cumsum(coin_state["responsibilities"], axis=0), axis=0) + 1
            
            coin_state["p_new_x"] = np.where(coin_state["context"] > coin_state["C"])[0]
            coin_state["p_old_x"] = np.where(coin_state["context"] <= coin_state["C"])[0]
            coin_state["C"][coin_state["p_new_x"]] = coin_state["C"][coin_state["p_new_x"]] + 1
            
            p_beta_x = coin_state["p_new_x"][coin_state["C"][coin_state["p_new_x"]] != self.max_contexts]
            inds = coin_state["context"][p_beta_x] - 1
            
            # sample the next stick-breaking weight
            sb_weight = np.random.beta(1, self.gamma_context * np.ones((len(p_beta_x), )))
            
            # update the global transition distribution
            coin_state["global_transition_probabilities"][inds+1, p_beta_x] = coin_state["global_transition_probabilities"][inds, p_beta_x] * (1 - sb_weight)
            coin_state["global_transition_probabilities"][inds, p_beta_x] = coin_state["global_transition_probabilities"][inds, p_beta_x] * sb_weight
            
            if coin_state["cues_exist"] and cue is not None:
                if cue > coin_state["Q"]:
                    # increment the cue context count
                    coin_state["Q"] += 1
                    
                    # sample the next stick-breaking weight
                    sb_weight = np.random.beta(1, self.gamma_cue * np.ones((self.particles, )))
                    
                    coin_state["global_cue_probabilities"][coin_state["Q"]+1, :] = coin_state["global_cue_probabilities"][coin_state["Q"], :] * (1 - sb_weight)
                    coin_state["global_cue_probabilities"][coin_state["Q"], :] = coin_state["global_cue_probabilities"][coin_state["Q"], :] * sb_weight
                    
            return coin_state
        
        def update_sufficient_statistics_for_parameters(self, coin_state: Dict[str, Any], cue: Optional[int] = None) -> Dict[str, Any]:
            # Check if cues are provided when cues_exist is True
            if coin_state["cues_exist"] and cue is None:
                raise ValueError("Cue must be provided when cues_exist is True.")
            
            # update sufficient statistics for the parameters of the global transition probabilities
            coin_state = self.update_sufficient_statistics_global_transition_probabilities(coin_state)
            
            # update sufficient statistics for the parameters of the global cue probabilities
            if coin_state["cues_exist"] and cue is not None:
                coin_state = self.update_sufficient_statistics_global_cue_probabilities(coin_state, cue)
                
            if coin_state["trial"] > 1:
                # update sufficient for the parameters of the state dynamics function
                coin_state = self.update_sufficient_statistics_dynamics(coin_state)
            
            if self.infer_bias and (coin_state["feedback_observed"][coin_state["trial"]-1]):
                coin_state = self.update_sufficient_statistics_bias(coin_state)
            
            return coin_state
        
        def update_sufficient_statistics_global_cue_probabilities(self, coin_state: Dict[str, Any], cue: int) -> Dict[str, Any]:
            inds_1 = coin_state["context"] - 1 # TODO: is the -1 right?
            inds_2 = cue * np.ones((self.particles, ), dtype=int)
            inds_3 = np.arange(self.particles)
            
            for i in range(self.particles):
                coin_state["n_cue"][inds_1[i], inds_2[i], inds_3[i]] = coin_state["n_cue"][inds_1[i], inds_2[i], inds_3[i]] + 1

            return coin_state
        
        def store_function(self, coin_state: Dict[str, Any], variable: str):
            if variable in ["Kalman_gain_given_cstar2", "state_given_cstar2"]:
                store_on = "previous_trial"
            else:
                store_on = "current_trial"
                
            if "stored" not in coin_state:
                coin_state["stored"] = {}
            
            # If not np.array, convert to np.array
            if not isinstance(coin_state[variable], np.ndarray):
                coin_state[variable] = np.array(coin_state[variable]) # Ensure variable is a valid np array
            
            if ((coin_state["trial"] == 1) and (store_on == "current_trial")) or ((coin_state["trial"] == 2) and (store_on == "previous_trial")):
                s = list(coin_state[variable].shape) + [1]
                coin_state["stored"][variable] = np.ones(s) * np.nan
            elif ((coin_state["trial"] > 1) and (store_on == "current_trial")) or ((coin_state["trial"] > 2) and (store_on == "previous_trial")):
                # Extend storage in last dimension, keeping previous values
                coin_state["stored"][variable] = np.concatenate(
                    [coin_state["stored"][variable], np.full(list(coin_state[variable].shape) + [1], np.nan)],
                    axis=-1
                )
                
            if store_on == "current_trial":
                trial = coin_state["trial"]
            elif store_on == "previous_trial":
                trial = coin_state["trial"] - 1
            
            coin_state["stored"][variable][..., trial-1] = coin_state[variable]
            
            return coin_state