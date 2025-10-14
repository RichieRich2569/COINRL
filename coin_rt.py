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
            # runs
            runs: int = 1, 
            # parallel processing
            max_cores: int = 1, 
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
                # runs
                runs=runs, 
                # parallel processing
                max_cores=max_cores, 
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
            # Mode number of instantiated contexts per trial
            self.mode_number_of_contexts = np.zeros((1, ), dtype=int) # shape (1, ) for initialization
            # Posterior over number of contexts
            self.posterior = np.zeros((self.max_contexts+1, 1))
            self.posterior_mean = np.zeros((1, ))
            self.posterior_mode = np.zeros((1, ), dtype=int)



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
            trial = self.coin_state["trial"]
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
                self.cues.insert(trial-1, cue)
            
            # Feedback observed or not
            temp = np.ones((trial, ))
            temp[:trial-1] = self.coin_state["feedback_observed"]
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
            S["weights"] = np.ones((self.runs, )) / self.runs
            S["properties"] = self
            return S
        
        def find_optimal_context_labels(self, S: Dict[str, Any]):
            """
               Find the optimal context labels for plotting by minimising the Hamming distance between context sequences across particles.
               Real-time option, which updates the optimal context labels once per trial.
            """
            # Obtain inds_resampled, only 1 run in RT version.
            inds_resampled = S["runs"][0]["resample_inds"]

            # A dict form {trial: array of shape [P,trial]} where we follow the context sequence for each particle up to the given trial
            context_sequence = self.context_sequence(S, inds_resampled)

            # Obtain the number of instantiated contexts for each particle as they vary per trial
            # - C: array of shape [R*P,T] with the number of instantiated contexts for each particle at each trial
            # - mode_number_of_contexts: array of shape [T] with modal number of instantiated contexts across particles at each trial
            C, _, _, mode_number_of_contexts = self.posterior_number_of_contexts(context_sequence, S)
            
            P = {}
            P["mode_number_of_contexts"] = mode_number_of_contexts
            
            # context label permutations

            # All possible permutations of context labels up to the maximum number of contexts
            L = np.array(list(permutations(np.arange(0,np.max(mode_number_of_contexts).astype(int)))))

            # Rearrange shape so that L has shape [max_mode_number_of_contexts, 1, num_permutations], also obtain number of permutations
            L = np.transpose(L[None], (2, 0, 1))
            n_perms = factorial(np.max(mode_number_of_contexts).astype(int))
            
            num_trials = len(self.perturbations)
            
            f = {}
            to_unique = {}
            from_unique = {}
            optimal_assignment = {}
            
            for i in range(num_trials):
                if np.mod(i+1, 50) == 0:
                    print(f"Finding optimal context labels (trial = {i+1})")
                
                # exclude sequences for which C > max(mode_number_of_context) as these sequences
                # (and their descendents) will never be analysed
                f[i] = np.where(C[:, i] <= np.max(mode_number_of_contexts))[0]
                
                # identify unique sequences (to avoid performing the same computations multiple times)
                unique_seqs, inds, reverse_inds = np.unique(
                    context_sequence[i][f[i]], axis=0, return_index=True, return_inverse=True
                )
                to_unique[i] = inds
                from_unique[i] = reverse_inds
                
                n_sequences = len(unique_seqs)
                
                # identify particles that have the same number of contexts as the most common number of
                # contexts (only contexts (only these particles will be analysed)
                valid_particle_inds = (C[f[i], i] == mode_number_of_contexts[i])
                
                if i == 0:
                    # hamming distances on trial 0
                    # dimension 2 of H considers all possible label permutations
                    H = (L[[0], :, :] != 0) * 1.0 # (1, 1, num_permutations)
                else:
                    # identify a valid parent of each unique sequence
                    # i.e., a sequence on the previous trial that is identical up to the previous trial
                    inds, _ = np.where(f[i-1][:, None] == inds_resampled[f[i][to_unique[i]], i][None])
                    parent = from_unique[i-1][inds]
                    
                    # pass Hamming distances from parents to children
                    inds_1 = np.tile(parent[:, None, None], [1, n_sequences, n_perms])
                    inds_2 = np.tile(parent[None, :, None], [n_sequences, 1, n_perms])
                    inds_3 = np.tile(np.arange(n_perms)[None, None], [n_sequences, n_sequences, 1])
                    
                    H_new = np.zeros((n_sequences, n_sequences, n_perms))
                    
                    for ii in range(n_sequences):
                        for jj in range(n_sequences):
                            for kk in range(n_perms):
                                H_new[ii, jj, kk] = H[inds_1[ii, jj, kk], inds_2[ii, jj, kk], inds_3[ii, jj, kk]]
                    
                    # recursively update Hamming distances
                    # dimension 2 of H considers all possible label permutations
                    for seq in range(n_sequences):
                        H_new[seq:, [seq], :] = H_new[seq:, [seq], :] + ((unique_seqs[seq, -1]-1) != L[unique_seqs[seq:, -1]-1, :, :]) * 1.0
                        H_new[seq, seq:, :] = H_new[seq:, seq, :] # by symmetry of Hamming distance
                    
                    H = H_new
                
                # compute the Hamming distance between each pair of sequences (after optimally permuting labels)
                H_optimal = np.min(H, axis=2)
                
                # count the number of times each unique sequence occurs
                sequence_count = np.sum(
                    from_unique[i][valid_particle_inds][:, None] == np.arange(len(unique_seqs))[None], 
                    axis=0, 
                )
                
                # compute the mean optimal Hamming distance of each sequence to all other sequences.
                # the distance from sequence i to sequence j is weighted by the number of times sequence j occurs.
                # if i == j, this weight is reduced by 1 so that the distance from one instance of sequence i to itself is ignored.
                H_mean = np.mean(H_optimal * (sequence_count[None] - np.eye(n_sequences)), axis=1)
                
                # assign infinite distance to invalid sequences 
                # i.e., sequences for which the number of contexts is not equal to the most common number of contexts
                H_mean[sequence_count == 0] = np.inf
                
                # find the index of the typical sequence 
                # (the sequence with minimum mean optimal Hamming distance to all other sequences)
                min_ind = np.argmin(H_mean, axis=0)
                
                # typical context sequence
                typical_sequence = unique_seqs[min_ind, :]
                
                # store the optimal permutation of labels for each sequence with respect to the typical sequence
                j = np.argmin(H[min_ind, :, :], axis=-1)
                optimal_assignment[i] = np.transpose(
                    L[:int(mode_number_of_contexts[i]), :, j].reshape((int(mode_number_of_contexts[i]), -1, 1)), 
                    [2, 0, 1], 
                )
        
            return P, S, optimal_assignment, from_unique, context_sequence, C
        
        def context_sequence(self, S: Dict[str, Any], inds_resampled: np.ndarray):
            """Reconstruct the context sequence for each particle in each run, after resampling. Rebuild for RT version."""
            
            # Update our context sequence dictionary
            i = self.trial
            self.context_seq[i] = np.zeros((self.particles, i+1), dtype=int)
            if i > 0:
                # Set the context sequence up to trial i-1 to be the same as that of previous trial sequence
                self.context_seq[i][:, :i] = self.context_seq[i-1][:, :]
                # Appropriately resample all context values according to inds_resampled
                self.context_seq[i][:, :] = self.context_seq[i][inds_resampled[:, i], :]
            # Set the context at trial i to be the context sampled at trial i
            self.context_seq[i][:, i] = S["runs"][0]["context"][:, i]

            return self.context_seq
        
        def posterior_number_of_contexts(self, context_sequence: Dict[int, Any], S: Dict[str, Any]): 
            """Compute the posterior over the number of contexts instantiated by the model at each trial, given the context sequences."""           
            # extend context tracking variables if number of trials greater than current size of C
            trial = self.trial

            # Ensure shape of all context tracking variables is identical in trial dimension
            if not (np.shape(self.C)[1] == np.shape(self.mode_number_of_contexts)[0] 
                    == np.shape(self.posterior_mean)[0] 
                    == np.shape(self.posterior_mode)[0] 
                    == np.shape(self.posterior)[1]):
                raise ValueError("Context tracking variables have inconsistent trial dimensions.")
            
            existing_trials = np.shape(self.C)[1]
            if existing_trials <  trial:
                self.C = np.hstack((self.C, np.zeros((self.particles * self.runs, trial - existing_trials), dtype=int)))
                self.mode_number_of_contexts = np.hstack((self.mode_number_of_contexts, np.zeros((trial - existing_trials, ), dtype=int)))
                self.posterior = np.hstack((self.posterior, np.zeros((self.max_contexts+1, trial - existing_trials))))
                self.posterior_mean = np.hstack((self.posterior_mean, np.zeros((trial - existing_trials, ))))
                self.posterior_mode = np.hstack((self.posterior_mode, np.zeros((trial - existing_trials, ), dtype=int)))

            for i in range(existing_trials, trial):
                self.C[:, i] = np.max(context_sequence[i], axis=1)

            particle_weight = np.repeat(S["weights"], self.particles) / self.particles
            
            
            for i in range(existing_trials, trial):
                for context in range(np.max(self.C[:, i])):
                    self.posterior[context, i] = np.sum((self.C[:, i] == (context+1)) * particle_weight)

                self.posterior_mean[i] = np.sum(np.arange(1,self.max_contexts+2) * self.posterior[:, i])
                self.posterior_mode[i] = np.argmax(self.posterior[:, i])+1 # contexts seen as 1 and not 0

            return self.C, self.posterior, self.posterior_mean, self.posterior_mode

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