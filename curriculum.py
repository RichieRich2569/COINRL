"""
curriculum.py

A sticky Markov curriculum over tasks, plus the metrics that ask whether COIN has learned
it. The chain is the *ground truth* the model is being tested against: it runs continuously
across rollout boundaries, so the only thing that ever restarts is the segment.

Pure numpy on purpose -- nothing here imports ``rl`` or ``realtimecoin``, so the schedule
and its metrics can be reasoned about (and unit-tested) without a COIN model in the loop.
The metric helpers take plain arrays, including the agents' ``[known..., nan..., novel]``
context layout, whose ``np.nan`` padding is treated as zero mass.
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Union

_EPS = 1e-12


def _as_rng(rng) -> np.random.Generator:
    """Accept a Generator, a seed, or None."""
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def sanitise_probabilities(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    NaN padding -> zero, negatives clipped, rows renormalised to sum to one.

    The agents' context vectors mark uninstantiated slots with ``np.nan`` (see
    ``rl.coin_context_vector``), which every metric below must read as "no mass here"
    rather than propagate. An all-zero row is left alone rather than divided by zero.

    Args:
        p (np.ndarray): Probabilities, possibly containing NaN padding.
        axis (int): Axis along which the probabilities sum to one.

    Returns:
        np.ndarray: A float array of the same shape, free of NaN.
    """
    q = np.nan_to_num(np.asarray(p, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    q = np.clip(q, 0.0, None)
    total = q.sum(axis=axis, keepdims=True)
    return np.divide(q, total, out=np.zeros_like(q), where=total > 0.0)


def stationary_distribution(transition_matrix: np.ndarray) -> np.ndarray:
    """
    Stationary distribution of a row-stochastic matrix, i.e. the leading left eigenvector.

    Args:
        transition_matrix (np.ndarray): An ``(n, n)`` row-stochastic matrix.

    Returns:
        np.ndarray: An ``(n,)`` probability vector satisfying ``v @ T = v``.
    """
    T = sanitise_probabilities(transition_matrix, axis=1)
    vals, vecs = np.linalg.eig(T.T)
    v = np.abs(np.real(vecs[:, int(np.argmin(np.abs(vals - 1.0)))]))
    return v / v.sum()


class MarkovTaskCurriculum:
    """
    A Markov chain over task indices, run continuously.

    One draw per segment, and the chain is never reset: rollout boundaries are an artefact
    of how ``train_step`` batches segments, not a fact about the contingency, so the model
    must be able to learn the transition structure *across* them. The curriculum keeps its
    own ``current`` state for exactly that reason -- the harness calls
    :meth:`sample_next` with no argument and the chain carries on.

    Args:
        transition_matrix (np.ndarray): An ``(n, n)`` row-stochastic matrix; entry
            ``[i, j]`` is ``P(next = j | prev = i)``.
        rng: A ``np.random.Generator``, a seed, or None.
    """

    def __init__(self, transition_matrix: np.ndarray, rng=None):
        T = np.asarray(transition_matrix, dtype=float)
        if T.ndim != 2 or T.shape[0] != T.shape[1] or T.shape[0] == 0:
            raise ValueError(f"transition_matrix must be square and non-empty, got {T.shape}.")
        if np.any(np.nan_to_num(T, nan=0.0) < 0.0):
            raise ValueError("transition_matrix must be non-negative.")
        rows = np.nan_to_num(T, nan=0.0).sum(axis=1)
        if np.any(rows <= 0.0):
            raise ValueError("every row of transition_matrix must carry some mass.")
        self.transition_matrix = sanitise_probabilities(T, axis=1)
        self.rng = _as_rng(rng)
        self.current: Optional[int] = None
        self._stationary: Optional[np.ndarray] = None

    @classmethod
    def sticky(cls, n_tasks: int, stay_prob: float, rng=None) -> "MarkovTaskCurriculum":
        """
        The usual construction: ``stay_prob`` on the diagonal, the remainder spread
        uniformly over the other tasks. Uniform stationary by symmetry.
        """
        n = int(n_tasks)
        if n < 1:
            raise ValueError("n_tasks must be at least 1.")
        p = float(stay_prob)
        if not 0.0 <= p <= 1.0:
            raise ValueError("stay_prob must lie in [0, 1].")
        off = 0.0 if n == 1 else (1.0 - p) / (n - 1)
        T = np.full((n, n), off)
        np.fill_diagonal(T, p if n > 1 else 1.0)
        return cls(T, rng=rng)

    @property
    def n_tasks(self) -> int:
        return int(self.transition_matrix.shape[0])

    @property
    def stationary(self) -> np.ndarray:
        """Stationary distribution, computed once."""
        if self._stationary is None:
            self._stationary = stationary_distribution(self.transition_matrix)
        return self._stationary

    def sample_initial(self) -> int:
        """Draw the first task from the stationary distribution and start the chain there."""
        self.current = int(self.rng.choice(self.n_tasks, p=self.stationary))
        return self.current

    def sample_next(self, prev_idx: Optional[int] = None) -> int:
        """
        Draw the next task index and advance the chain.

        Args:
            prev_idx (Optional[int]): The task to condition on. Defaults to the chain's own
                ``current``; with no history at all this is the stationary draw instead.

        Returns:
            int: The sampled task index, also stored as ``current``.
        """
        prev = self.current if prev_idx is None else int(prev_idx)
        if prev is None:
            return self.sample_initial()
        self.current = int(self.rng.choice(self.n_tasks, p=self.transition_matrix[prev]))
        return self.current

    def sample_block(self, n: int, prev_idx: Optional[int] = None) -> np.ndarray:
        """
        The next ``n`` task indices -- one rollout's worth of segments.

        Continues from ``prev_idx`` (or from ``current``), so calling this once per rollout
        is the same chain as calling :meth:`sample_next` ``n`` times.

        Returns:
            np.ndarray: An ``(n,)`` int array.
        """
        if prev_idx is not None:
            self.current = int(prev_idx)
        return np.array([self.sample_next() for _ in range(int(n))], dtype=int)


#----- Metrics: has COIN learned the chain? -----

def task_context_map(pi: np.ndarray, tasks: np.ndarray, n_tasks: Optional[int] = None) -> np.ndarray:
    """
    The context column each task is routed to, i.e. the modal ``argmax pi`` per task.

    COIN's context labels are its own; the curriculum's task indices are ours. This is the
    correspondence between the two, read off the trace rather than assumed. A task that
    never occurs maps to ``-1``.

    Args:
        pi (np.ndarray): ``(T, C)`` predicted context probabilities, one row per trial,
            in the agent layout (NaN padding allowed).
        tasks (np.ndarray): ``(T,)`` realised task indices.
        n_tasks (Optional[int]): Number of tasks. Defaults to ``max(tasks) + 1``.

    Returns:
        np.ndarray: An ``(n_tasks,)`` int array of context column indices.
    """
    p = sanitise_probabilities(np.atleast_2d(pi), axis=1)
    t = np.asarray(tasks, dtype=int).ravel()
    n = int(t.max()) + 1 if n_tasks is None else int(n_tasks)
    argmax = p.argmax(axis=1)
    out = np.full(n, -1, dtype=int)
    for i in range(n):
        hits = argmax[t == i]
        if hits.size:
            out[i] = int(np.bincount(hits).argmax())
    return out


def predictive_cross_entropy(
    pi: np.ndarray,
    tasks: np.ndarray,
    task_to_context: Optional[np.ndarray] = None,
    eps: float = _EPS,
) -> np.ndarray:
    """
    Per-trial cross-entropy (nats) of a predicted-pi sequence against the realised tasks.

    ``-log pi[t, c(task_t)]``, where ``c`` is the task-to-context correspondence. This is
    the quantity the curriculum is really testing: the one-step-ahead pi is formed *before*
    the segment is observed, so it can only be sharp if COIN has learned the chain.
    Compare it against :func:`oracle_cross_entropy` and :func:`stationary_cross_entropy`.

    Args:
        pi (np.ndarray): ``(T, C)`` predicted context probabilities (agent layout, NaN ok).
        tasks (np.ndarray): ``(T,)`` realised task indices.
        task_to_context (Optional[np.ndarray]): Task -> context column. Defaults to
            :func:`task_context_map` on this same trace.
        eps (float): Floor on the probability before the log, so an unrouted task costs a
            large finite number rather than an infinity.

    Returns:
        np.ndarray: A ``(T,)`` float array of cross-entropies in nats.
    """
    p = sanitise_probabilities(np.atleast_2d(pi), axis=1)
    t = np.asarray(tasks, dtype=int).ravel()
    if p.shape[0] != t.size:
        raise ValueError(f"pi has {p.shape[0]} rows but there are {t.size} tasks.")
    cmap = task_context_map(p, t) if task_to_context is None else np.asarray(task_to_context, dtype=int)
    cols = cmap[t]
    mass = np.where((cols >= 0) & (cols < p.shape[1]), p[np.arange(t.size), np.clip(cols, 0, p.shape[1] - 1)], 0.0)
    return -np.log(np.clip(mass, eps, None))


def oracle_cross_entropy(
    tasks: np.ndarray,
    transition_matrix: np.ndarray,
    stationary: Optional[np.ndarray] = None,
    eps: float = _EPS,
) -> np.ndarray:
    """
    The floor: per-trial cross-entropy under the true conditional row of the chain.

    ``-log T[task_{t-1}, task_t]``, with the first trial scored under the stationary
    distribution since it has no predecessor. No predictor that has only seen the past can
    beat this in expectation.

    Args:
        tasks (np.ndarray): ``(T,)`` realised task indices.
        transition_matrix (np.ndarray): The true ``(n, n)`` row-stochastic matrix.
        stationary (Optional[np.ndarray]): Distribution for trial 0. Defaults to the
            matrix's own stationary distribution.
        eps (float): Floor on the probability before the log.

    Returns:
        np.ndarray: A ``(T,)`` float array of cross-entropies in nats.
    """
    T = sanitise_probabilities(transition_matrix, axis=1)
    t = np.asarray(tasks, dtype=int).ravel()
    if t.size == 0:
        return np.zeros(0)
    stat = stationary_distribution(T) if stationary is None else sanitise_probabilities(stationary, axis=0)
    probs = np.empty(t.size)
    probs[0] = stat[t[0]]
    if t.size > 1:
        probs[1:] = T[t[:-1], t[1:]]
    return -np.log(np.clip(probs, eps, None))


def stationary_cross_entropy(
    tasks: np.ndarray,
    transition_matrix: np.ndarray,
    eps: float = _EPS,
) -> np.ndarray:
    """
    The ceiling worth beating: per-trial cross-entropy under the stationary distribution.

    A predictor scoring at this level knows how often each task occurs but nothing about
    the order, so it has learned the marginal and not the chain. Accepts either the
    transition matrix or a stationary vector directly.

    Args:
        tasks (np.ndarray): ``(T,)`` realised task indices.
        transition_matrix (np.ndarray): The true ``(n, n)`` matrix, or an ``(n,)``
            stationary vector.
        eps (float): Floor on the probability before the log.

    Returns:
        np.ndarray: A ``(T,)`` float array of cross-entropies in nats.
    """
    M = np.asarray(transition_matrix, dtype=float)
    stat = sanitise_probabilities(M, axis=0) if M.ndim == 1 else stationary_distribution(M)
    t = np.asarray(tasks, dtype=int).ravel()
    return -np.log(np.clip(stat[t], eps, None))


def empirical_transition_matrix(
    tasks: np.ndarray,
    n_tasks: Optional[int] = None,
    prior: float = 0.0,
) -> np.ndarray:
    """
    Count-based transition matrix of a realised task sequence.

    Useful both as a sanity check on the sampler and as the "what actually happened"
    reference when the realised run is short enough that it differs from the true matrix.

    Args:
        tasks (np.ndarray): ``(T,)`` realised task indices.
        n_tasks (Optional[int]): Matrix size. Defaults to ``max(tasks) + 1``.
        prior (float): Pseudo-count added to every entry; keeps unvisited rows uniform
            instead of all-zero.

    Returns:
        np.ndarray: An ``(n, n)`` row-stochastic matrix.
    """
    t = np.asarray(tasks, dtype=int).ravel()
    n = (int(t.max()) + 1 if t.size else 1) if n_tasks is None else int(n_tasks)
    counts = np.full((n, n), float(prior))
    if t.size > 1:
        np.add.at(counts, (t[:-1], t[1:]), 1.0)
    rows = counts.sum(axis=1, keepdims=True)
    uniform = np.full((n, n), 1.0 / n)
    return np.divide(counts, rows, out=uniform, where=rows > 0.0)


def transition_matrix_kl(
    inferred: np.ndarray,
    true: np.ndarray,
    weights: Optional[np.ndarray] = None,
    eps: float = _EPS,
) -> Tuple[np.ndarray, float]:
    """
    Compare an inferred transition matrix against the true one, row by row.

    Reports ``KL(true_row || inferred_row)`` in nats -- the direction that penalises an
    inferred model for putting no mass where the chain actually goes. The summary is the
    average over rows weighted by how often the chain visits them, so a rarely-visited row
    cannot dominate.

    Args:
        inferred (np.ndarray): The ``(n, n)`` matrix COIN holds (NaN padding allowed; slice
            it to the task block with :func:`reorder_transition_matrix` first).
        true (np.ndarray): The curriculum's own ``(n, n)`` matrix.
        weights (Optional[np.ndarray]): Row weights. Defaults to the true matrix's
            stationary distribution.
        eps (float): Floor on the inferred probabilities before the log.

    Returns:
        Tuple[np.ndarray, float]: Per-row KL ``(n,)`` and the weighted summary scalar.
    """
    P = sanitise_probabilities(true, axis=1)
    Q = sanitise_probabilities(inferred, axis=1)
    if P.shape != Q.shape:
        raise ValueError(f"shape mismatch: inferred {Q.shape} vs true {P.shape}.")
    ratio = np.log(np.clip(P, eps, None)) - np.log(np.clip(Q, eps, None))
    per_row = np.sum(np.where(P > 0.0, P * ratio, 0.0), axis=1)
    w = stationary_distribution(P) if weights is None else sanitise_probabilities(weights, axis=0)
    return per_row, float(np.dot(w, per_row))


def reorder_transition_matrix(matrix: np.ndarray, order: np.ndarray) -> np.ndarray:
    """
    Pull the task-ordered sub-block out of a context-indexed matrix.

    COIN's transition matrix is indexed by *its* contexts and is usually wider than the
    task set; ``order`` is the task-to-context map, so ``order[i]`` is the row and column
    belonging to task ``i``. Rows are renormalised afterwards, since dropping the other
    contexts (novel included) removes mass.

    Args:
        matrix (np.ndarray): A ``(C, C)`` matrix in the context frame (NaN padding allowed).
        order (np.ndarray): ``(n,)`` context index per task, as from :func:`task_context_map`.

    Returns:
        np.ndarray: An ``(n, n)`` row-stochastic matrix in task order.
    """
    M = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0)
    idx = np.asarray(order, dtype=int)
    if np.any(idx < 0) or np.any(idx >= M.shape[0]):
        raise ValueError("order contains a context index outside the matrix.")
    return sanitise_probabilities(M[np.ix_(idx, idx)], axis=1)
