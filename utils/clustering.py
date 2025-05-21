import numpy as np

# ------ K MEANS ------
def _kmeans_pp_init(x: np.ndarray, K: int, rng: np.random.Generator):
    """k‑means++ initialisation for 1‑D data (returns K starting centroids)."""
    centroids = np.empty(K, dtype=x.dtype)
    centroids[0] = rng.choice(x)

    # Squared distances to the closest chosen centroid
    d2 = (x - centroids[0]) ** 2

    for k in range(1, K):
        probs = d2 / d2.sum()
        centroids[k] = rng.choice(x, p=probs)
        d2 = np.minimum(d2, (x - centroids[k]) ** 2)

    return centroids

def remap_labels_by_first_appearance(labels):
    label_map = {}
    new_labels = []
    next_label = 0
    for label in labels:
        if label not in label_map:
            label_map[label] = next_label
            next_label += 1
        new_labels.append(label_map[label])
    return np.array(new_labels), label_map

def kmeans_1d(
    P,
    K: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int | None = None,
    verbose: bool = False,
):
    """
    Clusters 1‑D data P into K groups.

    Parameters
    ----------
    P : array‑like shape (N,)
        The data points (floats or ints).
    K : int
        Number of clusters.
    max_iter : int
        Maximum number of Expectation–Maximisation iterations.
    tol : float
        Convergence threshold on centroid shift.
    seed : int or None
        RNG seed for reproducibility.
    verbose : bool
        If True, prints loss and centroid shift each iteration.

    Returns
    -------
    labels : ndarray shape (N,)
        Cluster index (0..K‑1) for every point in P.
    centroids : ndarray shape (K,)
        Final centroid positions.
    """
    # --- prepare data & RNG ---
    x = np.asarray(P, dtype=float).ravel()
    N = x.size
    rng = np.random.default_rng(seed)

    # --- initial centroids ---
    centroids = _kmeans_pp_init(x, K, rng)

    for it in range(1, max_iter + 1):
        # ----- E‑step: assign each point to nearest centroid -----
        distances = np.abs(x[:, None] - centroids[None, :])  # (N, K)
        labels = distances.argmin(axis=1)                   # (N,)

        # ----- M‑step: recompute centroids (mean of assigned points) -----
        new_centroids = np.empty_like(centroids)
        for k in range(K):
            points_k = x[labels == k]
            # handle empty cluster ⇒ re‑initialise to random point
            new_centroids[k] = points_k.mean() if points_k.size else rng.choice(x)

        # ----- check convergence -----
        shift = np.abs(new_centroids - centroids).max()
        if verbose:
            inertia = ((x - new_centroids[labels]) ** 2).sum()
            print(f"iter {it:03d}  inertia={inertia:,.3f}  shift={shift:.5f}")
        if shift < tol:
            break
        centroids = new_centroids

        labels, map = remap_labels_by_first_appearance(labels) # Renaming by order of appearance
        new_centroids = centroids.copy()
        for key,value in map.items():
            new_centroids[value] = centroids[key]


    return labels, new_centroids