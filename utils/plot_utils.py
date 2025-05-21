import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

def plot_context_probabilities(
    prob: np.ndarray,
    param_vals: np.ndarray | None = None,
    training_data: np.ndarray | None = None,
    train_stride: int = 1000,
    ax: plt.Axes | None = None,
    base_maps: tuple[str, str] = ("tab10", "tab20"),
    novel_colour: str = "k",
    legend: bool = True,
    min_intensity: float = 0.3,
    fig_size: tuple[float, float] = (8, 6),
):
    """
    Make a single RGB heat-map where each context has its own hue and the
    brightness of that hue is the responsibility value.  The *last* context
    in `prob` is treated as “novel” and is shown in solid black.

    Parameters
    ----------
    prob : ndarray, shape (B, T, C)
        Responsibility values in [0, 1].
    param_vals : 1-D ndarray, length B, optional
        Values of the swept parameter (for x-tick labels).  If None,
        plain indices are used.
    training_data : 1-D ndarray, optional
        Raw data sequence used to place white-edged scatter markers
        (same units as `param_vals`).  If None, markers are skipped.
    train_stride : int, default 1000
        Plot every `train_stride`-th episode for the scatter markers.
    ax : matplotlib Axes, optional
        If provided, the plot is drawn on this Axes; otherwise a new
        figure+axes are created and returned.
    base_maps : (str, str), default ('tab10', 'tab20')
        Matplotlib categorical palettes from which colours are pulled.
        (You can replace with custom RGB tuples if you like.)
    novel_colour : str, default 'k'
        Colour swatch for the “novel” context (last index of `prob`).
    legend : bool, default True
        Whether to draw the context legend.
    min_intensity : float, default 0.3
        Minimum intensity for the known contexts.  For highlighting close to zero values.
    fig_size : (float, float), default (8, 6)
        Size of the figure in inches.  Ignored if `ax` is provided.
    """
    """

    Returns
    -------
    ax : matplotlib Axes
        The Axes that now contains the picture.
    """
    # ── 0.  sanity checks ───────────────────────────────────────────────
    if prob.ndim != 3:
        raise ValueError("prob must have shape (B, T, C)")
    B, T, C_full = prob.shape
    if C_full < 2:
        raise ValueError("Need at least two contexts (one novel).")

    # ── 1.  separate known vs. novel contexts ───────────────────────────
    known_pred = np.nan_to_num(prob[:, :, :-1])  # ignore last, replace NaN
    novel_pred = np.nan_to_num(prob[:, :, -1])   # (B, T)

    B, T, C = known_pred.shape  # C = C_full − 1

    # ── 2.  pick a distinct colour for each known context ───────────────
    cmap_A = plt.get_cmap(base_maps[0]).colors
    cmap_B = plt.get_cmap(base_maps[1]).colors
    base_rgbs = np.vstack((cmap_A, cmap_B))[:C, :3]    # at least C colours

    # ── 3.  build the RGB canvas ────────────────────────────────────────
    rgb_img = np.zeros((T, B, 3), dtype=float)         # rows=T, cols=B
    for c, col in enumerate(base_rgbs):
        # intensity transpose so time runs downward
        rgb_img += np.where(known_pred[:, :, c].T[..., None] > 0,
                            np.maximum(known_pred[:, :, c].T[..., None], min_intensity),
                            0) * col

    # add the novel context in plain black (scaled by its intensity)
    rgb_img += novel_pred.T[..., None] * np.array(to_rgb(novel_colour))

    # clip overflow
    rgb_img = np.clip(rgb_img, 0, 1)

    # mask cells where nothing happens (all contexts zero)
    mask = (np.nansum(prob, axis=2) == 0).T
    rgb_img[mask] = 0

    # ── 4.  plotting ────────────────────────────────────────────────────
    owned_fig = False
    if ax is None:
        owned_fig = True
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    im = ax.imshow(
        rgb_img,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xlabel("Parameter value")
    ax.set_ylabel("Episode")

    # axis ticks
    param_vals = param_vals if param_vals is not None \
                            else np.arange(B, dtype=float)
    ax.set_xticks(np.arange(0, B, max(1, B // 10)))
    ax.set_xticklabels([f"{v:.1f}" for v in
                        param_vals[::max(1, B // 10)]])
    ax.set_yticks(np.arange(0, T, max(1, T // 10)))
    ax.set_yticklabels(np.arange(0, T, max(1, T // 10)))

    # ── 5.  training-data scatter markers ───────────────────────────────
    if training_data is not None:
        d = training_data[::train_stride]
        nanidx = ~np.isnan(d)
        d = d[nanidx]  # remove NaN
        col_idx = np.rint(
            (d - param_vals.min()) /
            (param_vals.max() - param_vals.min()) *
            (param_vals.size - 1)
        ).astype(int)
        row_idx = np.arange(0, T, train_stride)
        row_idx = row_idx[nanidx]
        ax.scatter(
            col_idx, row_idx,
            facecolors="none",
            edgecolors="white",
            s=50, linewidths=1.5, marker="o",
        )

    # ── 6.  legend (hues for contexts + novel swatch) ───────────────────
    if legend:
        handles = [
            plt.Line2D([0], [0], marker="s", linestyle="",
                       markersize=10, markerfacecolor=base_rgbs[c],
                       markeredgecolor="k", label=f"Context {c+1}")
            for c in range(C)
        ]
        handles.append(
            plt.Line2D([0], [0], marker="s", linestyle="",
                       markersize=10, markerfacecolor=novel_colour,
                       markeredgecolor="k", label="Novel Context")
        )
        ax.legend(handles=handles, title="Contexts",
                  bbox_to_anchor=(1.02, 1), loc="upper left",
                  borderaxespad=0.)

    return ax

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

def plot_hor_context_prob(
    prob: np.ndarray,
    param_vals: np.ndarray | None = None,
    training_data: np.ndarray | None = None,
    train_stride: int = 1000,
    ax: plt.Axes | None = None,
    base_maps: tuple[str, str] = ("tab10", "tab20"),
    novel_colour: str = "k",
    legend: bool = True,
    min_intensity: float = 0.3,
    fig_size: tuple[float, float] = (8, 6),
):
    """
    Same as plot_context_probabilities, but with episodes (T) running left→right
    and swept parameter (B) running bottom→top.
    """
    if prob.ndim != 3:
        raise ValueError("prob must have shape (B, T, C)")
    B, T, C_full = prob.shape
    if C_full < 2:
        raise ValueError("Need at least two contexts (one novel).")

    # Separate known vs novel
    known_pred = np.nan_to_num(prob[:, :, :-1])  # (B, T, C-1)
    novel_pred = np.nan_to_num(prob[:, :, -1])   # (B, T)
    _, _, C = known_pred.shape

    # pick colours
    cmap_A = plt.get_cmap(base_maps[0]).colors
    cmap_B = plt.get_cmap(base_maps[1]).colors
    base_rgbs = np.vstack((cmap_A, cmap_B))[:C, :3]

    # build image with shape (rows = B params, cols = T time)
    rgb_img = np.zeros((B, T, 3), dtype=float)
    for c, col in enumerate(base_rgbs):
        intensity = np.maximum(known_pred[:, :, c], min_intensity)
        mask = known_pred[:, :, c] > 0
        rgb_img += (mask[..., None] * intensity[..., None]) * col

    # add novel in black
    rgb_img += novel_pred[..., None] * np.array(to_rgb(novel_colour))
    rgb_img = np.clip(rgb_img, 0, 1)

    # mask zero-total cells
    zero_mask = (np.nansum(prob, axis=2) == 0)
    rgb_img[zero_mask] = 0

    # plotting
    owned_fig = False
    if ax is None:
        owned_fig = True
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    im = ax.imshow(
        rgb_img,
        origin="lower",   # so param_idx=0 sits at bottom
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Parameter value")

    # ticks
    param_vals = param_vals if param_vals is not None else np.arange(B, dtype=float)
    # x-axis: episodes
    x_ticks = np.arange(0, T, max(1, T // 10))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)
    # y-axis: param_vals
    y_ticks = np.arange(0, B, max(1, B // 10))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{v:.1f}" for v in param_vals[y_ticks]])

    # scatter training-data markers
    if training_data is not None:
        d = training_data[::train_stride]
        nanidx = ~np.isnan(d)
        d_clean = d[nanidx]
        # map param value → param index
        y_idx = np.rint(
            (d_clean - param_vals.min()) /
            (param_vals.max() - param_vals.min()) *
            (param_vals.size - 1)
        ).astype(int)
        # x positions are episode indices
        x_idx = np.arange(0, T, train_stride)[nanidx]
        ax.scatter(
            x_idx, y_idx,
            facecolors="none",
            edgecolors="white",
            s=50, linewidths=1.5, marker="o",
        )

    # legend
    if legend:
        handles = [
            plt.Line2D([0], [0], marker="s", linestyle="",
                       markersize=10, markerfacecolor=base_rgbs[c],
                       markeredgecolor="k", label=f"Context {c+1}")
            for c in range(C)
        ]
        handles.append(
            plt.Line2D([0], [0], marker="s", linestyle="",
                       markersize=10, markerfacecolor=novel_colour,
                       markeredgecolor="k", label="Novel Context")
        )
        ax.legend(handles=handles, title="Contexts",
                  bbox_to_anchor=(1.02, 1), loc="upper left",
                  borderaxespad=0.)

    return ax
