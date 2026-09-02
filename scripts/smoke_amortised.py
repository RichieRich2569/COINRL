"""Smoke-test figures.ipynb's amortised runner against the current rl.py.

The full pilot is ~2 hours; this executes the same function body on a miniature
schedule in about a minute, which is enough to catch an API mismatch like the one
that made TRAIN_AMORTISED crash at agent construction (z_scale and the stripped
anchor/dispersion kwargs, removed by 8a2759a / ba254f6).

Usage: python smoke_amortised.py
"""
import sys
import time

REPO = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO)

import nbformat
import numpy as np
import torch


def main():
    torch.set_num_threads(2)

    import rl
    from rl import AmortisedCOINPPOAgent
    from baselines import fig3_common as f3
    from realtimecoin import RealTimeCOIN
    from tqdm.auto import tqdm

    nb = nbformat.read(REPO + r"\figures.ipynb", as_version=4)
    src = nb.cells[37].source
    body = src[:src.index("if TRAIN_AMORTISED:")]        # definition only

    ns = {
        "np": np, "torch": torch, "rl": rl, "f3": f3, "tqdm": tqdm,
        "AmortisedCOINPPOAgent": AmortisedCOINPPOAgent,
        "RealTimeCOIN": RealTimeCOIN,
        "TRAIN_AMORTISED": False,
        "__name__": "nbcell",
    }
    exec(compile(body, "figures.ipynb:cell37", "exec"), ns)
    run = ns["run_single_rep_amortised"]
    print("runner compiled and defined", flush=True)

    t0 = time.perf_counter()
    out = run(0, block_sizes=(2, 1, 1, 1, 1), n_segments=2, seg_steps=32,
              mini_epochs=1, mb_size=16, enc_steps=2, mb_segments=1,
              eval_episodes=2, eval_max_steps=50, max_episode_steps=50,
              progress=False)
    dt = time.perf_counter() - t0

    A = np.asarray(out["A_raw"], dtype=float)
    print(f"\nOK in {dt:.0f}s")
    print(f"  A_raw            {A.shape}   finite: {np.isfinite(A).all()}")
    print(f"  oracle_routed    {np.asarray(out['A_raw_oracle_routed']).shape}")
    print(f"  train_returns    {np.asarray(out['train_returns']).shape}")
    print(f"  coin_K_final     {out['coin_K_final']}")
    print(f"  meta keys        {len(out['meta'])}")
    stale = [k for k in ("z_scale", "anchor_window", "anchor_warmup", "disp_coef",
                         "inv_coef", "disp_target_sd") if k in out["meta"]]
    print(f"  stale meta keys  {stale or 'none'}")


if __name__ == "__main__":
    main()
