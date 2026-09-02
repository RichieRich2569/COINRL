"""Full amortised COIN-PPO pilot, run as a plain script rather than through nbclient.

Same function body as figures.ipynb cell 37 -- it is exec'd from the notebook, so this
cannot drift from what TRAIN_AMORTISED would run -- but driven directly so the job
streams progress. The nbclient route emits nothing for ~2 hours while the cell runs,
which is indistinguishable from a hang; this prints a line per checkpoint instead.

Writes models/<out>.npz. Does NOT touch the committed fig3_amortised_s*.npz.

Usage: python run_amortised_pilot.py [--seed 0] [--blocks 300 100 50 50 50]
"""
import argparse
import sys
import time

REPO = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO)

import nbformat
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blocks", type=int, nargs="+", default=[300, 100, 50, 50, 50])
    ap.add_argument("--kl-coef", type=float, default=1e-4,
                    help="the notebook constant is 0.0, which dates from the tanh"
                         " bound; since ba254f6 the KL is the only containment and"
                         " beta_window_check.py calibrates 1e-4")
    ap.add_argument("--prior-sd", type=float, default=0.5)
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(4)

    import rl
    from rl import AmortisedCOINPPOAgent
    from baselines import fig3_common as f3
    from realtimecoin import RealTimeCOIN
    from tqdm.auto import tqdm

    nb = nbformat.read(REPO + r"\figures.ipynb", as_version=4)
    src = nb.cells[37].source
    body = src[:src.index("if TRAIN_AMORTISED:")]
    ns = {"np": np, "torch": torch, "rl": rl, "f3": f3, "tqdm": tqdm,
          "AmortisedCOINPPOAgent": AmortisedCOINPPOAgent,
          "RealTimeCOIN": RealTimeCOIN, "TRAIN_AMORTISED": False,
          "__name__": "nbcell"}
    exec(compile(body, "figures.ipynb:cell37", "exec"), ns)
    run = ns["run_single_rep_amortised"]

    blocks = tuple(int(b) for b in args.blocks)
    out = args.out or f"models/amort_pilot_s{args.seed}.npz"
    print(f"amortised pilot: seed {args.seed}, blocks {blocks}, "
          f"kl_coef {args.kl_coef}, prior_sd {args.prior_sd}", flush=True)
    print(f"tasks: {f3.TASK_NAMES}", flush=True)

    t0 = time.perf_counter()
    res = run(args.seed, block_sizes=blocks, kl_coef=args.kl_coef,
              prior_sd=args.prior_sd, eval_episodes=args.eval_episodes,
              progress=True)
    dt = time.perf_counter() - t0

    A = np.asarray(res["A_raw"], dtype=float)
    per_task_final = np.nanmean(A[:, -1, :], axis=1)
    per_task_first = np.nanmean(A[:, 0, :], axis=1)
    print(f"\ncompleted in {dt / 60:.1f} min", flush=True)
    print(f"{'task':<18} {'ckpt 1':>8} {'final':>8}")
    for name, a, b in zip(f3.TASK_NAMES, per_task_first, per_task_final):
        print(f"{name:<18} {a:8.1f} {b:8.1f}")
    print(f"\nmean final {np.nanmean(per_task_final):.1f}   "
          f"K final {res['coin_K_final']}")

    save = {k: v for k, v in res.items() if k != "meta"}
    save["meta"] = np.array(res["meta"], dtype=object)
    np.savez(out, **save)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
