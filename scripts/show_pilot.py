"""Summarise a full amortised pilot npz: per-task retention profile and routing."""
import sys

import numpy as np

REPO = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO)

from baselines import fig3_common as f3


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "models/amort_pilot_s0.npz"
    d = np.load(path, allow_pickle=True)
    A = np.asarray(d["A_raw"], dtype=float)                 # (task, ckpt, episode)
    per = np.nanmean(A, axis=2)                             # (task, ckpt)
    print(f"{path}   A_raw {A.shape}   K final {int(d['coin_K_final'])}   "
          f"{float(d['total_seconds']) / 60:.0f} min")
    ck = np.asarray(d["checkpoints"]).tolist()
    head = "  ".join(f"{c:>7}" for c in ck)
    print(f"\n{'task':<18}{head}   <- rollouts trained")
    for name, row in zip(f3.TASK_NAMES, per):
        print(f"{name:<18}" + "  ".join(f"{v:7.1f}" for v in row))
    print(f"\n{'mean':<18}" + "  ".join(f"{v:7.1f}" for v in np.nanmean(per, axis=0)))

    if "A_raw_oracle_routed" in d:
        O = np.nanmean(np.asarray(d["A_raw_oracle_routed"], dtype=float), axis=2)
        print(f"\n{'oracle-routed':<18}" + "  ".join(f"{v:7.1f}"
                                                     for v in np.nanmean(O, axis=0)))
        print("(oracle = best head per cell; the gap to the row above is routing loss)")

    z = np.asarray(d["z"], dtype=float)
    print(f"\nz range over the run: {np.nanmin(z):+.2f} .. {np.nanmax(z):+.2f}"
          f"   (prior_sd 0.5 -> envelope about +-1)")
    kp = np.asarray(d["K_post"], dtype=float)
    print(f"K over the run: {int(np.nanmin(kp))} .. {int(np.nanmax(kp))}")


if __name__ == "__main__":
    main()
