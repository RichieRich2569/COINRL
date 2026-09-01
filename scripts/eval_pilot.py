"""Gate evaluation for the latest amortised pilot npz."""
import numpy as np

import pathlib, sys
_default = pathlib.Path(__file__).resolve().parent.parent / "models" / "fig3_amortised_s0.npz"
d = np.load(sys.argv[1] if len(sys.argv) > 1 else _default,
            allow_pickle=True)
names = list(d["task_names"])
A = d["A_raw"].mean(axis=2)
print("A_raw mean (rows=task, cols=ckpt 300/400/450/500/550):")
for i in range(5):
    print(f"  {names[i]:>16}: " + " ".join(f"{A[i, j]:8.1f}" for j in range(5)))

W = d["eval_w_mean"]
print("\ndominant eval head per (task, ckpt):")
for i in range(5):
    print(f"  {names[i]:>16}: " + "  ".join(
        f"{int(np.argmax(W[i, j]))}:{W[i, j].max():.2f}" for j in range(5)))

z, zsd = d["z"], d["z_sd"]
blocks = [(0, 300), (300, 400), (400, 450), (450, 500), (500, 550)]
print("\nz per block (mean +- sd | reported sd | last-10 mean):")
for i, (a, b) in enumerate(blocks):
    zz = z[a:b].ravel()
    print(f"  {names[i]:>16}: {zz.mean():+.3f} +- {zz.std():.3f} | "
          f"{zsd[a:b].mean():.3f} | {z[b - 10:b].mean():+.3f}")

kp = d["K_post"]
grew = np.where(np.diff(kp) > 0)[0] + 1
print(f"\ncontext births at rollouts {list(grew)} (block edges 300/400/450/500); "
      f"K final = {int(d['coin_K_final'])}")

rho = d["rho"]
print("train-time dominant context (last 10 rollouts of each block):")
for i, (a, b) in enumerate(blocks):
    r = np.nanmean(rho[b - 10:b], axis=(0, 1))
    print(f"  {names[i]:>16}: head {int(np.nanargmax(r))} w={np.nanmax(r):.2f} "
          f"full={np.round(r, 2)}")

if "enc_value_loss" in d.files:
    v = d["enc_value_loss"]
    ok = np.isfinite(v)
    print(f"\nenc_value_loss: logged {ok.sum()}/{v.size} rollouts, "
          f"first/last finite: {v[ok][0]:.1f} -> {v[ok][-1]:.1f}"
          if ok.any() else "\nenc_value_loss: never finite")
else:
    print("\nenc_value_loss NOT in npz (runner cell predates the stat)")

sh = d["eval_sharpen"]
print("median sharpen step at final ckpt per task:",
      np.round(np.nanmedian(sh[:, -1, :], axis=1), 1))

