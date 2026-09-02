"""Tabulate every pair_stream_test run in models/ on the three gates.

The recurring lesson of this work is that single-axis results mislead: the original
two gates passed while the agent was quietly broken, because they measure SEPARATION
and say nothing about whether task A's head survived. This prints all of it together,
including the oracle-routing column that distinguishes the two retention failures --
a head that was overwritten from a head that routing merely lost.

Usage: python summarise_arms.py [glob ...]
"""
import glob
import sys

import numpy as np


def load(path):
    d = np.load(path, allow_pickle=True)
    out = {"name": path.replace("\\", "/").split("/")[-1].replace(".npz", "")}
    code = np.asarray(d["code"], dtype=float)
    blocks = np.asarray(d["blocks"], dtype=int)
    sw = int(blocks[0][1])
    spread = np.asarray(d["spread"], dtype=float)
    pooled = float(np.sqrt(np.nanmean(spread[-1] ** 2)))
    gap_start = abs(code[sw, 0] - code[sw, 1])
    gap_end = abs(code[-1, 0] - code[-1, 1])
    out["gap"] = gap_end
    out["dprime"] = gap_end / max(pooled, 1e-9)
    out["diverge"] = gap_end > max(3 * pooled, gap_start + 0.2)
    ev = np.asarray(d["eval_returns"], dtype=float)
    heads = np.asarray(d["eval_heads"], dtype=int)
    out["evalA"], out["evalB"] = float(ev[0]), float(ev[1])
    out["separate"] = bool(heads[0] != heads[1])
    out["K"] = int(np.asarray(d["kpost"], dtype=int)[-1])
    if "head_ret" in d:
        hr = np.asarray(d["head_ret"], dtype=float)
        out["oracleA"] = float(np.nanmax(hr[0]))
        out["oracleB"] = float(np.nanmax(hr[1]))
        out["nheadsB"] = int(np.nansum(hr[1] > 140))     # heads that learned task B
    else:
        out["oracleA"] = out["oracleB"] = float("nan")
        out["nheadsB"] = -1
    out["retain"] = out["oracleA"] > 140.0
    return out


def main():
    pats = sys.argv[1:] or ["models/*.npz"]
    rows = []
    for p in sorted({f for pat in pats for f in glob.glob(pat)}):
        try:
            rows.append(load(p))
        except Exception as exc:                       # not a pair-test npz
            _ = exc
    if not rows:
        print("no pair-test runs found")
        return

    tick = lambda b: " ok " if b else "FAIL"
    print(f"{'arm':<34} {'div':>4} {'sep':>4} {'ret':>4} {'K':>3} "
          f"{'gap':>7} {'d-prime':>8} {'evalA':>7} {'evalB':>7} "
          f"{'oracA':>7} {'oracB':>7} {'B-heads':>7}")
    for r in sorted(rows, key=lambda r: (-r["oracleA"] if np.isfinite(r["oracleA"])
                                         else 0, r["name"])):
        print(f"{r['name'][:34]:<34} {tick(r['diverge']):>4} {tick(r['separate']):>4} "
              f"{tick(r['retain']):>4} {r['K']:>3} {r['gap']:>7.3f} "
              f"{r['dprime']:>8.1f} {r['evalA']:>7.1f} {r['evalB']:>7.1f} "
              f"{r['oracleA']:>7.1f} {r['oracleB']:>7.1f} {r['nheadsB']:>7d}")
    print("\nret = some head still scores >140 on task A under ORACLE routing;")
    print("B-heads = how many heads learned task B (>140). More than one means the")
    print("task spread across heads, which is how task A's head gets consumed.")


if __name__ == "__main__":
    main()
