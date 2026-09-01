"""Fig-3 v2 phase runner.

Executes a temp copy of figures.ipynb's Figure-3 section (imports cell + cells 30..N)
in a Jupyter kernel with chosen TRAIN_* flags flipped to True and optional source
substitutions applied. The repo notebook is never modified; npz/figure outputs land in
models/ and figures/ via the executed copy's cwd.

Usage:
  python run_phase.py --flags TRAIN_ORACLES --log oracles.ipynb [--last 51]
                      [--sub "FIG3_SEEDS = [0]::FIG3_SEEDS = FIG3_ALL_SEEDS"] ...
"""
import argparse
import re
import sys
import time

import nbformat
from nbclient import NotebookClient

REPO = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
NB = REPO + r"\figures.ipynb"
FIG3_FIRST = 30  # markdown '# Figure 3'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flags", nargs="*", default=[], help="TRAIN_* flags to set True")
    ap.add_argument("--sub", action="append", default=[],
                    help="literal source substitution OLD::NEW (applied to all cells)")
    ap.add_argument("--last", type=int, default=51, help="last notebook cell index to include")
    ap.add_argument("--log", required=True, help="path to write the executed notebook")
    args = ap.parse_args()

    nb = nbformat.read(NB, as_version=4)
    # Sanity: the section starts where we think it does.
    assert nb.cells[FIG3_FIRST].source.lstrip().startswith("# Figure 3"), \
        "cell 30 is no longer the Figure 3 header - update run_phase.py"

    cells = [nb.cells[1]] + nb.cells[FIG3_FIRST:args.last + 1]
    flipped = set()
    for c in cells:
        if c.cell_type != "code":
            continue
        src = c.source
        # notebook-frontend tqdm -> plain tqdm for headless execution logs
        src = src.replace("from tqdm.notebook import tqdm", "from tqdm import tqdm")
        for flag in args.flags:
            new, n = re.subn(rf"^({re.escape(flag)}\s*=\s*)False\b", r"\g<1>True", src,
                             flags=re.M)
            if n:
                flipped.add(flag)
                src = new
        for s in args.sub:
            old, _, newtxt = s.partition("::")
            if old in src:
                src = src.replace(old, newtxt)
                flipped.add(s[:40])
        c.source = src

    missing = [f for f in args.flags if f not in flipped]
    assert not missing, f"flags not found in section: {missing}"

    out = nbformat.v4.new_notebook(cells=cells, metadata=nb.metadata)
    client = NotebookClient(out, timeout=None, kernel_name="python3",
                            resources={"metadata": {"path": REPO}})
    t0 = time.time()
    print(f"[run_phase] executing {len(cells)} cells, flags={sorted(flipped)}", flush=True)
    try:
        client.execute()
        status = "OK"
    except Exception as e:  # noqa: BLE001 - report and persist the partial log
        status = f"FAILED: {type(e).__name__}: {e}"
    nbformat.write(out, args.log)
    dt = time.time() - t0
    print(f"[run_phase] {status} after {dt / 60:.1f} min; log -> {args.log}", flush=True)
    if status != "OK":
        sys.exit(1)


if __name__ == "__main__":
    main()

