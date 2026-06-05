"""Densify the high-bq region of the Scenario III training set and rebuild the dataset.

WHY
---
The XGBoost emulator over-extrapolates the log-likelihood in the data-sparse
corner of the domain (large bq, large |C1|): it invents a false global maximum
(logL ~ 26) above the true one (23.96), where SMEFT19 actually gives ~20. This
is the root cause of:
  * the spike-at-0 / gap in the Delta-chi2 histogram (Fig 1b / 7b);
  * a ~5% inflation of the 2-sigma contour toward higher bq in the maps (Fig 8a);
  * a biased MCMC posterior feeding the correlation matrices (Figs 10/11):
    31.6% of the chain sits at bq>2.5 vs 12.7% in the true posterior.

The principled fix is to give the emulator real SMEFT19 labels where it is
currently guessing. This script adds N_EXTRA points sampled in the slab
  C1 in [-0.30, 0],  C3 in [-0.30, 0],  bq in [BQ_LO, 3.20]
evaluates the exact likelihood, and splices them into the existing dataset so the
emulator learns the true (lower, ridge-shaped) surface there. Re-train with
FORCE_RETRAIN=True afterwards and regenerate Figs 1b/8a/10/11.

LAYOUT INVARIANT
----------------
The notebook's Fig 8a locates the three 2D plane grids as the LAST 3*GRID^2 rows
of combined_rotBIII_2025.dat. This script preserves that: it rebuilds the file as
    [original body] + [new high-bq points] + [original plane grids]
The original file is snapshotted once to `*.orig`; assembly always rebuilds from
that pristine snapshot, so re-running is idempotent (never double-inserts).

USAGE (paper-faithful env — needs SMEFT19 with the custom flavio/smelli; see
README_2025.md and BMesonsML/.venv-paper):

    .venv-paper/bin/python scripts/densify_highbq.py

    # tune via env vars:
    N_EXTRA=8000 BQ_LO=1.2 N_WORKERS=16 .venv-paper/bin/python scripts/densify_highbq.py

The extra evaluations stream to data/extra_highbq_2025.dat and are resumable
(re-run to continue an interrupted job). Delete that file to start the extra
sample over (e.g. after changing N_EXTRA / BQ_LO / SEED).
"""
import sys, os, csv, time, shutil, signal, warnings, multiprocessing
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
from scipy.stats.qmc import Sobol, scale

sys.path.insert(0, str(Path(__file__).resolve().parent))
from setup_likelihood import setup_global_likelihood
import SMEFT19

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "combined_rotBIII_2025.dat"
EXTRA = DATA / "extra_highbq_2025.dat"
ORIG = OUT.with_suffix(".dat.orig")          # pristine snapshot of the source dataset

# Physical domain
C1_RANGE = (-0.30, 0.00)
C3_RANGE = (-0.30, 0.00)
BQ_RANGE = (0.00, 3.20)

# Densification controls (env-overridable)
N_EXTRA = int(os.environ.get("N_EXTRA", "4096"))     # new SMEFT19 points (power of 2 -> balanced Sobol)
BQ_LO = float(os.environ.get("BQ_LO", "1.5"))        # lower edge of the slab to densify
GRID = int(os.environ.get("GRID", "40"))             # must match the dataset's plane grids
SEED = int(os.environ.get("SEED", "202"))            # distinct from the main generation seed (42)
N_WORKERS = int(os.environ.get("N_WORKERS", "12"))
EVAL_TIMEOUT = int(os.environ.get("EVAL_TIMEOUT", "30"))  # per-point wall cap (s); pathological
                                                          # flavio points can hang -> floor them

_LO = np.array([C1_RANGE[0], C3_RANGE[0], BQ_LO])
_HI = np.array([C1_RANGE[1], C3_RANGE[1], BQ_RANGE[1]])
_sc = SMEFT19.scenarios.massrotation


class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


def _eval(row):
    c1, c3, bq = row
    # Each forked worker runs tasks in its main thread, so SIGALRM caps a single
    # evaluation; a point where flavio stalls is floored instead of blocking the run.
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(EVAL_TIMEOUT)
    try:
        return max(float(SMEFT19.likelihood_global([c1, c3, 0.0, 0.0, 0.0, bq], _sc)), -200.0)
    except Exception:
        return -200.0
    finally:
        signal.alarm(0)


def _slab_points(n):
    """Low-discrepancy Sobol sample over the high-bq slab (full C1, C3)."""
    eng = Sobol(d=3, scramble=True, seed=SEED)
    u = eng.random(n)
    return scale(u, _LO, _HI)


def _count_lines(path):
    p = Path(path)
    if not p.exists():
        return 0
    with open(p) as f:
        return sum(1 for _ in f)


def assemble():
    """Rebuild OUT = [orig body] + [extra] + [orig plane grids], idempotently."""
    g3 = 3 * GRID * GRID
    if not ORIG.exists():
        shutil.copyfile(OUT, ORIG)                       # snapshot the pristine source once
        print(f"snapshot of original dataset -> {ORIG.name}", flush=True)
    orig_rows = ORIG.read_text().splitlines()
    if len(orig_rows) <= g3:
        sys.exit(f"{ORIG.name} has {len(orig_rows)} rows <= 3*GRID^2={g3}; wrong GRID?")
    body, grids = orig_rows[:-g3], orig_rows[-g3:]
    extra_rows = EXTRA.read_text().splitlines()
    with open(OUT, "w") as f:
        f.write("\n".join(body + extra_rows + grids) + "\n")
    print(f"\nrebuilt {OUT.name}: {len(body)} body + {len(extra_rows)} extra + "
          f"{len(grids)} grids = {len(body)+len(extra_rows)+len(grids)} rows", flush=True)
    print("plane grids remain the last 3*GRID^2 rows -> Fig 8a slicing unchanged.", flush=True)
    print("\nNEXT: in the notebook set FORCE_RETRAIN=True (keep FORCE_REBUILD=False) and "
          "re-run to retrain on the densified data and regenerate Figs 1b/8a/10/11.", flush=True)


def main():
    if not OUT.exists():
        sys.exit(f"missing {OUT}; generate the base dataset first (regenerate_dataset.py)")

    pts = _slab_points(N_EXTRA)
    done = _count_lines(EXTRA)
    print(f"densifying slab C1{C1_RANGE} x C3{C3_RANGE} x bq[{BQ_LO}, {BQ_RANGE[1]}] "
          f"with {N_EXTRA} SMEFT19 points (seed={SEED})", flush=True)
    if done >= N_EXTRA:
        print(f"extra file already complete ({done}/{N_EXTRA}); assembling.", flush=True)
        return assemble()
    if done:
        print(f"resuming from {done}/{N_EXTRA}", flush=True)

    print("building global likelihood in parent (inherited by workers via fork)...", flush=True)
    setup_global_likelihood()

    todo = pts[done:].tolist()
    t0 = time.time()
    ctx = multiprocessing.get_context("fork")
    pool = ctx.Pool(N_WORKERS)
    try:
        with open(EXTRA, "a", newline="") as f:
            w = csv.writer(f)
            for i, (row, lk) in enumerate(zip(todo, pool.imap(_eval, todo, chunksize=16)), done + 1):
                w.writerow([*row, lk])
                if i % 500 == 0 or i == N_EXTRA:
                    f.flush()
                    el = time.time() - t0
                    eta = el / (i - done) * (N_EXTRA - i)
                    print(f"  {i}/{N_EXTRA} | {el:.0f}s | ETA {eta:.0f}s", flush=True)
    finally:
        # forked SMEFT19/flavio workers carry threaded BLAS state and can spin on a
        # graceful pool join; terminate() reaps them immediately instead of hanging.
        pool.terminate()
        pool.join()
    print(f"\nextra evaluations done in {time.time()-t0:.0f}s -> {EXTRA.name}", flush=True)
    assemble()


if __name__ == "__main__":
    main()
