"""Regenerate the Scenario III training dataset with the 2025 measurements.

Run AFTER refit_bestfit.py (which writes data/rotBIII_2025.yaml). Produces
data/combined_rotBIII_2025.dat with columns (C1, C3, bq, logL).

--------------------------------------------------------------------------------
Sampling method (revised — why it differs from the original notebook)
--------------------------------------------------------------------------------
The original dataset was 3000 uniform-LHS points + three rigid 50x50 best-fit
plane grids + a uniform +-2 sigma "focus" box (and later a hand-tuned dense C1 grid).
Two problems came from that:
  * the rigid plane grids over-represent the best-fit coordinates -> the "columns of
    dots" artifact in the SHAP plots, and an emulator that is jagged near the peak;
  * the C1 grid spacing (0.006) was coarser than sigma_C1 (~0.015), so the narrow
    C1 ridge was under-resolved -> the C1 plateau, patched with a manual dense grid.

This version uses three complementary components:

  1. SOBOL global  (N_SOBOL): low-discrepancy quasi-random coverage of the full
     physical cube. Better, more uniform global coverage than plain LHS for the
     same point budget -> captures the overall likelihood shape.

  2. COVARIANCE-BASED local cloud (N_LOCAL): Gaussian samples drawn with the fit
     covariance Sigma around the best-fit, as a two-scale mix (tight core s=1.0 +
     broad shoulder s=2.2). Because it follows the actual likelihood ellipsoid it
     automatically densifies the narrow C1 direction WITHOUT a hand-tuned grid, and
     avoids the rigid-grid over-representation -> fixes the peak resolution and the
     SHAP artifact at the root.

  3. Three 2D PLANE grids (GRID x GRID, third parameter fixed at the best-fit): kept
     only so the notebook can still draw the "real SMEFT19 vs emulator" 2D likelihood
     maps (Fig 8a). Smaller than before (40x40). Their slice indices are printed at
     the end so the notebook's PLANES (s0/s1) can be updated.

Point budget defaults to ~13.8k (vs the paper's 10.5k); raise/lower N_SOBOL / N_LOCAL
/ GRID below (or via env vars) to trade emulator accuracy against generation time.

The global likelihood is built ONCE in the parent and inherited by workers via fork
(copy-on-write), so it is not rebuilt per worker.
"""
import sys, os, csv, time, warnings, multiprocessing
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import yaml
from scipy.stats.qmc import Sobol, scale

sys.path.insert(0, str(Path(__file__).resolve().parent))
from setup_likelihood import setup_global_likelihood
import SMEFT19

DATA = Path(__file__).resolve().parent.parent / "data"
BF_FILE = DATA / "rotBIII_2025.yaml"
OUT = DATA / "combined_rotBIII_2025.dat"

# Physical domain
C1_RANGE = (-0.30, 0.00)
C3_RANGE = (-0.30, 0.00)
BQ_RANGE = (0.00, 3.20)

# Sampling budget (override with env vars if desired)
N_SOBOL = int(os.environ.get("N_SOBOL", "4096"))   # global low-discrepancy (power of 2)
N_LOCAL = int(os.environ.get("N_LOCAL", "5000"))   # covariance cloud around best-fit
GRID = int(os.environ.get("GRID", "40"))           # each 2D plane grid is GRID x GRID
LOCAL_SCALES = (1.0, 2.2)                            # two-scale mix for the local cloud
SEED = 42
N_WORKERS = int(os.environ.get("N_WORKERS", "12"))

_LO = np.array([C1_RANGE[0], C3_RANGE[0], BQ_RANGE[0]])
_HI = np.array([C1_RANGE[1], C3_RANGE[1], BQ_RANGE[1]])
_sc = SMEFT19.scenarios.massrotation


def _eval(row):
    c1, c3, bq = row
    try:
        return max(float(SMEFT19.likelihood_global([c1, c3, 0.0, 0.0, 0.0, bq], _sc)), -200.0)
    except Exception:
        return -200.0


def _sobol_points(n):
    eng = Sobol(d=3, scramble=True, seed=SEED)
    u = eng.random(n)
    return scale(u, _LO, _HI)


def _local_cloud(bf, Sigma, n):
    """Two-scale Gaussian cloud around the best-fit, clipped to the physical domain."""
    rng = np.random.default_rng(SEED + 1)
    chol = np.linalg.cholesky(Sigma)
    out = []
    per = [n // 2, n - n // 2]
    for s, m in zip(LOCAL_SCALES, per):
        z = rng.standard_normal((m, 3))
        pts = bf + s * (z @ chol.T)
        out.append(pts)
    pts = np.vstack(out)
    return np.clip(pts, _LO, _HI)


def _plane_grids(bf, g):
    """Three g x g 2D grids; third parameter fixed at the best-fit (for Fig 8a maps)."""
    ranges = [C1_RANGE, C3_RANGE, BQ_RANGE]
    blocks = []
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        k = 3 - i - j
        xs = np.linspace(*ranges[i], g)
        ys = np.linspace(*ranges[j], g)
        pts = []
        for x in xs:
            for y in ys:
                row = [0.0, 0.0, 0.0]
                row[i] = x; row[j] = y; row[k] = float(bf[k])
                pts.append(row)
        blocks.append(np.array(pts))
    return blocks


def regenerate(out_path=OUT, bf_file=BF_FILE, setup=True):
    if not Path(bf_file).exists():
        sys.exit(f"missing {bf_file}; run scripts/refit_bestfit.py first")
    ell = yaml.safe_load(open(bf_file))
    bf = np.array(ell["bf"])
    eig = np.array([ell["d"][i][i] for i in range(3)])
    V = np.array(ell["v"])
    Sigma = V.T @ np.diag(1.0 / eig) @ V
    print(f"best-fit: C1={bf[0]:.4f} C3={bf[1]:.4f} bq={bf[2]:.4f}", flush=True)
    print(f"sigmas:   {np.sqrt(np.diag(Sigma))}", flush=True)

    sobol = _sobol_points(N_SOBOL)
    local = _local_cloud(bf, Sigma, N_LOCAL)
    grids = _plane_grids(bf, GRID)
    all_pts = np.vstack([sobol, local, *grids])

    # Layout (print so the notebook PLANES indices can be updated)
    g2 = GRID * GRID
    base = N_SOBOL + N_LOCAL
    print("\nlayout (row ranges):", flush=True)
    print(f"  sobol global : 0 : {N_SOBOL}", flush=True)
    print(f"  local cloud  : {N_SOBOL} : {base}", flush=True)
    print(f"  C1-C3 grid   : {base} : {base + g2}   (bq fixed)", flush=True)
    print(f"  C1-bq grid   : {base + g2} : {base + 2*g2}   (C3 fixed)", flush=True)
    print(f"  C3-bq grid   : {base + 2*g2} : {base + 3*g2}   (C1 fixed)", flush=True)
    print(f"  TOTAL        : {len(all_pts)}  (GRID={GRID})", flush=True)

    if setup:
        print("\nbuilding global likelihood in parent (inherited by workers via fork)...", flush=True)
        setup_global_likelihood()

    t0 = time.time()
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(N_WORKERS) as pool, open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        for done, (row, lk) in enumerate(
            zip(all_pts.tolist(), pool.imap(_eval, all_pts.tolist(), chunksize=16)), 1
        ):
            w.writerow([*row, lk])
            if done % 500 == 0 or done == len(all_pts):
                el = time.time() - t0
                eta = el / done * (len(all_pts) - done)
                print(f"  {done}/{len(all_pts)} | {el:.0f}s | ETA {eta:.0f}s", flush=True)
    print(f"\ndone in {time.time()-t0:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    regenerate()
