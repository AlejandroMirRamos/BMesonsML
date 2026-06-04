"""Recompute the Scenario III best-fit + covariance with the 2025 measurements.

Writes data/rotBIII_2025.yaml in the SMEFT19 ellipse format (bf, v, d, L) so the
notebook's existing reconstruction (Sigma = V.T @ diag(1/eig) @ V) works unchanged.
Does NOT overwrite the original rotBIII.yaml.
"""
import sys, warnings, time
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from setup_likelihood import setup_global_likelihood
import SMEFT19

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "rotBIII_2025.yaml"

sc = SMEFT19.scenarios.massrotation


def refit_bestfit(out_path=OUT, setup=True):
    if setup:
        setup_global_likelihood()

    def logL(x):
        return float(SMEFT19.likelihood_global([x[0], x[1], 0.0, 0.0, 0.0, x[2]], sc))

    def neg(x):
        return -logL(x)

    from scipy.optimize import minimize
    t_ev = time.time(); sm = logL([0.0, 0.0, 0.0]); dt = time.time() - t_ev
    print(f"logL(SM) = {sm:.4f}   (single eval = {dt:.2f}s)", flush=True)

    # A couple of starts to confirm convergence to the same interior minimum.
    starts = [(-0.186, -0.111, 0.82), (-0.205, -0.12, 0.64)]
    best = None
    for x0 in starts:
        r = minimize(neg, x0, method="Nelder-Mead",
                     options={"xatol": 5e-4, "fatol": 5e-4, "maxfev": 250})
        if best is None or r.fun < best.fun:
            best = r
        print(f"  start {x0} -> bf=({r.x[0]:+.4f},{r.x[1]:+.4f},{r.x[2]:+.4f}) logL={-r.fun:.4f}", flush=True)
    bf = best.x
    Lmin = best.fun  # = -logL(bf)
    logL_bf = -Lmin
    print(f"\nBEST-FIT: C1={bf[0]:.5f} C3={bf[1]:.5f} bq={bf[2]:.5f}", flush=True)
    print(f"logL_bf={logL_bf:.4f}  dchi2_SM={2*(logL_bf-sm):.2f}", flush=True)

    # Numerical Hessian of -logL at the minimum (finite differences)
    h = np.array([0.004, 0.006, 0.05])  # ~ step per parameter
    n = 3
    H = np.zeros((n, n))
    f0 = neg(bf)
    for i in range(n):
        ei = np.zeros(n); ei[i] = h[i]
        H[i, i] = (neg(bf + ei) - 2 * f0 + neg(bf - ei)) / h[i] ** 2
    for i in range(n):
        for j in range(i + 1, n):
            ei = np.zeros(n); ei[i] = h[i]
            ej = np.zeros(n); ej[j] = h[j]
            fpp = neg(bf + ei + ej); fpm = neg(bf + ei - ej)
            fmp = neg(bf - ei + ej); fmm = neg(bf - ei - ej)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * h[i] * h[j])

    cov = np.linalg.inv(H)
    # enforce symmetric positive-definite (guards against a noisy/ill-conditioned Hessian)
    cov = 0.5 * (cov + cov.T)
    w_eig, V_eig = np.linalg.eigh(cov)
    if np.any(w_eig <= 0):
        print(f"  warning: non-PD covariance (eigs={w_eig}); flooring", flush=True)
    w_eig = np.clip(w_eig, 1e-8, None)
    cov = (V_eig * w_eig) @ V_eig.T
    # cov = u·diag(d)·u.T (svd of symmetric PD). The notebook/regenerate reconstruct
    # Sigma = V.T·diag(1/eig)·V, so store V = u.T to make that recover cov exactly.
    u, d, _ = np.linalg.svd(cov)
    v = u.T
    d_ellipse = np.diag(1.0 / d)
    sig = np.sqrt(np.diag(cov))
    print(f"marginal sigmas: C1={sig[0]:.4f} C3={sig[1]:.4f} bq={sig[2]:.4f}", flush=True)
    _chk = np.sqrt(np.diag(v.T @ np.diag(1.0 / d) @ v))
    assert np.allclose(_chk, sig), f"covariance round-trip mismatch: {_chk} vs {sig}"

    values = {
        "name": "Mass Rotation fit, Scenario III (2025 measurements)",
        "L": float(Lmin),
        "fit": "rotBIII",
        "bf": [float(x) for x in bf],
        "v": [[float(x) for x in row] for row in v],
        "d": [[float(x) for x in row] for row in d_ellipse],
    }
    with open(out_path, "w") as f:
        yaml.dump(values, f)
    print(f"\nsaved -> {out_path}", flush=True)
    return values


if __name__ == "__main__":
    t0 = time.time()
    refit_bestfit()
    print(f"done in {time.time()-t0:.0f}s", flush=True)
