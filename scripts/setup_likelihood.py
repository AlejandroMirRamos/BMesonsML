"""Configure the SMEFT19 global likelihood with up-to-date (2025/2026) measurements.

Call `setup_global_likelihood()` once per process BEFORE any
`SMEFT19.likelihood_global(...)` evaluation. Used by the best-fit refit and the
dataset regeneration so both share identical experimental inputs.

Requires the paper-faithful environment (flavio@BctoJpsi + smelli@custom_measurements):
see BMesonsML/.venv-paper and BMesonsML/README_2025.md.

What it does:
  1. reads data/measurements_2025.yml into flavio's measurement registry
  2. restarts smelli's global likelihood, swapping the b->c tau nu R(D)/R(D*) sector
     to the HFLAV CKM2025 world average and adding the Belle II B+->K+ nu nu result.

R(J/psi): Rtaumu(Bc->J/psilnu) exists in flavio@BctoJpsi but is NOT part of smelli's
default sectors (and smelli's custom_likelihoods can only re-select observables already
present in a sector, not add a new one). It is therefore included as a standalone
Gaussian term added to the global log-likelihood, using the two reference measurements
directly (no hand-average): LHCb 0.71+-0.17+-0.18 [arXiv:1711.05623] and
CMS 0.49+-0.26 [arXiv:2510.21559]. The term is wired in by monkeypatching
SMEFT19.likelihood_global inside setup_global_likelihood(), so every call site (refit,
dataset regeneration, notebook) -- and every fork worker in regenerate_dataset.py, which
inherits the parent's patched module -- evaluates the same J/psi-augmented likelihood.
"""
import warnings
from pathlib import Path

import flavio
import SMEFT19

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MEAS_FILE = DATA_DIR / "measurements_2025.yml"

# Old tau-channel R(D)/R(D*) measurements to drop (replaced by the HFLAV CKM2025 WA).
# The Rmue / Rtaumu(B->D*lnu) measurements are kept (the WA does not cover those observables).
REMOVE = {
    "likelihood_rd_rds.yaml": [
        "BaBar RD 2012", "Belle RD had 2015", "Belle RD* had 2016", "Belle RD* sl 2019",
    ],
}
# New measurements to add to each likelihood sector.
ADD = {
    "likelihood_rd_rds.yaml": ["HFLAV CKM2025 RDRDstar"],
    "likelihood_bqnunu.yaml": ["BelleII BKnunu 2024"],
}

# --- R(J/psi) standalone term (see module docstring) -------------------------
# Exp-only Gaussian errors: the ~few % theory error is negligible vs the ~40% experimental one.
_JPSI_OBS = "Rtaumu(Bc->J/psilnu)"
_JPSI_MEAS = (
    (0.71, (0.17 ** 2 + 0.18 ** 2) ** 0.5),  # LHCb   arXiv:1711.05623
    (0.49, 0.26),                            # CMS    arXiv:2510.21559
)
_jpsi_installed = False


def _jpsi_logL(wc):
    pred = float(flavio.np_prediction(_JPSI_OBS, wc))
    chi2 = sum(((pred - v) / s) ** 2 for v, s in _JPSI_MEAS)
    return -0.5 * chi2


def _install_jpsi_term():
    """Monkeypatch SMEFT19.likelihood_global so the R(J/psi) term is added at every
    call site.  Idempotent.  Runs in the parent before regenerate_dataset.py forks,
    so the workers inherit the patched function."""
    global _jpsi_installed
    if _jpsi_installed:
        return
    _orig = SMEFT19.SMEFTglob.likelihood_global

    def likelihood_global(x, wfun):
        return _orig(x, wfun) + _jpsi_logL(wfun(x))

    SMEFT19.SMEFTglob.likelihood_global = likelihood_global
    SMEFT19.likelihood_global = likelihood_global
    _jpsi_installed = True


def setup_global_likelihood(verbose=True):
    # idempotent: only register the measurements once per kernel (re-reading would
    # raise on the duplicate measurement names)
    if "HFLAV CKM2025 RDRDstar" not in flavio.classes.Measurement.instances:
        flavio.measurements.read_file(str(MEAS_FILE))
    SMEFT19.SMEFTglob.restart_smelli(add_measurements=ADD, remove_measurements=REMOVE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        SMEFT19.SMEFTglob.gl.make_measurement()
    _install_jpsi_term()
    if verbose:
        print(f"global likelihood configured with {MEAS_FILE.name} (+ R(J/psi) term)", flush=True)
    return SMEFT19.SMEFTglob.gl
