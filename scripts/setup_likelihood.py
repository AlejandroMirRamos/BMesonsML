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

Note on R(J/psi): Rtaumu(Bc->J/psilnu) exists in flavio@BctoJpsi but is NOT part of
smelli's default global likelihood. Adding it requires editing smelli's bundled YAMLs,
which makes `import SMEFT19` fragile (the default GlobalLikelihood is built at import).
Given its ~40% experimental error its pull on (C1,C3,bq) is negligible, so it is left
out of the global fit. The measurement is still in measurements_2025.yml for reference.
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


def setup_global_likelihood(verbose=True):
    # idempotent: only register the measurements once per kernel (re-reading would
    # raise on the duplicate measurement names)
    if "HFLAV CKM2025 RDRDstar" not in flavio.classes.Measurement.instances:
        flavio.measurements.read_file(str(MEAS_FILE))
    SMEFT19.SMEFTglob.restart_smelli(add_measurements=ADD, remove_measurements=REMOVE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        SMEFT19.SMEFTglob.gl.make_measurement()
    if verbose:
        print(f"global likelihood configured with {MEAS_FILE.name}", flush=True)
    return SMEFT19.SMEFTglob.gl
