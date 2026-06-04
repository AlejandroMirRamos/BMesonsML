# Scenario III with up-to-date (2025/2026) measurements

This documents how to regenerate the BMesonsML training data and best-fit using the
**paper-faithful software stack** plus the latest experimental inputs, instead of the
stock `flavio 2.7.0 / smelli 2.4.2` originally in `.venv` (which lacks R(J/psi), the
Belle II B+->K+ nu nu result, and the updated R(D*), and whose cached best-fit
`data/rotBIII.yaml` is a C1=C3 local minimum — not Scenario III).

## 1. Environment (`.venv-paper`, Python 3.9)

Built with `uv` to match `SMEFT19/requirements.txt`:

- `flavio @ git+https://github.com/Jorge-Alda/flavio@BctoJpsi`   (adds `Rtaumu(Bc->J/psilnu)`)
- `smelli @ git+https://github.com/Jorge-Alda/smelli@custom_measurements`  (has `add_measurements`)
- `wilson==2.0`, `rundec==0.6`, `setuptools<81`, `iminuit>=2.11`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`

(Already created at `BMesonsML/.venv-paper`. To rebuild from scratch see `memory/bmesonml_paper_env.md`.)

## 2. Updated measurements — `data/measurements_2025.yml`

| Observable | Value | Source |
|---|---|---|
| R(D), R(D*) | 0.358 ± 0.024 , 0.281 ± 0.011 (corr −0.374) | HFLAV **CKM 2025** world average |
| BR(B+→K+νν̄) | (2.3 ± 0.7)×10⁻⁵ | Belle II 362 fb⁻¹ |
| R(J/ψ) | 0.52 ± 0.20 | LHCb/CMS WA (kept for reference; see note) |

`scripts/setup_likelihood.py` loads these into flavio and swaps them into smelli's
global likelihood (`likelihood_rd_rds.yaml`, `likelihood_bqnunu.yaml`) via
`restart_smelli`, dropping the superseded τ R(D)/R(D*) measurements.

**R(J/ψ) note:** it is *not* in smelli's default global likelihood, and adding it cleanly
requires editing smelli's bundled YAMLs (which makes `import SMEFT19` fragile). With a ~40%
error its pull on (C1,C3,βq) is negligible, so it is **excluded from the global fit**.
R(K)/R(K*): the newest LHCb results are SM-like and in Scenario III R(K)=R(K*)=1 exactly, so
they do not move the fit — smelli's defaults are kept.

## 3. How to run — just the notebook

Open `notebooks/xgboost_optimized.ipynb`, select the **`Python (bmesonml-paper)`**
kernel, and Run All. The first cell of Section 1 builds everything when the cached
files are missing (or when `FORCE_REBUILD=True`):

  best-fit + covariance -> `data/rotBIII_2025.yaml`   (~10-15 min)
  training dataset      -> `data/combined_rotBIII_2025.dat`   (~30-60 min)

then the XGBoost / SHAP / Fig 7 / correlation cells run on GPU in the same kernel.
Subsequent runs load the cached files (fast); set `FORCE_REBUILD=True` to redo.

The heavy logic is in `scripts/` and can also be run standalone:

```bash
cd BMesonsML
.venv-paper/bin/python scripts/refit_bestfit.py        # -> data/rotBIII_2025.yaml
.venv-paper/bin/python scripts/regenerate_dataset.py   # -> data/combined_rotBIII_2025.dat
```

Tune cost/accuracy with env vars, e.g. `N_SOBOL=2048 N_LOCAL=3000 GRID=30 N_WORKERS=16 .venv-paper/bin/python scripts/regenerate_dataset.py`.

## 4. Sampling method (revised)

The original 3000-LHS + three rigid 50×50 best-fit grids + uniform ±2σ box caused the
SHAP "columns" artifact and under-resolved the narrow C1 ridge (patched later with a
hand-tuned dense grid). The new scheme (`scripts/regenerate_dataset.py`):

1. **Sobol** global coverage of the cube — `N_SOBOL=4096`.
2. **Covariance-based** Gaussian cloud around the best-fit (two scales s=1.0, 2.2) —
   `N_LOCAL=5000`. Follows the real likelihood ellipsoid, so it densifies the narrow C1
   direction automatically and removes the rigid-grid over-representation.
3. Three 2D **plane grids** (`GRID=40`, 3×1600) kept only for the Fig 8a real-vs-emulator
   likelihood maps.

Default total ≈ **13.9k** points. The script prints the row ranges of each block.

## 5. Notebook wiring (already applied)

`notebooks/xgboost_optimized.ipynb` is already wired for this workflow:
- kernel set to `bmesonml-paper`; paths point to the `_2025` files; `GRID_SIZE=40`.
- the Section-1 cell calls `setup_global_likelihood` + `refit_bestfit` + `regenerate`
  from `scripts/` (build-once-or-load-cache).
- the old `local_dense_grid` cell is now a no-op.
- the Fig 8a `PLANES` slice indices are derived from the dataset
  (`_base = len(df_raw) - 3*GRID_SIZE**2`), so they stay correct if you change the
  point budget.

Everything (including GPU XGBoost training) runs in the single `bmesonml-paper` kernel.
