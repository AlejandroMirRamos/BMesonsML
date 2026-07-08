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

This exact stack is pinned in `requirements.txt`, so `pip install -r requirements.txt`
into a fresh **Python 3.9** venv reproduces it.

(Already created at `BMesonsML/.venv-paper`. To rebuild from scratch see `memory/bmesonml_paper_env.md`.)

## 2. Updated measurements — `data/measurements_2025.yml`

| Observable | Value | Source |
|---|---|---|
| R(D), R(D*) | 0.358 ± 0.024 , 0.281 ± 0.011 (corr −0.374) | HFLAV **CKM 2025** world average |
| BR(B+→K+νν̄) | (2.3 ± 0.7)×10⁻⁵ | Belle II 362 fb⁻¹ |
| R(J/ψ) | 0.71 ± 0.17 ± 0.18 (LHCb), 0.49 ± 0.26 (CMS) | included as a standalone term (see note) |

`scripts/setup_likelihood.py` loads these into flavio and swaps them into smelli's
global likelihood (`likelihood_rd_rds.yaml`, `likelihood_bqnunu.yaml`) via
`restart_smelli`, dropping the superseded τ R(D)/R(D*) measurements.

**R(J/ψ) note:** the observable `Rtaumu(Bc->J/psilnu)` is *not* in any smelli sector, so it
cannot be injected via `add_measurements`. It is included instead as a **standalone Gaussian
term** added to the global log-likelihood in `scripts/setup_likelihood.py`, using the two
reference measurements directly (LHCb and CMS, as independent constraints). The term is wired
in by monkeypatching `SMEFT19.likelihood_global` inside `setup_global_likelihood()`, so refit,
dataset regeneration and the notebook — and every fork worker — all evaluate it. Its ~40%
experimental error makes the pull on (C1,C3,βq) small, so the best-fit and contours shift only
slightly (Δχ²_SM ≈ 48 → 49).
R(K)/R(K*): the newest LHCb results are SM-like and in Scenario III R(K)=R(K*)=1 exactly, so
they do not move the fit — smelli's defaults are kept.

## 3. How to run — just the notebook

Open `notebooks/xgboost_optimized.ipynb`, select the **`Python (bmesonml-paper)`**
kernel, and Run All. The first cell of Section 1 builds everything when the cached
files are missing (or when `FORCE_REBUILD=True`):

  best-fit + covariance -> `data/rotBIII_2025.yaml`   (~10-15 min)
  training dataset      -> `data/combined_rotBIII_2025.dat`   (~30-60 min)

With `FORCE_DENSIFY=True` (the default), the next Section-1 cell then runs the
high-βq active-learning densification (`scripts/densify_highbq.py`): ~12k extra
SMEFT19 points streamed to `data/extra_highbq_2025.dat` (slow, but resumable if
interrupted) and spliced into the dataset. This removes the emulator
over-extrapolation spike in Fig 7b; see §4 below and the notebook §1 markdown.

The XGBoost / SHAP / Fig 7 / correlation cells then run in the same kernel
(XGBoost trains on CPU: the CUDA build available here trains poorly, R² ~ 0.5).
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

Default base total ≈ **13.9k** points. The script prints the row ranges of each block.

4. **High-βq densification** (`scripts/densify_highbq.py`): the base design leaves the
   weakly-constrained high-βq slab sparse, and there the emulator over-extrapolates the
   log-likelihood (a false maximum ≈ 25–26 vs a true ≈ 20–22), producing a spike at 0 in
   the Fig 7b Δχ² histogram and a biased high-βq posterior. The script adds ~12k exact
   SMEFT19 labels in that slab (resumable stream to `data/extra_highbq_2025.dat`) and
   splices them in **keeping the plane grids last**, so the Fig 8a slicing is unchanged.
   Final total ≈ **26k** points.

## 5. Notebook wiring (already applied)

`notebooks/xgboost_optimized.ipynb` is already wired for this workflow:
- kernel set to `bmesonml-paper`; paths point to the `_2025` files; `GRID_SIZE=40`.
- the Section-1 cell calls `setup_global_likelihood` + `refit_bestfit` + `regenerate`
  from `scripts/` (build-once-or-load-cache).
- the old `local_dense_grid` cell is now a no-op.
- the Fig 8a `PLANES` slice indices are derived from the dataset
  (`_base = len(df_raw) - 3*GRID_SIZE**2`), so they stay correct if you change the
  point budget.

Everything runs in the single `bmesonml-paper` kernel (XGBoost on CPU).
