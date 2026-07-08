"""
generate_dataset.py  –  Latin Hypercube Sampling + SMEFT19 likelihood (rotBIII)

Legacy standalone generator: ~10 000 points in the (C1, C3, bq) space with LHS,
evaluating the real SMEFT19 likelihood. Results are written incrementally so the
run can be interrupted and resumed. Superseded by regenerate_dataset.py.

Parameter-space ranges (same as fit3.py):
    C1  ∈ [-0.30,  0.00]
    C3  ∈ [-0.30,  0.00]
    bq  ∈ [  0.00, 3.20]

Usage (from any directory):
    python scripts/generate_dataset.py                        # 10 000 pts → data/lhs_rotBIII.dat
    python scripts/generate_dataset.py -n 5000 -o custom.dat  # custom output
    python scripts/generate_dataset.py --resume               # resume an interrupted run
"""

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
from scipy.stats.qmc import LatinHypercube, scale

# ── Physical domain ranges ────────────────────────────────────────────────────
BOUNDS_LO = np.array([-0.30, -0.30, 0.00])
BOUNDS_HI = np.array([ 0.00,  0.00, 3.20])
COLS = ['C1', 'C3', 'bq', 'likelihood']


def init_smeft():
    """Initialize the SMEFT19 global measurement (call once at startup)."""
    import SMEFT19
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        SMEFT19.SMEFTglob.gl.make_measurement()


def compute_likelihood(C1: float, C3: float, bq: float) -> float:
    """Evaluate the log-likelihood at a point. Returns -inf if the computation fails."""
    import SMEFT19
    from SMEFT19 import likelihood_global
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            lg = likelihood_global([C1, C3, 0.0, 0.0, 0.0, bq],
                                   SMEFT19.scenarios.massrotation)
            return max(float(lg), -200.0)
        except Exception:
            return float('-inf')


def count_existing_lines(path: str) -> int:
    """Count how many lines the file already has (to resume)."""
    p = Path(path)
    if not p.exists():
        return 0
    with open(p) as f:
        return sum(1 for _ in f)


def generate(n_points: int, output: str, seed: int = 42):
    print(f'Starting generation of {n_points} LHS points (seed={seed})')
    print(f'Output: {output}\n')

    init_smeft()

    sampler = LatinHypercube(d=3, seed=seed)
    unit_samples = sampler.random(n=n_points)
    samples = scale(unit_samples, BOUNDS_LO, BOUNDS_HI)  # shape (n_points, 3)

    start = count_existing_lines(output)
    if start > 0:
        print(f'Resuming from point {start}/{n_points}\n')

    with open(output, 'at', newline='') as f:
        writer = csv.writer(f)
        for i in range(start, n_points):
            C1, C3, bq = samples[i]
            lg = compute_likelihood(C1, C3, bq)
            writer.writerow([C1, C3, bq, lg])
            f.flush()
            if (i + 1) % 100 == 0:
                pct = 100 * (i + 1) / n_points
                print(f'  [{i+1:6d}/{n_points}]  {pct:5.1f}%  last lg={lg:.3f}',
                      flush=True)

    print(f'\nDone. {n_points} points written to "{output}".')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate an LHS dataset with the SMEFT19 likelihood (rotBIII).'
    )
    default_out = str(Path(__file__).parent.parent / 'data' / 'lhs_rotBIII.dat')
    parser.add_argument('-n', '--n-points', type=int, default=10_000,
                        help='Number of LHS points (default: 10 000)')
    parser.add_argument('-o', '--output',   default=default_out,
                        help=f'Output CSV file (default: {default_out})')
    parser.add_argument('-s', '--seed',     type=int, default=42,
                        help='Random seed for LHS (default: 42)')
    args = parser.parse_args()

    generate(args.n_points, args.output, args.seed)
