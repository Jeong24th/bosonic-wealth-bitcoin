"""
beta-m self-consistency test for the truncated geometric law.

For the one-parameter truncated geometric P_beta(k) on k=1..k_max, the
mean m(beta, k_max) is an analytic, strictly monotonic function of beta.
Given the Bayesian posterior estimate beta_i for each denomination-month
sample, the theoretical mean must coincide with the empirical mean;
equivalently, inverting the relation at the empirical mean yields a
second estimate beta_i^{(m)} that must match beta_i.

This is a self-consistency test that two-parameter phenomenological
alternatives (log-normal, negative binomial) cannot pass, because a
single empirical mean does not fix both of their parameters.

Expected input: a directory containing per-date summary.csv files with
columns (utxo, b_est, m, m_theory, beta_from_m, beta_diff).

Usage
-----
    python beta_m_consistency.py \\
        --fit-dir <path to fitting_data_final> \\
        --output  beta_m_all.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-dir", required=True, type=Path,
                    help="directory containing per-date summary.csv files")
    ap.add_argument("--output", default="beta_m_all.csv", type=Path,
                    help="aggregated output CSV")
    args = ap.parse_args()

    dates = sorted([p.name for p in args.fit_dir.iterdir()
                    if p.is_dir() and len(p.name) == 10])
    print(f"Dates available: {len(dates)}")

    frames = []
    for date_str in dates:
        pth = args.fit_dir / date_str / "summary.csv"
        if not pth.exists():
            continue
        df = pd.read_csv(pth)
        df["date"] = date_str
        frames.append(df)

    big = pd.concat(frames, ignore_index=True)
    print(f"Total samples: {len(big)}")

    big["rel_diff"] = big["beta_diff"].abs() / big["b_est"].abs()

    print("\n" + "=" * 60)
    print("  beta-m relation verification")
    print("=" * 60)
    print(f"  N                     = {len(big)}")
    print(f"  Mean |beta_diff|      = {big['beta_diff'].abs().mean():.2e}")
    print(f"  Max  |beta_diff|      = {big['beta_diff'].abs().max():.2e}")
    print(f"  Mean relative diff    = {big['rel_diff'].mean():.2e}")
    print(f"  Max  relative diff    = {big['rel_diff'].max():.2e}")

    r = np.corrcoef(big["b_est"], big["beta_from_m"])[0, 1]
    print(f"  Pearson r(b_est, beta_from_m) = {r:.10f}")

    for thr in [1e-2, 1e-3, 1e-4, 1e-5]:
        frac = (big["rel_diff"] < thr).mean() * 100
        print(f"  Fraction |rel_diff| < {thr:.0e}: {frac:.2f}%")

    big.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
