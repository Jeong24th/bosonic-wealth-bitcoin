"""
Aggregate beta_i across all denominations and months, then quantify
(a) denomination dependence of <beta_i> across 8 orders of magnitude in i, and
(b) time-stability of beta_i per denomination over 72 monthly snapshots.

Produces the per-denomination summary table that supports Section 2.4
"Universality of the inverse temperature across denominations and time"
in the paper.

Expected input: a directory containing per-date summary.csv files with
columns (utxo, b_est, ...).

Usage
-----
    python beta_i_universality.py \\
        --fit-dir <path to fitting_data_final> \\
        --output  beta_per_denomination.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-dir", required=True, type=Path,
                    help="directory containing per-date summary.csv files")
    ap.add_argument("--output", default="beta_per_denomination.csv", type=Path,
                    help="per-denomination summary CSV")
    args = ap.parse_args()

    dates = sorted([p.name for p in args.fit_dir.iterdir()
                    if p.is_dir() and len(p.name) == 10])
    print(f"Dates available: {len(dates)}")

    frames = []
    for d in dates:
        pth = args.fit_dir / d / "summary.csv"
        if not pth.exists():
            continue
        df = pd.read_csv(pth)
        df["date"] = d
        frames.append(df)

    big = pd.concat(frames, ignore_index=True)
    print(f"Total samples: {len(big)}")

    # Per-denomination aggregates across the 72 monthly snapshots
    g = (big.groupby("utxo")["b_est"]
            .agg(["mean", "std", "count"])
            .reset_index())
    g.columns = ["i", "beta_mean", "beta_std", "N"]
    g = g.sort_values("i").reset_index(drop=True)
    g["CV"] = g["beta_std"] / g["beta_mean"]

    # Denomination range and global mean
    print(f"\nDenomination range: i in [{g['i'].min()}, {g['i'].max()}]")
    print(f"<beta_i> range: [{g['beta_mean'].min():.4f}, "
          f"{g['beta_mean'].max():.4f}]")
    print(f"Global mean of <beta_i>: {g['beta_mean'].mean():.4f}")

    # Denomination-dependence diagnostics
    log_i = np.log10(g["i"].astype(float))
    log_b = np.log10(g["beta_mean"])
    r_log_log = float(np.corrcoef(log_i, log_b)[0, 1])
    r_lin_log = float(np.corrcoef(log_i, g["beta_mean"])[0, 1])
    a_ll, b_ll = np.polyfit(log_i, log_b, 1)
    print(f"\nPearson r(log i, log <beta>) = {r_log_log:.4f}")
    print(f"Pearson r(log i,     <beta>) = {r_lin_log:.4f}")
    print(f"power-law fit:  <beta_i> proportional to i^{a_ll:.4f}")

    # Time-stability diagnostics
    print(f"\nCoefficient of variation (std/mean) across denominations:")
    print(f"  median CV = {g['CV'].median():.4f}")
    print(f"  mean   CV = {g['CV'].mean():.4f}")
    print(f"  fraction CV < 0.10: {(g['CV'] < 0.10).mean() * 100:.1f}%")
    print(f"  fraction CV < 0.05: {(g['CV'] < 0.05).mean() * 100:.1f}%")

    g.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}  ({len(g)} rows)")


if __name__ == "__main__":
    main()
