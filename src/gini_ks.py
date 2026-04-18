"""
Gini coefficient and Kolmogorov-Smirnov consistency checks for the
truncated geometric hypothesis.

For each denomination-month sample, given the Bayesian-inferred beta_i,
we compute:

  * G_emp : empirical Gini coefficient of the smoothed P_*(k)
  * G_th  : theoretical Gini coefficient of the truncated geometric
            P_beta(k) with the fitted beta_i
  * KS    : discrete Kolmogorov-Smirnov statistic
            max_k |F_*(k) - F_beta(k)|

Outputs a CSV with per-sample values and prints summary statistics.

Usage
-----
    python gini_ks.py \\
        --data-dir <UTXO_distribution> \\
        --fit-dir  <fitting_data_final> \\
        --output   gini_ks.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def geometric_pmf(beta, K):
    k = np.arange(1, K + 1, dtype=float)
    logp = -beta * k
    logp -= logp.max()
    p = np.exp(logp)
    return p / p.sum()


def gini_from_pmf(p, k):
    m = np.sum(k * p)
    if m <= 0:
        return np.nan
    diff = np.abs(k[:, None] - k[None, :])
    mad = np.sum(p[:, None] * p[None, :] * diff)
    return mad / (2 * m)


def ks_discrete(p_emp, p_th):
    F_emp = np.cumsum(p_emp)
    F_th = np.cumsum(p_th)
    return float(np.max(np.abs(F_emp - F_th)))


def load_smooth(data_root, date_str, denom):
    path = data_root / date_str / f"common_smooth_dist_{denom}.csv"
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    k = df["x"].to_numpy(dtype=int)
    p = df["p(x)"].to_numpy(dtype=float)
    mask = p > 0
    return k[mask], p[mask]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="root of UTXO_distribution with per-date subfolders")
    ap.add_argument("--fit-dir", required=True, type=Path,
                    help="root of fitting_data_final with per-date summary.csv")
    ap.add_argument("--output", default="gini_ks.csv", type=Path)
    args = ap.parse_args()

    dates = sorted([p.name for p in args.fit_dir.iterdir()
                    if p.is_dir() and len(p.name) == 10])
    frames = []
    for d in dates:
        pth = args.fit_dir / d / "summary.csv"
        if pth.exists():
            df = pd.read_csv(pth)
            df["date"] = d
            frames.append(df)
    fits = pd.concat(frames, ignore_index=True)
    print(f"Samples: {len(fits)}")

    rows = []
    done = 0
    for _, row in fits.iterrows():
        done += 1
        denom = int(row["utxo"])
        date_str = row["date"]
        beta = float(row["b_est"])
        K = int(row["k_max"])

        k_s, p_s = load_smooth(args.data_dir, date_str, denom)
        if k_s is None or len(k_s) < 3:
            continue

        k_full = np.arange(1, K + 1, dtype=float)
        p_full = np.zeros(K, dtype=float)
        mask = (k_s >= 1) & (k_s <= K)
        p_full[k_s[mask] - 1] = p_s[mask]
        if p_full.sum() <= 0:
            continue
        p_full /= p_full.sum()

        p_th = geometric_pmf(beta, K)

        g_emp = gini_from_pmf(p_full, k_full)
        g_th = gini_from_pmf(p_th, k_full)
        ks = ks_discrete(p_full, p_th)

        rows.append({
            'date': date_str, 'denom': denom, 'K': K, 'beta': beta,
            'gini_emp': g_emp, 'gini_th': g_th, 'ks_stat': ks,
        })

        if done % 500 == 0:
            print(f"  {done}/{len(fits)}...", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    n = len(df)
    print(f"\n{'='*60}")
    print(f"  Gini and K-S statistics  (N = {n})")
    print(f"{'='*60}")
    r = np.corrcoef(df["gini_th"], df["gini_emp"])[0, 1]
    print(f"  Pearson r(Gini_th, Gini_emp) = {r:.6f}")
    for thr in [0.05, 0.10, 0.15]:
        frac = (df['ks_stat'] < thr).mean() * 100
        print(f"  Fraction KS < {thr}: {frac:.2f}%")
    print(f"  KS mean = {df['ks_stat'].mean():.4f}, median = {df['ks_stat'].median():.4f}")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
