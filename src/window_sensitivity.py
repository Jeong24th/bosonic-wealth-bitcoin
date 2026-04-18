"""
Smoothing window sensitivity for the geometric goodness-of-fit.

For window widths w in {1, 5, 9, 15, 21} (w=1 means no smoothing),
apply a centered moving average to the raw counts, redetermine the
per-denomination common k_max for that window, fit the truncated
geometric by MLE, and compute the Jensen-Shannon divergence against
the empirical distribution.

Reports the mean D_JS and the fraction of samples with D_JS below
selected thresholds as a function of w.

Usage
-----
    python window_sensitivity.py \\
        --data-dir <UTXO_distribution> \\
        --output window_sensitivity.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def load_raw_full(data_root, date_str, denom, K_outer=500):
    path = data_root / date_str / f"raw_dist_{denom}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    k = df["x"].to_numpy(dtype=int)
    n = df["y"].to_numpy(dtype=float)
    arr = np.zeros(K_outer, dtype=float)
    mask = (k >= 1) & (k <= K_outer)
    for ki, ni in zip(k[mask], n[mask]):
        arr[ki - 1] = ni
    return arr


def moving_average(arr, w):
    if w <= 1:
        return arr.copy()
    s = pd.Series(arr)
    return s.rolling(w, center=True, min_periods=1).mean().to_numpy()


def djs(p, q):
    mask = (p > 0) | (q > 0)
    p, q = p[mask], q[mask]
    m = 0.5 * (p + q)
    d = 0.0
    if np.any(p > 0):
        d += 0.5 * np.sum(p[p > 0] * np.log2(p[p > 0] / m[p > 0]))
    if np.any(q > 0):
        d += 0.5 * np.sum(q[q > 0] * np.log2(q[q > 0] / m[q > 0]))
    return d


def fit_geom_mle(n_K, K):
    k = np.arange(1, K + 1, dtype=float)

    def neg_ll(b):
        if b <= 1e-10:
            return 1e30
        log_pk = -b * k
        log_Z = np.log(np.sum(np.exp(log_pk - np.max(log_pk)))) + np.max(log_pk)
        return -float(np.sum(n_K * (log_pk - log_Z)))

    res = minimize_scalar(neg_ll, bounds=(1e-6, 10), method='bounded')
    return res.x


def geom_pmf(beta, K):
    k = np.arange(1, K + 1, dtype=float)
    logp = -beta * k
    logp -= logp.max()
    p = np.exp(logp)
    return p / p.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--output", default="window_sensitivity.csv", type=Path)
    ap.add_argument("--windows", default="1,5,9,15,21", type=str,
                    help="comma-separated list of window widths")
    args = ap.parse_args()

    kmax_df = pd.read_csv(args.data_dir / "common_kmax_by_utxo.csv")
    denoms = sorted(int(x) for x in kmax_df["utxo"])
    dates = sorted([p.name for p in args.data_dir.iterdir()
                    if p.is_dir() and len(p.name) == 10])

    print("Loading raw data...")
    raw_by_date = {}
    for date_str in dates:
        for denom in denoms:
            arr = load_raw_full(args.data_dir, date_str, denom)
            if arr is None:
                continue
            if arr.sum() >= 10:
                raw_by_date[(date_str, denom)] = arr

    print(f"Loaded {len(raw_by_date)} (date,denom) pairs")

    results = []
    for w in map(int, args.windows.split(",")):
        # determine per-window common k_max per denomination
        per_denom = {}
        for (date_str, denom), arr in raw_by_date.items():
            sm = moving_average(arr, w)
            k_nonzero = np.where(sm > 0)[0]
            if len(k_nonzero) == 0:
                continue
            per_denom.setdefault(denom, []).append(int(k_nonzero.max() + 1))
        kmax_for_w = {d: min(vals) for d, vals in per_denom.items()}

        djs_list = []
        for (date_str, denom), arr in raw_by_date.items():
            K = kmax_for_w.get(denom)
            if K is None or K < 3:
                continue
            sm = moving_average(arr, w)[:K]
            if sm.sum() <= 0:
                continue
            p_star = sm / sm.sum()
            beta = fit_geom_mle(sm, K)
            q = geom_pmf(beta, K)
            d = djs(p_star, q)
            djs_list.append(d)

        djs_arr = np.array(djs_list)
        n = len(djs_arr)
        # mean with inf filtered
        finite = djs_arr[np.isfinite(djs_arr)]
        mean_d = finite.mean() if len(finite) > 0 else np.nan
        frac_008 = (djs_arr < 0.08).mean() * 100 if n else 0
        frac_01 = (djs_arr < 0.10).mean() * 100 if n else 0
        print(f"w={w:2d}:  N={n}  mean D_JS={mean_d:.4f}  "
              f"<0.08 {frac_008:5.2f}%  <0.10 {frac_01:5.2f}%")

        results.append({
            'w': w, 'N': n,
            'mean_DJS': mean_d,
            'frac_lt_008': frac_008,
            'frac_lt_010': frac_01,
        })

    pd.DataFrame(results).to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
