"""
Alternative model comparison for the geometric distribution hypothesis.

Fits four candidate distributions to each denomination-month sample and
compares them using the Akaike information criterion (AIC):
  1. Truncated geometric (1 parameter: beta)
  2. Truncated Poisson (1 parameter: lambda)
  3. Truncated negative binomial (2 parameters: r, p)
  4. Discretised log-normal (2 parameters: mu, sigma)

The geometric is the theory-predicted one-parameter model from the
indistinguishability postulate; the others serve as null models.

Usage
-----
    python compare_models.py --data-dir <UTXO_distribution>
                             --output <model_comparison.csv>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize
from scipy.special import gammaln


def load_kmax_map(data_root):
    df = pd.read_csv(data_root / "common_kmax_by_utxo.csv")
    return {int(row["utxo"]): int(row["k_max_min"]) for _, row in df.iterrows()}


def load_raw(data_root, date_str, denom, k_max):
    path = data_root / date_str / f"raw_dist_{denom}.csv"
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    k = df["x"].to_numpy(dtype=int)
    n = df["y"].to_numpy(dtype=float)
    mask = (k >= 1) & (k <= k_max)
    k, n = k[mask], n[mask]
    if len(k) == 0:
        return None, None
    k_full = np.arange(1, k_max + 1)
    n_full = np.zeros(k_max)
    for ki, ni in zip(k, n):
        n_full[ki - 1] = ni
    return k_full, n_full


def fit_geometric(n, K):
    k_f = np.arange(1, K + 1, dtype=float)
    def neg_ll(b):
        if b <= 1e-10:
            return 1e30
        log_pk = -b * k_f
        log_Z = np.log(np.sum(np.exp(log_pk - np.max(log_pk)))) + np.max(log_pk)
        return -float(np.sum(n * (log_pk - log_Z)))
    res = minimize_scalar(neg_ll, bounds=(1e-6, 10), method="bounded")
    return {"name": "Geometric", "ll": -res.fun, "p": 1}


def fit_poisson(n, K):
    k_f = np.arange(1, K + 1, dtype=float)
    def neg_ll(lam):
        if lam <= 1e-10:
            return 1e30
        log_pk = -lam + k_f * np.log(lam) - gammaln(k_f + 1)
        log_Z = np.log(np.sum(np.exp(log_pk - np.max(log_pk)))) + np.max(log_pk)
        return -float(np.sum(n * (log_pk - log_Z)))
    res = minimize_scalar(neg_ll, bounds=(0.01, 500), method="bounded")
    return {"name": "Poisson", "ll": -res.fun, "p": 1}


def fit_negbin(n, K):
    k_f = np.arange(1, K + 1, dtype=float)
    def neg_ll(params):
        r, p = params
        if r <= 0.01 or p <= 1e-10 or p >= 1 - 1e-10:
            return 1e30
        log_pk = (gammaln(k_f + r) - gammaln(k_f + 1) - gammaln(r)
                  + k_f * np.log(p) + r * np.log(1 - p))
        log_Z = np.log(np.sum(np.exp(log_pk - np.max(log_pk)))) + np.max(log_pk)
        return -float(np.sum(n * (log_pk - log_Z)))
    best = None
    for r0 in [0.3, 0.5, 1.0, 2.0, 5.0, 10.0]:
        for p0 in [0.2, 0.4, 0.6, 0.8]:
            try:
                res = minimize(neg_ll, [r0, p0],
                               bounds=[(0.02, 100), (1e-6, 1 - 1e-6)],
                               method="L-BFGS-B")
                if best is None or res.fun < best.fun:
                    best = res
            except Exception:
                continue
    if best is None:
        return {"name": "NegBin", "ll": -np.inf, "p": 2}
    return {"name": "NegBin", "ll": -best.fun, "p": 2}


def fit_lognormal(n, K):
    k_f = np.arange(1, K + 1, dtype=float)
    def neg_ll(params):
        mu, sigma = params
        if sigma <= 0.01:
            return 1e30
        log_pk = -np.log(k_f) - (np.log(k_f) - mu)**2 / (2 * sigma**2)
        log_Z = np.log(np.sum(np.exp(log_pk - np.max(log_pk)))) + np.max(log_pk)
        return -float(np.sum(n * (log_pk - log_Z)))
    best = None
    for mu0 in [-1, 0, 0.5, 1, 2]:
        for s0 in [0.3, 0.5, 1, 2, 5]:
            try:
                res = minimize(neg_ll, [mu0, s0],
                               bounds=[(-10, 10), (0.02, 50)],
                               method="L-BFGS-B")
                if best is None or res.fun < best.fun:
                    best = res
            except Exception:
                continue
    if best is None:
        return {"name": "LogNormal", "ll": -np.inf, "p": 2}
    return {"name": "LogNormal", "ll": -best.fun, "p": 2}


def main():
    parser = argparse.ArgumentParser(
        description="Compare geometric vs alternative distributions (AIC)."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("model_comparison.csv"))
    args = parser.parse_args()

    kmax_map = load_kmax_map(args.data_dir)
    denoms = sorted(kmax_map.keys())
    dates = sorted([p.name for p in args.data_dir.iterdir()
                    if p.is_dir() and len(p.name) == 10])
    print(f"Dates: {len(dates)}, Denominations: {len(denoms)}")

    rows = []
    total = len(dates) * len(denoms)
    done = 0

    for date_str in dates:
        for denom in denoms:
            done += 1
            K = kmax_map[denom]
            k, n = load_raw(args.data_dir, date_str, denom, K)
            if k is None or np.sum(n) < 10:
                continue

            geo = fit_geometric(n, K)
            poi = fit_poisson(n, K)
            nb = fit_negbin(n, K)
            ln = fit_lognormal(n, K)

            aics = {r["name"]: 2 * r["p"] - 2 * r["ll"]
                    for r in [geo, poi, nb, ln]}
            winner = min(aics, key=aics.get)

            rows.append({
                "date": date_str, "denom": denom, "N": np.sum(n), "K": K,
                "AIC_Geo": aics["Geometric"], "AIC_Poi": aics["Poisson"],
                "AIC_NB": aics["NegBin"], "AIC_LN": aics["LogNormal"],
                "winner": winner,
            })
            if done % 500 == 0:
                print(f"  {done}/{total}...")

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    n_total = len(df)
    print(f"\nModel Comparison Summary (N = {n_total})")
    for model in ["Geometric", "Poisson", "NegBin", "LogNormal"]:
        count = (df["winner"] == model).sum()
        print(f"  {model:12s} wins: {count:5d} / {n_total}  "
              f"({100 * count / n_total:.1f}%)")

    print(f"\nMean delta AIC vs Geometric:")
    for col, label in [("AIC_Poi", "Poisson"), ("AIC_NB", "NegBin"),
                        ("AIC_LN", "LogNormal")]:
        d = (df[col] - df["AIC_Geo"]).mean()
        print(f"  {label:12s}: {d:+.0f}")

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
