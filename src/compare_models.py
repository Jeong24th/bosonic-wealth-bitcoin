"""
Alternative model comparison for the geometric distribution hypothesis.

Fits four candidate distributions to each denomination-month sample by
maximum-likelihood estimation on raw counts, then evaluates the
Jensen--Shannon divergence (D_JS, log_2) between each fitted model and
the smoothed empirical distribution.  This provides a uniform,
parameter-count-independent measure of distributional agreement.

Candidate models:
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
    """Load raw counts truncated at k_max."""
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


def load_smooth(data_root, date_str, denom):
    """Load smoothed empirical distribution p*(k)."""
    path = data_root / date_str / f"common_smooth_dist_{denom}.csv"
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    k = df["x"].to_numpy(dtype=int)
    p = df["p(x)"].to_numpy(dtype=float)
    return k, p


def djs(p, q):
    """Jensen-Shannon divergence with log_2."""
    mask = (p > 0) | (q > 0)
    p, q = p[mask], q[mask]
    m = 0.5 * (p + q)
    d = 0.0
    if np.any(p > 0):
        d += 0.5 * np.sum(p[p > 0] * np.log2(p[p > 0] / m[p > 0]))
    if np.any(q > 0):
        d += 0.5 * np.sum(q[q > 0] * np.log2(q[q > 0] / m[q > 0]))
    return d


def model_pmf(log_unnorm):
    """Normalise log-unnormalized PMF."""
    lp = log_unnorm - np.max(log_unnorm)
    pmf = np.exp(lp)
    return pmf / pmf.sum()


# ---- MLE fits on raw counts ----

def fit_geometric(n, K):
    k_f = np.arange(1, K + 1, dtype=float)
    def neg_ll(b):
        if b <= 1e-10:
            return 1e30
        log_pk = -b * k_f
        log_Z = np.log(np.sum(np.exp(log_pk - np.max(log_pk)))) + np.max(log_pk)
        return -float(np.sum(n * (log_pk - log_Z)))
    res = minimize_scalar(neg_ll, bounds=(1e-6, 10), method="bounded")
    return res.x


def fit_poisson(n, K):
    k_f = np.arange(1, K + 1, dtype=float)
    def neg_ll(lam):
        if lam <= 1e-10:
            return 1e30
        log_pk = -lam + k_f * np.log(lam) - gammaln(k_f + 1)
        log_Z = np.log(np.sum(np.exp(log_pk - np.max(log_pk)))) + np.max(log_pk)
        return -float(np.sum(n * (log_pk - log_Z)))
    res = minimize_scalar(neg_ll, bounds=(0.01, 500), method="bounded")
    return res.x


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
    for r0 in [0.5, 1.0, 2.0, 5.0]:
        for p0 in [0.3, 0.6, 0.9]:
            try:
                res = minimize(neg_ll, [r0, p0],
                               bounds=[(0.02, 100), (1e-6, 1 - 1e-6)],
                               method="L-BFGS-B")
                if best is None or res.fun < best.fun:
                    best = res
            except Exception:
                continue
    return best.x if best else np.array([1.0, 0.5])


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
    for mu0 in [0, 1, 2]:
        for s0 in [0.5, 1, 3]:
            try:
                res = minimize(neg_ll, [mu0, s0],
                               bounds=[(-10, 10), (0.02, 50)],
                               method="L-BFGS-B")
                if best is None or res.fun < best.fun:
                    best = res
            except Exception:
                continue
    return best.x if best else np.array([0.0, 1.0])


# ---- D_JS evaluation on smoothed data ----

def compute_djs_all(k_smooth, p_smooth, K, beta, lam, r_nb, p_nb, mu, sig):
    """Compute D_JS between smoothed empirical and each model PMF."""
    k_full = np.arange(1, K + 1, dtype=float)

    q_geo = model_pmf(-beta * k_full)
    q_poi = model_pmf(-lam + k_full * np.log(lam) - gammaln(k_full + 1))
    q_nb = model_pmf(gammaln(k_full + r_nb) - gammaln(k_full + 1) - gammaln(r_nb)
                     + k_full * np.log(p_nb) + r_nb * np.log(1 - p_nb))
    q_ln = model_pmf(-np.log(k_full) - (np.log(k_full) - mu)**2 / (2 * sig**2))

    mask = (k_smooth >= 1) & (k_smooth <= K)
    ks = k_smooth[mask]
    ps = p_smooth[mask]
    ps_sum = ps.sum()
    if ps_sum <= 0:
        return None
    ps_norm = ps / ps_sum
    idx = ks.astype(int) - 1

    results = {}
    for name, q_full in [("Geometric", q_geo), ("Poisson", q_poi),
                          ("NegBin", q_nb), ("LogNormal", q_ln)]:
        q_at_k = q_full[idx]
        q_sum = q_at_k.sum()
        if q_sum <= 0:
            results[name] = 1.0
            continue
        results[name] = djs(ps_norm, q_at_k / q_sum)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare geometric vs alternative distributions (D_JS)."
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

            # Step 1: MLE on raw counts
            k_raw, n_raw = load_raw(args.data_dir, date_str, denom, K)
            if k_raw is None or np.sum(n_raw) < 10:
                continue

            beta = fit_geometric(n_raw, K)
            lam = fit_poisson(n_raw, K)
            r_p = fit_negbin(n_raw, K)
            mu_sig = fit_lognormal(n_raw, K)

            # Step 2: D_JS on smoothed data
            k_s, p_s = load_smooth(args.data_dir, date_str, denom)
            if k_s is None or len(k_s) < 3:
                continue

            djs_vals = compute_djs_all(k_s, p_s, K, beta, lam,
                                       r_p[0], r_p[1], mu_sig[0], mu_sig[1])
            if djs_vals is None:
                continue

            winner = min(djs_vals, key=djs_vals.get)
            rows.append({
                "date": date_str, "denom": denom, "K": K,
                "DJS_Geo": djs_vals["Geometric"],
                "DJS_Poi": djs_vals["Poisson"],
                "DJS_NB": djs_vals["NegBin"],
                "DJS_LN": djs_vals["LogNormal"],
                "winner": winner,
            })
            if done % 500 == 0:
                print(f"  {done}/{total}...")

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    n_total = len(df)
    print(f"\nModel Comparison Summary — D_JS (N = {n_total})")
    for model in ["Geometric", "Poisson", "NegBin", "LogNormal"]:
        tag = {"Geometric": "Geo", "Poisson": "Poi",
               "NegBin": "NB", "LogNormal": "LN"}[model]
        col = f"DJS_{tag}"
        count = (df["winner"] == model).sum()
        print(f"  {model:12s} wins: {count:5d} / {n_total}  "
              f"({100 * count / n_total:.1f}%)  "
              f"mean D_JS = {df[col].mean():.4f}")

    print(f"\nMean delta D_JS vs Geometric:")
    for col, label in [("DJS_Poi", "Poisson"), ("DJS_NB", "NegBin"),
                        ("DJS_LN", "LogNormal")]:
        d = (df[col] - df["DJS_Geo"]).mean()
        print(f"  {label:12s}: {d:+.5f}")

    geo_close = ((df["DJS_Geo"] - df[["DJS_Geo", "DJS_NB", "DJS_LN", "DJS_Poi"]]
                  .min(axis=1)) < 0.01).sum()
    print(f"\nGeometric within 0.01 of best: {geo_close}/{n_total} "
          f"({100 * geo_close / n_total:.1f}%)")

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
