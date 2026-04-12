"""
Compute Jensen-Shannon divergence between empirical and fitted distributions.

For each UTXO denomination and monthly snapshot, this script:
  1. Loads the empirical distribution P*(k) and fitted beta from summary.csv.
  2. Constructs the theoretical P_beta(k) on a common k grid.
  3. Computes D_JS(P*, P_beta) using base-2 logarithms (D_JS in [0, 1]).
  4. Outputs a CSV with all JSD values, summary statistics, and a binned
     distribution table (Table 1 in the paper).

Note: D_JS uses log_2 for information-theoretic interpretation, while
the KL divergence in bayesian_fitting.py uses ln for direct likelihood
connection. The two are related by D_KL^{log2} = D_KL^{ln} / ln(2).

Usage
-----
    python compute_jsd.py --data-dir <UTXO_distribution>
                          --fit-dir <fitting_results>
                          --output <jsd_results.csv>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def to_utxo_key(v) -> str:
    try:
        return str(int(float(v)))
    except (ValueError, TypeError):
        return str(v)


def load_x_px(pstar_root: Path, date_str: str, utxo_str: str):
    csv_path = pstar_root / date_str / f"common_smooth_dist_{to_utxo_key(utxo_str)}.csv"
    df = pd.read_csv(csv_path)
    x = df["x"].to_numpy(dtype=float)
    p = df["p(x)"].to_numpy(dtype=float)
    p = np.clip(p, 0, None)
    s = p.sum()
    if s > 0:
        p = p / s
    return x, p


def load_kmax_map(kmax_csv: Path) -> dict:
    df = pd.read_csv(kmax_csv)
    df["utxo_key"] = df["utxo"].apply(to_utxo_key)
    return {row["utxo_key"]: int(row["k_max_min"]) for _, row in df.iterrows()}


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-15) -> float:
    """Jensen-Shannon divergence with base-2 logarithms."""
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    m = 0.5 * (p + q)
    return float(
        0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m))
    )


def available_dates(pstar_root: Path, beta_root: Path):
    def list_dates(root):
        return {p.name for p in root.iterdir() if p.is_dir() and len(p.name) == 10}
    return sorted(list_dates(pstar_root) & list_dates(beta_root))


def main():
    parser = argparse.ArgumentParser(
        description="Compute Jensen-Shannon divergence for all denominations."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("jsd_results.csv"))
    args = parser.parse_args()

    pstar_root = args.data_dir
    beta_root = args.fit_dir
    kmax_csv = pstar_root / "common_kmax_by_utxo.csv"

    kmax_map = load_kmax_map(kmax_csv)
    all_dates = available_dates(pstar_root, beta_root)

    print(f"Dates: {len(all_dates)}, Denominations: {len(kmax_map)}")

    rows = []
    for utxo_key, k_max_min in kmax_map.items():
        for date_str in all_dates:
            try:
                x, p_raw = load_x_px(pstar_root, date_str, utxo_key)
                summary = pd.read_csv(beta_root / date_str / "summary.csv")
                summary["utxo_key"] = summary["utxo"].apply(to_utxo_key)
                row = summary.loc[summary["utxo_key"] == utxo_key]
                if row.empty:
                    continue
                beta = float(row.iloc[0]["b_est"])

                kmax_avail = int(np.max(x))
                k_max = min(k_max_min, kmax_avail)
                if k_max <= 1:
                    continue

                # Build P* on common grid [1, k_max]
                k_grid = np.arange(1, k_max + 1, dtype=int)
                p_map = {int(k): float(v) for k, v in zip(x.astype(int), p_raw)}
                p_vec = np.array(
                    [max(p_map.get(k, 0.0), 0.0) for k in k_grid], dtype=float
                )
                s = p_vec.sum()
                if s > 0:
                    p_vec = p_vec / s

                # Build P_beta on common grid
                q_vec = np.exp(-beta * k_grid.astype(float))
                sq = q_vec.sum()
                if sq > 0:
                    q_vec = q_vec / sq

                djs = js_divergence(p_vec, q_vec)
                rows.append({
                    "utxo": utxo_key,
                    "date": date_str,
                    "beta": beta,
                    "k_max": k_max,
                    "D_JS": djs,
                })
            except Exception:
                continue

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} entries to {args.output}")

    # Summary statistics
    jsd = df["D_JS"].values
    n_total = len(jsd)
    print(f"\nJSD Summary (N = {n_total}):")
    print(f"  Mean:      {jsd.mean():.4f}")
    print(f"  Std. Dev.: {jsd.std():.4f}")
    print(f"  < 0.08:    {np.sum(jsd < 0.08)}/{n_total} "
          f"({100 * np.sum(jsd < 0.08) / n_total:.2f}%)")
    print(f"  < 0.10:    {np.sum(jsd < 0.10)}/{n_total} "
          f"({100 * np.sum(jsd < 0.10) / n_total:.2f}%)")

    # Binned distribution (Table 1 in the paper)
    bin_edges = np.arange(0.00, 0.13, 0.01)
    print(f"\n{'D_JS bin range':>20s} {'Count':>7s} {'Cumul.':>7s} {'Cumul. %':>10s}")
    print("-" * 48)
    cumul = 0
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == 0:
            count = int(np.sum((jsd >= lo) & (jsd <= hi)))
        else:
            count = int(np.sum((jsd > lo) & (jsd <= hi)))
        cumul += count
        pct = 100.0 * cumul / n_total if n_total > 0 else 0.0
        print(f"  ({lo:.2f}, {hi:.2f}] {count:7d} {cumul:7d} {pct:9.2f}%")


if __name__ == "__main__":
    main()
