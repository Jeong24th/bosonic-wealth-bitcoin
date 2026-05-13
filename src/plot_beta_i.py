"""
Generate Figure 7 of the main text and the time-stability figure of the
Supplementary Information.

  Figure 7 (main):
     (a) <beta_i> versus denomination i, with std error bars
     (b) Normalised trajectories beta_i(t)/<beta_i> for all 63 denominations

  SI time-stability figure:
     (a) CV_i = sigma(beta_i)/<beta_i> versus i
     (b) histogram of CV across the 63 denominations

Expected input:
  --beta-summary    per-denomination CSV from beta_i_universality.py
  --full-fits       per-date summary.csv root (for time evolution panel)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'figure.dpi': 150, 'savefig.dpi': 220, 'font.family': 'serif',
    'mathtext.fontset': 'cm',
})


def load_full(fit_dir: Path) -> pd.DataFrame:
    dates = sorted([p.name for p in fit_dir.iterdir()
                    if p.is_dir() and len(p.name) == 10])
    rows = []
    for d in dates:
        pth = fit_dir / d / "summary.csv"
        if not pth.exists():
            continue
        df = pd.read_csv(pth)
        df["date"] = d
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def fig_main(summary_csv: Path, full_csv: pd.DataFrame, out: Path):
    df = pd.read_csv(summary_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # (a) <beta_i> vs i
    ax = axes[0]
    ax.errorbar(df["i"], df["beta_mean"], yerr=df["beta_std"],
                fmt='o', ms=5, color='navy', ecolor='steelblue',
                capsize=2, alpha=0.85, label="per-denomination mean +/- std")
    ax.set_xscale('log')
    ax.set_xlabel(r"UTXO denomination $i$ (satoshi)")
    ax.set_ylabel(r"$\langle\beta_i\rangle$")
    ax.axhline(df["beta_mean"].mean(), color='red', linestyle='--', lw=1,
               label=fr"global mean $= {df['beta_mean'].mean():.3f}$")
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(r"(a) Denomination dependence of $\beta_i$")
    ax.set_ylim(0.18, 0.50)

    # (b) normalised trajectories for all 63 denominations
    ax = axes[1]
    denoms = sorted(full_csv["utxo"].unique())
    cmap = plt.cm.viridis
    log_d_min = np.log10(min(denoms))
    log_d_max = np.log10(max(denoms))
    for d in denoms:
        sub = full_csv[full_csv["utxo"] == d].sort_values("date")
        if len(sub) == 0:
            continue
        b = sub["b_est"].to_numpy()
        if b.mean() <= 0:
            continue
        months = np.arange(len(sub))
        col = cmap((np.log10(d) - log_d_min) /
                   max(log_d_max - log_d_min, 1e-9))
        ax.plot(months, b / b.mean(), '-', lw=0.6, color=col, alpha=0.55)
    ax.axhline(1.0, color='black', lw=1.1, linestyle='-', alpha=0.7)
    ax.axhspan(0.9, 1.1, color='red', alpha=0.06, label=r"$\pm 10\%$ band")
    ax.axhspan(0.95, 1.05, color='orange', alpha=0.08, label=r"$\pm 5\%$ band")
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=log_d_min, vmax=log_d_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10} i$", fontsize=11)
    ax.set_xlabel("Month index (Jan 2018 = 0)")
    ax.set_ylabel(r"$\beta_i(t)\,/\,\langle\beta_i\rangle$")
    ax.set_title(
        r"(b) Normalised time evolution of $\beta_i$ for all 63 denominations")
    ax.set_ylim(0.7, 1.3)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def fig_si(summary_csv: Path, out: Path):
    df = pd.read_csv(summary_csv)
    df["CV"] = df["beta_std"] / df["beta_mean"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    ax.scatter(df["i"], df["CV"] * 100, s=40, c='purple', alpha=0.85)
    ax.set_xscale('log')
    ax.set_xlabel(r"UTXO denomination $i$ (satoshi)")
    ax.set_ylabel(
        r"Coefficient of variation, $\sigma(\beta_i)/\langle\beta_i\rangle$ (\%)")
    ax.axhline(10, color='red', linestyle='--', lw=1, alpha=0.7,
               label='10\\% threshold')
    ax.axhline(5, color='orange', linestyle=':', lw=1, alpha=0.7,
               label='5\\% threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("(a) Time-stability per denomination")

    ax = axes[1]
    ax.hist(df["CV"] * 100, bins=20, color='purple',
            edgecolor='black', alpha=0.85)
    ax.axvline(df["CV"].median() * 100, color='blue', lw=1.2,
               label=fr"median = {df['CV'].median() * 100:.2f}\%")
    ax.axvline(10, color='red', linestyle='--', lw=1, alpha=0.7)
    ax.set_xlabel(r"$\sigma(\beta_i)/\langle\beta_i\rangle$ (\%)")
    ax.set_ylabel("Number of denominations")
    ax.set_title("(b) Distribution of CV across 63 denominations")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta-summary", required=True, type=Path,
                    help="per-denomination CSV from beta_i_universality.py")
    ap.add_argument("--fit-dir", required=True, type=Path,
                    help="root containing per-date summary.csv files")
    ap.add_argument("--out-main", default="beta_vs_i.png", type=Path)
    ap.add_argument("--out-si",   default="SI_beta_time_stability.png",
                    type=Path)
    args = ap.parse_args()

    full = load_full(args.fit_dir)
    fig_main(args.beta_summary, full, args.out_main)
    fig_si(args.beta_summary, args.out_si)


if __name__ == "__main__":
    main()
