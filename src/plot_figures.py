"""
Generate main-text figures (Figs. 1-6) for the paper.

Fig 1: Poisson vs Geometric (theoretical comparison, m = 50)
Fig 2: Log-scale empirical distributions (i = 10 and i = 1000 satoshi)
Fig 3: Bayesian posterior of beta_i
Fig 4: Geometric-distribution fits
Fig 5: Time evolution of empirical and theoretical means
Fig 6: Jensen-Shannon divergence time series

Usage
-----
    python plot_figures.py --data-dir <UTXO_distribution>
                           --fit-dir <fitting_results>
                           --output-dir <figures>
                           [--date 2020-06-01]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from scipy.special import factorial
from scipy.stats import gaussian_kde

matplotlib.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

SAVE_DPI = 300
COLOR_10 = "red"
COLOR_1000 = "blue"


# ================================================================= #
# Helper functions                                                    #
# ================================================================= #
def to_utxo_key(v) -> str:
    try:
        return str(int(float(v)))
    except (ValueError, TypeError):
        return str(v)


def load_x_px(pstar_root, date_str, utxo_str):
    csv_path = pstar_root / date_str / f"common_smooth_dist_{to_utxo_key(utxo_str)}.csv"
    df = pd.read_csv(csv_path)
    x = df["x"].to_numpy(dtype=float)
    p = df["p(x)"].to_numpy(dtype=float)
    p = np.clip(p, 0, None)
    s = p.sum()
    if s > 0:
        p = p / s
    return x, p


def load_beta(beta_root, date_str, utxo_str):
    df = pd.read_csv(beta_root / date_str / "summary.csv")
    df["utxo_key"] = df["utxo"].apply(to_utxo_key)
    row = df.loc[df["utxo_key"] == to_utxo_key(utxo_str)]
    return float(row.iloc[0]["b_est"])


def P_beta_model(beta, k_arr, k_max):
    """P_beta(k) = (e^beta - 1) / (1 - e^{-beta k_max}) * e^{-beta k}."""
    num = np.expm1(beta)
    den = -np.expm1(-beta * k_max)
    return (num / den) * np.exp(-beta * k_arr)


def m_theory(beta, k_max):
    """Theoretical mean of the truncated geometric distribution."""
    eb = np.exp(-beta)
    ebK = np.exp(-beta * k_max)
    ebK1 = np.exp(-beta * (k_max + 1))
    num = 1 - (k_max + 1) * ebK + k_max * ebK1
    den = (1 - eb) * (1 - ebK)
    return num / den


def available_dates(pstar_root, beta_root):
    def list_dates(root):
        return {p.name for p in root.iterdir() if p.is_dir() and len(p.name) == 10}
    return sorted(list_dates(pstar_root) & list_dates(beta_root))


def load_kmax_map(pstar_root):
    kmax_csv = pstar_root / "common_kmax_by_utxo.csv"
    df = pd.read_csv(kmax_csv)
    df["utxo_key"] = df["utxo"].apply(to_utxo_key)
    return {row["utxo_key"]: int(row["k_max_min"]) for _, row in df.iterrows()}


def js_divergence(p, q, eps=1e-15):
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m))


# ================================================================= #
# Figure 1: Poisson vs Geometric (simulation check)                  #
# ================================================================= #
def plot_fig1(out_dir, N=10_000, M=500_000):
    print("Figure 1: Poisson vs Geometric simulation...")
    m_avg = M / N
    rng = np.random.default_rng(42)

    # (a) Poisson equilibrium sample
    holdings_a = rng.poisson(m_avg, size=N)
    diff = M - holdings_a.sum()
    for _ in range(abs(diff)):
        idx = rng.integers(N)
        if diff > 0:
            holdings_a[idx] += 1
        elif holdings_a[idx] > 0:
            holdings_a[idx] -= 1

    k_max_a = int(np.max(holdings_a))
    k_vals_a = np.arange(0, k_max_a + 1)
    hist_a = np.bincount(holdings_a, minlength=k_max_a + 1) / N
    P_poisson = np.exp(-m_avg) * m_avg ** k_vals_a / factorial(k_vals_a, exact=False)

    # (b) Geometric equilibrium sample
    p_geom_param = 1.0 / (m_avg + 1)
    holdings_b = rng.geometric(p_geom_param, size=N) - 1
    diff = M - holdings_b.sum()
    for _ in range(abs(diff)):
        idx = rng.integers(N)
        if diff > 0:
            holdings_b[idx] += 1
        elif holdings_b[idx] > 0:
            holdings_b[idx] -= 1

    k_max_b = int(np.max(holdings_b))
    k_vals_b = np.arange(0, k_max_b + 1)
    hist_b = np.bincount(holdings_b, minlength=k_max_b + 1) / N
    P_geom = (1.0 / (m_avg + 1)) * (m_avg / (m_avg + 1)) ** k_vals_b

    # Plot (a)
    fig_a, ax_a = plt.subplots(figsize=(7, 6))
    ax_a.plot(k_vals_a, hist_a, "*", color="black", markersize=4, label="simulation")
    ax_a.plot(k_vals_a, P_poisson, "-", color="red", linewidth=2, label="Poisson")
    ax_a.set_xlabel(r"holdings $k$", fontsize=18)
    ax_a.set_ylabel(r"$P_{\mathrm{Poisson}}(k)$", fontsize=18)
    ax_a.tick_params(axis="both", which="major", labelsize=14)
    ax_a.legend(fontsize=14, frameon=True)
    ax_a.set_xlim(0, max(k_vals_a[hist_a > 1e-6]))
    plt.tight_layout()
    fig_a.savefig(out_dir / "fig1a_poisson.png", dpi=SAVE_DPI)
    plt.close(fig_a)

    # Plot (b)
    fig_b, ax_b = plt.subplots(figsize=(7, 6))
    ax_b.plot(k_vals_b, hist_b, "*", color="black", markersize=4, label="simulation")
    ax_b.plot(k_vals_b, P_geom, "-", color="blue", linewidth=2, label="Geometric")
    ax_b.set_xlabel(r"holdings $k$", fontsize=18)
    ax_b.set_ylabel(r"$P_{\mathrm{Geom.}}(k)$", fontsize=18)
    ax_b.tick_params(axis="both", which="major", labelsize=14)
    ax_b.legend(fontsize=14, frameon=True)
    ax_b.set_xlim(0, min(400, max(k_vals_b[hist_b > 1e-6])))
    plt.tight_layout()
    fig_b.savefig(out_dir / "fig1b_geometric.png", dpi=SAVE_DPI)
    plt.close(fig_b)
    print("  Saved fig1a_poisson.png, fig1b_geometric.png")


# ================================================================= #
# Figure 2: Log-scale empirical distributions                        #
# ================================================================= #
def plot_fig2(pstar_root, out_dir, date_str):
    print("Figure 2: Log-scale empirical distributions...")
    for utxo_str, color, fname in [
        ("10", COLOR_10, "fig2a_dist_10.png"),
        ("1000", COLOR_1000, "fig2b_dist_1000.png"),
    ]:
        x, p = load_x_px(pstar_root, date_str, utxo_str)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.semilogy(x, p, "*", color=color, markersize=6)
        ax.set_xlabel(r"holdings $k$", fontsize=18)
        ax.set_ylabel(r"$P_{\star}(k)$", fontsize=18)
        ax.set_title(rf"$i = {utxo_str}$ satoshi, June 1st 2020", fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=14)
        plt.tight_layout()
        fig.savefig(out_dir / fname, dpi=SAVE_DPI)
        plt.close(fig)
        print(f"  Saved {fname}")


# ================================================================= #
# Figure 3: Bayesian posterior of beta                                #
# ================================================================= #
def plot_fig3(beta_root, out_dir, date_str):
    print("Figure 3: Bayesian posteriors...")
    for utxo_str, fname in [
        ("10", "fig3a_posterior_10.png"),
        ("1000", "fig3b_posterior_1000.png"),
    ]:
        raw = np.loadtxt(
            beta_root / date_str / f"chain_{date_str}_{to_utxo_key(utxo_str)}.txt"
        )
        w_col = raw[:, 0]
        beta_col = raw[:, 2]

        sorted_idx = np.argsort(beta_col)
        sorted_b = beta_col[sorted_idx]
        sorted_w = w_col[sorted_idx]
        cumw = np.cumsum(sorted_w) / np.sum(sorted_w)

        b_est = sorted_b[np.searchsorted(cumw, 0.50)]
        lo = sorted_b[np.searchsorted(cumw, 0.16)]
        hi = sorted_b[np.searchsorted(cumw, 0.84)]

        kde = gaussian_kde(beta_col, weights=w_col)
        x_grid = np.linspace(lo - 3 * (hi - lo), hi + 3 * (hi - lo), 500)
        pdf = kde(x_grid)
        pdf = pdf / pdf.max()

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(x_grid, pdf, "-", color="black", linewidth=2)
        ax.axvline(b_est, color="black", linewidth=1.5, linestyle="-")
        ax.axvline(lo, color="black", linewidth=1.0, linestyle="--")
        ax.axvline(hi, color="black", linewidth=1.0, linestyle="--")
        ax.set_xlabel(r"$\beta_i$", fontsize=18)
        ax.set_ylabel(r"posterior", fontsize=18)
        ax.set_title(rf"$i = {utxo_str}$ satoshi, June 1st 2020", fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.set_yticks([])
        delta_hi = hi - b_est
        delta_lo = b_est - lo
        ann = rf"$\beta_i = {b_est:.5f}^{{+{delta_hi:.5f}}}_{{-{delta_lo:.5f}}}$"
        ax.text(
            0.95, 0.90, ann, transform=ax.transAxes, fontsize=16,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"),
        )
        ax.set_xticks([lo, b_est, hi])
        ax.set_xticklabels([f"{lo:.5f}", f"{b_est:.5f}", f"{hi:.5f}"], fontsize=12)
        plt.tight_layout()
        fig.savefig(out_dir / fname, dpi=SAVE_DPI)
        plt.close(fig)
        print(f"  Saved {fname}")


# ================================================================= #
# Figure 4: Geometric-distribution fits                               #
# ================================================================= #
def plot_fig4(pstar_root, beta_root, out_dir, date_str):
    print("Figure 4: Geometric fits...")
    for utxo_str, color, fname in [
        ("10", COLOR_10, "fig4a_fit_10.png"),
        ("1000", COLOR_1000, "fig4b_fit_1000.png"),
    ]:
        x, p_obs = load_x_px(pstar_root, date_str, utxo_str)
        b_est = load_beta(beta_root, date_str, utxo_str)
        k_max = float(np.max(x))

        x_fine = np.linspace(1, k_max, 500)
        p_fit_fine = P_beta_model(b_est, x_fine, k_max)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(
            x_fine, p_fit_fine, "-", color=color, linewidth=1.5,
            label=rf"fitting with $\langle \beta_i \rangle = {b_est:.5f}$",
        )
        ax.plot(x, p_obs, "*", color=color, markersize=6, label="data")
        ax.set_xlabel(r"$k$", fontsize=18)
        ax.set_ylabel(r"$P(k)$", fontsize=18)
        ax.set_title(rf"$i = {utxo_str}$ satoshi, June 1st 2020", fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=14)
        legend = ax.legend(fontsize=14, frameon=True)
        legend.get_frame().set_edgecolor("gray")
        plt.tight_layout()
        fig.savefig(out_dir / fname, dpi=SAVE_DPI)
        plt.close(fig)
        print(f"  Saved {fname}")


# ================================================================= #
# Figure 5: Time evolution of means                                   #
# ================================================================= #
def plot_fig5(pstar_root, beta_root, out_dir):
    print("Figure 5: Time evolution of means...")
    all_dates = available_dates(pstar_root, beta_root)

    for utxo_str, color, fname in [
        ("10", COLOR_10, "fig5a_timeevol_10.png"),
        ("1000", COLOR_1000, "fig5b_timeevol_1000.png"),
    ]:
        rows = []
        for date_str in all_dates:
            try:
                x, p = load_x_px(pstar_root, date_str, utxo_str)
                beta = load_beta(beta_root, date_str, utxo_str)
            except Exception:
                continue
            k_max = float(np.max(x))
            m_star = np.sum(x * p)
            m_theo = m_theory(beta, k_max)
            rows.append({"date": date_str, "m_star": m_star, "m_theory": m_theo})

        if not rows:
            continue

        df = pd.DataFrame(rows).sort_values("date")
        df["date_dt"] = pd.to_datetime(df["date"])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            df["date_dt"], df["m_star"], "*", color=color, markersize=5,
            label=r"$m_i^{\star}$ (data)",
        )
        ax.plot(
            df["date_dt"], df["m_theory"], "-", color=color, linewidth=1.5,
            label=r"$m_i$ (geometric)",
        )
        ax.set_xlabel(r"Date", fontsize=18)
        ax.set_ylabel(r"$m_i$", fontsize=18)
        ax.set_title(rf"$i = {utxo_str}$ satoshi", fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=14)
        legend = ax.legend(fontsize=14, frameon=True)
        legend.get_frame().set_edgecolor("gray")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        plt.tight_layout()
        fig.savefig(out_dir / fname, dpi=SAVE_DPI)
        plt.close(fig)
        print(f"  Saved {fname}")


# ================================================================= #
# Figure 6: Jensen-Shannon divergence time series                     #
# ================================================================= #
def plot_fig6(pstar_root, beta_root, out_dir):
    print("Figure 6: JSD time series...")
    kmax_map = load_kmax_map(pstar_root)
    all_dates = available_dates(pstar_root, beta_root)

    for utxo_str, color, fname in [
        ("10", COLOR_10, "fig6a_jsd_10.png"),
        ("1000", COLOR_1000, "fig6b_jsd_1000.png"),
    ]:
        utxo_key = to_utxo_key(utxo_str)
        k_max_min = kmax_map.get(utxo_key)
        if k_max_min is None:
            continue

        rows = []
        for date_str in all_dates:
            try:
                x, p_raw = load_x_px(pstar_root, date_str, utxo_str)
                beta = load_beta(beta_root, date_str, utxo_str)
            except Exception:
                continue
            kmax_avail = int(np.max(x))
            k_max = min(k_max_min, kmax_avail)
            if k_max <= 1:
                continue

            k_grid = np.arange(1, k_max + 1, dtype=int)
            p_map = {int(k): float(v) for k, v in zip(x.astype(int), p_raw)}
            p_vec = np.array(
                [max(p_map.get(k, 0.0), 0.0) for k in k_grid], dtype=float
            )
            s = p_vec.sum()
            if s > 0:
                p_vec = p_vec / s
            q_vec = np.exp(-beta * k_grid.astype(float))
            sq = q_vec.sum()
            if sq > 0:
                q_vec = q_vec / sq
            djs = js_divergence(p_vec, q_vec)
            rows.append({"date": date_str, "D_JS": djs})

        if not rows:
            continue

        df = pd.DataFrame(rows).sort_values("date")
        df["date_dt"] = pd.to_datetime(df["date"])
        mean_jsd = df["D_JS"].mean()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["date_dt"], df["D_JS"], "o", color=color, markersize=5)
        ax.set_xlabel(r"Date", fontsize=18)
        ax.set_ylabel(r"$D_\mathrm{JS}$", fontsize=18)
        ax.set_title(rf"$i = {utxo_str}$ satoshi", fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=14)

        mean_label = rf"$\langle D_\mathrm{{JS}} \rangle = {mean_jsd:.5f}$"
        mean_proxy = Line2D(
            [0], [0], linestyle="none", marker="", color="none", label=mean_label
        )
        legend = ax.legend(
            handles=[mean_proxy], fontsize=16, frameon=True,
            handlelength=0, handletextpad=0,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("black")

        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        plt.tight_layout()
        fig.savefig(out_dir / fname, dpi=SAVE_DPI)
        plt.close(fig)
        print(f"  Saved {fname}")


# ================================================================= #
# Main                                                                #
# ================================================================= #
def main():
    parser = argparse.ArgumentParser(description="Generate paper figures.")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Root of UTXO_distribution.")
    parser.add_argument("--fit-dir", type=Path, required=True,
                        help="Root of fitting results.")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"),
                        help="Directory to save figures.")
    parser.add_argument("--date", type=str, default="2020-06-01",
                        help="Snapshot date for Figs. 2-4.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_fig1(args.output_dir)
    plot_fig2(args.data_dir, args.output_dir, args.date)
    plot_fig3(args.fit_dir, args.output_dir, args.date)
    plot_fig4(args.data_dir, args.fit_dir, args.output_dir, args.date)
    plot_fig5(args.data_dir, args.fit_dir, args.output_dir)
    plot_fig6(args.data_dir, args.fit_dir, args.output_dir)
    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
