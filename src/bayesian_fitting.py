"""
Bayesian parameter inference for the truncated geometric distribution.

For each UTXO denomination and monthly snapshot, this script fits an
inverse-temperature parameter beta by minimising the Kullback-Leibler
divergence between the empirical occupancy distribution P*(k) and the
truncated geometric model P_beta(k) = C_beta * exp(-beta * k).

The KL divergence is defined with the natural logarithm (ln), so that
the multinomial log-likelihood satisfies ln L = -N * D_KL + const.
The Jensen-Shannon divergence (computed separately in compute_jsd.py)
uses log_2, giving D_JS in [0, 1].

The fit is embedded in a Bayesian framework using dynamic nested
sampling (dynesty), yielding posterior distributions for beta.

Usage
-----
    python bayesian_fitting.py --data-dir <UTXO_distribution>
                               --output-dir <fitting_results>
                               [--workers 4]
"""
import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from pathlib import Path

import numpy as np
import pandas as pd
import dynesty
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)


# ------------------------------------------------------------------ #
# Prior                                                                #
# ------------------------------------------------------------------ #
def prior_transform(u):
    """Uniform prior: beta ~ U(0, 1)."""
    return np.array([1.0 * u[0]])


# ------------------------------------------------------------------ #
# Truncated geometric model                                           #
# ------------------------------------------------------------------ #
def make_p_model(k_min: float, K: int):
    """Return P_beta(k) for k in [k_min, k_min + K - 1]."""
    def p_model(b: float, x_arr: np.ndarray) -> np.ndarray:
        if abs(b) < 1e-12:
            return np.full_like(x_arr, 1.0 / K, dtype=float)
        num = -np.expm1(-b)       # 1 - e^{-b}
        den = -np.expm1(-b * K)   # 1 - e^{-bK}
        p = np.exp(-b * (x_arr - k_min)) * (num / den)
        return np.clip(p, 1e-300, None)
    return p_model


# ------------------------------------------------------------------ #
# Process one date folder                                              #
# ------------------------------------------------------------------ #
def process_one_date(ddir: Path, results_root: Path, q):
    """Fit all denominations for a single monthly snapshot."""
    try:
        date_str = ddir.name
        out_dir = results_root / date_str
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_files = sorted(ddir.glob("common_smooth_dist_*.csv"))
        if not csv_files:
            q.put(("warn", date_str, f"No common_smooth_dist CSV in {ddir}"))
            q.put(("done", date_str))
            return date_str

        q.put(("init", date_str, len(csv_files)))

        rows = []
        done = 0

        for csv_path in csv_files:
            try:
                m = re.match(r"common_smooth_dist_(.+)\.csv$", csv_path.name)
                utxo_str = m.group(1) if m else csv_path.stem

                df = pd.read_csv(csv_path, sep=None, engine="python")
                if "x" not in df.columns or "p(x)" not in df.columns:
                    q.put(("warn", date_str, f"Missing columns in {csv_path.name}"))
                else:
                    x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
                    p_obs = pd.to_numeric(
                        df["p(x)"], errors="coerce"
                    ).fillna(0.0).to_numpy(dtype=float)

                    p_obs = np.clip(p_obs, 0.0, None)
                    s = p_obs.sum()
                    if s <= 0:
                        q.put(("warn", date_str, f"Sum of p(x) = 0: {csv_path.name}"))
                    else:
                        p_obs = p_obs / s

                        # Effective counts for the multinomial likelihood
                        if "y" in df.columns:
                            counts_raw = pd.to_numeric(
                                df["y"], errors="coerce"
                            ).fillna(0.0).to_numpy(dtype=float)
                            N_eff = counts_raw.sum()
                            counts = counts_raw if N_eff > 0 else p_obs * float(len(x))
                        else:
                            counts = p_obs * float(len(x))
                        counts = np.clip(
                            np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0),
                            0.0, None,
                        )

                        k_min = 1.0
                        k_max = float(np.max(x))
                        K = int(k_max - k_min + 1)
                        p_model = make_p_model(k_min, K)

                        def loglike(theta):
                            b = float(theta[0])
                            p_hat = p_model(b, x)
                            if (not np.all(np.isfinite(p_hat))) or np.any(p_hat <= 0):
                                return -np.inf
                            return float(np.sum(counts * np.log(p_hat)))

                        rng = np.random.default_rng(42)
                        sampler = dynesty.DynamicNestedSampler(
                            loglike, prior_transform, ndim=1,
                            bound="multi", sample="rwalk", rstate=rng,
                        )
                        sampler.run_nested(dlogz_init=0.01, print_progress=False)
                        results = sampler.results

                        samples = results.samples[:, 0]
                        weights = np.exp(results.logwt - results.logz[-1])
                        weights /= np.sum(weights)
                        b_est = float(np.sum(samples * weights))
                        b_std = float(
                            np.sqrt(np.sum((samples - b_est) ** 2 * weights))
                        )

                        m_data = float(np.sum(x * p_obs))

                        # Save MCMC chain
                        chain_path = out_dir / f"chain_{date_str}_{utxo_str}.txt"
                        with open(chain_path, "w") as f:
                            for s_val, w, l in zip(samples, weights, results.logl):
                                f.write(f"{w:.8e} {-l:.8e} {s_val:.8e}\n")

                        q.put(("save", date_str, "file", str(chain_path)))

                        rows.append({
                            "utxo": utxo_str,
                            "b_est": b_est,
                            "b_std": b_std,
                            "k_max": k_max,
                            "m": m_data,
                        })

            except Exception as e:
                q.put(("warn", date_str, f"{csv_path.name} error: {e}"))
            finally:
                done += 1
                q.put(("progress", date_str, done))

        if rows:
            summary_df = pd.DataFrame(rows)[["utxo", "b_est", "b_std", "k_max", "m"]]
            summary_path = out_dir / "summary.csv"
            summary_df.to_csv(summary_path, index=False)
            q.put(("save", date_str, "summary", str(summary_path)))

        q.put(("done", date_str))
        return date_str

    except Exception as e:
        ds = ddir.name if isinstance(ddir, Path) else str(ddir)
        q.put(("fail", ds, str(e)))
        q.put(("done", ds))
        return ds


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="Bayesian fitting of truncated geometric distributions "
        "to Bitcoin UTXO occupancy data."
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Root of UTXO_distribution (contains date sub-folders).",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write fitting results.",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers (default: 4).",
    )
    args = parser.parse_args()

    console = Console()
    base_dir = args.data_dir
    results_dir = args.output_dir
    max_workers = args.workers

    if not base_dir.exists():
        console.print(f"[red]Data directory not found:[/red] {base_dir}")
        return
    results_dir.mkdir(parents=True, exist_ok=True)

    date_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir()])
    if not date_dirs:
        console.print("[yellow]No date folders found.[/yellow]")
        return

    with Manager() as m:
        q = m.Queue()

        overall_prog = Progress(
            TextColumn("[bold]Overall[/bold]"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        )
        running_prog = Progress(
            TextColumn("[cyan]{task.fields[date]}[/cyan]"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        )

        with overall_prog, running_prog:
            overall = overall_prog.add_task("overall", total=len(date_dirs))
            slot_ids = []
            for _ in range(max_workers):
                tid = running_prog.add_task(
                    "slot", total=1, visible=False, date=""
                )
                slot_ids.append(tid)

            free_slots = slot_ids[:]
            date_to_slot = {}

            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = [
                    ex.submit(process_one_date, d, results_dir, q)
                    for d in date_dirs
                ]

                finished_dates = 0
                while finished_dates < len(futures):
                    kind, date_str, *rest = q.get()

                    if kind == "init":
                        (total_utxo,) = rest
                        if not free_slots:
                            continue
                        sid = free_slots.pop(0)
                        date_to_slot[date_str] = sid
                        running_prog.update(
                            sid, total=total_utxo, completed=0,
                            visible=True, fields={"date": date_str},
                        )
                    elif kind == "progress":
                        (done_count,) = rest
                        sid = date_to_slot.get(date_str)
                        if sid is not None:
                            running_prog.update(sid, completed=done_count)
                    elif kind == "warn":
                        (msg,) = rest
                        console.log(f"[yellow][{date_str}] {msg}[/yellow]")
                    elif kind == "fail":
                        (msg,) = rest
                        console.log(f"[red][{date_str}] Failed: {msg}[/red]")
                    elif kind == "done":
                        finished_dates += 1
                        overall_prog.update(overall, completed=finished_dates)
                        sid = date_to_slot.pop(date_str, None)
                        if sid is not None:
                            running_prog.update(
                                sid, total=1, completed=0,
                                visible=False, fields={"date": ""},
                            )
                            free_slots.append(sid)

    console.print("[bold green]Bayesian fitting complete.[/bold green]")


if __name__ == "__main__":
    main()
