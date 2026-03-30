"""
Minimal exchange simulations for distinguishable and identical wealth.

Two agent-based models that conserve both the total number of holders N
and total wealth M:

1. Random wealth-to-receiver (distinguishable/Poisson):
   At each step, a random unit is reassigned to a random holder.

2. Random sender-to-receiver (identical/bosonic/geometric):
   At each step, a sender with k >= 1 transfers one unit to a random holder.

Usage
-----
    python simulate.py [--N 10000] [--M 500000] [--steps 100000000]
                       [--output-dir ../results/simulations]
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.special import factorial


def simulate_wealth_to_receiver(N, M, steps, rng):
    """Poisson equilibrium via random wealth-to-receiver exchange."""
    holdings = np.zeros(N, dtype=np.int64)
    # Initialise: distribute M units uniformly
    for _ in range(M):
        holdings[rng.integers(N)] += 1

    for _ in range(steps):
        # Pick a random unit from the pool (select holder proportional to wealth)
        cumw = np.cumsum(holdings)
        r = rng.integers(1, M + 1)
        sender = np.searchsorted(cumw, r)
        if sender >= N:
            sender = N - 1
        if holdings[sender] > 0:
            holdings[sender] -= 1
            receiver = rng.integers(N)
            holdings[receiver] += 1

    return holdings


def simulate_sender_to_receiver(N, M, steps, rng):
    """Geometric equilibrium via random sender-to-receiver exchange."""
    holdings = np.zeros(N, dtype=np.int64)
    for _ in range(M):
        holdings[rng.integers(N)] += 1

    for _ in range(steps):
        # Pick a sender with k >= 1
        nonzero = np.where(holdings > 0)[0]
        if len(nonzero) == 0:
            continue
        sender = rng.choice(nonzero)
        receiver = rng.integers(N)
        holdings[sender] -= 1
        holdings[receiver] += 1

    return holdings


def main():
    parser = argparse.ArgumentParser(
        description="Minimal exchange simulations (Poisson & geometric)."
    )
    parser.add_argument("--N", type=int, default=10_000, help="Number of holders.")
    parser.add_argument("--M", type=int, default=500_000, help="Total wealth units.")
    parser.add_argument("--steps", type=int, default=100_000_000, help="Exchange steps.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/simulations"),
        help="Directory to save simulation results.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    m_avg = args.M / args.N

    # --- Poisson (wealth-to-receiver) ---
    print(f"Running wealth-to-receiver simulation (N={args.N}, M={args.M}, "
          f"steps={args.steps})...")
    h_poisson = simulate_wealth_to_receiver(args.N, args.M, args.steps, rng)
    k_max_p = int(np.max(h_poisson))
    k_vals_p = np.arange(0, k_max_p + 1)
    hist_p = np.bincount(h_poisson, minlength=k_max_p + 1) / args.N
    P_poisson_theory = (
        np.exp(-m_avg) * m_avg ** k_vals_p / factorial(k_vals_p, exact=False)
    )

    np.savetxt(
        args.output_dir / "poisson_simulation.csv",
        np.column_stack([k_vals_p, hist_p, P_poisson_theory]),
        header="k,P_sim,P_theory", delimiter=",", comments="",
    )
    print(f"  Saved: {args.output_dir / 'poisson_simulation.csv'}")

    # --- Geometric (sender-to-receiver) ---
    print(f"Running sender-to-receiver simulation...")
    h_geom = simulate_sender_to_receiver(args.N, args.M, args.steps, rng)
    k_max_g = int(np.max(h_geom))
    k_vals_g = np.arange(0, k_max_g + 1)
    hist_g = np.bincount(h_geom, minlength=k_max_g + 1) / args.N
    P_geom_theory = (1.0 / (m_avg + 1)) * (m_avg / (m_avg + 1)) ** k_vals_g

    np.savetxt(
        args.output_dir / "geometric_simulation.csv",
        np.column_stack([k_vals_g, hist_g, P_geom_theory]),
        header="k,P_sim,P_theory", delimiter=",", comments="",
    )
    print(f"  Saved: {args.output_dir / 'geometric_simulation.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
