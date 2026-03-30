# Empirical confirmation of bosonic wealth statistics in Bitcoin UTXOs

Code and data analysis pipeline for the paper:

> C. Park, C. J. Tessone, Y. Zhang, and J.-H. Park, "Empirical confirmation of bosonic wealth statistics in Bitcoin UTXOs" (2025).

## Overview

This repository provides the code for:

1. **Bayesian inference** of the inverse-temperature parameter $\beta$ for truncated geometric distributions fitted to Bitcoin UTXO occupancy data.
2. **Numerical simulations** of minimal exchange processes (Poisson and geometric equilibria).
3. **Jensen--Shannon divergence** computation between empirical and fitted distributions across all denominations and time snapshots.
4. **Figure generation** for the main text (Figs. 1--6).

## Repository structure

```
bosonic-wealth-bitcoin/
  src/
    bayesian_fitting.py   # Bayesian parameter inference (dynesty)
    simulate.py           # Minimal exchange simulations
    compute_jsd.py        # Jensen-Shannon divergence analysis
    plot_figures.py        # Main-text figures (Figs. 1-6)
  data/                    # UTXO distribution data (see below)
  results/                 # Fitting results (generated)
  figures/                 # Output figures (generated)
  requirements.txt
  LICENSE
  README.md
```

## Data

The empirical data are constructed from Bitcoin mainnet using [BlockSci](https://github.com/citp/BlockSci) (v0.21.0). For each monthly snapshot (January 2018 -- December 2023), the UTXO set is parsed and denomination-resolved occupancy distributions are computed for 63 representative face values.

**Expected data layout:**

```
data/
  UTXO_distribution/
    common_kmax_by_utxo.csv
    2018-01-01/
      common_smooth_dist_10.csv
      common_smooth_dist_1000.csv
      ...
    2018-02-01/
      ...
  fitting_results/
    2018-01-01/
      summary.csv
      chain_2018-01-01_10.txt
      ...
```

Each `common_smooth_dist_{i}.csv` contains columns `x` (holdings $k$) and `p(x)` (empirical probability), with moving-average smoothing (window = 9) and a common $k_{\max}$ across all dates for each denomination.

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, NumPy, Pandas, Matplotlib, SciPy, dynesty, rich.

LaTeX is required for figure rendering (`text.usetex = True`).

## Usage

### 1. Bayesian fitting

```bash
python src/bayesian_fitting.py \
    --data-dir data/UTXO_distribution \
    --output-dir results/fitting_results \
    --workers 4
```

Fits a truncated geometric distribution $P_\beta(k) = C_\beta e^{-\beta k}$ to each denomination and snapshot using dynamic nested sampling. Outputs MCMC chains and summary statistics (`summary.csv`) per date.

### 2. Numerical simulations

```bash
python src/simulate.py --N 10000 --M 500000 --steps 100000000
```

Runs both exchange processes:
- **Wealth-to-receiver** (distinguishable units $\to$ Poisson equilibrium)
- **Sender-to-receiver** (identical units $\to$ geometric equilibrium)

### 3. Jensen--Shannon divergence

```bash
python src/compute_jsd.py \
    --data-dir data/UTXO_distribution \
    --fit-dir results/fitting_results \
    --output results/jsd_results.csv
```

### 4. Generate figures

```bash
python src/plot_figures.py \
    --data-dir data/UTXO_distribution \
    --fit-dir results/fitting_results \
    --output-dir figures
```

## Data availability

The Bitcoin blockchain data used in this study are publicly available via the Bitcoin network.

## Citation

If you use this code, please cite:

```bibtex
@article{park2025bosonic,
  title={Empirical confirmation of bosonic wealth statistics in Bitcoin UTXOs},
  author={Park, Chanhee and Tessone, Claudio J. and Zhang, Yu and Park, Jeong-Hyuck},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
