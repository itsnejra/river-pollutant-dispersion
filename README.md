# River Pollutant Dispersion Simulator

A Monte Carlo simulation framework for modelling **1D advection–diffusion of chemical pollutants** in a river watershed. The tool quantifies pollutant arrival time and peak concentration at a downstream water-intake point under three climate scenarios (dry, normal, rainy) using stochastic river-flow modelling, surrogate metamodels, sensitivity analysis, and differential-evolution optimisation.

> **Academic context** — Topic 12: *Pollutant Dispersion in a River Watershed*  
> Course: Computer Modelling and Simulation  
> Authors: Nejra Smajlović (136)

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Docker](#docker)
- [Simulation Pipeline](#simulation-pipeline)
- [Configuration](#configuration)
- [Output](#output)
- [Dependencies](#dependencies)

---

## Features

| Capability | Details |
|---|---|
| **Stochastic flow** | Ornstein–Uhlenbeck process (exact discretisation, not Euler–Maruyama) |
| **PDE solver** | Upwind-FTCS finite-difference scheme for the 1D ADE |
| **Monte Carlo** | Adaptive N-determination via sequential confidence-interval narrowing |
| **Metamodels** | Latin-Hypercube sampling + GP / Random Forest / Gradient Boosting surrogates |
| **Sensitivity** | One-At-a-Time (OAT) parameter sweep |
| **Optimisation** | Differential evolution for worst-case arrival time and peak concentration |
| **What-if** | Exact grid sweep over pairs of parameters |
| **Visualisation** | Space-time heatmaps, MC distributions, metamodel parity plots, contour maps |

---

## Architecture

```
river-pollutant-dispersion/
├── main.py                    # 8-step pipeline orchestrator
├── pyproject.toml             # Project metadata & dependencies (uv)
├── Dockerfile                 # Container image
├── docker-compose.yml         # Compose for one-command runs
└── src/
    ├── config.py              # All physical constants & climate scenarios
    ├── logger.py              # Structured logging
    ├── stochastic_flow.py     # Ornstein-Uhlenbeck river-flow model
    ├── advection_diffusion.py # 1D ADE finite-difference solver
    ├── simulator.py           # Single-replication orchestrator
    ├── replication_manager.py # Monte Carlo manager & N-determination
    ├── statistics.py          # CI, Shapiro-Wilk, summary tables
    ├── metamodel.py           # LHS + GP / RF / GB surrogate training
    ├── sensitivity.py         # OAT sensitivity analyser
    ├── optimizer.py           # Differential evolution & what-if grid
    └── plotter.py             # All matplotlib/seaborn visualisations
```

---

## Quick Start

**Requirements:** Python 3.11+, [uv](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/itsnejra/river-pollutant-dispersion.git
cd river-pollutant-dispersion

# Install dependencies
uv sync

# Run the full pipeline (~5-15 min depending on hardware)
uv run python main.py
```

Output figures are written to `images/` as `.pdf`, `.svg`, and `.png`.

---

## Docker

```bash
# Build and run
docker compose up --build

# Or with plain Docker
docker build -t river-pollutant .
docker run --rm -v "$(pwd)/images:/app/images" river-pollutant
```

---

## Simulation Pipeline

The `main.py` orchestrator runs eight sequential steps:

| Step | Name | Description |
|------|------|-------------|
| 1 | **Deterministic baseline** | ADE solved with constant mean velocity for each climate scenario |
| 2 | **Warmup validation + N replications** | Welch's method validates OU warmup; sequential CI determines required N |
| 3 | **Monte Carlo** | N stochastic replications × 3 climate scenarios |
| 4 | **Statistical analysis** | 95 % CI, Shapiro–Wilk normality test, summary tables |
| 5 | **Metamodel training** | LHS design + GP, RF, GB trained; best model chosen by CV R² |
| 6 | **OAT sensitivity** | 51-point sweep ±30 % around nominal for each parameter |
| 7 | **Optimisation** | Differential evolution finds worst-case t_arrival and C_peak |
| 8 | **What-if analysis** | Exact 30×30 grid sweeps over (u_base, D) and (mass, u_base) |

---

## Configuration

All physical and numerical parameters are centralised in [`src/config.py`](src/config.py).

### Climate Scenarios

| Scenario | u_base (m/s) | σ_u (m/s) | D (m²/s) |
|----------|-------------|-----------|----------|
| Dry | 0.30 | 0.05 | 20 |
| Normal | 0.50 | 0.10 | 50 |
| Rainy | 0.75 | 0.20 | 80 |

### Key Physical Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| River length | 50 km | 1D domain extent |
| Spatial nodes | 101 | Grid resolution |
| Water intake | 40 km | Point of interest |
| Pollutant mass | 1 000 kg | Instantaneous release |
| Release position | 5 km | Upstream spill location |
| Time step dt | 150 s | Satisfies CFL & diffusion stability |
| Simulation time | 72 h | Total integration window |

### Numerical Stability

The finite-difference scheme requires:

```
Courant number:   Co = u·dt/dx ≤ 1.0
Diffusion number: d  = D·dt/dx² ≤ 0.5
```

Both conditions are checked at initialisation and raise `ValueError` on violation.

---

## Output

After a successful run the `images/` directory contains (among others):

| Figure | Content |
|--------|---------|
| `spacetime_*.pdf` | Space-time concentration heatmap per scenario |
| `snapshots_*.pdf` | Spatial profiles at selected time steps |
| `scenario_comparison.pdf` | Overlay of all three deterministic baselines |
| `velocity_series.pdf` | OU stochastic velocity time series |
| `replication_convergence.pdf` | CI width vs. number of replications |
| `warmup_welch.pdf` | Welch's periodogram for warmup validation |
| `mc_distributions.pdf` | Histograms of t_arrival and C_peak per scenario |
| `mc_confidence_bands.pdf` | MC 95 % confidence bands on concentration curves |
| `metamodel_validation.pdf` | Parity plots for all surrogates and targets |
| `sensitivity_*.pdf` | OAT sensitivity tornado charts |
| `whatif_*.pdf` | Contour maps of t_arrival and C_peak |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Vectorised numerics |
| `scipy` | `lfilter` for OU discretisation, `differential_evolution`, stats |
| `scikit-learn` | GP, Random Forest, Gradient Boosting metamodels; LHS sampling |
| `matplotlib` | All figures |
| `seaborn` | Distribution plots |
| `pandas` | Summary tables |
| `simpy` | (available for discrete-event extensions) |

---

## License

MIT — see [LICENSE](LICENSE) for details.
