"""
Orkestracija cjelokupnog simulacijskog pipelinea.

Tema 12: Širenje zagađivača u riječnom slivu
Studenti: Nejra Smajlović (136)

Pokretanje:
    python main.py

Figure se snimaju u images/ folder kao .pdf, .svg i .png.
"""

import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")

import numpy as np

from src.config import (
    CLIMATE_SCENARIOS,
    DomainConfig,
    FlowConfig,
    PollutantConfig,
    SimulationConfig,
    TransportConfig,
)
from src.logger import get_logger
from src.metamodel import PollutantMetamodel
from src.optimizer import OptimizationResult, ScenarioOptimizer
from src.plotter import SimulationPlotter
from src.replication_manager import MonteCarloManager, ReplicationSummary
from src.sensitivity import OATSensitivityAnalyzer
from src.simulator import RiverSimulation
from src.statistics import print_mc_summary
from src.stochastic_flow import OrnsteinUhlenbeckProcess

logger = get_logger(__name__)

REFERENCE_SCENARIO_KEY: str = "normal"


def _section(title: str) -> None:
    bar = "=" * 65
    logger.info(bar)
    logger.info(f"  {title}")
    logger.info(bar)


def _elapsed(t0: float) -> str:
    elapsed = time.time() - t0
    return f"{elapsed:.1f} s" if elapsed < 60 else f"{elapsed/60:.1f} min ({elapsed:.0f} s)"


def run_pipeline() -> None:
    """Orkestrira cijeli simulacijski pipeline.

    Pipeline koraci:
      1. DETERMINISTIČKI BASELINE — 3 klimatska scenarija
      2. WARMUP VALIDACIJA + N REPLIKACIJA
      3. MONTE CARLO — N replikacija × 3 scenarija
      4. STATISTIČKA ANALIZA — CI, Shapiro-Wilk
      5. METAMODEL — LHS + GP + RF + GB
      6. OAT ANALIZA OSJETLJIVOSTI
      7. OPTIMIZACIJA — diferencijalna evolucija
      8. WHAT-IF ANALIZA — egzaktni grid sweep
    """
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    total_start = time.time()

    logger.info("#" * 65)
    logger.info("  SIMULACIJA SIRENJA ZAGADJIVACA U RIJECNOM SLIVU")
    logger.info("  Tema 12 — Nejra Smajlovic")
    logger.info("#" * 65)

    # -------------------------------------------------------------------------
    # INICIJALIZACIJA
    # -------------------------------------------------------------------------
    domain: DomainConfig = DomainConfig()
    transport: TransportConfig = TransportConfig()
    pollutant: PollutantConfig = PollutantConfig()
    sim: SimulationConfig = SimulationConfig()
    plotter: SimulationPlotter = SimulationPlotter(domain)

    logger.info(f"Domen: L={domain.length/1e3:.0f} km, dx={domain.dx:.0f} m, N={domain.n_nodes}")
    logger.info(f"Vodovod na x={domain.x_interest/1e3:.0f} km")
    logger.info(f"dt={sim.dt:.0f} s, T={sim.t_total/3600:.0f} h, N_steps={sim.n_steps}")

    # -------------------------------------------------------------------------
    # 1. DETERMINISTIČKI BASELINE
    # -------------------------------------------------------------------------
    _section("KORAK 1: DETERMINISTICKI BASELINE SIMULACIJE")
    t0 = time.time()

    det_results = {}
    for key, scenario in CLIMATE_SCENARIOS.items():
        logger.info(f"Scenarij: {scenario.name}")
        rng = np.random.default_rng(0)
        sim_obj = RiverSimulation.from_scenario(scenario, domain, pollutant, sim, rng)
        result = sim_obj.run_deterministic()
        det_results[key] = result
        if result.arrived:
            logger.info(f"  -> t_arrival={result.t_arrival/3600:.2f} h, C_peak={result.C_peak:.4f} mg/m3")
        else:
            logger.info("  -> Zagadivac NIJE stigao!")
        plotter.plot_spacetime(result, key, scenario)
        plotter.plot_snapshots(result, key, scenario)

    plotter.plot_scenario_comparison(det_results)

    stoch_results = {}
    for key, scenario in CLIMATE_SCENARIOS.items():
        rng = np.random.default_rng(42)
        sim_obj = RiverSimulation.from_scenario(scenario, domain, pollutant, sim, rng)
        stoch_results[key] = sim_obj.run()
    plotter.plot_velocity_series(stoch_results, dt=sim.dt)

    logger.info(f"Korak 1 zavrsen za {_elapsed(t0)}.")

    # -------------------------------------------------------------------------
    # 2. WARMUP VALIDACIJA + N REPLIKACIJA
    # -------------------------------------------------------------------------
    _section("KORAK 2: WARMUP VALIDACIJA + N REPLIKACIJA")
    t0 = time.time()

    scenario_ref = CLIMATE_SCENARIOS[REFERENCE_SCENARIO_KEY]
    mc_mgr = MonteCarloManager(domain, transport, pollutant, sim, base_seed=42)

    flow_ref = FlowConfig(u_base=scenario_ref.u_base, sigma_u=scenario_ref.sigma_u)
    ou_ref = OrnsteinUhlenbeckProcess(flow_ref, sim, np.random.default_rng(0))
    logger.info(f"OU tau = {ou_ref.relaxation_time/3600:.2f} h")
    logger.info(f"Stacionarna N(mu={ou_ref.stationary_mean:.2f}, sigma={ou_ref.stationary_std:.3f}) m/s")
    logger.info(f"Warmup = {sim.warmup_time/ou_ref.relaxation_time:.1f} tau")

    n_required, rel_widths = mc_mgr.determine_n_replications(
        scenario_ref, target_rel_width=0.05, min_n=10
    )
    logger.info(f"Minimalni N = {n_required} replikacija")

    plotter.plot_replication_convergence(rel_widths, n_required)

    ou_welch = OrnsteinUhlenbeckProcess(flow_ref, sim, np.random.default_rng(777))
    u_welch = ou_welch.generate(sim.n_warmup_steps * 4 + 500)
    plotter.plot_warmup_welch(u_welch, sim.dt, sim.n_warmup_steps, scenario_ref)

    logger.info(f"Korak 2 zavrsen za {_elapsed(t0)}.")

    # -------------------------------------------------------------------------
    # 3. MONTE CARLO
    # -------------------------------------------------------------------------
    _section(f"KORAK 3: MONTE CARLO ({n_required} replikacija x 3 scenarija)")
    t0 = time.time()

    mc_summaries: dict[str, ReplicationSummary] = {}
    for key, scenario in CLIMATE_SCENARIOS.items():
        logger.info(f"-> {scenario.name}")
        mc_summaries[key] = mc_mgr.run_scenario(scenario, n_replications=n_required)
        s = mc_summaries[key]
        logger.info(
            f"   t_arrival: {s.t_arrival_mean/3600:.2f} +/- {s.t_arrival_std/3600:.2f} h  "
            f"[{s.t_arrival_ci_95[0]/3600:.2f}, {s.t_arrival_ci_95[1]/3600:.2f}]"
        )
        logger.info(f"   C_peak: {s.C_peak_mean:.4f} +/- {s.C_peak_std:.4f} mg/m3")
        logger.info(f"   P(dolazak) = {s.arrival_probability*100:.1f}%")

    logger.info(f"Korak 3 zavrsen za {_elapsed(t0)}.")

    # -------------------------------------------------------------------------
    # 4. STATISTIČKA ANALIZA
    # -------------------------------------------------------------------------
    _section("KORAK 4: STATISTICKA ANALIZA")
    t0 = time.time()

    print_mc_summary(mc_summaries)
    plotter.plot_mc_distributions(mc_summaries)
    plotter.plot_mc_confidence_bands(mc_summaries)

    logger.info(f"Korak 4 zavrsen za {_elapsed(t0)}.")

    # -------------------------------------------------------------------------
    # 5. METAMODEL
    # -------------------------------------------------------------------------
    _section("KORAK 5: METAMODEL (LHS + GP + RF + GB)")
    t0 = time.time()

    meta: PollutantMetamodel = PollutantMetamodel(
        domain=domain,
        pollutant=pollutant,
        sim_config=sim,
        n_samples=sim.n_metamodel_samples,
        rng=np.random.default_rng(sim.random_seed),
    )

    logger.info("Trening metamodela (moze potrajati ~2-5 min)...")
    meta_results = meta.train()

    for mname, tdict in meta_results.items():
        for tname, mr in tdict.items():
            logger.info(
                f"  {mname:<22} | {tname:<9} | "
                f"R2_CV={mr.r2_cv:.4f} | R2_test={mr.r2_test:.4f}"
            )
    logger.info(f"Odabran: '{meta.best_model_name}'")

    plotter.plot_metamodel_validation(meta_results)

    logger.info(f"Korak 5 zavrsen za {_elapsed(t0)}.")

    # -------------------------------------------------------------------------
    # 6. OAT ANALIZA OSJETLJIVOSTI
    # -------------------------------------------------------------------------
    _section("KORAK 6: OAT ANALIZA OSJETLJIVOSTI")
    t0 = time.time()

    nominal = {
        "u_base":  CLIMATE_SCENARIOS["normal"].u_base,
        "sigma_u": CLIMATE_SCENARIOS["normal"].sigma_u,
        "D":       CLIMATE_SCENARIOS["normal"].D,
        "mass":    1_000.0,
    }

    analyzer: OATSensitivityAnalyzer = OATSensitivityAnalyzer(
        meta, nominal_params=nominal, n_sweep=51
    )
    sens_results = analyzer.analyze()
    analyzer.print_summary(sens_results)
    plotter.plot_sensitivity_analysis(sens_results, analyzer)

    logger.info(f"Korak 6 zavrsen za {_elapsed(t0)}.")

    # -------------------------------------------------------------------------
    # 7. OPTIMIZACIJA
    # -------------------------------------------------------------------------
    _section("KORAK 7: OPTIMIZACIJA (DIFERENCIJALNA EVOLUCIJA)")
    t0 = time.time()

    optimizer: ScenarioOptimizer = ScenarioOptimizer(meta)

    opt_arrival: OptimizationResult = optimizer.find_worst_case_arrival(
        domain=domain, sim_config=sim
    )
    logger.info(f"min t_arrival (egzaktno) = {opt_arrival.optimal_value/3600:.2f} h")
    for p, v in opt_arrival.optimal_params.items():
        logger.info(f"    {p} = {v:.4f}")

    opt_peak: OptimizationResult = optimizer.find_worst_case_peak(
        domain=domain, sim_config=sim
    )
    logger.info(
        f"max C_peak (egzaktno) = {opt_peak.verified_value:.1f} mg/m3  "
        f"(greska: {opt_peak.surrogate_error_pct:.1f}%)"
    )
    for p, v in opt_peak.optimal_params.items():
        logger.info(f"    {p} = {v:.4f}")

    logger.info(f"Korak 7 zavrsen za {_elapsed(t0)}.")

    # -------------------------------------------------------------------------
    # 8. WHAT-IF ANALIZA
    # -------------------------------------------------------------------------
    _section("KORAK 8: WHAT-IF ANALIZA (EGZAKTNI GRID SWEEP)")
    t0 = time.time()

    base_params = {
        "u_base":  CLIMATE_SCENARIOS["normal"].u_base,
        "sigma_u": CLIMATE_SCENARIOS["normal"].sigma_u,
        "D":       CLIMATE_SCENARIOS["normal"].D,
        "mass":    1_000.0,
    }

    logger.info("Sweep 1: u_base x D ...")
    gx, gy, Zt, Zc = optimizer.what_if_grid_exact(
        "u_base", "D", base_params, domain, sim, n_grid=30
    )
    plotter.plot_whatif_contours(gx, gy, Zt, Zc, "u_base", "D")

    logger.info("Sweep 2: mass x u_base ...")
    gx2, gy2, Zt2, Zc2 = optimizer.what_if_grid_exact(
        "mass", "u_base", base_params, domain, sim, n_grid=30
    )
    plotter.plot_whatif_contours(gx2, gy2, Zt2, Zc2, "mass", "u_base")

    logger.info(f"Korak 8 zavrsen za {_elapsed(t0)}.")

    # -------------------------------------------------------------------------
    # FINALNI SAŽETAK
    # -------------------------------------------------------------------------
    total_time = time.time() - total_start
    n_figs = len(list(plotter._out.glob("*.pdf")))

    logger.info("=" * 65)
    logger.info(f"  Pipeline zavrsen za {total_time/60:.1f} min")
    logger.info(f"  Generisano figura: {n_figs}  (images/*.pdf + *.svg + *.png)")
    logger.info(f"  min t_arrival = {opt_arrival.optimal_value/3600:.2f} h")
    logger.info(f"  max C_peak    = {opt_peak.verified_value:.1f} mg/m3")
    logger.info("=" * 65)


def main() -> None:
    """Entry point za pokretanje simulacijskog pipelinea."""
    run_pipeline()


if __name__ == "__main__":
    main()
