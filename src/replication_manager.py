"""replication_manager.py - Upravljač Monte Carlo replikacijama."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.stats as stats

from src.config import (
    ClimateScenario,
    DomainConfig,
    PollutantConfig,
    SimulationConfig,
    TransportConfig,
)
from src.advection_diffusion import SimulationResult
from src.simulator import RiverSimulation

logger = logging.getLogger(__name__)


@dataclass
class ReplicationSummary:
    """Statistički sažetak Monte Carlo eksperimenta.

    Attributes:
        n_replications: Ukupan broj pokrennutih replikacija.
        t_arrival_mean: Srednja vrijednost vremena dolaska [s].
        t_arrival_std: Standardna devijacija vremena dolaska [s].
        t_arrival_ci_95: 95% interval pouzdanosti za t_arrival [s].
        t_arrival_ci_relative_width: Relativna polu-širina 95% CI.
        C_peak_mean: Srednja pik koncentracija [mg/m³].
        C_peak_std: Std pik koncentracije [mg/m³].
        C_peak_ci_95: 95% CI za C_peak [mg/m³].
        t_peak_mean: Srednje vrijeme pika [s].
        t_peak_std: Std vremena pika [s].
        arrival_probability: Udio replikacija gdje zagađivač stiže do vodovoda.
        t_arrivals: Sirovi niz vremena dolaska za sve replikacije.
        C_peaks: Sirovi niz pik koncentracija za sve replikacije.
        t_peaks: Sirovi niz vremena pikova.
        results: Kompletni rezultati svih replikacija.
    """

    n_replications: int
    t_arrival_mean: float
    t_arrival_std: float
    t_arrival_ci_95: tuple[float, float]
    t_arrival_ci_relative_width: float
    C_peak_mean: float
    C_peak_std: float
    C_peak_ci_95: tuple[float, float]
    t_peak_mean: float
    t_peak_std: float
    arrival_probability: float
    t_arrivals: npt.NDArray[np.float64]
    C_peaks: npt.NDArray[np.float64]
    t_peaks: npt.NDArray[np.float64]
    results: list[SimulationResult]


class MonteCarloManager:
    """Pokretanje i analiza Monte Carlo replikacija za dati klimatski scenario.

    Args:
        domain: Konfiguracija prostornog domena.
        transport: Bazni transport.
        pollutant: Početni uvjeti ispuštanja zagađivača.
        sim_config: Numerički parametri simulacije.
        base_seed: Bazni seed; seed i-te replikacije = base_seed + i.
    """

    def __init__(
        self,
        domain: DomainConfig,
        transport: TransportConfig,
        pollutant: PollutantConfig,
        sim_config: SimulationConfig,
        base_seed: int = 42,
    ) -> None:
        self._domain = domain
        self._transport = transport
        self._pollutant = pollutant
        self._sim = sim_config
        self._base_seed = base_seed

    def run_scenario(
        self,
        scenario: ClimateScenario,
        n_replications: int | None = None,
    ) -> ReplicationSummary:
        """Pokretanje Monte Carlo replikacija za dati klimatski scenario.

        Args:
            scenario: Klimatski scenario s hidrološkim parametrima.
            n_replications: Broj replikacija. Default: sim_config.n_replications.

        Returns:
            Statistički sažetak svih replikacija.
        """
        n_rep = n_replications or self._sim.n_replications
        results: list[SimulationResult] = []

        logger.info(f"  Pokretanje {n_rep} replikacija za '{scenario.name}'...")
        for i in range(n_rep):
            rng = np.random.default_rng(self._base_seed + i)
            sim = RiverSimulation.from_scenario(
                scenario, self._domain, self._pollutant, self._sim, rng
            )
            results.append(sim.run())
            if (i + 1) % 25 == 0:
                logger.info(f"    Završeno {i+1}/{n_rep}")

        return self._compute_summary(results)

    def determine_n_replications(
        self,
        scenario: ClimateScenario,
        target_rel_width: float = 0.05,
        min_n: int = 10,
    ) -> tuple[int, list[float]]:
        """Sekvencijalno određivanje minimalnog broja replikacija.

        Adaptivni while algoritam — dodaje replikacije jednu po jednu dok
        relativna polu-širina 95% CI za t_arrival ne padne ispod ciljne
        vrijednosti (gamma).

        Kriterijum zaustavljanja:
            h / |X̄| = (t_crit * SE) / mean(t_arrivals) <= target_rel_width

        Args:
            scenario: Klimatski scenario koji se koristi za procjenu.
            target_rel_width: Ciljana relativna greška gamma (default: 5%).
            min_n: Minimalni broj replikacija.

        Returns:
            Tuple (broj_replikacija, historija_relativnih_širina).
        """
        t_arrivals: list[float] = []
        rel_widths: list[float] = []

        relativna_greska: float = float("inf")
        n: int = 0

        while n < min_n or relativna_greska > target_rel_width:
            rng = np.random.default_rng(self._base_seed + n)
            sim = RiverSimulation.from_scenario(
                scenario, self._domain, self._pollutant, self._sim, rng
            )
            result = sim.run()
            if result.arrived:
                t_arrivals.append(result.t_arrival)
            n += 1

            if len(t_arrivals) >= 2:
                arr = np.array(t_arrivals)
                standardna_greska: float = float(stats.sem(arr))
                t_krit: float = float(stats.t.ppf((1 + 0.95) / 2, df=len(arr) - 1))
                h: float = t_krit * standardna_greska
                relativna_greska = h / float(arr.mean())

                if len(t_arrivals) >= min_n:
                    rel_widths.append(relativna_greska)

                    if n % 10 == 0:
                        logger.info(f"Iteracija {n:3d}: Relativna greška = {relativna_greska * 100:.2f}%")

        if t_arrivals and relativna_greska <= target_rel_width:
            logger.info(
                f"  Konvergencija pri n={n}: "
                f"rel. greška = {relativna_greska * 100:.2f}%"
            )

        return n, rel_widths

    @staticmethod
    def _compute_summary(results: list[SimulationResult]) -> ReplicationSummary:
        """Računanje statističkih sažetaka iz liste rezultata replikacija.

        Args:
            results: Lista rezultata svih replikacija.

        Returns:
            Statistički sažetak Monte Carlo eksperimenta.

        Raises:
            RuntimeError: Ako manje od 2 replikacije dostignu tačku interesa.
        """
        arrived = [r for r in results if r.arrived]
        if len(arrived) < 2:
            raise RuntimeError(
                f"Premalo replikacija dostiglo tačku interesa ({len(arrived)})."
            )

        t_arrivals: npt.NDArray[np.float64] = np.array(
            [r.t_arrival for r in arrived]
        )
        C_peaks: npt.NDArray[np.float64] = np.array([r.C_peak for r in arrived])
        t_peaks: npt.NDArray[np.float64] = np.array([r.t_peak for r in arrived])

        n = len(t_arrivals)

        t_mean = float(np.mean(t_arrivals))
        t_std = float(np.std(t_arrivals, ddof=1))
        t_ci_lo, t_ci_hi = stats.t.interval(
            confidence=0.95,
            df=n - 1,
            loc=t_mean,
            scale=stats.sem(t_arrivals),
        )
        t_ci: tuple[float, float] = (float(t_ci_lo), float(t_ci_hi))
        h_t = t_ci_hi - t_mean
        t_rel_w = float(h_t / t_mean) if t_mean > 0 else np.inf

        c_mean = float(np.mean(C_peaks))
        c_std = float(np.std(C_peaks, ddof=1))
        c_ci_lo, c_ci_hi = stats.t.interval(
            confidence=0.95,
            df=n - 1,
            loc=c_mean,
            scale=stats.sem(C_peaks),
        )
        c_ci: tuple[float, float] = (float(c_ci_lo), float(c_ci_hi))

        return ReplicationSummary(
            n_replications=len(results),
            t_arrival_mean=t_mean,
            t_arrival_std=t_std,
            t_arrival_ci_95=t_ci,
            t_arrival_ci_relative_width=t_rel_w,
            C_peak_mean=c_mean,
            C_peak_std=c_std,
            C_peak_ci_95=c_ci,
            t_peak_mean=float(np.mean(t_peaks)),
            t_peak_std=float(np.std(t_peaks, ddof=1)),
            arrival_probability=len(arrived) / len(results),
            t_arrivals=t_arrivals,
            C_peaks=C_peaks,
            t_peaks=t_peaks,
            results=results,
        )
