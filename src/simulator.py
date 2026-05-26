"""simulator.py - Orkestrator jedne simulacijske replikacije."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.config import (
    ClimateScenario,
    DomainConfig,
    FlowConfig,
    PollutantConfig,
    SimulationConfig,
    TransportConfig,
)
from src.advection_diffusion import ADESolver, SimulationResult
from src.stochastic_flow import OrnsteinUhlenbeckProcess


class RiverSimulation:
    """Orkestrator jedne simulacije širenja zagađivača u rijeci.

    Jedna replikacija se sastoji od:
        1. Warmup faza: OU proces se pokreće n_warmup_steps koraka.
        2. Simulacijska faza: ADE se rješava s post-warmup serijom brzina.

    Args:
        domain: Konfiguracija prostornog domena.
        flow: Parametri stohastičke brzine (OU proces).
        transport: Parametri transporta (koeficijent disperzije).
        pollutant: Početni uvjeti ispuštanja zagađivača.
        sim_config: Numerički parametri simulacije.
        rng: Generator slučajnih brojeva (jedinstven po replikaciji).
    """

    def __init__(
        self,
        domain: DomainConfig,
        flow: FlowConfig,
        transport: TransportConfig,
        pollutant: PollutantConfig,
        sim_config: SimulationConfig,
        rng: np.random.Generator,
    ) -> None:
        self._ou = OrnsteinUhlenbeckProcess(flow, sim_config, rng)
        self._solver = ADESolver(domain, transport, pollutant, sim_config)
        self._sim = sim_config

    @classmethod
    def from_scenario(
        cls,
        scenario: ClimateScenario,
        domain: DomainConfig,
        pollutant: PollutantConfig,
        sim_config: SimulationConfig,
        rng: np.random.Generator,
    ) -> "RiverSimulation":
        """Konstruisanje simulacije iz imenovanog klimatskog scenarija.

        Args:
            scenario: Predefinisani hidrološki scenario.
            domain: Konfiguracija prostornog domena.
            pollutant: Početni uvjeti ispuštanja zagađivača.
            sim_config: Numerički parametri simulacije.
            rng: Generator slučajnih brojeva.

        Returns:
            Inicijalizovana RiverSimulation instanca.
        """
        flow = FlowConfig(u_base=scenario.u_base, sigma_u=scenario.sigma_u)
        transport = TransportConfig(D=scenario.D)
        return cls(domain, flow, transport, pollutant, sim_config, rng)

    def run(self) -> SimulationResult:
        """Pokretanje jedne stohastičke simulacije (s OU procesom).

        Returns:
            Kompletni rezultati simulacije.
        """
        velocities: npt.NDArray[np.float64] = self._ou.generate_with_warmup(
            n_steps=self._sim.n_steps,
            n_warmup=self._sim.n_warmup_steps,
        )
        return self._solver.solve(velocities)

    def run_deterministic(self) -> SimulationResult:
        """Pokretanje determinističke simulacije sa konstantnom (srednjom) brzinom.

        Returns:
            Kompletni rezultati simulacije s konstantnom brzinom.
        """
        u_mean = self._ou.stationary_mean
        velocities: npt.NDArray[np.float64] = np.full(
            self._sim.n_steps, u_mean, dtype=np.float64
        )
        return self._solver.solve(velocities)
