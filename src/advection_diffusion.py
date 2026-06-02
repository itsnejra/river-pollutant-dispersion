"""advection_diffusion.py - Solver konačnih razlika za 1D ADE.

Rješava PDE:
    dC/dt + u(t) * dC/dx = D * d²C/dx²

Numerička shema
---------------
Advekcija  : Prva-reda upwind shema (unatražna razlika, za u > 0)
Difuzija   : Centralne razlike, FTCS (Forward-Time Central-Space)

Kompletna shema (vektorizovano za unutrašnje čvorove):
    C_i^{n+1} = C_i^n - Co*(C_i^n - C_{i-1}^n) + d*(C_{i+1}^n - 2C_i^n + C_{i-1}^n)

Uvjeti stabilnosti:
    Co = u*dt/dx <= 1.0  (Courantov broj)
    d  = D*dt/dx² <= 0.5 (difuzijski broj)

Granični uvjeti: zero-gradient Neumann (C_0 = C_1, C_{N-1} = C_{N-2}).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.config import DomainConfig, PollutantConfig, SimulationConfig, TransportConfig


@dataclass
class SimulationResult:
    """Kontejner izlaznih podataka jedne simulacijske replikacije.

    Attributes:
        C_history: Polje koncentracije [mg/m³], shape (n_saved, n_nodes).
        t_history: Vremena snimljenih snimaka [s], shape (n_saved,).
        C_at_interest: Koncentracija na tački interesa kroz vrijeme [mg/m³].
        velocity_history: Serija brzine rijeke [m/s], shape (n_steps,).
        t_arrival: Prvo vrijeme [s] kad koncentracija prelazi prag detekcije.
        C_peak: Maksimalna koncentracija na tački interesa [mg/m³].
        t_peak: Vrijeme nastupa pika koncentracije [s].
        arrived: Da li je zagađivač dostigao tačku interesa.
    """

    C_history: npt.NDArray[np.float64]
    t_history: npt.NDArray[np.float64]
    C_at_interest: npt.NDArray[np.float64]
    velocity_history: npt.NDArray[np.float64]
    t_arrival: float
    C_peak: float
    t_peak: float
    arrived: bool


class ADESolver:
    """Eksplicitni solver konačnih razlika za 1D ADE s varijabilnom brzinom.

    Koristi kombinovanu upwind-FTCS shemu. Vremenski korak dt mora
    zadovoljiti oba uvjeta stabilnosti:
        d  = D*dt/dx² <= 0.5
        Co = u*dt/dx  <= 1.0  (za sve u <= u_max)

    Args:
        domain: Konfiguracija prostornog domena.
        transport: Parametri transporta (koeficijent disperzije).
        pollutant: Početni uvjeti ispuštanja zagađivača.
        sim_config: Numerički parametri simulacije.
    """

    def __init__(
        self,
        domain: DomainConfig,
        transport: TransportConfig,
        pollutant: PollutantConfig,
        sim_config: SimulationConfig,
    ) -> None:
        self._domain = domain
        self._transport = transport
        self._pollutant = pollutant
        self._sim = sim_config

        self._dx: float = domain.dx
        self._dt: float = sim_config.dt
        self._D: float = transport.D
        self._xi: int = domain.x_interest_index
        self._threshold: float = sim_config.detection_threshold
        self._r: float = self._D * self._dt / self._dx**2

        self._validate_stability()

    def _validate_stability(self) -> None:
        """Provjera difuzijskog uvjeta stabilnosti d = D*dt/dx² <= 0.5.

        Raises:
            ValueError: Ako je difuzijski broj van stabilnog opsega.
        """
        if self._r > 0.5:  # diffusion stability
            raise ValueError(
                f"Difuzijska nestabilnost: d = {self._r:.4f} > 0.5. "
                f"Smanjite dt={self._dt} ili povećajte dx={self._dx:.1f}."
            )

    def _build_initial_condition(self) -> npt.NDArray[np.float64]:
        """Konstruisanje početnog Gaussovog profila koncentracije.

        Returns:
            Početna koncentracija [mg/m³], shape (n_nodes,).
        """
        x = self._domain.x_grid
        x0 = self._pollutant.release_position
        sigma = self._pollutant.release_width
        A = self._pollutant.cross_section_area
        mass_mg = self._pollutant.mass * 1e6

        return (mass_mg / (A * sigma * np.sqrt(2.0 * np.pi))) * np.exp(
            -0.5 * ((x - x0) / sigma) ** 2
        )

    def solve(self, velocities: npt.NDArray[np.float64]) -> SimulationResult:
        """Integracija ADE s podrazumijevanim Gaussovim početnim uvjetom.

        Args:
            velocities: Brzina rijeke [m/s] za svaki vremenski korak,
                shape (n_steps,).

        Returns:
            Kompletni rezultati simulacije.
        """
        return self.solve_from(self._build_initial_condition(), velocities)

    def solve_from(
        self,
        C_init: npt.NDArray[np.float64],
        velocities: npt.NDArray[np.float64],
    ) -> SimulationResult:
        """Integracija ADE naprijed u vremenu iz proizvoljnog početnog polja.

        Args:
            C_init: Početno polje koncentracije [mg/m³], shape (n_nodes,).
            velocities: Brzina rijeke [m/s] za svaki vremenski korak.

        Returns:
            Kompletni rezultati simulacije.
        """
        n_steps: int = self._sim.n_steps
        n_nodes: int = self._domain.n_nodes
        save_every: int = self._sim.save_every
        xi: int = self._xi
        r: float = self._r
        dx: float = self._dx
        dt: float = self._dt

        C = C_init.copy()

        n_saved: int = n_steps // save_every + 1
        C_history: npt.NDArray[np.float64] = np.empty(
            (n_saved, n_nodes), dtype=np.float64
        )
        t_history: npt.NDArray[np.float64] = np.empty(n_saved, dtype=np.float64)
        C_at_xi: npt.NDArray[np.float64] = np.empty(n_saved, dtype=np.float64)

        C_history[0] = C
        t_history[0] = 0.0
        C_at_xi[0] = C[xi]

        save_idx: int = 1
        t_arrival: float = np.nan
        C_peak: float = 0.0
        t_peak: float = np.nan
        arrived: bool = False

        for step in range(n_steps):
            u = velocities[step]
            co = u * dt / dx

            C_new: npt.NDArray[np.float64] = np.empty_like(C)
            C_new[1:-1] = (
                C[1:-1]
                - co * (C[1:-1] - C[:-2])
                + r * (C[2:] - 2.0 * C[1:-1] + C[:-2])
            )
            C_new[0] = C_new[1]
            C_new[-1] = C_new[-2]
            np.maximum(C_new, 0.0, out=C_new)
            C = C_new

            t_current = (step + 1) * dt
            c_xi = C[xi]

            if not arrived and c_xi > self._threshold:
                t_arrival = t_current
                arrived = True

            if c_xi > C_peak:
                C_peak = c_xi
                t_peak = t_current

            if (step + 1) % save_every == 0 and save_idx < n_saved:
                C_history[save_idx] = C
                t_history[save_idx] = t_current
                C_at_xi[save_idx] = c_xi
                save_idx += 1

        return SimulationResult(
            C_history=C_history[:save_idx],
            t_history=t_history[:save_idx],
            C_at_interest=C_at_xi[:save_idx],
            velocity_history=velocities,
            t_arrival=t_arrival,
            C_peak=C_peak,
            t_peak=t_peak,
            arrived=arrived,
        )

    def check_courant(self, u_max: float) -> None:
        """Warn if the Courant number exceeds 1 for the given peak velocity."""
        co = u_max * self._dt / self._dx
        if co > 1.0:
            import warnings
            warnings.warn(
                f"Courant number Co={co:.3f} > 1.0 at u_max={u_max:.2f} m/s. "
                "Consider reducing dt for numerical accuracy.",
                stacklevel=2,
            )
