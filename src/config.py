"""config.py - Centralizirana konfiguracija simulacije širenja zagađivača u rijeci.

Sve fizikalne konstante, numerički parametri i klimatski scenariji definirani
su isključivo ovdje. Ostatak koda importuje iz ovog modula.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class DomainConfig:
    """Konfiguracija 1D prostornog domena rijeke.

    Attributes:
        length: Ukupna dužina rijeke [m].
        n_nodes: Broj prostornih čvorova mreže.
        x_interest: Lokacija gradskog vodovoda [m].
    """

    length: float = 50_000.0
    n_nodes: int = 101
    x_interest: float = 40_000.0

    @property
    def dx(self) -> float:
        """Prostorni korak [m]."""
        return self.length / (self.n_nodes - 1)

    @property
    def x_grid(self) -> npt.NDArray[np.float64]:
        """Uniformna prostorna mreža [m]."""
        return np.linspace(0.0, self.length, self.n_nodes)

    @property
    def x_interest_index(self) -> int:
        """Indeks mrežnog čvora najbliži tački interesa."""
        return int(round(self.x_interest / self.dx))


@dataclass
class FlowConfig:
    """Parametri stohastičke brzine toka rijeke (Ornstein-Uhlenbeck proces).

    Model:
        dU(t) = theta(mu - U(t)) dt + sigma dW(t)

    gdje je theta brzina povratka ka srednoj vrijednosti, mu dugoročna srednja
    brzina, a sigma volatilnost (intenzitet slučajnih fluktuacija zbog kiša).

    Attributes:
        u_base: Dugoročna srednja brzina (mu) [m/s].
        theta: Brzina povratka ka sredini (theta) [1/s].
        sigma_u: Volatilnost OU procesa (sigma) [m/s].
        u_min: Fizikalna donja granica brzine [m/s].
        u_max: Fizikalna gornja granica (poplavni vrh Bosne) [m/s].
    """

    u_base: float = 0.50
    theta: float = 1 / 3600
    sigma_u: float = 0.10
    u_min: float = 0.05
    u_max: float = 3.00


@dataclass
class TransportConfig:
    """Parametri transporta zagađivača u rijeci.

    Attributes:
        D: Longitudinalni koeficijent disperzije [m²/s].
    """

    D: float = 50.0


@dataclass
class PollutantConfig:
    """Početni uvjeti ispuštanja hemijskog zagađivača.

    Zagađivač se modelira kao Gaussov puls mase M0 ispušten u tački x0.
    Početni profil koncentracije:
        C(x, 0) = M0 / (A * sigma0 * sqrt(2*pi)) * exp(-(x - x0)^2 / (2*sigma0^2))

    Attributes:
        mass: Ukupna ispuštena masa [kg].
        release_position: Pozicija ispuštanja [m].
        release_width: Standardna devijacija početnog Gaussovog pulsa [m].
        river_width: Prosječna širina rijeke [m].
        river_depth: Prosječna dubina rijeke [m].
    """

    mass: float = 1_000.0
    release_position: float = 5_000.0
    release_width: float = 1_500.0
    river_width: float = 50.0
    river_depth: float = 1.16

    @property
    def cross_section_area(self) -> float:
        """Površina poprečnog presjeka rijeke [m²]."""
        return self.river_width * self.river_depth


@dataclass
class SimulationConfig:
    """Numerički i statistički parametri kontrole simulacije.

    Attributes:
        t_total: Ukupno trajanje simulacije [s].
        dt: Vremenski korak (delta-t) [s].
        n_replications: Broj Monte Carlo replikacija.
        warmup_time: Period zagrijavanja OU procesa [s].
        random_seed: Bazni seed za reproducibilnost.
        save_every: Čuvati polje svakih N koraka.
        detection_threshold: Prag detekcije dolaska [mg/m³].
        n_metamodel_samples: LHS tačke za trening metamodela.
    """

    t_total: float = 3 * 86_400.0
    dt: float = 150.0
    n_replications: int = 100
    warmup_time: float = 10_800.0
    random_seed: int = 42
    save_every: int = 6
    detection_threshold: float = 1e-3
    n_metamodel_samples: int = 400

    @property
    def n_steps(self) -> int:
        """Ukupan broj vremenskih koraka."""
        return int(self.t_total / self.dt)

    @property
    def n_warmup_steps(self) -> int:
        """Broj koraka perioda zagrijavanja."""
        return int(self.warmup_time / self.dt)


@dataclass
class ClimateScenario:
    """Imenovani hidrološki/klimatski scenario.

    Attributes:
        name: Naziv scenarija.
        u_base: Srednja brzina toka [m/s].
        sigma_u: Volatilnost brzine [m/s].
        D: Koeficijent disperzije [m²/s].
        color: Boja za vizualizacije (hex string).
        description: Kratak opis scenarija.
    """

    name: str
    u_base: float
    sigma_u: float
    D: float
    color: str
    description: str


CLIMATE_SCENARIOS: dict[str, ClimateScenario] = {
    "dry": ClimateScenario(
        name="Suša (Dry Season)",
        u_base=0.30,
        sigma_u=0.05,
        D=20.0,
        color="#D97706",
        description="Nizak protok, minimalna turbulencija, visoka retencija",
    ),
    "normal": ClimateScenario(
        name="Normalni uvjeti",
        u_base=0.50,
        sigma_u=0.10,
        D=50.0,
        color="#2563EB",
        description="Prosječni hidrološki režim",
    ),
    "rainy": ClimateScenario(
        name="Velika kiša / Poplava",
        u_base=0.75,
        sigma_u=0.20,
        D=80.0,
        color="#059669",
        description="Povišen protok nakon intenzivnih padavina",
    ),
}
