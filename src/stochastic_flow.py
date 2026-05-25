"""stochastic_flow.py - Stohastički model brzine toka rijeke.

Implementira Ornstein-Uhlenbeck (OU) proces za modeliranje vremenski
varijabilne brzine rijeke, uključujući efekte kiša i promjene protoka.

Matematički model:
    dU(t) = theta*(mu - U(t)) dt + sigma dW(t)

Koristi se egzaktna diskretizacija (ne Euler-Maruyama):
    U(t+dt) | U(t) ~ N(m(t), v)
    m(t) = mu + (U(t) - mu) * exp(-theta*dt)
    v    = sigma² * (1 - exp(-2*theta*dt)) / (2*theta)

Period zagrijavanja: >= 3*tau (tri vremena relaksacije) za stacionarnu distribuciju.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.signal import lfilter

from src.config import FlowConfig, SimulationConfig


class OrnsteinUhlenbeckProcess:
    """Egzaktna simulacija Ornstein-Uhlenbeck stohastičkog procesa.

    Koristi egzaktnu (ne Euler-Maruyama) diskretizaciju koja čuva
    statistička svojstva OU procesa za bilo koji vremenski korak.

    Args:
        flow_config: Parametri dugoročne srednje, brzine povratka i volatilnosti.
        sim_config: Vremenski korak i kontrolni parametri simulacije.
        rng: Generator slučajnih brojeva (za reproducibilnost).
    """

    def __init__(
        self,
        flow_config: FlowConfig,
        sim_config: SimulationConfig,
        rng: np.random.Generator,
    ) -> None:
        self._mu = flow_config.u_base
        self._theta = flow_config.theta
        self._sigma = flow_config.sigma_u * np.sqrt(2.0 * self._theta)
        self._u_min = flow_config.u_min
        self._u_max = flow_config.u_max
        self._dt = sim_config.dt
        self._rng = rng

        self._exp_decay: float = float(np.exp(-self._theta * self._dt))
        variance_exact = (
            self._sigma**2
            * (1.0 - np.exp(-2.0 * self._theta * self._dt))
            / (2.0 * self._theta)
        )
        self._transition_std: float = float(np.sqrt(variance_exact))

    @property
    def stationary_mean(self) -> float:
        """Dugoročna srednja vrijednost stacionarne distribucije [m/s]."""
        return self._mu

    @property
    def stationary_std(self) -> float:
        """Standardna devijacija stacionarne distribucije [m/s]."""
        return self._sigma / np.sqrt(2.0 * self._theta)

    @property
    def relaxation_time(self) -> float:
        """Vremenska konstanta relaksacije tau = 1/theta [s]."""
        return 1.0 / self._theta

    def generate(
        self,
        n_steps: int,
        u0: float | None = None,
    ) -> npt.NDArray[np.float64]:
        """Generisanje vremenske serije brzine dužine n_steps koraka.

        Implementacija je potpuno vektorizovana korištenjem
        scipy.signal.lfilter (C-implementiran IIR filter).

        Args:
            n_steps: Broj vremenskih koraka.
            u0: Početna brzina. Podrazumijevano: stacionarna srednja vrijednost.

        Returns:
            Niz brzina [m/s] za svaki vremenski korak, shape (n_steps,).
        """
        u0_val: float = self._mu if u0 is None else float(u0)
        noise: npt.NDArray[np.float64] = (
            self._rng.standard_normal(n_steps) * self._transition_std
        )

        b: float = self._exp_decay
        zi: npt.NDArray[np.float64] = np.array([b * (u0_val - self._mu)])
        Y, _ = lfilter([1.0], [1.0, -b], noise, zi=zi)

        return np.clip(Y + self._mu, self._u_min, self._u_max)

    def generate_with_warmup(
        self,
        n_steps: int,
        n_warmup: int,
    ) -> npt.NDArray[np.float64]:
        """Generisanje post-warmup serije brzine, odbacujući prijelaznu fazu.

        Args:
            n_steps: Željeni broj koraka u izlaznoj seriji (nakon warmup-a).
            n_warmup: Broj warmup koraka koji se odbacuju (preporuka: >= 3*tau/dt).

        Returns:
            Serija brzine u stacionarnom stanju, shape (n_steps,).
        """
        total = self.generate(n_steps + n_warmup)
        return total[n_warmup:]
