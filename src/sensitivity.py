"""sensitivity.py - Analiza osjetljivosti (One-At-a-Time, OAT).

Varijacijom jednog parametra u datom rasponu uz fiksne ostale
procjenjujemo normalizovani indeks osjetljivosti:

    S_i = (deltaY / Y0) / (delta_x_i / x_i0)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.metamodel import PollutantMetamodel

logger = logging.getLogger(__name__)


@dataclass
class SensitivityResult:
    """Rezultat OAT analize osjetljivosti za jedan parametar.

    Attributes:
        param_name: Naziv parametra.
        param_values: Niz vrijednosti parametra u opsegu sweep-a.
        t_arrival_values: Predikcije t_arrival za svaku vrijednost parametra.
        C_peak_values: Predikcije C_peak za svaku vrijednost parametra.
        t_arrival_sensitivity: Normalizovani indeks osjetljivosti za t_arrival.
        C_peak_sensitivity: Normalizovani indeks osjetljivosti za C_peak.
        t_arrival_range: (min, max) t_arrival u opsegu perturbacije.
        C_peak_range: (min, max) C_peak u opsegu perturbacije.
    """

    param_name: str
    param_values: npt.NDArray[np.float64]
    t_arrival_values: npt.NDArray[np.float64]
    C_peak_values: npt.NDArray[np.float64]
    t_arrival_sensitivity: float
    C_peak_sensitivity: float
    t_arrival_range: tuple[float, float]
    C_peak_range: tuple[float, float]


class OATSensitivityAnalyzer:
    """One-At-a-Time (OAT) analiza osjetljivosti metamodela.

    Args:
        metamodel: Treniran surrogate model.
        nominal_params: Nominalne (bazne) vrijednosti svih parametara.
        n_sweep: Broj evaluacijskih tačaka po parametru (default: 51).
    """

    PARAM_LABELS: dict[str, str] = {
        "u_base": "Srednja brzina toka u [m/s]",
        "sigma_u": "Volatilnost brzine sigma [m/s]",
        "D": "Koeficijent disperzije D [m²/s]",
        "mass": "Ispuštena masa M0 [kg]",
    }

    def __init__(
        self,
        metamodel: PollutantMetamodel,
        nominal_params: dict[str, float],
        n_sweep: int = 51,
    ) -> None:
        self._meta = metamodel
        self._nominal = nominal_params
        self._n_sweep = n_sweep
        self._param_names = list(metamodel.PARAM_BOUNDS.keys())

        x_nom: npt.NDArray[np.float64] = np.array(
            [[nominal_params[p] for p in self._param_names]]
        )
        self._t0: float = float(self._meta.predict_t_arrival(x_nom)[0])
        self._c0: float = float(self._meta.predict_C_peak(x_nom)[0])

    @property
    def baseline_t_arrival(self) -> float:
        """Baseline t_arrival [s] u nominalnoj tački."""
        return self._t0

    @property
    def baseline_C_peak(self) -> float:
        """Baseline C_peak [mg/m³] u nominalnoj tački."""
        return self._c0

    def analyze(self) -> list[SensitivityResult]:
        """Pokretanje OAT sweep-a za sve parametre.

        Returns:
            Lista SensitivityResult, sortirana po apsolutnoj osjetljivosti
            na t_arrival (opadajuće).
        """
        results: list[SensitivityResult] = []

        for param in self._param_names:
            lo, hi = self._meta.PARAM_BOUNDS[param]
            sweep_values: npt.NDArray[np.float64] = np.linspace(lo, hi, self._n_sweep)

            X_sweep: npt.NDArray[np.float64] = np.tile(
                [self._nominal[p] for p in self._param_names],
                (self._n_sweep, 1),
            )
            col_idx = self._param_names.index(param)
            X_sweep[:, col_idx] = sweep_values

            t_vals: npt.NDArray[np.float64] = self._meta.predict_t_arrival(X_sweep)
            c_vals: npt.NDArray[np.float64] = self._meta.predict_C_peak(X_sweep)

            x0 = self._nominal[param]
            dx = hi - lo

            t_range_half = (t_vals.max() - t_vals.min()) / 2.0
            c_range_half = (c_vals.max() - c_vals.min()) / 2.0

            if self._t0 != 0 and dx != 0 and x0 != 0:
                s_t = (t_range_half / self._t0) / (dx / 2.0 / x0)
            else:
                s_t = 0.0

            if self._c0 != 0 and dx != 0 and x0 != 0:
                s_c = (c_range_half / self._c0) / (dx / 2.0 / x0)
            else:
                s_c = 0.0

            results.append(SensitivityResult(
                param_name=param,
                param_values=sweep_values,
                t_arrival_values=t_vals,
                C_peak_values=c_vals,
                t_arrival_sensitivity=float(s_t),
                C_peak_sensitivity=float(s_c),
                t_arrival_range=(float(t_vals.min()), float(t_vals.max())),
                C_peak_range=(float(c_vals.min()), float(c_vals.max())),
            ))

        results.sort(key=lambda r: abs(r.t_arrival_sensitivity), reverse=True)
        return results

    def print_summary(self, results: list[SensitivityResult]) -> None:
        """Ispis normalizovanih indeksa osjetljivosti na konzolu.

        Args:
            results: Lista rezultata OAT analize.
        """
        logger.info("\n" + "=" * 65)
        logger.info("  OAT ANALIZA OSJETLJIVOSTI — NORMALIZOVANI INDEKSI")
        logger.info("=" * 65)
        nominal_fmt = {k: round(v, 4) for k, v in self._nominal.items()}
        logger.info(f"  Bazna tačka: {nominal_fmt}")
        logger.info(f"  Baseline t_arrival = {self._t0/3600:.2f} h")
        logger.info(f"  Baseline C_peak    = {self._c0:.4f} mg/m³")
        logger.info("-" * 65)
        logger.info(f"  {'Parametar':<35} {'S(t_arr)':>10} {'S(C_peak)':>10}")
        logger.info("-" * 65)
        for r in results:
            label = self.PARAM_LABELS.get(r.param_name, r.param_name)
            logger.info(
                f"  {label:<35} {r.t_arrival_sensitivity:>10.4f} "
                f"{r.C_peak_sensitivity:>10.4f}"
            )
        logger.info("=" * 65 + "\n")
