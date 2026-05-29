"""optimizer.py - Optimizacija scenarija korištenjem metamodela.

Koristi diferencijalnu evoluciju (scipy.optimize.differential_evolution) za
globalnu optimizaciju nad parametarskim prostorom.

Ciljevi optimizacije:
- Najgori scenarij (min t_arrival): najraniji mogući dolazak zagađivača.
- Najgori scenarij (max C_peak): maksimalna moguća pik koncentracija.
- What-if analiza: grid sweep dva parametra uz fiksne ostale.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.optimize as opt

from src.config import DomainConfig, FlowConfig, PollutantConfig, SimulationConfig, TransportConfig
from src.logger import get_logger
from src.metamodel import PollutantMetamodel
from src.simulator import RiverSimulation

logger = get_logger(__name__)


@dataclass
class OptimizationResult:
    """Ishod jedne optimizacijske procedure.

    Attributes:
        objective: Opis cilja optimizacije.
        optimal_params: Optimalne vrijednosti parametara.
        optimal_value: Vrijednost funkcije cilja (surrogate predikcija).
        verified_value: Egzaktna vrijednost dobivena pravom simulacijom.
        surrogate_error_pct: Greška surrogate u odnosu na egzaktnu vrijednost [%].
        success: Da li je algoritam konvergirao.
        message: Poruka optimizatora o statusu konvergencije.
        n_evaluations: Ukupan broj evaluacija funkcije cilja.
    """

    objective: str
    optimal_params: dict[str, float]
    optimal_value: float
    verified_value: float
    surrogate_error_pct: float
    success: bool
    message: str
    n_evaluations: int


class ScenarioOptimizer:
    """Optimizator scenarija baziran na metamodelu.

    Koristi globalni algoritam diferencijalne evolucije (DE) koji je
    robustan za višemodalne, nehomogene prostorne funkcije cilja.

    Args:
        metamodel: Treniran metamodel (mora biti pozvan train() prije).
        seed: Seed za reproducibilnost.
    """

    def __init__(self, metamodel: PollutantMetamodel, seed: int = 42) -> None:
        self._meta = metamodel
        self._bounds = list(metamodel.PARAM_BOUNDS.values())
        self._param_names = list(metamodel.PARAM_BOUNDS.keys())
        self._seed = seed

    def _verify_with_simulation(
        self,
        params: dict[str, float],
        domain: DomainConfig,
        sim_config: SimulationConfig,
    ) -> tuple[float, float]:
        """Verifikacija optimalnih parametara egzaktnom determinističkom simulacijom.

        Args:
            params: Optimalni parametri iz DE optimizacije.
            domain: Konfiguracija prostornog domena.
            sim_config: Numerički parametri simulacije.

        Returns:
            tuple (t_arrival_exact [s], C_peak_exact [mg/m³]).
        """
        flow = FlowConfig(
            u_base=float(params["u_base"]),
            sigma_u=float(params["sigma_u"]),
        )
        transport = TransportConfig(D=float(params["D"]))
        pollutant = PollutantConfig(mass=float(params["mass"]))
        rng = np.random.default_rng(self._seed)
        sim = RiverSimulation(domain, flow, transport, pollutant, sim_config, rng)
        result = sim.run_deterministic()
        return result.t_arrival, result.C_peak

    def _wrap_objective(
        self,
        predictor: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        sign: float = 1.0,
    ) -> Callable[[npt.NDArray[np.float64]], float]:
        """Omotavač predikcijskog poziva u skalarnu funkciju cilja.

        Args:
            predictor: Metamodel prediktor koji prima array shape (1, n_features).
            sign: 1.0 za minimizaciju, -1.0 za maksimizaciju.

        Returns:
            Skalarna funkcija cilja za scipy.optimize.
        """
        def objective(x: npt.NDArray[np.float64]) -> float:
            """Skalarna funkcija cilja za scipy.optimize.differential_evolution."""
            return sign * float(predictor(x.reshape(1, -1))[0])

        return objective

    def find_worst_case_arrival(
        self,
        domain: DomainConfig | None = None,
        sim_config: SimulationConfig | None = None,
    ) -> OptimizationResult:
        """Nalaženje parametara koji minimiziraju t_arrival (najraniji dolazak).

        Args:
            domain: Prostorni domen. Ako proslijeđen, koristi egzaktnu simulaciju.
            sim_config: Numerički parametri. Ako proslijeđen, koristi egzaktnu simulaciju.

        Returns:
            Rezultat optimizacije s egzaktnom vrijednošću.
        """
        if domain is not None and sim_config is not None:
            logger.info("Optimizacija: minimizacija t_arrival (egzaktna ADE simulacija)...")

            def obj_exact(x: npt.NDArray[np.float64]) -> float:
                params_x = dict(zip(self._param_names, x))
                t_val, _ = self._verify_with_simulation(params_x, domain, sim_config)
                return float(t_val)

            de_result = opt.differential_evolution(
                obj_exact,
                bounds=self._bounds,
                seed=42,
                maxiter=500,
                tol=1e-4,
                popsize=10,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,
                workers=1,
            )
            params = dict(zip(self._param_names, de_result.x))
            exact_val = float(de_result.fun)
            logger.info(f"Egzaktni minimum: t_arrival = {exact_val/3600:.2f} h")
            return OptimizationResult(
                objective="Minimizacija t_arrival — egzaktna ADE simulacija",
                optimal_params=params,
                optimal_value=exact_val,
                verified_value=exact_val,
                surrogate_error_pct=0.0,
                success=bool(de_result.success),
                message=str(de_result.message),
                n_evaluations=int(de_result.nfev),
            )
        else:
            logger.info("Optimizacija: minimizacija t_arrival (surrogate)...")
            obj = self._wrap_objective(self._meta.predict_t_arrival, sign=1.0)
            de_result = opt.differential_evolution(
                obj, bounds=self._bounds, seed=42, maxiter=1000, tol=1e-6,
                popsize=15, mutation=(0.5, 1.0), recombination=0.7,
                polish=True, workers=1,
            )
            params = dict(zip(self._param_names, de_result.x))
            return OptimizationResult(
                objective="Minimizacija t_arrival — surrogate model",
                optimal_params=params,
                optimal_value=float(de_result.fun),
                verified_value=float(de_result.fun),
                surrogate_error_pct=0.0,
                success=bool(de_result.success),
                message=str(de_result.message),
                n_evaluations=int(de_result.nfev),
            )

    def find_worst_case_peak(
        self,
        domain: DomainConfig | None = None,
        sim_config: SimulationConfig | None = None,
    ) -> OptimizationResult:
        """Nalaženje parametara koji maksimiziraju C_peak (najveća koncentracija).

        Args:
            domain: Prostorni domen za verifikacijsku simulaciju.
            sim_config: Numerički parametri za verifikacijsku simulaciju.

        Returns:
            Rezultat optimizacije s optimalnim parametrima i verificiranom vrijednošću.
        """
        logger.info("Optimizacija: maksimizacija C_peak (najgori slučaj)...")
        obj = self._wrap_objective(self._meta.predict_C_peak, sign=-1.0)

        de_result = opt.differential_evolution(
            obj,
            bounds=self._bounds,
            seed=42,
            maxiter=1000,
            tol=1e-6,
            popsize=15,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True,
            workers=1,
        )

        params = dict(zip(self._param_names, de_result.x))
        surrogate_val = float(-de_result.fun)

        if domain is not None and sim_config is not None:
            _, c_exact = self._verify_with_simulation(params, domain, sim_config)
            err_pct = abs(surrogate_val - c_exact) / c_exact * 100
            logger.info(
                f"Verifikacija: surrogate={surrogate_val:.1f} mg/m³, "
                f"egzaktno={c_exact:.1f} mg/m³, greška={err_pct:.1f}%"
            )
        else:
            c_exact = surrogate_val
            err_pct = 0.0

        return OptimizationResult(
            objective="Maksimizacija C_peak (najveća pik koncentracija)",
            optimal_params=params,
            optimal_value=surrogate_val,
            verified_value=c_exact,
            surrogate_error_pct=err_pct,
            success=bool(de_result.success),
            message=str(de_result.message),
            n_evaluations=int(de_result.nfev),
        )

    def what_if_grid(
        self,
        param_x: str,
        param_y: str,
        base_params: dict[str, float],
        n_grid: int = 35,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Sweep dva parametra po mreži uz fiksne ostale (what-if analiza).

        Args:
            param_x: Naziv parametra na x-osi.
            param_y: Naziv parametra na y-osi.
            base_params: Fiksne vrijednosti preostalih parametara.
            n_grid: Rezolucija mreže po svakoj osi.

        Returns:
            Tuple (grid_x, grid_y, Z_t_arrival [h], Z_C_peak [mg/m³]).
        """
        bounds = self._meta.PARAM_BOUNDS
        x_vals: npt.NDArray[np.float64] = np.linspace(*bounds[param_x], n_grid)
        y_vals: npt.NDArray[np.float64] = np.linspace(*bounds[param_y], n_grid)
        grid_x, grid_y = np.meshgrid(x_vals, y_vals)

        n_points = n_grid * n_grid
        X_sweep: npt.NDArray[np.float64] = np.empty(
            (n_points, len(self._param_names))
        )

        for col, name in enumerate(self._param_names):
            if name == param_x:
                X_sweep[:, col] = grid_x.ravel()
            elif name == param_y:
                X_sweep[:, col] = grid_y.ravel()
            else:
                lo, hi = bounds[name]
                X_sweep[:, col] = base_params.get(name, (lo + hi) / 2.0)

        Z_t = self._meta.predict_t_arrival(X_sweep).reshape(n_grid, n_grid)
        Z_c = self._meta.predict_C_peak(X_sweep).reshape(n_grid, n_grid)

        return grid_x, grid_y, Z_t, Z_c

    def what_if_grid_exact(
        self,
        param_x: str,
        param_y: str,
        base_params: dict[str, float],
        domain: DomainConfig,
        sim_config: SimulationConfig,
        n_grid: int = 30,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Egzaktni what-if sweep koristeći pravu determinističku ADE simulaciju.

        Args:
            param_x: Naziv parametra na x-osi.
            param_y: Naziv parametra na y-osi.
            base_params: Fiksne nominalne vrijednosti preostalih parametara.
            domain: Konfiguracija prostornog domena rijeke.
            sim_config: Numerički parametri simulacije.
            n_grid: Rezolucija mreže po svakoj osi.

        Returns:
            Tuple (grid_x, grid_y, Z_t_arrival [s], Z_C_peak [mg/m³]).
        """
        bounds = self._meta.PARAM_BOUNDS

        x_vals: npt.NDArray[np.float64] = np.linspace(*bounds[param_x], n_grid)
        y_vals: npt.NDArray[np.float64] = np.linspace(*bounds[param_y], n_grid)
        grid_x, grid_y = np.meshgrid(x_vals, y_vals)

        Z_t = np.full((n_grid, n_grid), np.nan, dtype=np.float64)
        Z_c = np.full((n_grid, n_grid), np.nan, dtype=np.float64)

        n_total = n_grid * n_grid
        logger.info(
            f"Egzaktni what-if grid: {param_x} x {param_y}, "
            f"{n_grid}x{n_grid}={n_total} determinističkih simulacija..."
        )

        u_base_fix  = base_params.get("u_base",  0.50)
        sigma_u_fix = base_params.get("sigma_u", 0.10)
        D_fix       = base_params.get("D",       50.0)
        mass_fix    = base_params.get("mass",  1000.0)

        for j, y_val in enumerate(y_vals):
            for i, x_val in enumerate(x_vals):
                u_base  = x_val if param_x == "u_base"  else (y_val if param_y == "u_base"  else u_base_fix)
                sigma_u = x_val if param_x == "sigma_u" else (y_val if param_y == "sigma_u" else sigma_u_fix)
                D       = x_val if param_x == "D"       else (y_val if param_y == "D"       else D_fix)
                mass    = x_val if param_x == "mass"    else (y_val if param_y == "mass"    else mass_fix)

                flow      = FlowConfig(u_base=float(u_base), sigma_u=float(sigma_u))
                transport = TransportConfig(D=float(D))
                pollutant = PollutantConfig(mass=float(mass))

                rng = np.random.default_rng(self._seed)
                sim = RiverSimulation(domain, flow, transport, pollutant, sim_config, rng)
                result = sim.run_deterministic()

                if result.arrived:
                    Z_t[j, i] = result.t_arrival
                    Z_c[j, i] = result.C_peak

            if (j + 1) % 5 == 0 or j == n_grid - 1:
                logger.info(f"  Red {j+1}/{n_grid} završen.")

        n_arrived = int(np.sum(~np.isnan(Z_t)))
        logger.info(
            f"Egzaktni grid završen: {n_arrived}/{n_total} tačaka "
            f"gdje zagađivač stiže do vodovoda."
        )

        return grid_x, grid_y, Z_t, Z_c
