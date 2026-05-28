"""metamodel.py - Surrogate model za brzu predikciju ishoda simulacije.

Workflow metamodeliranja:
1. Generisanje dizajna eksperimenta metodom Latinskog hiperkuba (LHS).
2. Evaluacija determinističkog ADE solvera u svakoj tački dizajna.
3. Trening i unakrsna validacija surrogate modela (GP, RF, GB, HistGB).
4. Izvještavanje R², MAE na CV i test skupu.
5. Izlaganje najboljeg modela za optimizaciju i what-if analizu.

Prostor parametara:
    u_base  in [0.20, 1.50] m/s
    sigma_u in [0.02, 0.40] m/s
    D       in [10.0, 120.0] m²/s
    mass    in [200, 3000] kg
"""

from __future__ import annotations

import copy
import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats.qmc import LatinHypercube, scale
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    DomainConfig,
    FlowConfig,
    PollutantConfig,
    SimulationConfig,
    TransportConfig,
)
from src.advection_diffusion import ADESolver
from src.stochastic_flow import OrnsteinUhlenbeckProcess

logger = logging.getLogger(__name__)


@dataclass
class MetamodelResult:
    """Metrike evaluacije jednog treniranog surrogate modela.

    Attributes:
        name: Naziv modela.
        target: Ciljni izlaz ('t_arrival' ili 'C_peak').
        r2_cv: Unakrsno-validirani R².
        mae_cv: Unakrsno-validirani MAE.
        r2_test: R² na hold-out test skupu.
        mae_test: MAE na hold-out test skupu.
        y_true: Stvarne vrijednosti (test skup).
        y_pred: Predviđene vrijednosti (test skup).
        model: Fit pipeline objekat.
    """

    name: str
    target: str
    r2_cv: float
    mae_cv: float
    r2_test: float
    mae_test: float
    y_true: npt.NDArray[np.float64]
    y_pred: npt.NDArray[np.float64]
    model: Any


class PollutantMetamodel:
    """Surrogate model za simulator disperzije zagađivača u rijeci.

    Kombinuje LHS dizajn eksperimenta sa četiri surrogate algoritma
    (GP, RF, GB, HistGB) kako bi brzo predvidio t_arrival i C_peak za
    proizvoljne kombinacije ulaznih parametara.

    Args:
        domain: Konfiguracija prostornog domena.
        pollutant: Početni uvjeti ispuštanja zagađivača.
        sim_config: Numerički parametri simulacije.
        n_samples: Broj LHS tačaka dizajna.
        rng: Generator slučajnih brojeva.
    """

    PARAM_BOUNDS: dict[str, tuple[float, float]] = {
        "u_base": (0.20, 1.50),
        "sigma_u": (0.02, 0.40),
        "D": (10.0, 120.0),
        "mass": (200.0, 3000.0),
    }

    def __init__(
        self,
        domain: DomainConfig,
        pollutant: PollutantConfig,
        sim_config: SimulationConfig,
        n_samples: int = 200,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._domain = domain
        self._pollutant = pollutant
        self._sim = sim_config
        self._n_samples = n_samples
        self._rng = rng or np.random.default_rng(42)

        self._design_data: pd.DataFrame | None = None
        self._results: dict[str, dict[str, MetamodelResult]] = {}
        self._best_model_t: Any = None
        self._best_model_c: Any = None
        self._best_model_name: str = ""

    def generate_design(self) -> pd.DataFrame:
        """Kreiranje Latinskog hiperkuba sa n_samples tačaka.

        Returns:
            Tablica dizajna s kolonama: u_base, sigma_u, D, mass.
        """
        sampler = LatinHypercube(d=4, seed=int(self._rng.integers(int(1e6))))
        unit_sample = sampler.random(n=self._n_samples)

        lower = np.array([v[0] for v in self.PARAM_BOUNDS.values()])
        upper = np.array([v[1] for v in self.PARAM_BOUNDS.values()])
        scaled = scale(unit_sample, lower, upper)

        return pd.DataFrame(scaled, columns=list(self.PARAM_BOUNDS.keys()))

    def evaluate_design(self, design: pd.DataFrame) -> pd.DataFrame:
        """Evaluacija stohastičnog ADE solvera u svakoj tački dizajna.

        Args:
            design: Tablica LHS tačaka dizajna.

        Returns:
            Dizajn sa dodanim kolonama t_arrival i C_peak.
        """
        n = len(design)
        t_arrivals: npt.NDArray[np.float64] = np.full(n, np.nan)
        C_peaks: npt.NDArray[np.float64] = np.zeros(n)

        logger.info(f"  Evaluiranje {n} LHS tačaka dizajna...")

        design_values: npt.NDArray[np.float64] = design[
            ["u_base", "sigma_u", "D", "mass"]
        ].values

        for i in range(n):
            u_base, sigma_u, D, mass = design_values[i]

            point_rng = np.random.default_rng(i * 997 + 1)
            flow = FlowConfig(u_base=float(u_base), sigma_u=float(sigma_u))
            transport = TransportConfig(D=float(D))
            pollutant = PollutantConfig(
                mass=float(mass),
                release_position=self._pollutant.release_position,
                release_width=self._pollutant.release_width,
                river_width=self._pollutant.river_width,
                river_depth=self._pollutant.river_depth,
            )

            try:
                ou = OrnsteinUhlenbeckProcess(flow, self._sim, point_rng)
                velocities = ou.generate_with_warmup(
                    self._sim.n_steps, self._sim.n_warmup_steps
                )
                solver = ADESolver(self._domain, transport, pollutant, self._sim)
                sim_result = solver.solve(velocities)

                t_arrivals[i] = sim_result.t_arrival if sim_result.arrived else np.nan
                C_peaks[i] = sim_result.C_peak
            except ValueError:
                pass

            if (i + 1) % 50 == 0:
                logger.info(f"    Evaluirano {i+1}/{n} tačaka")

        result_df = design.copy()
        result_df["t_arrival"] = t_arrivals
        result_df["C_peak"] = C_peaks
        result_df.dropna(subset=["t_arrival"], inplace=True)
        result_df.reset_index(drop=True, inplace=True)
        logger.info(f"  Validnih tačaka: {len(result_df)}/{n}")
        return result_df

    def _build_pipelines(self) -> dict[str, Pipeline]:
        """Instanciranje pipeline-ova surrogate modela.

        Returns:
            Rječnik naziv -> sklearn Pipeline za svaki algoritam.
        """
        return {
            "Gaussian Process": Pipeline([
                ("scaler", StandardScaler()),
                ("model", GaussianProcessRegressor(
                    kernel=ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
                           + WhiteKernel(noise_level=0.1),
                    n_restarts_optimizer=5,
                    random_state=42,
                    normalize_y=True,
                )),
            ]),
            "Random Forest": Pipeline([
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(
                    n_estimators=500,
                    min_samples_leaf=1,
                    max_features=0.6,
                    max_depth=None,
                    random_state=42,
                    n_jobs=-1,
                )),
            ]),
            "Gradient Boosting": Pipeline([
                ("scaler", StandardScaler()),
                ("model", GradientBoostingRegressor(
                    n_estimators=500,
                    max_depth=5,
                    learning_rate=0.03,
                    subsample=0.8,
                    min_samples_leaf=2,
                    random_state=42,
                )),
            ]),
            "HistGradient Boosting": Pipeline([
                ("scaler", StandardScaler()),
                ("model", HistGradientBoostingRegressor(
                    max_iter=500,
                    max_depth=6,
                    learning_rate=0.03,
                    min_samples_leaf=10,
                    l2_regularization=0.1,
                    random_state=42,
                )),
            ]),
        }

    def train(self) -> dict[str, dict[str, MetamodelResult]]:
        """Generisanje podataka, trening svih surrogate modela i CV evaluacija.

        Returns:
            Ugniježđeni rječnik results[model_name][target] -> MetamodelResult.
        """
        logger.info("  Generisanje LHS dizajna...")
        self._design_data = self.generate_design()
        logger.info("  Pokretanje simulatora na tackama dizajna...")
        evaluated = self.evaluate_design(self._design_data)

        feature_cols = list(self.PARAM_BOUNDS.keys())
        X: npt.NDArray[np.float64] = evaluated[feature_cols].values
        y_t: npt.NDArray[np.float64] = evaluated["t_arrival"].values
        y_c: npt.NDArray[np.float64] = evaluated["C_peak"].values

        y_c_safe: npt.NDArray[np.float64] = np.log1p(np.clip(y_c, 0, None))
        y_t_safe: npt.NDArray[np.float64] = np.log1p(np.clip(y_t, 0, None))
        targets: dict[str, npt.NDArray[np.float64]] = {
            "t_arrival": y_t_safe,
            "C_peak": y_c_safe,
        }

        n = len(X)
        rng_split = np.random.default_rng(42)
        test_indices = rng_split.choice(n, size=n // 5, replace=False)
        test_mask: npt.NDArray[np.bool_] = np.zeros(n, dtype=bool)
        test_mask[test_indices] = True
        X_train: npt.NDArray[np.float64] = X[~test_mask]
        X_test: npt.NDArray[np.float64] = X[test_mask]

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        all_results: dict[str, dict[str, MetamodelResult]] = {}

        warnings.filterwarnings("ignore", category=UserWarning)

        for model_name, pipe_template in self._build_pipelines().items():
            all_results[model_name] = {}
            logger.info(f"  Trening '{model_name}'...")

            for target_name, y_all in targets.items():
                y_train = y_all[~test_mask]
                y_test = y_all[test_mask]

                pipe = copy.deepcopy(pipe_template)

                cv_r2 = cross_val_score(
                    pipe, X_train, y_train, cv=kf, scoring="r2"
                ).mean()
                cv_mae = -cross_val_score(
                    pipe, X_train, y_train, cv=kf,
                    scoring="neg_mean_absolute_error",
                ).mean()

                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)

                mae_test: float = float(mean_absolute_error(y_test, y_pred))
                r2_test: float = float(r2_score(y_test, y_pred))

                logger.info(
                    f"    [{target_name}] "
                    f"R2_CV={cv_r2:.4f}  MAE_CV={cv_mae:.4f}  "
                    f"R2_test={r2_test:.4f}  MAE_test={mae_test:.4f}"
                )

                all_results[model_name][target_name] = MetamodelResult(
                    name=model_name,
                    target=target_name,
                    r2_cv=float(cv_r2),
                    mae_cv=float(cv_mae),
                    r2_test=r2_test,
                    mae_test=mae_test,
                    y_true=y_test,
                    y_pred=y_pred,
                    model=pipe,
                )

        self._results = all_results

        valid_models = {
            m: np.mean([
                all_results[m]["t_arrival"].r2_cv,
                all_results[m]["C_peak"].r2_cv,
            ])
            for m in all_results
            if np.isfinite(all_results[m]["t_arrival"].r2_cv)
            and np.isfinite(all_results[m]["C_peak"].r2_cv)
        }
        self._best_model_name = (
            max(valid_models, key=lambda m: valid_models[m])
            if valid_models
            else "Random Forest"
        )
        self._best_model_t = all_results[self._best_model_name]["t_arrival"].model
        self._best_model_c = all_results[self._best_model_name]["C_peak"].model
        self._t_log_transformed = True
        logger.info(f"  Najbolji model: '{self._best_model_name}'")

        return all_results

    @property
    def results(self) -> dict[str, dict[str, MetamodelResult]]:
        """Rezultati treninga i evaluacije svih surrogate modela."""
        if not self._results:
            raise RuntimeError("Pozovite train() prije pristupa rezultatima.")
        return self._results

    @property
    def design_data(self) -> pd.DataFrame | None:
        """Evaluirani LHS dizajn (s izlaznim kolonama)."""
        return self._design_data

    @property
    def best_model_name(self) -> str:
        """Ime najboljeg surrogate modela."""
        return self._best_model_name

    def predict_t_arrival(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Predikcija vremena dolaska [s] za nove vektore parametara.

        Args:
            X: Parametarski vektori [u_base, sigma_u, D, mass], shape (n, 4).

        Returns:
            Predviđena vremena dolaska [s], shape (n,).
        """
        if self._best_model_t is None:
            raise RuntimeError("Metamodel nije treniran. Pozovite train().")
        y_log = self._best_model_t.predict(X)
        return np.expm1(y_log)

    def predict_C_peak(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Predikcija pik koncentracije [mg/m³] za nove vektore parametara.

        Args:
            X: Parametarski vektori [u_base, sigma_u, D, mass], shape (n, 4).

        Returns:
            Predviđene pik koncentracije [mg/m³], shape (n,).
        """
        if self._best_model_c is None:
            raise RuntimeError("Metamodel nije treniran. Pozovite train().")
        y_log = self._best_model_c.predict(X)
        return np.expm1(y_log)
