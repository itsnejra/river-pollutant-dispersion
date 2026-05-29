"""plotter.py — Sve vizualizacije projekta.

Generisane figure (images/):
1.  spacetime_{scenario}        — x-t heatmap + profil na vodovodu
2.  scenario_comparison         — poređenje 3 scenarija
3.  snapshots_{scenario}        — profili koncentracije u 6 trenutaka
4.  velocity_series             — stohastičke serije brzina (OU)
5.  replication_convergence     — konvergencija CI po N replikacija
5b. warmup_welch_{scenario}     — Welchov warmup grafik
6.  mc_distributions            — histogrami t_arrival i C_peak
7.  mc_confidence_bands         — mean ± 95% CI pojasevi
8.  metamodel_validation        — predicted vs simulated scatter
9.  sensitivity_analysis        — OAT spider + tornado dijagram
10. whatif_{px}_vs_{py}         — konturni dijagrami what-if analize

Svaka figura se sprema kao .pdf, .svg i .png u images/ folder.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.colors import LogNorm
from scipy.stats import t as t_dist

from src.sensitivity import OATSensitivityAnalyzer, SensitivityResult
from src.config import CLIMATE_SCENARIOS, ClimateScenario, DomainConfig
from src.advection_diffusion import SimulationResult
from src.metamodel import MetamodelResult
from src.replication_manager import ReplicationSummary

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("images")


def _setup_style() -> None:
    """Postavljanje konzistentnog stila svih figura."""
    matplotlib.use("Agg")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
    })


def save_current_figure(stem: str, out_dir: Path = OUTPUT_DIR) -> Path:
    """Sprema trenutnu figuru kao .pdf, .svg i .png u out_dir.

    Args:
        stem: Osnovno ime fajla bez ekstenzije.
        out_dir: Direktorij za snimanje (default: images/).

    Returns:
        Putanja do .pdf verzije figure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".svg", ".png"):
        plt.savefig(out_dir / f"{stem}{ext}", bbox_inches="tight")
    return out_dir / f"{stem}.pdf"


class SimulationPlotter:
    """Centralizovani objekt za kreiranje svih vizualizacija projekta.

    Sve metode snimaju figure u output_dir kao .pdf, .svg i .png.

    Args:
        domain: Prostorni domen (za x_grid i x_interest).
        output_dir: Direktorij za snimanje. Default: images/.
    """

    def __init__(
        self,
        domain: DomainConfig,
        output_dir: Path = OUTPUT_DIR,
    ) -> None:
        self._domain = domain
        self._out = output_dir
        self._out.mkdir(parents=True, exist_ok=True)
        _setup_style()

    def _save(self, fig: plt.Figure, stem: str) -> Path:
        """Sprema figuru kao .pdf, .svg i .png u izlazni direktorij.

        Args:
            fig: Matplotlib figura za snimanje.
            stem: Osnovno ime fajla bez ekstenzije.

        Returns:
            Putanja do .pdf verzije.
        """
        for ext in (".pdf", ".svg", ".png"):
            fig.savefig(self._out / f"{stem}{ext}", bbox_inches="tight")
        logger.info(f"  [ok] {stem}.pdf / .svg / .png")
        return self._out / f"{stem}.pdf"

    def plot_spacetime(
        self,
        result: SimulationResult,
        scenario_key: str,
        scenario: ClimateScenario,
    ) -> Path:
        """Figura 1: Heatmap koncentracije u prostoru i vremenu + profil na vodovodu.

        Args:
            result: Rezultati jedne simulacijske replikacije.
            scenario_key: Ključ scenarija (npr. 'dry', 'normal', 'rainy').
            scenario: Klimatski scenario s metapodacima.

        Returns:
            Putanja do snimljene .pdf figure.
        """
        x_km = self._domain.x_grid / 1e3
        t_h = result.t_history / 3600.0
        x_int_km = self._domain.x_interest / 1e3

        fig = plt.figure(figsize=(13, 9))
        gs = gridspec.GridSpec(
            2, 2, figure=fig,
            height_ratios=[2.5, 1],
            width_ratios=[1, 0.03],
            hspace=0.35, wspace=0.05,
        )
        ax_heat = fig.add_subplot(gs[0, 0])
        ax_cbar = fig.add_subplot(gs[0, 1])
        ax_prof = fig.add_subplot(gs[1, 0])

        C = result.C_history
        C_clipped = np.where(C > 0, C, np.nan)
        vmin = np.nanpercentile(C_clipped, 2)
        vmax = np.nanpercentile(C_clipped, 99.5)
        if np.isnan(vmin) or vmin <= 0:
            vmin = 1e-4

        im = ax_heat.pcolormesh(
            x_km, t_h, C,
            cmap="plasma",
            norm=LogNorm(vmin=max(vmin, 1e-5), vmax=max(vmax, 1e-4)),
            shading="auto",
        )
        plt.colorbar(im, cax=ax_cbar, label="Koncentracija [mg/m³]")
        ax_heat.axvline(x_int_km, color="white", ls="--", lw=1.8,
                        label=f"Vodovod (x={x_int_km:.0f} km)")
        if result.arrived:
            ax_heat.axhline(result.t_arrival / 3600, color="#00FF88",
                            ls=":", lw=1.5, label=f"t_arr={result.t_arrival/3600:.1f} h")
        ax_heat.set_xlabel("Pozicija duž rijeke [km]")
        ax_heat.set_ylabel("Vrijeme [h]")
        ax_heat.set_title(
            f"Prostorno-vremenski profil koncentracije\n{scenario.name}",
            fontweight="bold",
        )
        ax_heat.legend(loc="upper left", fontsize=9, framealpha=0.7)

        ax_prof.plot(t_h, result.C_at_interest, color=scenario.color, lw=2)
        ax_prof.fill_between(t_h, result.C_at_interest, alpha=0.2, color=scenario.color)
        if result.arrived:
            ax_prof.axvline(result.t_arrival / 3600, color="red", ls="--",
                            lw=1.5, label=f"Dolazak: {result.t_arrival/3600:.1f} h")
            ax_prof.axhline(result.C_peak, color="purple", ls=":",
                            lw=1.5, label=f"Pik: {result.C_peak:.3f} mg/m³")
        ax_prof.set_xlabel("Vrijeme [h]")
        ax_prof.set_ylabel("C [mg/m³]")
        ax_prof.set_title(f"Koncentracija na vodovodu (x={x_int_km:.0f} km)")
        ax_prof.legend(fontsize=9)

        fig.suptitle(
            f"Širenje zagađivača — {scenario.name}",
            fontsize=14, fontweight="bold", y=1.01,
        )

        path = self._save(fig, f"spacetime_{scenario_key}")
        plt.close(fig)
        return path

    def plot_scenario_comparison(
        self,
        results: dict[str, SimulationResult],
    ) -> Path:
        """Figura 2: Poređenje determinističkih profila tri klimatska scenarija.

        Args:
            results: Rječnik {scenario_key: SimulationResult}.

        Returns:
            Putanja do snimljene .pdf figure.
        """
        x_km = self._domain.x_grid / 1e3
        x_int_km = self._domain.x_interest / 1e3

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_t, ax_x = axes

        for key, res in results.items():
            sc = CLIMATE_SCENARIOS[key]
            t_h = res.t_history / 3600.0
            ax_t.plot(t_h, res.C_at_interest, color=sc.color, lw=2.2, label=sc.name)
            if res.arrived:
                ax_t.axvline(res.t_arrival / 3600, color=sc.color, ls="--", lw=1.2, alpha=0.6)
            ax_x.plot(x_km, res.C_history[-1], color=sc.color, lw=2.2, label=sc.name)

        ax_t.set_xlabel("Vrijeme [h]")
        ax_t.set_ylabel("Koncentracija [mg/m³]")
        ax_t.set_title("Koncentracija na gradskom vodovodu C(t)", fontweight="bold")
        ax_t.legend(framealpha=0.85)
        ax_t.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

        ax_x.axvline(x_int_km, color="gray", ls=":", lw=1.5, label="Vodovod")
        ax_x.set_xlabel("Pozicija [km]")
        ax_x.set_ylabel("Koncentracija [mg/m³]")
        ax_x.set_title("Finalni prostorni profil C(x, t=72 h)", fontweight="bold")
        ax_x.legend(framealpha=0.85)

        fig.suptitle(
            "Poređenje klimatskih scenarija — deterministička analiza",
            fontsize=14, fontweight="bold",
        )
        fig.tight_layout()

        path = self._save(fig, "scenario_comparison")
        plt.close(fig)
        return path

    def plot_snapshots(
        self,
        result: SimulationResult,
        scenario_key: str,
        scenario: ClimateScenario,
        n_snaps: int = 6,
    ) -> Path:
        """Figura 3: Prostorni profil C(x) u n_snaps vremenskih tačaka.

        Args:
            result: Rezultati simulacije.
            scenario_key: Ključ scenarija.
            scenario: Klimatski scenario s metapodacima.
            n_snaps: Broj snimaka (default: 6).

        Returns:
            Putanja do snimljene .pdf figure.
        """
        n_saved = len(result.t_history)
        indices = np.linspace(0, n_saved - 1, n_snaps, dtype=int)
        x_km = self._domain.x_grid / 1e3
        x_int_km = self._domain.x_interest / 1e3
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_snaps))

        fig, ax = plt.subplots(figsize=(11, 6))

        for snap_i, color in zip(indices, colors):
            t_h = result.t_history[snap_i] / 3600.0
            ax.plot(x_km, result.C_history[snap_i], color=color,
                    lw=2.0, label=f"t = {t_h:.1f} h")

        ax.axvline(x_int_km, color="red", ls="--", lw=1.5,
                   label=f"Vodovod ({x_int_km:.0f} km)")
        ax.set_xlabel("Pozicija duž rijeke [km]")
        ax.set_ylabel("Koncentracija [mg/m³]")
        ax.set_title(f"Evolucija prostornog profila — {scenario.name}", fontweight="bold")
        ax.legend(loc="upper right", fontsize=9, ncol=2)
        ax.set_xlim(0, self._domain.length / 1e3)

        fig.tight_layout()
        path = self._save(fig, f"snapshots_{scenario_key}")
        plt.close(fig)
        return path

    def plot_velocity_series(
        self,
        results: dict[str, SimulationResult],
        dt: float = 300.0,
    ) -> Path:
        """Figura 4: OU stohastičke serije brzine za sva tri scenarija.

        Args:
            results: Rječnik {scenario_key: SimulationResult}.
            dt: Vremenski korak [s].

        Returns:
            Putanja do snimljene .pdf figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax_ts, ax_hist = axes

        for key, res in results.items():
            sc = CLIMATE_SCENARIOS[key]
            v = res.velocity_history
            t_h = np.arange(len(v)) * dt / 3600.0
            mask = t_h <= 24
            ax_ts.plot(t_h[mask], v[mask], color=sc.color, lw=0.8, alpha=0.85, label=sc.name)
            ax_hist.hist(v, bins=50, density=True, alpha=0.4,
                         color=sc.color, label=sc.name, edgecolor="none")

        ax_ts.set_xlabel("Vrijeme [h]")
        ax_ts.set_ylabel("Brzina toka [m/s]")
        ax_ts.set_title("Stohastička serija brzine rijeke U(t)\n(prvih 24 h)", fontweight="bold")
        ax_ts.legend(fontsize=9)

        ax_hist.set_xlabel("Brzina toka [m/s]")
        ax_hist.set_ylabel("Gustoća vjerovatnoće")
        ax_hist.set_title("Stacionarna distribucija brzine\n(Ornstein-Uhlenbeck)", fontweight="bold")
        ax_hist.legend(fontsize=9)

        fig.suptitle("Stohastički model brzine rijeke (OU proces)", fontsize=13, fontweight="bold")
        fig.tight_layout()

        path = self._save(fig, "velocity_series")
        plt.close(fig)
        return path

    def plot_replication_convergence(
        self,
        rel_widths: list[float],
        n_required: int,
        target_width: float = 0.05,
        min_n: int = 10,
    ) -> Path:
        """Figura 5: Konvergencija relativne greške 95% CI po broju replikacija.

        Args:
            rel_widths: Lista relativnih grešaka h/mean.
            n_required: Ukupan broj replikacija pri konvergenciji.
            target_width: Ciljana relativna greška gamma (default: 5%).
            min_n: Minimalni N pri prvom mjerenju.

        Returns:
            Putanja do snimljene .pdf figure.
        """
        x_osa = np.arange(min_n, min_n + len(rel_widths))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            x_osa, [w * 100 for w in rel_widths],
            color="#2563EB", lw=2.2, marker="o", ms=4,
            label="Relativna greška h / |X̄| [%]",
        )
        ax.axhline(
            target_width * 100, color="red", ls="--", lw=2.0,
            label=f"Ciljana granica (γ = {target_width*100:.0f}%)",
        )
        if n_required <= x_osa[-1]:
            ax.axvline(
                n_required, color="green", ls=":", lw=2.0,
                label=f"Konvergencija: N = {n_required}",
            )

        ax.set_xlabel("Broj replikacija N", fontsize=12)
        ax.set_ylabel("Relativna polu-širina 95% CI [%]", fontsize=12)
        ax.set_title(
            "Sekvencijalno uzorkovanje — određivanje broja replikacija\n"
            "while petlja: dodaje replikacije dok h/|X̄| > γ = 5%",
            fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle=":", alpha=0.7)

        fig.tight_layout()
        path = self._save(fig, "replication_convergence")
        plt.close(fig)
        return path

    def plot_warmup_welch(
        self,
        u_series: npt.NDArray[np.float64],
        dt: float,
        n_warmup_steps: int,
        scenario: ClimateScenario,
        window: int = 72,
    ) -> Path:
        """Figura 5b: Welchov moving-average grafik za detekciju warmup perioda.

        Args:
            u_series: Cijela OU serija (warmup + simulacija) [m/s].
            dt: Vremenski korak [s].
            n_warmup_steps: Broj warmup koraka koji se odbacuju.
            scenario: Klimatski scenario s metapodacima.
            window: Širina prozora W za pokretni prosjek (broj koraka).

        Returns:
            Putanja do snimljene .pdf figure.
        """
        t_h = np.arange(len(u_series)) * dt / 3600.0
        df_u = pd.Series(u_series)
        u_ma = df_u.rolling(window=window, min_periods=1).mean().values

        t_warmup_h = n_warmup_steps * dt / 3600.0
        u_mean = scenario.u_base
        u_std = scenario.sigma_u

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(t_h, u_series, color=scenario.color, alpha=0.18, lw=0.7,
                label="OU serija (sirovi podaci)")
        ax.plot(t_h, u_ma, color=scenario.color, lw=2.4,
                label=f"Pokretni prosjek (w={window} koraka = {window*dt/3600:.1f} h)")
        ax.axvline(t_warmup_h, color="red", ls="--", lw=2.0,
                   label=f"Warmup granica = {t_warmup_h:.1f} h (3τ)")
        ax.axhline(u_mean, color="black", ls="-", lw=1.5, alpha=0.7,
                   label=f"μ = {u_mean:.3f} m/s")
        ax.axhspan(u_mean - u_std, u_mean + u_std, alpha=0.10,
                   color="gray", label=f"μ ± σ = [{u_mean-u_std:.2f}, {u_mean+u_std:.2f}] m/s")

        ax.set_xlabel("Simulacijsko vrijeme [h]")
        ax.set_ylabel("Brzina toka u(t) [m/s]")
        ax.set_title(
            f"Welchova metoda detekcije warmup perioda — {scenario.name}\n"
            f"OU proces: μ={u_mean:.3f} m/s, σ={u_std:.3f} m/s, τ=1 h  |  "
            f"Warmup = 3τ = {t_warmup_h:.1f} h",
            fontweight="bold",
        )
        ax.legend(fontsize=9, loc="upper right")
        ax.set_ylim(bottom=0)

        fig.tight_layout()
        path = self._save(fig, f"warmup_welch_{scenario.name.split()[0].lower()}")
        plt.close(fig)
        return path

    def plot_mc_distributions(
        self,
        summaries: dict[str, ReplicationSummary],
    ) -> Path:
        """Figura 6: Histogrami t_arrival i C_peak za sve scenarije.

        Args:
            summaries: Rječnik {key: ReplicationSummary}.

        Returns:
            Putanja do snimljene .pdf figure.
        """
        scenario_keys = list(summaries.keys())
        n_sc = len(scenario_keys)

        fig, axes = plt.subplots(2, n_sc, figsize=(5 * n_sc, 9))
        if n_sc == 1:
            axes = axes.reshape(2, 1)

        for j, key in enumerate(scenario_keys):
            sc = CLIMATE_SCENARIOS[key]
            s = summaries[key]

            ax = axes[0, j]
            t_h = s.t_arrivals / 3600.0
            ax.hist(t_h, bins=20, color=sc.color, alpha=0.75,
                    edgecolor="white", linewidth=0.5)
            ax.axvline(t_h.mean(), color="black", lw=2, ls="-",
                       label=f"μ = {t_h.mean():.2f} h")
            ci_lo, ci_hi = s.t_arrival_ci_95
            ax.axvspan(ci_lo / 3600, ci_hi / 3600, alpha=0.15, color="black",
                       label="95% CI")
            ax.set_xlabel("Vrijeme dolaska [h]")
            ax.set_ylabel("Frekvencija")
            ax.set_title(f"{sc.name}\nt_arrival distribucija", fontweight="bold")
            ax.legend(fontsize=8)

            ax = axes[1, j]
            ax.hist(s.C_peaks, bins=20, color=sc.color, alpha=0.75,
                    edgecolor="white", linewidth=0.5)
            ax.axvline(s.C_peaks.mean(), color="black", lw=2, ls="-",
                       label=f"μ = {s.C_peaks.mean():.4f}")
            ci_lo_c, ci_hi_c = s.C_peak_ci_95
            ax.axvspan(ci_lo_c, ci_hi_c, alpha=0.15, color="black", label="95% CI")
            ax.set_xlabel("Pik koncentracija [mg/m³]")
            ax.set_ylabel("Frekvencija")
            ax.set_title(f"{sc.name}\nC_peak distribucija", fontweight="bold")
            ax.legend(fontsize=8)

        fig.suptitle(
            "Monte Carlo distribucije izlaznih veličina\n"
            f"(N = {list(summaries.values())[0].n_replications} replikacija po scenariju)",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()

        path = self._save(fig, "mc_distributions")
        plt.close(fig)
        return path

    def plot_mc_confidence_bands(
        self,
        summaries: dict[str, ReplicationSummary],
    ) -> Path:
        """Figura 7: Mean +/- 95% CI pojasevi koncentracije C(t) na vodovodu.

        Args:
            summaries: Rječnik {key: ReplicationSummary}.

        Returns:
            Putanja do snimljene .pdf figure.
        """
        fig, axes = plt.subplots(1, len(summaries), figsize=(6 * len(summaries), 6),
                                 sharey=False)
        if len(summaries) == 1:
            axes = [axes]

        for ax, (key, s) in zip(axes, summaries.items()):
            sc = CLIMATE_SCENARIOS[key]
            arrived_results = [r for r in s.results if r.arrived]
            if not arrived_results:
                continue

            t_h = arrived_results[0].t_history / 3600.0
            C_mat = np.array([r.C_at_interest for r in arrived_results])

            mean_c = C_mat.mean(axis=0)
            std_c = C_mat.std(axis=0, ddof=1)
            t_crit = t_dist.ppf(0.975, df=len(arrived_results) - 1)
            se = std_c / np.sqrt(len(arrived_results))

            ax.plot(t_h, mean_c, color=sc.color, lw=2.5, label="Srednja vrijednost")
            ax.fill_between(
                t_h,
                mean_c - t_crit * se,
                mean_c + t_crit * se,
                alpha=0.25, color=sc.color, label="95% CI pojasevi",
            )

            n_show = min(10, len(arrived_results))
            rng = np.random.default_rng(0)
            idxs = rng.choice(len(arrived_results), n_show, replace=False)
            for idx in idxs:
                ax.plot(t_h, C_mat[idx], color=sc.color, lw=0.6, alpha=0.25)

            ax.set_xlabel("Vrijeme [h]")
            ax.set_ylabel("Koncentracija [mg/m³]")
            ax.set_title(f"{sc.name}\n(N={len(arrived_results)} replikacija)", fontweight="bold")
            ax.legend(fontsize=9)

        fig.suptitle(
            "Monte Carlo — pojasevi pouzdanosti koncentracije na vodovodu",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()

        path = self._save(fig, "mc_confidence_bands")
        plt.close(fig)
        return path

    def plot_metamodel_validation(
        self,
        meta_results: dict[str, dict[str, MetamodelResult]],
    ) -> Path:
        """Figura 8: Predicted vs. Simulated scatter plot za sve modele i targete.

        Args:
            meta_results: Ugniježđeni rječnik results[model_name][target] = MetamodelResult.

        Returns:
            Putanja do snimljene .pdf figure.
        """
        model_names = list(meta_results.keys())
        targets = ["t_arrival", "C_peak"]
        target_labels = {
            "t_arrival": "Vrijeme dolaska [h]",
            "C_peak": "Pik koncentracija [mg/m³]",
        }
        n_models = len(model_names)
        n_targets = len(targets)

        fig, axes = plt.subplots(n_targets, n_models,
                                 figsize=(5 * n_models, 5 * n_targets))
        if n_models == 1:
            axes = axes.reshape(n_targets, 1)

        colors_m = ["#2563EB", "#059669", "#D97706", "#7C3AED"]

        for row, target in enumerate(targets):
            for col, (mname, color) in enumerate(zip(model_names, colors_m)):
                ax = axes[row, col]
                mr = meta_results[mname][target]

                y_true = mr.y_true.copy()
                y_pred = mr.y_pred.copy()

                if target == "t_arrival":
                    y_true = np.expm1(np.clip(y_true, 0, None)) / 3600.0
                    y_pred = np.expm1(np.clip(y_pred, 0, None)) / 3600.0
                elif target == "C_peak":
                    y_true = np.expm1(np.clip(y_true, 0, None))
                    y_pred = np.expm1(np.clip(y_pred, 0, None))

                vmin = min(y_true.min(), y_pred.min())
                vmax = max(y_true.max(), y_pred.max())
                margin = (vmax - vmin) * 0.05

                ax.scatter(y_true, y_pred, color=color, s=45, alpha=0.75,
                           edgecolors="white", linewidths=0.4)
                ax.plot([vmin - margin, vmax + margin],
                        [vmin - margin, vmax + margin],
                        "k--", lw=1.5, label="y = x (idealan)")

                ax.set_xlim(vmin - margin, vmax + margin)
                ax.set_ylim(vmin - margin, vmax + margin)
                ax.set_xlabel(f"Simulirano — {target_labels[target]}")
                ax.set_ylabel(f"Predviđeno — {target_labels[target]}")
                ax.set_title(
                    f"{mname}\n"
                    f"R²_CV={mr.r2_cv:.3f}  MAE_CV={mr.mae_cv:.2e}\n"
                    f"R²_test={mr.r2_test:.3f}  MAE_test={mr.mae_test:.2e}",
                    fontsize=9, fontweight="bold",
                )
                ax.legend(fontsize=8)
                ax.set_aspect("equal", adjustable="box")

        fig.suptitle(
            "Validacija surrogate (metamodel) modela\n"
            "Predicted vs. Simulated — hold-out test skup",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()

        path = self._save(fig, "metamodel_validation")
        plt.close(fig)
        return path

    def plot_sensitivity_analysis(
        self,
        sensitivity_results: list[SensitivityResult],
        analyzer: OATSensitivityAnalyzer,
    ) -> Path:
        """Figura 9: OAT spider plot + tornado dijagram.

        Args:
            sensitivity_results: Lista rezultata OAT analize.
            analyzer: OAT analizator s baseline vrijednostima.

        Returns:
            Putanja do snimljene .pdf figure.
        """
        PARAM_LABELS = OATSensitivityAnalyzer.PARAM_LABELS

        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        ax_spider, ax_tornado = axes

        colors_s = ["#2563EB", "#D97706", "#059669", "#7C3AED"]
        t0 = analyzer.baseline_t_arrival / 3600.0

        for sr, color in zip(sensitivity_results, colors_s):
            t_vals_h = sr.t_arrival_values / 3600.0
            norm_x = np.linspace(0, 1, len(sr.param_values))
            ax_spider.plot(norm_x, t_vals_h, color=color, lw=2.2,
                           label=PARAM_LABELS.get(sr.param_name, sr.param_name))

        ax_spider.axhline(t0, color="gray", ls="--", lw=1.5,
                          label=f"Baseline = {t0:.1f} h")
        ax_spider.set_xlabel("Normalizovana vrijednost parametra [0=min, 1=max]")
        ax_spider.set_ylabel("Predviđeno t_arrival [h]")
        ax_spider.set_title("OAT Spider plot — Osjetljivost t_arrival", fontweight="bold")
        ax_spider.legend(fontsize=8, loc="best")

        params_ordered = [PARAM_LABELS.get(r.param_name, r.param_name)
                          for r in sensitivity_results]
        s_t = [r.t_arrival_sensitivity for r in sensitivity_results]
        s_c = [r.C_peak_sensitivity for r in sensitivity_results]

        y_pos = np.arange(len(params_ordered))
        bar_h = 0.35
        ax_tornado.barh(y_pos + bar_h / 2, s_t, bar_h,
                        color="#2563EB", alpha=0.8, label="S(t_arrival)")
        ax_tornado.barh(y_pos - bar_h / 2, s_c, bar_h,
                        color="#D97706", alpha=0.8, label="S(C_peak)")
        ax_tornado.set_yticks(y_pos)
        ax_tornado.set_yticklabels(params_ordered, fontsize=9)
        ax_tornado.axvline(0, color="black", lw=0.8)
        ax_tornado.set_xlabel("Normalizovani indeks osjetljivosti S_i")
        ax_tornado.set_title("Tornado dijagram — Ranking parametara", fontweight="bold")
        ax_tornado.legend(fontsize=9)
        ax_tornado.invert_yaxis()

        fig.suptitle("OAT Analiza osjetljivosti metamodela", fontsize=13, fontweight="bold")
        fig.tight_layout()

        path = self._save(fig, "sensitivity_analysis")
        plt.close(fig)
        return path

    def plot_whatif_contours(
        self,
        grid_x: npt.NDArray[np.float64],
        grid_y: npt.NDArray[np.float64],
        Z_t: npt.NDArray[np.float64],
        Z_c: npt.NDArray[np.float64],
        param_x: str,
        param_y: str,
        param_units: dict[str, str] | None = None,
    ) -> Path:
        """Figura 10: Konturni dijagram what-if analize (sweep dva parametra).

        Args:
            grid_x: Meshgrid koordinate x-osi.
            grid_y: Meshgrid koordinate y-osi.
            Z_t: Predikcije t_arrival na mreži.
            Z_c: Predikcije C_peak na mreži.
            param_x: Naziv parametra na x-osi.
            param_y: Naziv parametra na y-osi.
            param_units: Opcionalni rječnik {param: jedinica}.

        Returns:
            Putanja do snimljene .pdf figure.
        """
        units = param_units or {
            "u_base": "m/s", "sigma_u": "m/s",
            "D": "m²/s", "mass": "kg",
        }
        LABELS = OATSensitivityAnalyzer.PARAM_LABELS

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_t, ax_c = axes

        Z_t_h = Z_t / 3600.0
        cf1 = ax_t.contourf(grid_x, grid_y, Z_t_h, levels=25, cmap="RdYlGn_r")
        ct1 = ax_t.contour(grid_x, grid_y, Z_t_h, levels=10, colors="black",
                            linewidths=0.5, alpha=0.5)
        ax_t.clabel(ct1, fmt="%.1f h", fontsize=7)
        plt.colorbar(cf1, ax=ax_t, label="t_arrival [h]")
        ax_t.set_xlabel(f"{LABELS.get(param_x, param_x)} [{units.get(param_x, '')}]")
        ax_t.set_ylabel(f"{LABELS.get(param_y, param_y)} [{units.get(param_y, '')}]")
        ax_t.set_title("Predviđeno vrijeme dolaska zagađivača\nt_arrival [h]", fontweight="bold")

        cf2 = ax_c.contourf(grid_x, grid_y, Z_c, levels=25, cmap="YlOrRd")
        ct2 = ax_c.contour(grid_x, grid_y, Z_c, levels=10, colors="black",
                            linewidths=0.5, alpha=0.5)
        ax_c.clabel(ct2, fmt="%.3f", fontsize=7)
        plt.colorbar(cf2, ax=ax_c, label="C_peak [mg/m³]")
        ax_c.set_xlabel(f"{LABELS.get(param_x, param_x)} [{units.get(param_x, '')}]")
        ax_c.set_ylabel(f"{LABELS.get(param_y, param_y)} [{units.get(param_y, '')}]")
        ax_c.set_title("Predviđena maksimalna pik koncentracija\nC_peak [mg/m³]", fontweight="bold")

        fig.suptitle(
            f"What-If analiza: {LABELS.get(param_x, param_x)} x "
            f"{LABELS.get(param_y, param_y)}",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()

        path = self._save(fig, f"whatif_{param_x}_vs_{param_y}")
        plt.close(fig)
        return path
