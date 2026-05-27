"""statistics.py - Statistička analiza Monte Carlo rezultata."""

from __future__ import annotations

import logging
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as stats

from src.replication_manager import ReplicationSummary

logger = logging.getLogger(__name__)


def shapiro_wilk_test(
    data: npt.NDArray[np.float64],
    alpha: float = 0.05,
) -> dict[str, float | bool | str]:
    """Shapiro-Wilk test normalnosti distribucije.

    Args:
        data: Niz mjerenja.
        alpha: Nivo značajnosti (default: 0.05).

    Returns:
        Rječnik s ključevima: statistic, p_value, is_normal, interpretation.
    """
    stat, p_val = stats.shapiro(data)
    is_normal = bool(p_val > alpha)
    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "is_normal": is_normal,
        "interpretation": (
            f"Distribucija je {'normalna' if is_normal else 'nije normalna'} "
            f"na nivou značajnosti alpha={alpha} (p={p_val:.4f})"
        ),
    }


def descriptive_stats(
    data: npt.NDArray[np.float64],
    label: str = "",
) -> pd.Series:
    """Računanje kompletnih deskriptivnih statistika.

    Args:
        data: Sirovi podaci.
        label: Naziv serije.

    Returns:
        Statistike: n, mean, std, cv, min, q25, median, q75, max, skewness, kurtosis.
    """
    n = len(data)
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1))
    return pd.Series(
        {
            "n": n,
            "mean": mean,
            "std": std,
            "cv_%": 100.0 * std / mean if mean != 0 else np.nan,
            "min": float(np.min(data)),
            "q25": float(np.percentile(data, 25)),
            "median": float(np.median(data)),
            "q75": float(np.percentile(data, 75)),
            "max": float(np.max(data)),
            "skewness": float(stats.skew(data)),
            "kurtosis": float(stats.kurtosis(data)),
        },
        name=label,
    )


def build_summary_table(
    summaries: dict[str, ReplicationSummary],
    time_unit: str = "h",
) -> pd.DataFrame:
    """Kreiranje sažetne tabele poređenja Monte Carlo rezultata svih scenarija.

    Args:
        summaries: Rječnik {ime_scenarija: ReplicationSummary}.
        time_unit: Jedinica za prikaz vremena ('h' ili 's').

    Returns:
        Tabela s jednom vrstom po scenariju.
    """
    factor = 1 / 3600.0 if time_unit == "h" else 1.0
    rows = []
    for key, s in summaries.items():
        ci_lo = s.t_arrival_ci_95[0] * factor
        ci_hi = s.t_arrival_ci_95[1] * factor
        sw = shapiro_wilk_test(s.t_arrivals)
        rows.append({
            "Scenario": key,
            f"t_arrival_mean [{time_unit}]": round(s.t_arrival_mean * factor, 2),
            f"t_arrival_std [{time_unit}]": round(s.t_arrival_std * factor, 2),
            f"95% CI [{time_unit}]": f"[{ci_lo:.2f}, {ci_hi:.2f}]",
            "CI_rel_width_%": round(s.t_arrival_ci_relative_width * 100, 2),
            "C_peak_mean [mg/m³]": round(s.C_peak_mean, 4),
            "C_peak_std [mg/m³]": round(s.C_peak_std, 4),
            "Arrival_prob": round(s.arrival_probability, 3),
            "Shapiro_p": round(float(sw["p_value"]), 4),
            "Normal_dist": sw["is_normal"],
        })
    return pd.DataFrame(rows).set_index("Scenario")


def print_mc_summary(
    summaries: dict[str, ReplicationSummary],
    scenario_names: dict[str, str] | None = None,
) -> None:
    """Ispis formatiranog sažetka Monte Carlo analize na konzolu.

    Args:
        summaries: Rječnik {ključ_scenarija: ReplicationSummary}.
        scenario_names: Opcionalna mapa ključeva na duge nazive.
    """
    logger.info("\n" + "=" * 70)
    logger.info("  MONTE CARLO STATISTIČKI SAŽETAK")
    logger.info("=" * 70)
    df = build_summary_table(summaries, time_unit="h")
    logger.info("\n" + df.to_string())
    logger.info("=" * 70 + "\n")
