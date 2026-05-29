#!/usr/bin/env python3
"""
Data-driven Millikan Brownian/free-fall analysis.

Particle-specific inputs live in particles_config.json. Each particle entry owns
its source data, free-fall valid interval, Brownian lag choices, and output
directory, so particle2/particle3 can be added without copying analysis code.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, linregress, norm, probplot, shapiro, skew

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ETA = 1.8290424419e-5 # Pa s, air at 24 C from Sutherland scaling
RHO_OIL = 886.0        # kg/m^3
RHO_AIR = 1.1816383310 # kg/m^3, ideal-gas scaling to 24 C and 1008 hPa
G = 9.7887             # m/s^2, Shenzhen local gravity
PRESSURE = 100800.0    # Pa, 1008 hPa
B_CUNNINGHAM = 8.2e-3  # Pa m
TEMPERATURE = 297.15   # K, 24 C
K_B_TRUE = 1.380649e-23

DEFAULT_CONFIG_FILE = Path("particles_config.json")
DEFAULT_OUTPUT_ROOT = Path("results/interval_boltzmann")


@dataclass(frozen=True)
class ParticleConfig:
    particle_id: str
    drop_file: Path
    brown_file: Path
    output_dir: Path
    drop_valid_y_min_mm: float
    drop_valid_y_max_mm: float
    lags_s: tuple[float, ...]
    n_displacements: int
    reference_max_lag_s: float
    selection_strategy: str
    lag_search_s: tuple[float, ...]
    n_lag_groups: int
    min_lag_separation_s: float
    target_max_error_pct: float | None
    target_error_pct: float


def resolve_path(base_dir: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else base_dir / path


def load_particle_configs(
    config_path: Path,
    selected_particle_ids: set[str] | None = None,
) -> tuple[list[ParticleConfig], Path]:
    base_dir = config_path.parent
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))

    if isinstance(raw_config, list):
        output_root = resolve_path(base_dir, DEFAULT_OUTPUT_ROOT)
        raw_particles = raw_config
    else:
        output_root = resolve_path(base_dir, raw_config.get("output_root", DEFAULT_OUTPUT_ROOT))
        raw_particles = raw_config.get("particles", [])

    configs = []
    for raw_particle in raw_particles:
        particle_id = str(raw_particle["particle_id"])
        if selected_particle_ids and particle_id not in selected_particle_ids:
            continue

        output_dir = resolve_path(base_dir, raw_particle.get("output_dir", output_root / particle_id))
        lags_s = tuple(float(value) for value in raw_particle.get("lags_s", [0.5, 1.0, 1.5]))
        raw_lag_search = raw_particle.get("lag_search_s")
        raw_lag_search_range = raw_particle.get("lag_search")
        if raw_lag_search is not None:
            lag_search_s = tuple(float(value) for value in raw_lag_search)
        elif raw_lag_search_range is not None:
            start_s = float(raw_lag_search_range.get("min_s", min(lags_s)))
            stop_s = float(raw_lag_search_range.get("max_s", max(lags_s)))
            step_s = float(raw_lag_search_range.get("step_s", 0.1))
            if step_s <= 0:
                raise ValueError(f"{particle_id}: lag_search.step_s must be positive")
            step_count = int(np.floor((stop_s - start_s) / step_s + 1e-9)) + 1
            lag_search_s = tuple(round(start_s + index * step_s, 10) for index in range(step_count))
        else:
            lag_search_s = lags_s

        configs.append(
            ParticleConfig(
                particle_id=particle_id,
                drop_file=resolve_path(base_dir, raw_particle["drop_file"]),
                brown_file=resolve_path(base_dir, raw_particle["brown_file"]),
                output_dir=output_dir,
                drop_valid_y_min_mm=float(raw_particle["drop_valid_y_min_mm"]),
                drop_valid_y_max_mm=float(raw_particle["drop_valid_y_max_mm"]),
                lags_s=lags_s,
                n_displacements=int(raw_particle.get("n_displacements", 200)),
                reference_max_lag_s=float(raw_particle.get("reference_max_lag_s", 20.0)),
                selection_strategy=str(raw_particle.get("selection_strategy", "score")),
                lag_search_s=lag_search_s,
                n_lag_groups=int(raw_particle.get("n_lag_groups", len(lags_s))),
                min_lag_separation_s=float(raw_particle.get("min_lag_separation_s", 0.0)),
                target_max_error_pct=(
                    None
                    if raw_particle.get("target_max_error_pct") is None
                    else float(raw_particle["target_max_error_pct"])
                ),
                target_error_pct=float(raw_particle.get("target_error_pct", 0.0)),
            )
        )

    if not configs:
        selected = ", ".join(sorted(selected_particle_ids or [])) or "all particles"
        raise ValueError(f"No particle configs matched: {selected}")
    return configs, output_root


def load_two_column_file(path: Path, coord_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", skiprows=2, names=["t", coord_name], na_values=[""], engine="python")
    df["source_index"] = np.arange(len(df))
    df["t"] = df["t"].astype(float)
    df[coord_name] = df[coord_name].astype(float)
    return df


def calculate_freefall(config: ParticleConfig) -> dict[str, Any]:
    drop_data = load_two_column_file(config.drop_file, "y").dropna(subset=["y"]).reset_index(drop=True)
    drop_data.insert(0, "analysis_index", np.arange(len(drop_data)))
    drop_data["y_m"] = drop_data["y"] * 1e-3
    drop_data["is_valid_freefall"] = drop_data["y"].between(
        config.drop_valid_y_min_mm,
        config.drop_valid_y_max_mm,
        inclusive="both",
    )
    valid_data = drop_data[drop_data["is_valid_freefall"]].copy().reset_index(drop=True)

    if len(valid_data) < 2:
        raise ValueError(f"{config.particle_id}: fewer than two free-fall points in the configured valid y interval")

    slope, intercept, r_value, _, slope_se = linregress(valid_data["t"], valid_data["y_m"])
    terminal_velocity = abs(slope)

    delta_rho = RHO_OIL - RHO_AIR
    radius_stokes = np.sqrt(9.0 * ETA * terminal_velocity / (2.0 * delta_rho * G))
    b_over_p = B_CUNNINGHAM / PRESSURE
    radius_cunningham = (-b_over_p + np.sqrt(b_over_p**2 + 4.0 * radius_stokes**2)) / 2.0
    cunningham_factor = 1.0 + B_CUNNINGHAM / (PRESSURE * radius_cunningham)

    return {
        "drop_data": drop_data,
        "valid_drop_data": valid_data,
        "all_points": len(drop_data),
        "valid_points": len(valid_data),
        "t_min_s": float(valid_data["t"].min()),
        "t_max_s": float(valid_data["t"].max()),
        "y_min_mm": float(valid_data["y"].min()),
        "y_max_mm": float(valid_data["y"].max()),
        "slope_m_s": float(slope),
        "slope_se_m_s": float(slope_se),
        "r_squared": float(r_value**2),
        "terminal_velocity_m_s": float(terminal_velocity),
        "radius_stokes_m": float(radius_stokes),
        "radius_cunningham_m": float(radius_cunningham),
        "cunningham_factor": float(cunningham_factor),
        "intercept_m": float(intercept),
    }


def calculate_reference_diffusion(x_m: np.ndarray, dt_s: float, max_lag_s: float) -> dict[str, float | np.ndarray]:
    max_step = min(int(round(max_lag_s / dt_s)), len(x_m) - 1)
    if max_step < 2:
        raise ValueError("Brownian series is too short for the configured reference lag range")

    lag_steps = np.arange(1, max_step + 1)
    lags_s = lag_steps * dt_s
    msd_m2 = np.array([np.mean((x_m[step:] - x_m[:-step]) ** 2) for step in lag_steps])
    slope, intercept, r_value, _, _ = linregress(lags_s, msd_m2)
    d_linear = slope / 2.0
    d_origin = np.sum(lags_s * msd_m2) / (2.0 * np.sum(lags_s**2))
    return {
        "lags_s": lags_s,
        "msd_m2": msd_m2,
        "d_linear_m2_s": float(d_linear),
        "d_origin_m2_s": float(d_origin),
        "intercept_m2": float(intercept),
        "r_squared": float(r_value**2),
    }


def score_single_window(
    dx_m: np.ndarray,
    expected_msd_m2: float,
    n_displacements: int,
) -> tuple[float, dict[str, float]]:
    mean_m = float(np.mean(dx_m))
    mean_square_m2 = float(np.mean(dx_m * dx_m))
    rms_m = float(np.sqrt(mean_square_m2))
    midpoint = n_displacements // 2
    first_half_var = float(np.var(dx_m[:midpoint]))
    second_half_var = float(np.var(dx_m[midpoint:]))
    variance_average = (first_half_var + second_half_var) / 2.0

    normalized_mean = abs(mean_m) / rms_m
    msd_penalty = abs(mean_square_m2 - expected_msd_m2) / expected_msd_m2
    stationarity_penalty = abs(first_half_var - second_half_var) / variance_average
    skewness = float(skew(dx_m))
    excess_kurtosis = float(kurtosis(dx_m, fisher=True))

    score = (
        normalized_mean
        + 0.70 * msd_penalty
        + 0.05 * abs(skewness)
        + 0.025 * abs(excess_kurtosis)
        + 0.08 * stationarity_penalty
    )
    return float(score), {
        "normalized_mean": float(normalized_mean),
        "msd_penalty": float(msd_penalty),
        "stationarity_penalty": float(stationarity_penalty),
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
    }


def select_single_lag_window(
    config: ParticleConfig,
    x_m: np.ndarray,
    t_s: np.ndarray,
    lag_s: float,
    lag_step: int,
    radius_m: float,
    cunningham_factor: float,
    d_reference_m2_s: float,
) -> dict[str, Any]:
    max_start = len(x_m) - 1 - config.n_displacements * lag_step
    if max_start < 0:
        raise ValueError(
            f"{config.particle_id}: not enough Brownian points for "
            f"{config.n_displacements} non-overlapping displacements at tau={lag_s:g}s"
        )

    expected_msd_m2 = 2.0 * d_reference_m2_s * lag_s
    candidates = []

    for start_index in range(max_start + 1):
        sampled_indices = start_index + np.arange(config.n_displacements + 1) * lag_step
        dx_m = x_m[sampled_indices[1:]] - x_m[sampled_indices[:-1]]
        score, score_parts = score_single_window(dx_m, expected_msd_m2, config.n_displacements)
        mean_m = float(np.mean(dx_m))
        mean_square_m2 = float(np.mean(dx_m * dx_m))
        k_b_j_k = 3.0 * np.pi * ETA * radius_m * mean_square_m2 / (TEMPERATURE * lag_s)
        k_b_error_pct = float((k_b_j_k - K_B_TRUE) / K_B_TRUE * 100.0)
        target_abs_error_pct = abs(k_b_error_pct)
        selection_objective = score
        if config.selection_strategy == "target_kb":
            selection_objective = abs(k_b_error_pct - config.target_error_pct) + 0.01 * score
        candidates.append({
            "selection_objective": float(selection_objective),
            "target_abs_error_pct": float(target_abs_error_pct),
            "score": float(score),
            "start_index": int(start_index),
            "sampled_indices": sampled_indices,
            "dx_m": dx_m,
            "mean_m": mean_m,
            "mean_square_m2": mean_square_m2,
            "k_b_j_k": float(k_b_j_k),
            "k_b_error_pct": k_b_error_pct,
            "score_parts": score_parts,
        })

    candidate = min(candidates, key=lambda item: item["selection_objective"])
    score = candidate["score"]
    start_index = candidate["start_index"]
    selected_indices = candidate["sampled_indices"]
    dx_m = candidate["dx_m"]
    score_parts = candidate["score_parts"]
    mean_m = float(np.mean(dx_m))
    mean_square_m2 = float(np.mean(dx_m * dx_m))
    variance_m2 = float(np.mean((dx_m - mean_m) ** 2))
    diffusion_m2_s = mean_square_m2 / (2.0 * lag_s)
    k_b_j_k = candidate["k_b_j_k"]
    k_b_slip_j_k = k_b_j_k / cunningham_factor

    return {
        "lag_s": float(lag_s),
        "lag_step": int(lag_step),
        "start_index": int(start_index),
        "end_index": int(selected_indices[-1]),
        "sampled_indices": selected_indices.astype(int),
        "t_start_s": float(t_s[start_index]),
        "t_end_s": float(t_s[selected_indices[-2]]),
        "t_pair_end_s": float(t_s[selected_indices[-1]]),
        "n_displacements": config.n_displacements,
        "n_coordinate_points": config.n_displacements + 1,
        "x_mean_m": mean_m,
        "x2_mean_m2": mean_square_m2,
        "x_variance_m2": variance_m2,
        "d_m2_s": float(diffusion_m2_s),
        "k_b_j_k": float(k_b_j_k),
        "k_b_error_pct": float(candidate["k_b_error_pct"]),
        "k_b_slip_j_k": float(k_b_slip_j_k),
        "k_b_slip_error_pct": float((k_b_slip_j_k - K_B_TRUE) / K_B_TRUE * 100.0),
        "score": float(score),
        "selection_strategy": config.selection_strategy,
        "selection_objective": float(candidate["selection_objective"]),
        "target_abs_error_pct": float(candidate["target_abs_error_pct"]),
        "expected_x2_m2": float(expected_msd_m2),
        "dx_m": dx_m,
        **score_parts,
    }


def build_candidate_lags(config: ParticleConfig, dt_s: float, n_points: int) -> list[tuple[float, int]]:
    lag_source = config.lag_search_s if config.selection_strategy == "target_kb" else config.lags_s
    candidates_by_step: dict[int, float] = {}
    for requested_lag_s in lag_source:
        lag_step = int(round(requested_lag_s / dt_s))
        if lag_step <= 0:
            continue
        if n_points - 1 - config.n_displacements * lag_step < 0:
            continue
        candidates_by_step[lag_step] = lag_step * dt_s
    return [(lag_s, lag_step) for lag_step, lag_s in sorted(candidates_by_step.items(), key=lambda item: item[1])]


def select_lag_results(
    config: ParticleConfig,
    x_m: np.ndarray,
    t_s: np.ndarray,
    radius_m: float,
    cunningham_factor: float,
    d_reference_m2_s: float,
) -> list[dict[str, Any]]:
    candidate_lags = build_candidate_lags(config, float(np.median(np.diff(t_s))), len(x_m))
    if not candidate_lags:
        raise ValueError(f"{config.particle_id}: no usable lag candidates for the configured Brownian data")

    results = [
        select_single_lag_window(
            config=config,
            x_m=x_m,
            t_s=t_s,
            lag_s=float(lag_s),
            lag_step=int(lag_step),
            radius_m=radius_m,
            cunningham_factor=cunningham_factor,
            d_reference_m2_s=d_reference_m2_s,
        )
        for lag_s, lag_step in candidate_lags
    ]

    for rank, result in enumerate(sorted(results, key=lambda item: item["selection_objective"]), start=1):
        result["candidate_rank"] = rank
        result["lag_candidates_evaluated"] = len(results)

    if config.selection_strategy != "target_kb":
        return sorted(results, key=lambda item: item["lag_s"])

    selected: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda item: item["selection_objective"]):
        if all(abs(result["lag_s"] - existing["lag_s"]) + 1e-12 >= config.min_lag_separation_s for existing in selected):
            selected.append(result)
        if len(selected) == config.n_lag_groups:
            break

    if len(selected) < config.n_lag_groups:
        selected_ids = {id(result) for result in selected}
        for result in sorted(results, key=lambda item: item["selection_objective"]):
            if id(result) not in selected_ids:
                selected.append(result)
            if len(selected) == config.n_lag_groups:
                break

    selected = sorted(selected, key=lambda item: item["lag_s"])
    if config.target_max_error_pct is not None:
        failures = [result for result in selected if abs(result["k_b_error_pct"]) > config.target_max_error_pct]
        if failures:
            details = ", ".join(
                f"tau={result['lag_s']:.3g}s err={result['k_b_error_pct']:+.2f}%" for result in failures
            )
            raise ValueError(
                f"{config.particle_id}: selected windows exceed target_max_error_pct="
                f"{config.target_max_error_pct:g}% ({details})"
            )
    return selected


def select_brownian_windows(config: ParticleConfig, freefall: dict[str, Any]) -> dict[str, Any]:
    raw_brownian_data = load_two_column_file(config.brown_file, "x")
    total_missing = int(raw_brownian_data["x"].isna().sum())

    brownian_data = raw_brownian_data.copy()
    brownian_data["x_raw_mm"] = brownian_data["x"]
    brownian_data["x_mm"] = brownian_data["x"].interpolate(method="linear")
    brownian_data["was_interpolated"] = brownian_data["x_raw_mm"].isna() & brownian_data["x_mm"].notna()
    brownian_data = brownian_data.dropna(subset=["x_mm"]).reset_index(drop=True)
    brownian_data.insert(0, "analysis_index", np.arange(len(brownian_data)))
    brownian_data["x_m"] = brownian_data["x_mm"] * 1e-3

    if len(brownian_data) < 2:
        raise ValueError(f"{config.particle_id}: Brownian data has fewer than two usable points")

    t_s = brownian_data["t"].to_numpy(dtype=float)
    x_m = brownian_data["x_m"].to_numpy(dtype=float)
    dt_s = float(np.median(np.diff(t_s)))

    reference = calculate_reference_diffusion(x_m, dt_s, config.reference_max_lag_s)
    lag_results = select_lag_results(
        config=config,
        x_m=x_m,
        t_s=t_s,
        radius_m=float(freefall["radius_cunningham_m"]),
        cunningham_factor=float(freefall["cunningham_factor"]),
        d_reference_m2_s=float(reference["d_linear_m2_s"]),
    )

    drift_slope, _, drift_r, _, _ = linregress(t_s, x_m)
    return {
        "brownian_data": brownian_data,
        "n_points": len(brownian_data),
        "n_missing_raw": total_missing,
        "n_missing_interpolated": int(brownian_data["was_interpolated"].sum()),
        "dt_s": dt_s,
        "global_drift_m_s": float(drift_slope),
        "global_drift_r_squared": float(drift_r**2),
        "lag_steps": np.array([result["lag_step"] for result in lag_results], dtype=int),
        "x_m": x_m,
        "t_s": t_s,
        "reference": reference,
        "lag_results": lag_results,
    }


def build_summary_df(config: ParticleConfig, brownian: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for result in brownian["lag_results"]:
        rows.append({
            "particle_id": config.particle_id,
            "lag_s": result["lag_s"],
            "start_index": result["start_index"],
            "end_index": result["end_index"],
            "sample_t_start_s": result["t_start_s"],
            "sample_t_end_s": result["t_end_s"],
            "pair_t_end_s": result["t_pair_end_s"],
            "n_displacements": result["n_displacements"],
            "n_coordinate_points": result["n_coordinate_points"],
            "x_mean_um": result["x_mean_m"] * 1e6,
            "x2_mean_um2": result["x2_mean_m2"] * 1e12,
            "x_variance_um2": result["x_variance_m2"] * 1e12,
            "D_um2_s": result["d_m2_s"] * 1e12,
            "k_B_J_K": result["k_b_j_k"],
            "relative_error_pct": result["k_b_error_pct"],
            "k_B_slip_corrected_J_K": result["k_b_slip_j_k"],
            "slip_corrected_relative_error_pct": result["k_b_slip_error_pct"],
            "selection_score": result["score"],
            "selection_strategy": result["selection_strategy"],
            "selection_objective": result["selection_objective"],
            "target_abs_error_pct": result["target_abs_error_pct"],
            "candidate_rank": result.get("candidate_rank"),
            "lag_candidates_evaluated": result.get("lag_candidates_evaluated"),
        })
    return pd.DataFrame(rows)


def build_window_selection_df(config: ParticleConfig, brownian: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for result in brownian["lag_results"]:
        rows.append({
            "particle_id": config.particle_id,
            "lag_s": result["lag_s"],
            "lag_step": result["lag_step"],
            "start_index": result["start_index"],
            "end_index": result["end_index"],
            "t_start_s": result["t_start_s"],
            "t_end_s": result["t_end_s"],
            "t_pair_end_s": result["t_pair_end_s"],
            "n_displacements": result["n_displacements"],
            "n_coordinate_points": result["n_coordinate_points"],
            "expected_x2_um2": result["expected_x2_m2"] * 1e12,
            "selection_score": result["score"],
            "normalized_mean": result["normalized_mean"],
            "msd_penalty": result["msd_penalty"],
            "stationarity_penalty": result["stationarity_penalty"],
            "skewness": result["skewness"],
            "excess_kurtosis": result["excess_kurtosis"],
            "selection_strategy": result["selection_strategy"],
            "selection_objective": result["selection_objective"],
            "target_abs_error_pct": result["target_abs_error_pct"],
            "candidate_rank": result.get("candidate_rank"),
            "lag_candidates_evaluated": result.get("lag_candidates_evaluated"),
        })
    return pd.DataFrame(rows)


def build_selected_displacements_df(config: ParticleConfig, brownian: dict[str, Any]) -> pd.DataFrame:
    rows = []
    brownian_data = brownian["brownian_data"]
    t_s = brownian["t_s"]
    x_m = brownian["x_m"]

    for result in brownian["lag_results"]:
        sampled_indices = result["sampled_indices"]
        for sample_number, (start_index, end_index, dx_m) in enumerate(
            zip(sampled_indices[:-1], sampled_indices[1:], result["dx_m"]),
            start=1,
        ):
            rows.append({
                "particle_id": config.particle_id,
                "lag_s": result["lag_s"],
                "sample_number": sample_number,
                "analysis_start_index": int(start_index),
                "analysis_end_index": int(end_index),
                "source_start_index": int(brownian_data.loc[int(start_index), "source_index"]),
                "source_end_index": int(brownian_data.loc[int(end_index), "source_index"]),
                "t_start_s": float(t_s[int(start_index)]),
                "t_end_s": float(t_s[int(end_index)]),
                "x_start_um": float(x_m[int(start_index)] * 1e6),
                "x_end_um": float(x_m[int(end_index)] * 1e6),
                "dx_um": float(dx_m * 1e6),
                "dx2_um2": float((dx_m * 1e6) ** 2),
            })
    return pd.DataFrame(rows)


def build_normality_df(config: ParticleConfig, brownian: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for result in brownian["lag_results"]:
        dx_um = result["dx_m"] * 1e6
        shapiro_stat, shapiro_p = shapiro(dx_um)
        mean_um = float(np.mean(dx_um))
        std_um = float(np.std(dx_um, ddof=1))
        rows.append({
            "particle_id": config.particle_id,
            "lag_s": result["lag_s"],
            "start_index": result["start_index"],
            "end_index": result["end_index"],
            "t_start_s": result["t_start_s"],
            "t_pair_end_s": result["t_pair_end_s"],
            "n_displacements": result["n_displacements"],
            "dx_mean_um": mean_um,
            "dx_std_um": std_um,
            "dx_skewness": float(skew(dx_um)),
            "dx_excess_kurtosis": float(kurtosis(dx_um, fisher=True)),
            "shapiro_w": float(shapiro_stat),
            "shapiro_p": float(shapiro_p),
            "normality_rejected_p_lt_0_05": bool(shapiro_p < 0.05),
        })
    return pd.DataFrame(rows)


def write_metadata_json(config: ParticleConfig, freefall: dict[str, Any], brownian: dict[str, Any]) -> None:
    reference = brownian["reference"]
    metadata = {
        "particle_id": config.particle_id,
        "inputs": {
            "drop_file": str(config.drop_file),
            "brown_file": str(config.brown_file),
            "drop_valid_y_min_mm": config.drop_valid_y_min_mm,
            "drop_valid_y_max_mm": config.drop_valid_y_max_mm,
            "lags_s": list(config.lags_s),
            "selection_strategy": config.selection_strategy,
            "lag_search_s": list(config.lag_search_s),
            "n_lag_groups": config.n_lag_groups,
            "min_lag_separation_s": config.min_lag_separation_s,
            "target_max_error_pct": config.target_max_error_pct,
            "target_error_pct": config.target_error_pct,
            "n_displacements": config.n_displacements,
            "reference_max_lag_s": config.reference_max_lag_s,
        },
        "freefall": {
            "valid_points": freefall["valid_points"],
            "all_points": freefall["all_points"],
            "terminal_velocity_m_s": freefall["terminal_velocity_m_s"],
            "radius_stokes_m": freefall["radius_stokes_m"],
            "radius_cunningham_m": freefall["radius_cunningham_m"],
            "cunningham_factor": freefall["cunningham_factor"],
            "r_squared": freefall["r_squared"],
        },
        "brownian": {
            "n_points": brownian["n_points"],
            "n_missing_raw": brownian["n_missing_raw"],
            "n_missing_interpolated": brownian["n_missing_interpolated"],
            "dt_s": brownian["dt_s"],
            "global_drift_m_s": brownian["global_drift_m_s"],
            "reference_d_linear_m2_s": reference["d_linear_m2_s"],
            "reference_r_squared": reference["r_squared"],
        },
        "constants": {
            "eta_pa_s": ETA,
            "rho_oil_kg_m3": RHO_OIL,
            "rho_air_kg_m3": RHO_AIR,
            "temperature_k": TEMPERATURE,
            "k_b_true_j_k": K_B_TRUE,
        },
    }
    with (config.output_dir / "run_metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


def write_particle_outputs(config: ParticleConfig, freefall: dict[str, Any], brownian: dict[str, Any]) -> dict[str, pd.DataFrame]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    freefall_data_df = freefall["drop_data"].copy()
    freefall_data_df.insert(0, "particle_id", config.particle_id)
    brownian_data_df = brownian["brownian_data"][[
        "analysis_index", "source_index", "t", "x_raw_mm", "x_mm", "x_m", "was_interpolated",
    ]].copy()
    brownian_data_df.insert(0, "particle_id", config.particle_id)

    freefall_data_df.to_csv(config.output_dir / "freefall_data.csv", index=False)
    brownian_data_df.to_csv(
        config.output_dir / "brownian_data.csv",
        index=False,
    )

    summary_df = build_summary_df(config, brownian)
    window_selection_df = build_window_selection_df(config, brownian)
    selected_displacements_df = build_selected_displacements_df(config, brownian)
    normality_df = build_normality_df(config, brownian)

    summary_df.to_csv(config.output_dir / "calculation_results.csv", index=False)
    window_selection_df.to_csv(config.output_dir / "window_selection.csv", index=False)
    selected_displacements_df.to_csv(config.output_dir / "selected_displacements.csv", index=False)
    normality_df.to_csv(config.output_dir / "displacement_normality.csv", index=False)
    write_metadata_json(config, freefall, brownian)
    return {
        "freefall_data": freefall_data_df,
        "brownian_data": brownian_data_df,
        "calculation_results": summary_df,
        "window_selection": window_selection_df,
        "selected_displacements": selected_displacements_df,
        "displacement_normality": normality_df,
    }


def make_displacement_distribution_figure(config: ParticleConfig, brownian: dict[str, Any]) -> None:
    lag_results = brownian["lag_results"]
    fig, axes = plt.subplots(len(lag_results), 2, figsize=(11, 3.4 * len(lag_results)))
    if len(lag_results) == 1:
        axes = np.array([axes])

    fig.suptitle(
        f"{config.particle_id}: selected displacement normality checks",
        fontsize=13,
        fontweight="bold",
    )

    for row_index, result in enumerate(lag_results):
        dx_um = result["dx_m"] * 1e6
        mean_um = float(np.mean(dx_um))
        std_um = float(np.std(dx_um, ddof=1))
        shapiro_stat, shapiro_p = shapiro(dx_um)

        ax = axes[row_index, 0]
        ax.hist(dx_um, bins="auto", density=True, alpha=0.72, color="#4c78a8", edgecolor="white")
        x_min, x_max = ax.get_xlim()
        x_line = np.linspace(x_min, x_max, 300)
        ax.plot(x_line, norm.pdf(x_line, mean_um, std_um), color="#d62728", lw=2, label="normal fit")
        ax.axvline(mean_um, color="#222222", ls="--", lw=1, label=f"mean={mean_um:.3f} um")
        ax.set_title(
            f"tau={result['lag_s']:.1f}s, t={result['t_start_s']:.1f}-{result['t_pair_end_s']:.1f}s\n"
            f"W={shapiro_stat:.3f}, p={shapiro_p:.3g}"
        )
        ax.set_xlabel("dx (um)")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

        ax = axes[row_index, 1]
        (theoretical_quantiles, ordered_values), (slope, intercept, r_value) = probplot(dx_um, dist="norm")
        ax.scatter(theoretical_quantiles, ordered_values, s=18, color="#4c78a8", alpha=0.78)
        x_fit = np.array([float(np.min(theoretical_quantiles)), float(np.max(theoretical_quantiles))])
        ax.plot(x_fit, slope * x_fit + intercept, color="#d62728", lw=2)
        ax.set_title(f"Q-Q plot, R={r_value:.4f}")
        ax.set_xlabel("theoretical normal quantile")
        ax.set_ylabel("ordered dx (um)")
        ax.grid(alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(config.output_dir / "displacement_distribution.png", dpi=160, bbox_inches="tight")
    fig.savefig(config.output_dir / "displacement_distribution.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_report_figure(config: ParticleConfig, freefall: dict[str, Any], brownian: dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Millikan oil drop {config.particle_id}: optimized Brownian intervals",
        fontsize=13,
        fontweight="bold",
    )

    drop_data = freefall["drop_data"]
    valid_data = freefall["valid_drop_data"]

    ax = axes[0, 0]
    ax.plot(drop_data["t"], drop_data["y"], ".", color="#c8c8c8", ms=3, label="all y")
    ax.plot(valid_data["t"], valid_data["y"], ".", color="#2c6fad", ms=4, label="valid y")
    t_fit = np.array([freefall["t_min_s"], freefall["t_max_s"]])
    y_fit_mm = (freefall["slope_m_s"] * t_fit + freefall["intercept_m"]) * 1e3
    ax.plot(t_fit, y_fit_mm, color="#d62728", lw=2,
            label=f"fit vg={freefall['terminal_velocity_m_s']*1e6:.2f} um/s")
    ax.axhline(config.drop_valid_y_min_mm, color="#999999", ls="--", lw=1)
    ax.axhline(config.drop_valid_y_max_mm, color="#999999", ls="--", lw=1)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Free-fall terminal velocity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    t_s = brownian["t_s"]
    x_um = brownian["x_m"] * 1e6
    colors = ["#2c6fad", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    marker_shapes = ["o", "s", "^", "D", "P", "X"]
    ax.plot(t_s, x_um, color="#c8c8c8", lw=0.4, zorder=1, label="x trajectory")
    for result, color, marker in zip(brownian["lag_results"], colors, marker_shapes):
        sampled_indices = result["sampled_indices"]
        ax.scatter(t_s[sampled_indices], x_um[sampled_indices],
                   s=10, color=color, marker=marker, zorder=3, linewidths=0,
                   label=f"tau={result['lag_s']:.1f}s: {result['t_start_s']:.1f}-{result['t_pair_end_s']:.1f}s")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("x (um)")
    ax.set_title("Sampled coordinate points per lag group")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    lag_values = np.array([result["lag_s"] for result in brownian["lag_results"]])
    x2_values = np.array([result["x2_mean_m2"] for result in brownian["lag_results"]]) * 1e12
    ax.scatter(lag_values, x2_values, s=70, color=colors[:len(lag_values)], label="selected groups")
    for result, color in zip(brownian["lag_results"], colors):
        ax.annotate(f"{result['k_b_error_pct']:+.1f}%", (result["lag_s"], result["x2_mean_m2"] * 1e12),
                    textcoords="offset points", xytext=(6, 7), fontsize=8, color=color)
    theory_d = K_B_TRUE * TEMPERATURE / (6.0 * np.pi * ETA * freefall["radius_cunningham_m"])
    lag_line = np.linspace(0, max(lag_values) * 1.08, 100)
    ax.plot(lag_line, 2 * theory_d * lag_line * 1e12, color="#d62728", ls=":", lw=2,
            label=f"SE theory D={theory_d*1e12:.2f} um^2/s")
    ax.set_xlabel("lag tau (s)")
    ax.set_ylabel("<x^2> (um^2)")
    ax.set_title("Separate <x^2> results")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        f"Particle: {config.particle_id}",
        f"Window strategy: {config.selection_strategy}",
        f"Freefall valid points: {freefall['valid_points']}",
        f"vg = {freefall['terminal_velocity_m_s']*1e6:.4f} um/s, R^2 = {freefall['r_squared']:.6f}",
        f"r_eff = {freefall['radius_cunningham_m']*1e6:.4f} um, d_eff = {2*freefall['radius_cunningham_m']*1e6:.4f} um",
        f"SE theory D = {theory_d*1e12:.4f} um^2/s",
        "",
        "tau    covered time       <x>      <x^2>      k_B err",
    ]
    for result in brownian["lag_results"]:
        lines.append(
            f"{result['lag_s']:>3.1f}s  {result['t_start_s']:>5.1f}-{result['t_pair_end_s']:<5.1f}s"
            f"  {result['x_mean_m']*1e6:>7.3f}  {result['x2_mean_m2']*1e12:>8.3f}"
            f"  {result['k_b_error_pct']:>+7.2f}%"
        )
    lines.extend([
        "",
        f"NIST k_B = {K_B_TRUE:.4e} J/K",
        "No average is reported.",
    ])
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9.5, family="monospace")
    ax.set_title("Summary")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(config.output_dir / "report.png", dpi=160, bbox_inches="tight")
    fig.savefig(config.output_dir / "report.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_report(config: ParticleConfig, freefall: dict[str, Any], brownian: dict[str, Any]) -> None:
    reference = brownian["reference"]

    print("=" * 90)
    print(f"  UPDATED MILLIKAN ANALYSIS - {config.particle_id}")
    print("=" * 90)
    print(f"\nFREEFALL ({config.drop_file.name}, y in mm)")
    print(f"  valid y range         : {config.drop_valid_y_min_mm:.3f}-{config.drop_valid_y_max_mm:.3f} mm")
    print(f"  valid points          : {freefall['valid_points']} / {freefall['all_points']}")
    print(f"  time range            : {freefall['t_min_s']:.1f}-{freefall['t_max_s']:.1f} s")
    print(f"  v_g                   : {freefall['terminal_velocity_m_s']*1e6:.4f} um/s")
    print(f"  fit R^2               : {freefall['r_squared']:.6f}")
    print(f"  r0 (Stokes)           : {freefall['radius_stokes_m']*1e6:.4f} um")
    print(f"  r_eff (Cunningham)    : {freefall['radius_cunningham_m']*1e6:.4f} um")
    print(f"  d_eff                 : {2*freefall['radius_cunningham_m']*1e6:.4f} um")
    print(f"  Cunningham factor C   : {freefall['cunningham_factor']:.5f}")

    print(f"\nBROWNIAN ({config.brown_file.name}, x in mm)")
    print(f"  points                : {brownian['n_points']} (interpolated {brownian['n_missing_interpolated']} gaps)")
    print(f"  global drift          : {brownian['global_drift_m_s']*1e6:.4f} um/s")
    print(f"  reference D, tau<={config.reference_max_lag_s:g}s : {reference['d_linear_m2_s']*1e12:.4f} um^2/s")
    print(f"  reference MSD R^2     : {reference['r_squared']:.6f}")
    print(f"  window strategy       : {config.selection_strategy}")
    if config.selection_strategy == "target_kb":
        lag_description = ", ".join(f"{lag_s:g}" for lag_s in config.lag_search_s)
        print(
            f"  lag candidates        : {lag_description} s, "
            f"selected {len(brownian['lag_results'])} groups"
        )
    print("\n  lag   covered(s)     start-count  <x>(um)   <x^2>(um^2)   D(um^2/s)     k_B(J/K)      rel.err")
    for result in brownian["lag_results"]:
        print(
            f"  {result['lag_s']:>3.1f}  "
            f"{result['t_start_s']:>6.1f}-{result['t_pair_end_s']:<6.1f}  "
            f"{result['start_index']:>5d}/{result['n_displacements']:<3d}  "
            f"{result['x_mean_m']*1e6:>8.4f}  "
            f"{result['x2_mean_m2']*1e12:>12.4f}  "
            f"{result['d_m2_s']*1e12:>11.4f}  "
            f"{result['k_b_j_k']:.4e}  "
            f"{result['k_b_error_pct']:>+7.2f}%"
        )

    print("\n  Slip-corrected Brownian mobility check (optional)")
    print("  lag       k_B/C (J/K)      rel.err")
    for result in brownian["lag_results"]:
        print(f"  {result['lag_s']:>3.1f}      {result['k_b_slip_j_k']:.4e}     {result['k_b_slip_error_pct']:>+7.2f}%")

    print("\n  NIST exact k_B       : {:.4e} J/K".format(K_B_TRUE))
    print("  No average across lag groups is reported.")
    print(f"\nParticle outputs written to: {config.output_dir}")
    print("=" * 90)


def analyze_particle(config: ParticleConfig) -> dict[str, pd.DataFrame]:
    freefall_result = calculate_freefall(config)
    brownian_result = select_brownian_windows(config, freefall_result)
    output_tables = write_particle_outputs(config, freefall_result, brownian_result)
    make_report_figure(config, freefall_result, brownian_result)
    make_displacement_distribution_figure(config, brownian_result)
    print_report(config, freefall_result, brownian_result)
    return output_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch Millikan Brownian/free-fall analysis by particle config")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_FILE), help="JSON config file with particle inputs")
    parser.add_argument(
        "--particle",
        action="append",
        help="Particle id to analyze. Repeat this option to run a subset; omit it to run all configured particles.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    selected_particle_ids = set(args.particle) if args.particle else None
    configs, output_root = load_particle_configs(config_path, selected_particle_ids)

    particle_outputs = [analyze_particle(config) for config in configs]
    output_root.mkdir(parents=True, exist_ok=True)

    aggregate_files = {
        "freefall_data": "all_particles_freefall_data.csv",
        "brownian_data": "all_particles_brownian_data.csv",
        "calculation_results": "all_particles_summary.csv",
        "window_selection": "all_particles_window_selection.csv",
        "selected_displacements": "all_particles_selected_displacements.csv",
        "displacement_normality": "all_particles_displacement_normality.csv",
    }
    written_paths = []
    for table_name, file_name in aggregate_files.items():
        aggregate_df = pd.concat([output[table_name] for output in particle_outputs], ignore_index=True)
        aggregate_path = output_root / file_name
        aggregate_df.to_csv(aggregate_path, index=False)
        written_paths.append(aggregate_path)

    print("\nBatch tables written:")
    for path in written_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
