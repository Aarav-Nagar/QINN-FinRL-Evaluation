#!/usr/bin/env python
"""Validate and summarize the post-hoc market-state trend audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ANN = "ANN signal"
MPS = "QINN-MPS signal"
BENCHMARK = "Equal-weight buy-and-hold"
EXPECTED_CONDITIONS = {"Base FinRL", ANN, MPS, BENCHMARK}
SEEDS = list(range(10))
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_802
ANNUAL_METRICS = (
    "period_return",
    "sharpe",
    "max_drawdown",
    "annualized_turnover",
    "total_cost",
)
WINDOWS = {
    "2017-2018": {
        "years": (2017, 2018),
        "test_start": "2017-01-03",
        "test_end": "2018-12-28",
    },
    "2019-2020": {
        "years": (2019, 2020),
        "test_start": "2019-01-02",
        "test_end": "2020-12-31",
    },
    "2021-2022": {
        "years": (2021, 2022),
        "test_start": "2021-01-04",
        "test_end": "2022-12-30",
    },
}
STATE_LABELS = {
    "benchmark_direction": ("negative", "nonnegative"),
    "benchmark_volatility": ("low", "high"),
    "benchmark_drawdown": ("at_or_above_prior_peak", "below_prior_peak"),
    "benchmark_return_tail": ("bottom_decile", "middle_80pct", "top_decile"),
}
REQUIRED_CURVE_COLUMNS = {
    "date",
    "account_value",
    "daily_return",
    "daily_turnover",
    "transaction_cost",
    "condition",
    "seed",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_bootstrap(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if len(array) != len(SEEDS) or not np.isfinite(array).all():
        raise ValueError("Bootstrap requires ten finite matched-seed effects")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        len(array),
        size=(BOOTSTRAP_SAMPLES, len(array)),
    )
    means = array[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def _require_columns(frame: pd.DataFrame, required: set[str], context: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{context} missing columns: {sorted(missing)}")


def load_window(
    label: str,
    run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    if label not in WINDOWS:
        raise ValueError(f"Unexpected window: {label}")
    curve_path = run_dir / "equity_curves.csv"
    annual_path = run_dir / "annual_period_metrics.csv"
    manifest_path = run_dir / "run_manifest.json"
    for path in (curve_path, annual_path, manifest_path):
        if not path.is_file():
            raise ValueError(f"{label} source is missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiment = manifest["experiment"]
    expected = WINDOWS[label]
    for field in ("test_start", "test_end"):
        if experiment.get(field) != expected[field]:
            raise ValueError(
                f"{label} {field} drifted: "
                f"{experiment.get(field)!r} != {expected[field]!r}"
            )
    if experiment.get("ppo_seeds") != SEEDS:
        raise ValueError(f"{label} PPO seeds drifted")
    if experiment.get("ppo_timesteps") != 20_000:
        raise ValueError(f"{label} PPO budget drifted")
    if experiment.get("mps_bond_dimension") != 2:
        raise ValueError(f"{label} MPS dimension drifted")
    if manifest.get("runtime", {}).get("status") != "completed":
        raise ValueError(f"{label} run is not complete")

    curves = pd.read_csv(curve_path)
    _require_columns(curves, REQUIRED_CURVE_COLUMNS, f"{label} curves")
    if set(curves["condition"].unique()) != EXPECTED_CONDITIONS:
        raise ValueError(f"{label} conditions drifted")
    curves["date"] = pd.to_datetime(curves["date"], errors="raise")
    numeric = [
        "account_value",
        "daily_return",
        "daily_turnover",
        "transaction_cost",
    ]
    if not np.isfinite(curves[numeric].to_numpy(dtype=float)).all():
        raise ValueError(f"{label} curves contain non-finite values")
    if curves.duplicated(["condition", "seed", "date"]).any():
        raise ValueError(f"{label} curves contain duplicate keys")

    benchmark = curves[curves["condition"] == BENCHMARK].copy()
    if benchmark["date"].duplicated().any():
        raise ValueError(f"{label} benchmark has duplicate dates")
    benchmark_dates = benchmark["date"].sort_values().tolist()
    expected_years = set(expected["years"])
    if set(benchmark["date"].dt.year.unique()) != expected_years:
        raise ValueError(f"{label} evaluation years drifted")

    for condition in (ANN, MPS):
        selected = curves[curves["condition"] == condition]
        observed_seeds = sorted(selected["seed"].astype(int).unique().tolist())
        if observed_seeds != SEEDS:
            raise ValueError(f"{label} seeds drifted for {condition}")
        for seed in SEEDS:
            dates = (
                selected[selected["seed"].astype(int) == seed]["date"]
                .sort_values()
                .tolist()
            )
            if dates != benchmark_dates:
                raise ValueError(
                    f"{label} date grid differs for {condition}, seed {seed}"
                )

    annual = pd.read_csv(annual_path)
    _require_columns(
        annual,
        {"condition", "seed", "year", *ANNUAL_METRICS},
        f"{label} annual metrics",
    )
    annual = annual[annual["condition"].isin((ANN, MPS))].copy()
    expected_keys = {
        (condition, seed, year)
        for condition in (ANN, MPS)
        for seed in SEEDS
        for year in expected["years"]
    }
    observed_keys = set(
        annual[["condition", "seed", "year"]]
        .itertuples(index=False, name=None)
    )
    if observed_keys != expected_keys:
        raise ValueError(f"{label} annual condition/seed/year keys drifted")

    return curves, annual, [curve_path, annual_path, manifest_path]


def benchmark_state_masks(
    curves: pd.DataFrame,
) -> dict[tuple[str, str], pd.Series]:
    benchmark = (
        curves[curves["condition"] == BENCHMARK]
        .sort_values("date")
        .set_index("date")
    )
    returns = benchmark["daily_return"].astype(float)
    account = benchmark["account_value"].astype(float)
    volatility = returns.rolling(20).std(ddof=1) * math.sqrt(252)
    volatility_median = float(volatility.dropna().median())
    lower_tail = float(returns.quantile(0.10))
    upper_tail = float(returns.quantile(0.90))
    running_peak = account.cummax()

    masks = {
        ("benchmark_direction", "negative"): returns < 0,
        ("benchmark_direction", "nonnegative"): returns >= 0,
        ("benchmark_volatility", "low"): volatility.notna()
        & (volatility <= volatility_median),
        ("benchmark_volatility", "high"): volatility.notna()
        & (volatility > volatility_median),
        ("benchmark_drawdown", "at_or_above_prior_peak"): account
        >= running_peak,
        ("benchmark_drawdown", "below_prior_peak"): account < running_peak,
        ("benchmark_return_tail", "bottom_decile"): returns <= lower_tail,
        ("benchmark_return_tail", "middle_80pct"): (returns > lower_tail)
        & (returns < upper_tail),
        ("benchmark_return_tail", "top_decile"): returns >= upper_tail,
    }

    for family, labels in STATE_LABELS.items():
        family_masks = [masks[(family, label)] for label in labels]
        coverage = pd.concat(family_masks, axis=1).sum(axis=1)
        expected = volatility.notna() if family == "benchmark_volatility" else True
        if isinstance(expected, bool):
            if not (coverage == 1).all():
                raise ValueError(f"{family} masks do not partition the dates")
        elif not (coverage[expected] == 1).all() or (coverage[~expected] != 0).any():
            raise ValueError(f"{family} masks do not partition valid dates")
    return masks


def _downside_deviation(returns: pd.Series) -> float:
    downside = np.minimum(returns.to_numpy(dtype=float), 0.0)
    return float(np.sqrt(np.mean(np.square(downside))) * math.sqrt(252))


def score_market_states(label: str, curves: pd.DataFrame) -> pd.DataFrame:
    masks = benchmark_state_masks(curves)
    rows: list[dict[str, object]] = []
    for (family, state), mask in masks.items():
        dates = mask.index[mask]
        if len(dates) == 0:
            raise ValueError(f"{label} {family}/{state} has no observations")
        for seed in SEEDS:
            condition_returns: dict[str, pd.Series] = {}
            for condition in (ANN, MPS):
                selected = (
                    curves[
                        (curves["condition"] == condition)
                        & (curves["seed"].astype(int) == seed)
                    ]
                    .set_index("date")
                    .loc[dates, "daily_return"]
                    .astype(float)
                )
                condition_returns[condition] = selected
            ann = condition_returns[ANN]
            mps = condition_returns[MPS]
            difference = mps - ann
            ann_downside = _downside_deviation(ann)
            mps_downside = _downside_deviation(mps)
            rows.append(
                {
                    "window": label,
                    "state_family": family,
                    "state": state,
                    "seed": seed,
                    "observations": len(dates),
                    "ann_mean_daily_return": float(ann.mean()),
                    "mps_mean_daily_return": float(mps.mean()),
                    "mps_minus_ann_mean_daily_return": float(difference.mean()),
                    "mps_minus_ann_annualized_mean_return": float(
                        difference.mean() * 252
                    ),
                    "ann_annualized_downside_deviation": ann_downside,
                    "mps_annualized_downside_deviation": mps_downside,
                    "mps_minus_ann_downside_deviation": (
                        mps_downside - ann_downside
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_market_states(seed_effects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in seed_effects.groupby(
        ["window", "state_family", "state"],
        sort=True,
    ):
        values = group["mps_minus_ann_annualized_mean_return"]
        lower, upper = paired_bootstrap(values)
        mean = float(values.mean())
        rows.append(
            {
                "window": keys[0],
                "state_family": keys[1],
                "state": keys[2],
                "observations": int(group["observations"].iloc[0]),
                "n_matched_seeds": len(group),
                "mean_mps_minus_ann_annualized_return": mean,
                "median_mps_minus_ann_annualized_return": float(values.median()),
                "positive_seeds": int((values > 0).sum()),
                "negative_seeds": int((values < 0).sum()),
                "zero_seeds": int((values == 0).sum()),
                "paired_seed_bootstrap_95pct_lower": lower,
                "paired_seed_bootstrap_95pct_upper": upper,
                "mean_mps_minus_ann_downside_deviation": float(
                    group["mps_minus_ann_downside_deviation"].mean()
                ),
                "mean_effect_sign": (
                    "positive" if mean > 0 else "negative" if mean < 0 else "zero"
                ),
            }
        )
    result = pd.DataFrame(rows)
    expected_cells = seed_effects["window"].nunique() * sum(
        len(labels) for labels in STATE_LABELS.values()
    )
    if len(result) != expected_cells:
        raise ValueError(
            f"Expected {expected_cells} market-state cells, found {len(result)}"
        )
    return result


def cross_window_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in summary.groupby(["state_family", "state"], sort=True):
        if set(group["window"]) != set(WINDOWS):
            raise ValueError(f"Cross-window cell is incomplete: {keys}")
        values = group["mean_mps_minus_ann_annualized_return"]
        rows.append(
            {
                "state_family": keys[0],
                "state": keys[1],
                "n_windows": len(group),
                "equal_window_mean_annualized_return_difference": float(
                    values.mean()
                ),
                "minimum_window_difference": float(values.min()),
                "maximum_window_difference": float(values.max()),
                "positive_windows": int((values > 0).sum()),
                "negative_windows": int((values < 0).sum()),
                "zero_windows": int((values == 0).sum()),
                "sign_consistent_across_windows": bool(
                    (values > 0).all() or (values < 0).all() or (values == 0).all()
                ),
            }
        )
    return pd.DataFrame(rows)


def direction_contribution(summary: pd.DataFrame) -> pd.DataFrame:
    """Exactly reconcile the direction-state effects to each full window."""

    direction = summary[
        summary["state_family"] == "benchmark_direction"
    ].copy()
    rows: list[dict[str, object]] = []
    for window, group in direction.groupby("window", sort=True):
        indexed = group.set_index("state")
        if set(indexed.index) != set(STATE_LABELS["benchmark_direction"]):
            raise ValueError(f"{window} direction states are incomplete")
        total_observations = int(indexed["observations"].sum())
        negative_weight = float(
            indexed.loc["negative", "observations"] / total_observations
        )
        nonnegative_weight = 1.0 - negative_weight
        negative_effect = float(
            indexed.loc[
                "negative",
                "mean_mps_minus_ann_annualized_return",
            ]
        )
        nonnegative_effect = float(
            indexed.loc[
                "nonnegative",
                "mean_mps_minus_ann_annualized_return",
            ]
        )
        negative_contribution = negative_weight * negative_effect
        nonnegative_contribution = nonnegative_weight * nonnegative_effect
        rows.append(
            {
                "window": window,
                "observations": total_observations,
                "negative_day_fraction": negative_weight,
                "negative_day_effect": negative_effect,
                "negative_day_contribution": negative_contribution,
                "nonnegative_day_fraction": nonnegative_weight,
                "nonnegative_day_effect": nonnegative_effect,
                "nonnegative_day_contribution": nonnegative_contribution,
                "reconciled_full_window_annualized_mean_difference": (
                    negative_contribution + nonnegative_contribution
                ),
                "negative_offset_fraction_of_nonnegative_contribution": (
                    abs(negative_contribution / nonnegative_contribution)
                ),
            }
        )
    return pd.DataFrame(rows)


def _direct_annual_metrics(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = curves[curves["condition"].isin((ANN, MPS))].copy()
    selected["year"] = selected["date"].dt.year
    for (condition, seed, year), group in selected.groupby(
        ["condition", "seed", "year"]
    ):
        group = group.sort_values("date")
        returns = group["daily_return"].astype(float)
        wealth = (1.0 + returns).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        standard_deviation = float(returns.std(ddof=1))
        rows.append(
            {
                "condition": condition,
                "seed": int(seed),
                "year": int(year),
                "period_return": float(wealth.iloc[-1] - 1.0),
                "sharpe": float(
                    returns.mean() / standard_deviation * math.sqrt(252)
                ),
                "max_drawdown": float(drawdown.min()),
                "annualized_turnover": float(
                    group["daily_turnover"].mean() * 252
                ),
                "total_cost": float(group["transaction_cost"].sum()),
            }
        )
    return pd.DataFrame(rows)


def validate_and_pair_annual(
    label: str,
    curves: pd.DataFrame,
    saved: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["condition", "seed", "year"]
    direct = _direct_annual_metrics(curves)
    merged = saved.merge(
        direct,
        on=keys,
        how="outer",
        suffixes=("_saved", "_direct"),
        validate="one_to_one",
    )
    for metric in ANNUAL_METRICS:
        if not np.allclose(
            merged[f"{metric}_saved"],
            merged[f"{metric}_direct"],
            rtol=1e-11,
            atol=1e-11,
        ):
            raise ValueError(f"{label} saved annual {metric} does not reproduce")

    pivot = saved.pivot(index=["seed", "year"], columns="condition")
    rows: list[dict[str, object]] = []
    for seed, year in pivot.index:
        for metric in ANNUAL_METRICS:
            ann = float(pivot.loc[(seed, year), (metric, ANN)])
            mps = float(pivot.loc[(seed, year), (metric, MPS)])
            rows.append(
                {
                    "window": label,
                    "year": int(year),
                    "seed": int(seed),
                    "metric": metric,
                    "ann": ann,
                    "mps": mps,
                    "mps_minus_ann": mps - ann,
                }
            )
    return pd.DataFrame(rows)


def summarize_annual(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in paired.groupby(["window", "year", "metric"], sort=True):
        values = group["mps_minus_ann"]
        lower, upper = paired_bootstrap(values)
        rows.append(
            {
                "window": keys[0],
                "year": int(keys[1]),
                "metric": keys[2],
                "n_matched_seeds": len(group),
                "mean_difference": float(values.mean()),
                "positive_seeds": int((values > 0).sum()),
                "negative_seeds": int((values < 0).sum()),
                "zero_seeds": int((values == 0).sum()),
                "paired_seed_bootstrap_95pct_lower": lower,
                "paired_seed_bootstrap_95pct_upper": upper,
            }
        )
    expected = len(WINDOWS) * 2 * len(ANNUAL_METRICS)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} annual summaries, found {len(rows)}")
    return pd.DataFrame(rows)


def inference_summary(
    market_cross_window: pd.DataFrame,
    annual_summary: pd.DataFrame,
) -> dict[str, object]:
    consistent = market_cross_window[
        market_cross_window["sign_consistent_across_windows"]
    ]
    annual_return = annual_summary[annual_summary["metric"] == "period_return"]
    sharpe = annual_summary[annual_summary["metric"] == "sharpe"]
    return {
        "analysis_role": "post-hoc exploratory market-state trend audit",
        "windows": list(WINDOWS),
        "ppo_seeds_per_window": SEEDS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "market_state_cells": int(
            len(WINDOWS) * sum(len(labels) for labels in STATE_LABELS.values())
        ),
        "consistent_market_state_signs": [
            {
                "state_family": row.state_family,
                "state": row.state,
                "sign": (
                    "positive"
                    if row.positive_windows == len(WINDOWS)
                    else "negative"
                    if row.negative_windows == len(WINDOWS)
                    else "zero"
                ),
                "equal_window_mean_annualized_return_difference": (
                    row.equal_window_mean_annualized_return_difference
                ),
            }
            for row in consistent.itertuples(index=False)
        ],
        "calendar_years_with_positive_mean_return_difference": int(
            (annual_return["mean_difference"] > 0).sum()
        ),
        "calendar_years_with_negative_mean_return_difference": int(
            (annual_return["mean_difference"] < 0).sum()
        ),
        "calendar_years_with_positive_mean_sharpe_difference": int(
            (sharpe["mean_difference"] > 0).sum()
        ),
        "calendar_years_with_negative_mean_sharpe_difference": int(
            (sharpe["mean_difference"] < 0).sum()
        ),
        "interpretation": (
            "Descriptive conditional evidence only; states overlap, windows are "
            "few, and seed intervals do not represent calendar uncertainty."
        ),
    }


def write_outputs(
    run_dirs: dict[str, Path],
    output_dir: Path,
    protocol: Path,
) -> dict[str, Path]:
    loaded = {
        label: load_window(label, run_dirs[label])
        for label in WINDOWS
    }
    seed_frames = []
    annual_frames = []
    source_paths: list[Path] = [protocol]
    for label, (curves, annual, paths) in loaded.items():
        seed_frames.append(score_market_states(label, curves))
        annual_frames.append(validate_and_pair_annual(label, curves, annual))
        source_paths.extend(paths)

    seed_effects = pd.concat(seed_frames, ignore_index=True)
    market_summary = summarize_market_states(seed_effects)
    cross_summary = cross_window_summary(market_summary)
    direction_summary = direction_contribution(market_summary)
    annual_paired = pd.concat(annual_frames, ignore_index=True)
    annual_summary = summarize_annual(annual_paired)
    inference = inference_summary(cross_summary, annual_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "market_state_seed_effects": output_dir / "market_state_seed_effects.csv",
        "market_state_summary": output_dir / "market_state_summary.csv",
        "market_state_cross_window": output_dir / "market_state_cross_window.csv",
        "direction_contribution": output_dir / "direction_contribution.csv",
        "calendar_year_paired_metrics": output_dir
        / "calendar_year_paired_metrics.csv",
        "calendar_year_summary": output_dir / "calendar_year_summary.csv",
        "inference": output_dir / "market_state_inference.json",
        "manifest": output_dir / "market_state_manifest.json",
    }
    seed_effects.to_csv(paths["market_state_seed_effects"], index=False)
    market_summary.to_csv(paths["market_state_summary"], index=False)
    cross_summary.to_csv(paths["market_state_cross_window"], index=False)
    direction_summary.to_csv(paths["direction_contribution"], index=False)
    annual_paired.to_csv(paths["calendar_year_paired_metrics"], index=False)
    annual_summary.to_csv(paths["calendar_year_summary"], index=False)
    paths["inference"].write_text(
        json.dumps(inference, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    root = protocol.resolve().parent.parent
    script = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "analysis_role": inference["analysis_role"],
        "source_sha256": {
            path.resolve().relative_to(root).as_posix(): sha256(path)
            for path in sorted(source_paths + [script])
        },
        "output_sha256": {
            path.resolve().relative_to(root).as_posix(): sha256(path)
            for name, path in sorted(paths.items())
            if name != "manifest"
        },
        "row_counts": {
            "market_state_seed_effects": len(seed_effects),
            "market_state_summary": len(market_summary),
            "market_state_cross_window": len(cross_summary),
            "direction_contribution": len(direction_summary),
            "calendar_year_paired_metrics": len(annual_paired),
            "calendar_year_summary": len(annual_summary),
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-2017-2018",
        type=Path,
        default=Path("results/robustness/shifted"),
    )
    parser.add_argument(
        "--window-2019-2020",
        type=Path,
        default=Path("results/robustness/equal_windows/2019-2020"),
    )
    parser.add_argument(
        "--window-2021-2022",
        type=Path,
        default=Path("results/robustness/equal_windows/2021-2022"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/MARKET_STATE_TREND_PROTOCOL.md"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/robustness/market_states"),
    )
    args = parser.parse_args()
    paths = write_outputs(
        {
            "2017-2018": args.window_2017_2018,
            "2019-2020": args.window_2019_2020,
            "2021-2022": args.window_2021_2022,
        },
        args.output_dir,
        args.protocol,
    )
    print(f"Wrote {len(paths)} guarded market-state artifacts.")


if __name__ == "__main__":
    main()
