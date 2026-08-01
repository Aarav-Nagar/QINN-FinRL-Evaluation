#!/usr/bin/env python
"""Re-score every cumulative year-end prefix of the frozen primary run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


CONDITIONS = ["Base FinRL", "ANN signal", "QINN-MPS signal"]
BENCHMARK_CONDITION = "Equal-weight buy-and-hold"
SEEDS = list(range(10))
END_YEARS = [2019, 2020, 2021, 2022, 2023]
METRICS = [
    "annual_return",
    "sharpe",
    "max_drawdown",
    "annualized_turnover",
    "total_cost",
]
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_728
SOURCE_KEYS = {
    "run_manifest.json": "results/final/run_manifest.json",
    "ppo_backtest_metrics.csv": "results/final/ppo_backtest_metrics.csv",
    "equity_curves.csv": "results/final/equity_curves.csv",
}


def canonical_sha256(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix.lower() not in {".png", ".pdf"}:
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _require_equal(actual: object, expected: object, context: str) -> None:
    if actual != expected:
        raise ValueError(f"{context}: expected {expected!r}, found {actual!r}")


def _verify_frozen_inputs(
    run_dir: Path, freeze_path: Path
) -> dict[str, object]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    frozen_hashes = freeze.get("sha256", {})
    for filename, freeze_key in SOURCE_KEYS.items():
        if freeze_key not in frozen_hashes:
            raise ValueError(f"Freeze registry is missing {freeze_key}")
        observed = canonical_sha256(run_dir / filename)
        _require_equal(observed, frozen_hashes[freeze_key], freeze_key)
    return freeze


def _validate_manifest(manifest: dict[str, object]) -> None:
    _require_equal(
        manifest.get("runtime", {}).get("status"),
        "completed",
        "primary runtime status",
    )
    experiment = manifest["experiment"]
    expected = {
        "window_name": "primary",
        "ppo_seeds": SEEDS,
        "ppo_timesteps": 20_000,
        "mps_bond_dimension": 2,
        "representation_seed": 2026,
        "transaction_cost": 0.001,
        "initial_amount": 1_000_000,
        "test_start": "2019-01-02",
        "test_end": "2023-12-28",
    }
    for field, value in expected.items():
        _require_equal(experiment.get(field), value, f"primary {field}")


def validate_curves(curves: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "account_value",
        "gross_traded_notional",
        "daily_turnover",
        "transaction_cost",
        "condition",
        "seed",
    }
    if missing := required - set(curves.columns):
        raise ValueError(f"Primary curves missing columns: {sorted(missing)}")
    validated = curves.copy()
    validated["date"] = pd.to_datetime(validated["date"])
    validated["seed"] = validated["seed"].astype(int)
    observed_conditions = set(validated["condition"].unique())
    allowed_conditions = {*CONDITIONS, BENCHMARK_CONDITION}
    if unexpected := observed_conditions - allowed_conditions:
        raise ValueError(
            f"Primary curves contain unexpected conditions: {sorted(unexpected)}"
        )
    if missing_conditions := set(CONDITIONS) - observed_conditions:
        raise ValueError(
            f"Primary curves missing conditions: {sorted(missing_conditions)}"
        )
    if validated.duplicated(["condition", "seed", "date"]).any():
        raise ValueError("Primary curves contain duplicate condition/seed/date rows")

    reference: pd.DatetimeIndex | None = None
    for condition in CONDITIONS:
        observed_seeds = sorted(
            validated.loc[validated["condition"] == condition, "seed"]
            .unique()
            .tolist()
        )
        _require_equal(observed_seeds, SEEDS, f"primary seeds for {condition}")
        for seed in SEEDS:
            dates = pd.DatetimeIndex(
                validated.loc[
                    (validated["condition"] == condition)
                    & (validated["seed"] == seed),
                    "date",
                ].sort_values()
            )
            if reference is None:
                reference = dates
            elif not dates.equals(reference):
                raise ValueError(
                    f"Date grid drift for condition={condition!r}, seed={seed}"
                )
    if BENCHMARK_CONDITION in observed_conditions:
        benchmark = validated[validated["condition"] == BENCHMARK_CONDITION]
        _require_equal(
            sorted(benchmark["seed"].unique().tolist()),
            [-1],
            "benchmark seeds",
        )
        benchmark_dates = pd.DatetimeIndex(benchmark["date"].sort_values())
        if reference is not None and not benchmark_dates.equals(reference):
            raise ValueError("Date grid drift for equal-weight benchmark")

    if reference is None or reference.empty:
        raise ValueError("Primary curves are empty")
    _require_equal(reference.min().date().isoformat(), "2019-01-02", "start date")
    _require_equal(reference.max().date().isoformat(), "2023-12-28", "end date")
    observed_years = sorted(set(reference.year.tolist()))
    _require_equal(observed_years, END_YEARS, "available evaluation years")
    return validated[validated["condition"].isin(CONDITIONS)].copy()


def compute_prefix_metrics(prefix: pd.DataFrame) -> dict[str, float | int]:
    ordered = prefix.sort_values("date")
    if len(ordered) < 2:
        raise ValueError("A horizon prefix needs at least two account observations")
    account = ordered["account_value"].to_numpy(dtype=float)
    if not np.isfinite(account).all() or (account <= 0).any():
        raise ValueError("Account values must be positive and finite")
    returns = pd.Series(account).pct_change().fillna(0.0)
    step_count = len(ordered) - 1
    years = max(step_count / 252.0, 1 / 252.0)
    total_return = account[-1] / account[0] - 1.0
    annual_return = (1.0 + total_return) ** (1.0 / years) - 1.0
    return_std = float(returns.std(ddof=1))
    annual_volatility = return_std * math.sqrt(252)
    sharpe = (
        float(returns.mean() / return_std * math.sqrt(252))
        if return_std > 0
        else float("nan")
    )
    wealth = pd.Series(account)
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "observations": int(len(ordered)),
        "trading_steps": int(step_count),
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "gross_traded_notional": float(
            ordered["gross_traded_notional"].astype(float).sum()
        ),
        "cumulative_turnover": float(
            ordered["daily_turnover"].astype(float).sum()
        ),
        "annualized_turnover": float(
            ordered["daily_turnover"].astype(float).sum() / step_count * 252
        ),
        "total_cost": float(ordered["transaction_cost"].astype(float).sum()),
    }


def score_horizons(curves: pd.DataFrame) -> pd.DataFrame:
    validated = validate_curves(curves)
    rows: list[dict[str, object]] = []
    for horizon_years, end_year in enumerate(END_YEARS, start=1):
        cutoff = validated.loc[validated["date"].dt.year <= end_year, "date"].max()
        if pd.isna(cutoff) or int(cutoff.year) != end_year:
            raise ValueError(f"Missing year-end cutoff for {end_year}")
        for (condition, seed), group in validated.groupby(
            ["condition", "seed"], sort=False
        ):
            prefix = group[group["date"] <= cutoff]
            rows.append(
                {
                    "horizon_years": horizon_years,
                    "end_year": end_year,
                    "start_date": prefix["date"].min().date().isoformat(),
                    "end_date": prefix["date"].max().date().isoformat(),
                    "condition": condition,
                    "seed": int(seed),
                    **compute_prefix_metrics(prefix),
                }
            )
    scored = pd.DataFrame(rows)
    _require_equal(
        len(scored),
        len(END_YEARS) * len(CONDITIONS) * len(SEEDS),
        "horizon score row count",
    )
    return scored.sort_values(
        ["horizon_years", "condition", "seed"]
    ).reset_index(drop=True)


def validate_full_horizon(
    scored: pd.DataFrame, final_metrics: pd.DataFrame
) -> None:
    final = final_metrics[final_metrics["seed"] >= 0].copy()
    required = {"condition", "seed", *METRICS}
    if missing := required - set(final.columns):
        raise ValueError(f"Final metrics missing columns: {sorted(missing)}")
    if final.duplicated(["condition", "seed"]).any():
        raise ValueError("Final metrics contain duplicate condition/seed rows")
    full = scored[scored["horizon_years"] == 5]
    merged = full.merge(
        final[["condition", "seed", *METRICS]],
        on=["condition", "seed"],
        suffixes=("_prefix", "_frozen"),
        validate="one_to_one",
    )
    _require_equal(len(merged), 30, "five-year validation row count")
    for metric in METRICS:
        if not np.allclose(
            merged[f"{metric}_prefix"].to_numpy(dtype=float),
            merged[f"{metric}_frozen"].to_numpy(dtype=float),
            rtol=1e-11,
            atol=1e-11,
            equal_nan=True,
        ):
            maximum = float(
                np.max(
                    np.abs(
                        merged[f"{metric}_prefix"].to_numpy(dtype=float)
                        - merged[f"{metric}_frozen"].to_numpy(dtype=float)
                    )
                )
            )
            raise ValueError(
                f"Five-year {metric} does not reproduce frozen metrics; "
                f"maximum absolute difference={maximum}"
            )


def paired_bootstrap(values: pd.Series) -> tuple[float, float]:
    array = values.to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_SAMPLES, len(array)))
    means = array[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def summarize_scores(
    scored: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    condition_rows: list[dict[str, object]] = []
    paired_frames: list[pd.DataFrame] = []
    paired_summary_rows: list[dict[str, object]] = []

    for horizon in range(1, 6):
        selected_horizon = scored[scored["horizon_years"] == horizon]
        for condition in CONDITIONS:
            selected = selected_horizon[
                selected_horizon["condition"] == condition
            ]
            row: dict[str, object] = {
                "horizon_years": horizon,
                "end_year": int(selected["end_year"].iloc[0]),
                "end_date": selected["end_date"].iloc[0],
                "condition": condition,
                "n_seeds": len(selected),
            }
            for metric in METRICS:
                row[f"{metric}_mean"] = float(selected[metric].mean())
                row[f"{metric}_std"] = float(selected[metric].std(ddof=1))
            condition_rows.append(row)

        pivot = selected_horizon.pivot(index="seed", columns="condition")
        paired = pd.DataFrame(
            {"horizon_years": horizon, "seed": SEEDS}
        )
        for metric in METRICS:
            column = f"mps_minus_ann_{metric}"
            paired[column] = (
                pivot[metric]["QINN-MPS signal"] - pivot[metric]["ANN signal"]
            ).to_numpy()
            differences = paired[column]
            lower, upper = paired_bootstrap(differences)
            paired_summary_rows.append(
                {
                    "horizon_years": horizon,
                    "end_year": int(selected_horizon["end_year"].iloc[0]),
                    "metric": metric,
                    "n_matched_seeds": len(differences),
                    "mean_difference": float(differences.mean()),
                    "standard_deviation": float(differences.std(ddof=1)),
                    "median_difference": float(differences.median()),
                    "positive_seeds": int((differences > 0).sum()),
                    "negative_seeds": int((differences < 0).sum()),
                    "zero_seeds": int((differences == 0).sum()),
                    "paired_seed_bootstrap_95pct_lower": lower,
                    "paired_seed_bootstrap_95pct_upper": upper,
                }
            )
        paired_frames.append(paired)

    paired_summary = pd.DataFrame(paired_summary_rows)
    sharpe = paired_summary[paired_summary["metric"] == "sharpe"].sort_values(
        "horizon_years"
    )
    differences = sharpe["mean_difference"].to_numpy(dtype=float)
    signs = np.sign(differences)
    inference = {
        "analysis_role": "post-hoc exploratory nested evaluation-horizon analysis",
        "source": "frozen 2019-2023 primary equity curves; no retraining",
        "horizons_years": list(range(1, 6)),
        "organizing_metric": "paired MPS-minus-ANN annualized Sharpe",
        "mean_sharpe_differences": differences.tolist(),
        "mps_higher_mean_sharpe_horizons": sharpe.loc[
            sharpe["mean_difference"] > 0, "horizon_years"
        ].astype(int).tolist(),
        "ann_higher_mean_sharpe_horizons": sharpe.loc[
            sharpe["mean_difference"] < 0, "horizon_years"
        ].astype(int).tolist(),
        "adjacent_mean_sign_changes": int(np.sum(signs[1:] != signs[:-1])),
        "sharpe_intervals_including_zero": sharpe.loc[
            (sharpe["paired_seed_bootstrap_95pct_lower"] <= 0)
            & (sharpe["paired_seed_bootstrap_95pct_upper"] >= 0),
            "horizon_years",
        ].astype(int).tolist(),
        "bootstrap_samples_per_horizon_metric": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "interpretation_boundary": (
            "All cumulative prefixes are reported. They are nested, dependent, "
            "and post hoc; changes cannot establish a causal horizon effect or "
            "a stable representation advantage."
        ),
    }
    return (
        pd.DataFrame(condition_rows),
        pd.concat(paired_frames, ignore_index=True),
        paired_summary,
        inference,
    )


def write_outputs(
    run_dir: Path,
    freeze_path: Path,
    protocol_path: Path,
    output_dir: Path,
) -> None:
    _verify_frozen_inputs(run_dir, freeze_path)
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    _validate_manifest(manifest)
    curves = pd.read_csv(run_dir / "equity_curves.csv")
    final_metrics = pd.read_csv(run_dir / "ppo_backtest_metrics.csv")
    scored = score_horizons(curves)
    validate_full_horizon(scored, final_metrics)
    condition, paired, paired_summary, inference = summarize_scores(scored)

    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "horizon_seed_metrics.csv": scored,
        "horizon_condition_summary.csv": condition,
        "horizon_paired_seed_effects.csv": paired,
        "horizon_paired_metric_summary.csv": paired_summary,
    }
    for filename, frame in frames.items():
        frame.to_csv(output_dir / filename, index=False, lineterminator="\n")
    inference_path = output_dir / "nested_horizon_inference.json"
    inference_path.write_bytes(
        (json.dumps(inference, indent=2) + "\n").encode("utf-8")
    )
    evidence_manifest = {
        "artifact": "nested_evaluation_horizon_analysis",
        "analysis_role": "post-hoc exploratory",
        "protocol": str(protocol_path.as_posix()),
        "fixed_source": "results/final/equity_curves.csv",
        "retraining": False,
        "horizons_years": list(range(1, 6)),
        "inputs": {
            "protocol": canonical_sha256(protocol_path),
            "scientific_freeze": canonical_sha256(freeze_path),
            **{
                filename: canonical_sha256(run_dir / filename)
                for filename in SOURCE_KEYS
            },
        },
        "outputs": {
            filename: canonical_sha256(output_dir / filename)
            for filename in frames
        }
        | {inference_path.name: canonical_sha256(inference_path)},
        "reporting_rule": (
            "Report all one- through five-year cumulative prefixes; do not "
            "select a stopping point; interpret only as nested post-hoc evidence."
        ),
    }
    (output_dir / "nested_horizon_manifest.json").write_bytes(
        (json.dumps(evidence_manifest, indent=2) + "\n").encode("utf-8")
    )
    sharpe = paired_summary[paired_summary["metric"] == "sharpe"]
    print(sharpe.to_string(index=False))
    print()
    print(json.dumps(inference, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_outputs(
        args.run_dir,
        args.freeze,
        args.protocol,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
