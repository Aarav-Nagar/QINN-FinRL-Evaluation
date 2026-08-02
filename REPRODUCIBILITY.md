# Reproducibility Notes

## Reference environment

- Python: 3.12.10
- NumPy: 2.4.2
- pandas: 3.0.0
- Matplotlib: 3.10.8
- PyTorch: 2.10.0+cpu
- Stable-Baselines3: 2.9.0
- Gymnasium: 1.2.3
- pytest: 9.0.3
- FinRL commit: `2334a5fe6d30629157f13c3b0319e1637e15e123`
- Representation seed: 2026
- PPO seeds: 0, 1, and 2 for the reference run; 0--9 for final and shifted evaluations

The exact direct package versions used to verify the saved artifacts are in
`requirements-lock.txt`. The broader compatible minimums are in
`requirements.txt`.

## Clean-checkout verification

From a directory outside the source repository, run:

```powershell
git clone https://github.com/Aarav-Nagar/QINN-FinRL-Evaluation.git qinn-clean
Set-Location qinn-clean
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m pip check
python scripts\verify_scientific_results_freeze.py
python -m pytest -q
```

`requirements-lock.txt` pins every direct package used by the pipeline and
tests. The installer resolves their transitive dependencies; a successful
`pip check` confirms that resolved environment has no broken requirements.
The lock is not an offline wheel archive, so network access is required when
the packages are not already cached.

Long-running matrices should write resumable raw checkpoints under the
repository-local, gitignored `local_runs/` directory. Keeping active
checkpoints inside the repository tree prevents task-workspace cleanup from
removing a run directory between PPO training and its first checkpoint. Only
validated, compact summaries and provenance manifests belong in `results/`.
Checkpoint basenames are deliberately compact (`metrics.partial.csv`,
`curves.partial.csv`, and `config.partial.json`) so an atomic temporary file
does not exceed the default Windows path limit after a descriptive matrix job
identifier is appended. The loader remains compatible with the longer legacy
checkpoint basenames.

## Compute and runtime record

The experiment script explicitly sets PPO to `device="cpu"` and restricts
PyTorch to one thread. Encoder execution accepts `--encoder-device auto`,
`cpu`, or `cuda`; an explicit CUDA request fails if the installed PyTorch build
cannot access CUDA. New runs write requested/resolved devices, package/runtime
details, the source commit, UTC timestamps, and elapsed time to
`run_status.json` and `run_manifest.json`.

The reference artifacts were verified on an Intel Core 7 240H system (10
physical cores, 16 logical processors) using PyTorch `2.10.0+cpu`; they did not
use a GPU. Wall-clock duration was not recorded for that older reference run,
so no retrospective runtime estimate is claimed.

The same host exposes an RTX 5060 Laptop GPU. An isolated PyTorch
`2.10.0+cu130` environment successfully executed CUDA operations, but a matched
10-epoch engineering benchmark made the signal pipeline slower on CUDA (13.60
seconds) than CPU (5.50 seconds). The encoders are too small to amortize device
transfer and kernel-launch overhead in this configuration. Expanded runs
therefore retain CPU by default; the measurement is preserved under
`results/smoke/2026-07-27/device_benchmark.csv`.

## Data

The experiment uses the processed Nasdaq 2013-2023 dataset hosted at:

https://huggingface.co/datasets/benstaf/nasdaq_2013_2023

The pipeline verifies these SHA-256 checksums before training:

| File | SHA-256 |
|---|---|
| `train_data_2013_2018.csv` | `92eb993137595e9c461091e9f3569295d0050f5536c75b707468ba6d2197657b` |
| `trade_data_2019_2023.csv` | `01587b66236b5563df8f871f0110bbf752f1c593427a346192c20e271efffd3b` |

## Reference command

From the repository root:

```powershell
python -m pip install -r requirements-lock.txt
python run_experiment.py `
  --data-dir .cache\data `
  --finrl-dir .cache\FinRL `
  --output-dir results `
  --timesteps 5000 `
  --seeds 0 1 2 `
  --bond-dimension 4 `
  --encoder-device auto
python -m pytest -q test_experiment.py test_smoke_matrix.py
```

The run downloads the data, checks out the recorded FinRL commit, trains both
encoders, trains all PPO conditions, and regenerates the contents of
`results/`.

## Expanded-study pilot commands

The expanded study selected 20,000 PPO steps after the matched budget pilot.
The prespecified bond-dimension pilot can be planned and resumed with:

```powershell
python scripts/run_experiment_matrix.py `
  --phase dimension-pilot `
  --data-dir .cache\data `
  --finrl-dir .cache\FinRL `
  --output-root local_runs\dimension-pilot `
  --timesteps 20000 `
  --bond-dimensions 2 4 8 `
  --seeds 0 1 2 `
  --encoder-epochs 60 `
  --encoder-patience 10 `
  --encoder-batch-size 512 `
  --encoder-device cpu
```

The matrix writes its resolved job plan before execution, skips matching
completed jobs, and refuses to overwrite stale configurations. Generate the
guarded capacity summary with:

```powershell
python scripts/summarize_dimension_pilot.py `
  results\pilots\2026-07-28\raw\dimension-pilot_steps20000_bd2_seeds0-1-2_epochs60_batch512_cpu `
  results\pilots\2026-07-28\raw\dimension-pilot_steps20000_bd4_seeds0-1-2_epochs60_batch512_cpu `
  results\pilots\2026-07-28\raw\dimension-pilot_steps20000_bd8_seeds0-1-2_epochs60_batch512_cpu `
  --summary-output .cache\reproduced-capacity\dimension_summary.csv `
  --paired-output .cache\reproduced-capacity\dimension_paired.csv `
  --manifest-output .cache\reproduced-capacity\dimension_manifest.json
```

The summarizer verifies matched seeds, fixed non-dimension settings, completed
manifests, all three prespecified dimensions, required result schemas, and
invariant Base/ANN controls before writing evidence. It hashes each source
manifest and result table into a generated provenance manifest. The primary
dimension is chosen from validation MSE, parameter count, and fit time using
the rule frozen in
`docs/EXPERIMENT_PROTOCOL.md`; test-period and trading metrics cannot drive the
selection.

The completed pilot selected bond dimension 2. Plan or resume the final matched
evaluation with:

```powershell
python scripts/run_experiment_matrix.py `
  --phase final-evaluation `
  --data-dir .cache\data `
  --finrl-dir .cache\FinRL `
  --output-root local_runs\final-evaluation `
  --timesteps 20000 `
  --bond-dimensions 2 `
  --seeds 0 1 2 3 4 5 6 7 8 9 `
  --encoder-epochs 60 `
  --encoder-patience 10 `
  --encoder-batch-size 512 `
  --encoder-device cpu
```

After the run reaches `completed`, validate and reproduce the prespecified
final artifacts in a disposable cache directory with:

```powershell
python scripts/summarize_final_evaluation.py `
  results\final `
  --condition-output .cache\reproduced-primary\condition_summary.csv `
  --paired-output .cache\reproduced-primary\paired_seed_effects.csv `
  --inference-output .cache\reproduced-primary\primary_inference.json `
  --manifest-output .cache\reproduced-primary\final_manifest.json `
  --artifact-name final_ten_seed_evaluation
python scripts/plot_final_effect.py `
  .cache\reproduced-primary\paired_seed_effects.csv `
  .cache\reproduced-primary\primary_inference.json `
  --png-output .cache\reproduced-primary\final_paired_effect.png `
  --pdf-output .cache\reproduced-primary\final_paired_effect.pdf
```

The final summarizer rejects incomplete runs, any seed set other than 0--9,
budgets other than 20,000 steps, MPS dimensions other than 2, duplicate or
unmatched condition/seed rows, and missing schemas. It records a deterministic
paired-seed bootstrap interval, exact two-sided sign test, and hashes of source
and output artifacts. The interval describes training-seed uncertainty on one
fixed historical split; it is not a population-level causal interval.

If training writes every required table but a later plotting step fails, the
completed endpoints can be finalized without retraining:

```powershell
python run_experiment.py `
  --data-dir . `
  --finrl-dir . `
  --output-dir <completed-run-directory> `
  --finalize-existing
```

Recovery requires the full configured metrics and equity-curve key sets, all
required tabular artifacts, a still-running status, and no existing manifest.
It preserves the training commit, records the finalization commit and recovery
basis, regenerates plots with Windows-safe filenames, and refuses to estimate
runtime fields that were not persisted.

## Shifted-window robustness run

The non-overlapping dates in `docs/ROBUSTNESS_PROTOCOL.md` are implemented as
the named `shifted` window. The runner rejects date drift from that protocol
and excludes encoder rows whose next-day target crosses either the fitting or
validation boundary. Plan the complete matched matrix with:

```powershell
python scripts/run_experiment_matrix.py `
  --phase temporal-robustness `
  --window shifted `
  --data-dir .cache\data `
  --finrl-dir .cache\FinRL `
  --output-root local_runs\temporal-robustness `
  --timesteps 20000 `
  --bond-dimensions 2 `
  --seeds 0 1 2 3 4 5 6 7 8 9 `
  --encoder-epochs 60 `
  --encoder-patience 10 `
  --encoder-batch-size 512 `
  --encoder-device cpu `
  --dry-run
```

Remove `--dry-run` only after inspecting `matrix_plan.json`. The resulting job
identifier contains `shifted`, and `run_status.json` records the exact dates,
window name, device, seeds, budget, and bond dimension.

On Windows, the equivalent durable launcher writes all output to
`work\temporal_robustness_2026-07-30\matrix.log`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_temporal_robustness.ps1
```

The completed shifted run is published under `results/robustness/shifted/`.
Regenerate its guarded summaries and paired-effect figure with:

```powershell
python scripts/summarize_final_evaluation.py results/robustness/shifted `
  --condition-output .cache/reproduced-shifted/condition_summary.csv `
  --paired-output .cache/reproduced-shifted/paired_seed_effects.csv `
  --inference-output .cache/reproduced-shifted/robustness_inference.json `
  --manifest-output .cache/reproduced-shifted/robustness_manifest.json `
  --artifact-name shifted_window_ten_seed_evaluation
python scripts/plot_final_effect.py `
  .cache/reproduced-shifted/paired_seed_effects.csv `
  .cache/reproduced-shifted/robustness_inference.json `
  --png-output .cache/reproduced-shifted/shifted_paired_effect.png `
  --pdf-output .cache/reproduced-shifted/shifted_paired_effect.pdf
```

The same guarded summarizer rejects incomplete status, configuration drift,
unmatched seeds, duplicate condition/seed rows, and missing schemas for both
primary and shifted evaluations. Distinct artifact labels prevent the
secondary robustness evidence from being mistaken for the primary estimand.

## Equal-length temporal robustness panel

The extension in `docs/EQUAL_WINDOW_PROTOCOL.md` reuses the completed
2017--2018 shifted cell and adds 2019--2020 and 2021--2022. Each cell has a
four-year PPO training span, three encoder-fit years, one encoder-validation
year, a two-year evaluation, 20,000 PPO steps, and matched seeds 0--9.

On Windows, launch or resume both new matrices with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_equal_window_robustness.ps1
```

Matching completed jobs are skipped; incomplete jobs resume only when their
saved configuration matches. The two published run bundles are under
`results/robustness/equal_windows/2019-2020/` and `2021-2022/`.

Regenerate the guarded panel and paired-effect figure with:

```powershell
python scripts\summarize_equal_windows.py `
  --window-2017-2018 results\robustness\shifted `
  --window-2019-2020 results\robustness\equal_windows\2019-2020 `
  --window-2021-2022 results\robustness\equal_windows\2021-2022 `
  --output-dir .cache\reproduced-equal-windows `
  --protocol docs\EQUAL_WINDOW_PROTOCOL.md
python scripts\plot_equal_windows.py `
  --paired .cache\reproduced-equal-windows\window_paired_seed_effects.csv `
  --summary .cache\reproduced-equal-windows\window_paired_metric_summary.csv `
  --png-output .cache\reproduced-equal-windows\equal_window_paired_effect.png `
  --pdf-output .cache\reproduced-equal-windows\equal_window_paired_effect.pdf
```

The summarizer rejects incomplete status, date or control drift, missing or
duplicate condition/seed keys, and missing ANN/MPS signal rows. It hashes the
protocol, all three source manifests, raw portfolio metrics, signal metrics,
and every derived table. The panel is exploratory: all windows are reported,
the cells are not pooled, and no market-regime cause is inferred.


## Post-hoc nested evaluation-horizon diagnostic

This deterministic analysis uses the frozen primary curves and does not train
an encoder or PPO policy. It reports every one- through five-year cumulative
prefix so the requested one- through three-year view cannot stop selectively.

```powershell
python scripts\summarize_nested_horizons.py `
  --run-dir results\final `
  --freeze results\SCIENTIFIC_RESULTS_FREEZE.json `
  --protocol docs\NESTED_HORIZON_PROTOCOL.md `
  --output-dir .cache\reproduced-nested-horizons
python scripts\plot_nested_horizons.py `
  --paired .cache\reproduced-nested-horizons\horizon_paired_seed_effects.csv `
  --summary .cache\reproduced-nested-horizons\horizon_paired_metric_summary.csv `
  --png-output .cache\reproduced-nested-horizons\nested_horizon.png `
  --pdf-output .cache\reproduced-nested-horizons\nested_horizon.pdf `
  --manifest .cache\reproduced-nested-horizons\nested_horizon_manifest.json
```

The scorer validates the scientific-freeze hashes, primary configuration,
three PPO conditions, seeds 0--9, common daily grid, all five year-end
cutoffs, and exact reproduction of the frozen five-year metrics. It explicitly
validates and excludes the saved equal-weight benchmark from the PPO
condition comparison. The manifest binds all generated tables and both figure
formats to their sources and scripts.

The prefixes share policies, seeds, dates, and observations. Treat them as
post-hoc descriptive evidence, not independent replications, a causal trend,
or an optimized evaluation horizon.

## Post-hoc market-state trend audit

This analysis uses the saved daily curves from all three non-overlapping
two-year windows. Benchmark-only thresholds define direction, trailing
volatility, drawdown, and return-tail states before ANN and MPS differences
are calculated.

```powershell
python scripts\summarize_market_states.py `
  --window-2017-2018 results\robustness\shifted `
  --window-2019-2020 results\robustness\equal_windows\2019-2020 `
  --window-2021-2022 results\robustness\equal_windows\2021-2022 `
  --protocol docs\MARKET_STATE_TREND_PROTOCOL.md `
  --output-dir .cache\reproduced-market-states
```

The scorer requires completed manifests, the frozen dates and controls, seeds
0--9, identical daily grids, all prespecified states, and exact reproduction of
the saved annual metrics. It writes 270 state/seed effects, 27 state summaries,
300 calendar-year paired metric rows, 30 annual summaries, a three-row exact
direction decomposition, and a manifest binding every source and output hash.

The analysis is post hoc. State categories overlap, only three two-year
windows are available, and seed-bootstrap intervals do not capture historical
calendar uncertainty. It supports descriptive failure analysis only.

## Fixed experimental settings

| Setting | Reference artifact | Expanded study |
|---|---:|---:|
| Initial portfolio | $1,000,000 | $1,000,000 |
| Transaction cost | 0.10% per executed buy or sell | 0.10% per executed buy or sell |
| Maximum order | 100 shares per asset per step | 100 shares per asset per step |
| PPO training budget | 5,000 environment steps | 20,000 environment steps |
| PPO update epochs | 3 | 3 |
| Reward scaling | `1e-4` | `1e-4` |
| MPS bond dimension | 4 | Selected by the frozen capacity rule |
| ANN parameters | 369 | 369 |
| MPS parameters | 369 | Capacity-dependent |

The detailed feature lists, ticker universe, state formulas, dates, checksums,
and known limitations are recorded in `results/final/run_manifest.json` for
the corrected primary, `results/robustness/shifted/run_manifest.json` for the
secondary window, and the two complete manifests under
`results/robustness/equal_windows/` for the equal-length extension.

## Integrity tests

`test_experiment.py` checks:

- matching ANN and MPS parameter counts;
- normalized MPS feature-map behavior;
- Base and signal-agent state dimensions;
- executed-trade turnover calculations;
- calendar-period metrics; and
- seed-summary confidence interval calculations.

It also checks sensitivity-configuration validation, MPS capacity changes,
explicit device resolution, temporal target boundaries, and runtime metadata.
`test_smoke_matrix.py` ensures reduced smoke runs are comparable and remain
marked as non-evidentiary. `test_experiment_matrix.py` verifies resumable
orchestration and stale-result protection. `test_budget_pilot.py` and
`test_dimension_pilot.py` verify guarded pilot summaries. The final-evaluation
and figure tests validate matched ten-seed evidence, distinct primary/shifted
provenance labels, deterministic inference, and paper-facing plots.
`test_equal_window_summary.py`, `test_equal_window_figure.py`, and
`test_equal_window_evidence.py` additionally bind the three fixed windows,
every requested paired metric, prediction quality, input/output hashes,
manuscript values, and PNG/PDF structure.
`test_nested_horizon_summary.py`, `test_nested_horizon_figure.py`, and
`test_nested_horizon_evidence.py` validate all five cumulative prefixes,
benchmark exclusion, exact five-year reproduction, paired metrics, source and
output hashes, post-hoc paper wording, and manifest-bound figure files.
`test_market_state_trends.py` validates state partitions, matched-seed scoring,
the additive direction reconciliation, deterministic intervals, complete
output row counts, and every source/output hash.

## Scope

The included results are a reproducible historical experiment, not evidence of
live trading performance. The MPS implementation is quantum-inspired and runs
on classical hardware.
