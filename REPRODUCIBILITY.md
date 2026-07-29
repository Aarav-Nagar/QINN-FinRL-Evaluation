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
- PPO seeds: 0, 1, and 2

The exact direct package versions used to verify the saved artifacts are in
`requirements-lock.txt`. The broader compatible minimums are in
`requirements.txt`.

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
  --output-root work\dimension-pilot `
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
  work\dimension-pilot\dimension-pilot_steps20000_bd2_seeds0-1-2_epochs60_batch512_cpu `
  work\dimension-pilot\dimension-pilot_steps20000_bd4_seeds0-1-2_epochs60_batch512_cpu `
  work\dimension-pilot\dimension-pilot_steps20000_bd8_seeds0-1-2_epochs60_batch512_cpu `
  --summary-output results\pilots\dimension_summary.csv `
  --paired-output results\pilots\dimension_paired.csv `
  --manifest-output results\pilots\dimension_manifest.json
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
  --output-root work\final-evaluation `
  --timesteps 20000 `
  --bond-dimensions 2 `
  --seeds 0 1 2 3 4 5 6 7 8 9 `
  --encoder-epochs 60 `
  --encoder-patience 10 `
  --encoder-batch-size 512 `
  --encoder-device cpu
```

After the run reaches `completed`, validate and publish the prespecified final
artifacts with:

```powershell
python scripts/summarize_final_evaluation.py `
  work\final-evaluation\final-evaluation_steps20000_bd2_seeds0-1-2-3-4-5-6-7-8-9_epochs60_batch512_cpu `
  --condition-output results\final\condition_summary.csv `
  --paired-output results\final\paired_seed_effects.csv `
  --inference-output results\final\primary_inference.json `
  --manifest-output results\final\artifact_manifest.json
python scripts/plot_final_effect.py `
  results\final\paired_seed_effects.csv `
  results\final\primary_inference.json `
  --png-output results\figures\final_paired_effect.png `
  --pdf-output results\figures\final_paired_effect.pdf
```

The final summarizer rejects incomplete runs, any seed set other than 0--9,
budgets other than 20,000 steps, MPS dimensions other than 2, duplicate or
unmatched condition/seed rows, and missing schemas. It records a deterministic
paired-seed bootstrap interval, exact two-sided sign test, and hashes of source
and output artifacts. The interval describes training-seed uncertainty on one
fixed historical split; it is not a population-level causal interval.

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
and known limitations are recorded in `results/run_manifest.json`.

## Integrity tests

`test_experiment.py` checks:

- matching ANN and MPS parameter counts;
- normalized MPS feature-map behavior;
- Base and signal-agent state dimensions;
- executed-trade turnover calculations;
- calendar-period metrics; and
- seed-summary confidence interval calculations.

It also checks sensitivity-configuration validation, MPS capacity changes,
explicit device resolution, and runtime metadata. `test_smoke_matrix.py`
ensures reduced smoke runs are comparable and remain marked as non-evidentiary.
`test_experiment_matrix.py` verifies resumable orchestration and stale-result
protection. `test_budget_pilot.py` and `test_dimension_pilot.py` verify the
guarded expanded-study summaries and paired comparisons.

## Scope

The included results are a reproducible historical experiment, not evidence of
live trading performance. The MPS implementation is quantum-inspired and runs
on classical hardware.
