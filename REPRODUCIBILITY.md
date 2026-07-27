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

## Fixed experimental settings

| Setting | Value |
|---|---:|
| Initial portfolio | $1,000,000 |
| Transaction cost | 0.10% per executed buy or sell |
| Maximum order | 100 shares per asset per step |
| PPO training budget | 5,000 environment steps |
| PPO update epochs | 3 |
| Reward scaling | `1e-4` |
| MPS bond dimension | 4 |
| ANN parameters | 369 |
| MPS parameters | 369 |

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

## Scope

The included results are a reproducible historical experiment, not evidence of
live trading performance. The MPS implementation is quantum-inspired and runs
on classical hardware.
