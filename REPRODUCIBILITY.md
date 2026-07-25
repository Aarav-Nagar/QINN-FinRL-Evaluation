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
PyTorch to one thread. The saved artifacts were verified on an Intel Core 7
240H system (10 physical cores, 16 logical processors); the pipeline did not
use a GPU. Wall-clock duration was not recorded when the saved experiment was
run, so no runtime estimate is claimed. Future runs should record start time,
end time, and peak memory alongside the run manifest.

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
  --seeds 0 1 2
python -m pytest -q test_experiment.py
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

## Scope

The included results are a reproducible historical experiment, not evidence of
live trading performance. The MPS implementation is quantum-inspired and runs
on classical hardware.
