# QINN vs ANN as FinRL state signals

This package implements Professor Xiao-Yang Liu's suggested next experiment:
test whether a quantum-inspired representation improves sequential trading
decisions when injected as a signal into a FinRL benchmark.

The experiment compares three otherwise matched PPO agents:

1. Base FinRL market state.
2. Base state plus an ANN next-day-return signal.
3. Base state plus a quantum-inspired matrix-product-state (MPS) signal.

The ANN and MPS encoders each contain exactly 369 trainable parameters and use
the same 13 inputs, training dates, target, optimizer family, and validation
period. The PPO conditions use the same 15 stocks, official FinRL indicators,
transaction cost (0.10%), algorithm, policy network, timesteps, and seeds.

## Honest scope

This is a quantum-inspired tensor-network study running entirely on classical
hardware. It does not use qubits and cannot demonstrate quantum advantage.
The MPS design is motivated by Liu and Fang's NeurIPS 2020 workshop paper and
by the quantum-tensor-network workshops Professor Liu recommended.

## Verified result

- Mean out-of-sample Sharpe: ANN **0.735**, QINN-MPS **0.694**, Base FinRL
  **0.559**.
- QINN-MPS trailed ANN in all three paired PPO seeds.
- Paired annualized return difference, MPS minus ANN: **-0.92 percentage
  points**, with a 20-day block-bootstrap 95% interval of **[-4.63, +2.63]**.
- Mean annualized turnover: ANN **27.32%**, QINN-MPS **24.14%**, Base FinRL
  **26.43%**.

The result is a controlled negative finding, not a general claim against
tensor-network methods.

## Reproduce

Create a Python 3.12 environment and install the requirements:

```powershell
python -m pip install -r requirements.txt
python run_experiment.py `
  --data-dir .cache\data `
  --finrl-dir .cache\FinRL `
  --output-dir results `
  --timesteps 5000 `
  --seeds 0 1 2
python -m pytest -q test_experiment.py
```

The script downloads and checksum-verifies the processed Nasdaq dataset used
for this FinRL Contest 2025-aligned evaluation, checks out a pinned FinRL
commit, trains the representations, trains PPO agents, and writes all metrics
and figures. The completed reference run uses seeds 0, 1, and 2 and is included
in `results/`.

Each PPO update uses 3 optimization epochs. The 5,000-step budget is a scoped
CPU benchmark, not a claim that every policy has converged.

## Chronology

- Representation fit: 2013-2017.
- Representation early stopping: 2018.
- PPO training: 2013-2018.
- Locked out-of-sample PPO test: 2019-2023.

No 2019-2023 return target is used to fit either signal or PPO policy.

## Main files

- `run_experiment.py`: complete reproducible pipeline.
- `test_experiment.py`: focused integrity tests.
- `research_report.md`: interpretation, source grounding, and limitations.
- `technical_report.docx` and `technical_report.pdf`: submission-ready report.
- `professor_followup.md`: final email to accompany the repository and report.
- `results/ppo_backtest_metrics.csv`: per-seed portfolio results.
- `results/annual_period_metrics.csv`: calendar-year results by condition and
  seed.
- `results/condition_seed_summary.csv`: means and exploratory 95% seed
  intervals.
- `results/annual_period_seed_summary.csv`: calendar-year seed summaries.
- `results/signal_metrics.csv`: predictive metrics before RL.
- `results/ann_vs_mps_block_bootstrap.csv`: paired 20-day block bootstrap.
- `results/run_manifest.json`: pinned commits, checksums, settings, and
  limitations.

## Turnover and costs

The evaluation records the shares actually executed by FinRL. Daily gross
turnover is:

```text
sum(abs(executed shares) * pre-trade price) / beginning-of-step portfolio value
```

Annualized turnover is mean daily turnover multiplied by 252. Both buys and
sells contribute. The daily output also records executed notional and costs,
which reconcile to the configured 0.10% rate.

Nothing in this repository is financial advice.
