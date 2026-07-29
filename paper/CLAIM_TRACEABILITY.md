# Paper Claim and Artifact Traceability

This registry prevents manuscript claims from drifting away from generated
evidence. `Ready` means the cited artifact exists and supports the bounded
wording below; it does not mean the final manuscript sentence is frozen.

## Claims

| ID | Intended claim | Evidence | Status and wording boundary |
|---|---|---|---|
| C01 | The study uses 15 Nasdaq stocks, a 2013-2018 PPO training period, and a 2019-2023 test period. | `results/run_manifest.json` | Ready. Describe this as one fixed historical split. |
| C02 | ANN and dimension-4 MPS encoders have 369 trainable parameters in the reference comparison. | `results/run_manifest.json`, `results/signal_metrics.csv` | Ready only for dimension 4. Other MPS dimensions are capacity sensitivity, not parameter matched. |
| C03 | The reference 5k-step, three-seed MPS condition did not improve mean Sharpe relative to ANN. | `results/condition_seed_summary.csv`, `results/ppo_backtest_metrics.csv` | Ready. Bound the claim to this configuration and split. |
| C04 | The reference moving-block-bootstrap interval for MPS-minus-ANN annualized return includes zero. | `results/ann_vs_mps_block_bootstrap.csv`, `results/run_manifest.json` | Ready. Treat the interval as uncertainty description, not proof of no effect. |
| C05 | The matched budget pilot selected 20,000 PPO steps because outcomes and diagnostics still changed between 10k and 20k. | `results/pilots/2026-07-27/budget_summary.csv`, `results/pilots/2026-07-27/budget_paired.csv`, `docs/DECISIONS.md` | Ready. State explicitly that convergence was not established. |
| C06 | MPS validation behavior and computational cost vary with bond dimension; the frozen validation/parsimony rule selected dimension 2. | `results/pilots/2026-07-28/dimension_summary.csv`, `results/pilots/2026-07-28/dimension_manifest.json`, `docs/DECISIONS.md` | Ready. All dimensions were within 1% validation MSE; dimension 2 had the fewest parameters. Trading outcomes did not drive selection. |
| C07 | The final primary effect is the paired MPS-minus-ANN Sharpe difference across ten matched PPO seeds. | Planned final seed-level and aggregate artifacts | Pending. No value, sign, or uncertainty statement is authorized yet. |
| C08 | Fixed-split conclusions are or are not directionally stable under a shifted market-period evaluation. | Planned rolling/expanding-window artifacts | Pending. Do not imply temporal robustness from annual slices of the same fixed test period. |
| C09 | The implementation is a classical MPS/tensor-network simulation and did not execute a quantum circuit or use quantum hardware. | `run_experiment.py`, `results/run_manifest.json`, `docs/EXPERIMENT_PROTOCOL.md` | Ready and required in the abstract/methods or limitations. |
| C10 | The RTX 5060 was available but CPU was retained after a matched encoder benchmark was faster. | `results/smoke/2026-07-27/device_benchmark.csv`, `REPRODUCIBILITY.md` | Ready as reproducibility context only; it is not a model-performance result. |

## Tables

| ID | Planned content | Source | Status |
|---|---|---|---|
| T01 | Data splits, state construction, costs, PPO budget, seeds, and encoder controls | `results/run_manifest.json`, `docs/EXPERIMENT_PROTOCOL.md`, `docs/DECISIONS.md` | Ready for methods; final run manifest still pending |
| T02 | Bond-dimension validation, parameter, runtime, and descriptive trading sensitivity | `results/pilots/2026-07-28/dimension_summary.csv`, `results/pilots/2026-07-28/dimension_paired.csv` | Ready |
| T03 | Ten-seed Base, ANN, and selected-MPS portfolio outcomes with paired differences | Planned final evaluation artifacts | Pending |
| T04 | Shifted-window robustness summary | Planned temporal-robustness artifacts | Pending |

## Figures

| ID | Planned content | Source | Status |
|---|---|---|---|
| F01 | MPS validation error and runtime by bond dimension | `results/pilots/2026-07-28/dimension_summary.csv` | Data ready; final figure pending |
| F02 | Paired ten-seed MPS-minus-ANN primary effect with uncertainty | Planned final paired artifact | Pending |
| F03 | Fixed-split versus shifted-window comparison, if legible within the page budget | Planned robustness artifacts | Pending |

## Primary references verified for drafting

- Schulman et al., *Proximal Policy Optimization Algorithms*,
  arXiv:1707.06347.
- Liu et al., *FinRL: A Deep Reinforcement Learning Library for Automated Stock
  Trading in Quantitative Finance*, arXiv:2011.09607.
- Stoudenmire and Schwab, *Supervised Learning with Quantum-Inspired Tensor
  Networks*, arXiv:1605.05775.
- Biamonte et al., *Quantum Machine Learning*, Nature 549, 195-202 (2017),
  doi:10.1038/nature23474.

These references establish methods and context. They do not provide evidence
for this repository's empirical results.
