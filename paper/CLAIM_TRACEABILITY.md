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
| C07 | In the corrected ten-seed primary evaluation, mean MPS-minus-ANN Sharpe was -0.0366, with a paired-seed bootstrap 95% interval from -0.1129 to 0.0448; MPS was higher in 3/10 seeds. | `results/final/paired_seed_effects.csv`, `results/final/primary_inference.json`, `results/final/final_manifest.json` | Ready. Bound to dimension 2, 20k PPO steps, one fixed historical split; do not call the interval population-level or causal. |
| C08 | In the prespecified shifted 2017--2018 evaluation, mean MPS-minus-ANN Sharpe was +0.0862, MPS was higher in 9/10 seeds, and the seed-bootstrap interval was [0.0011, 0.1945]; the annualized return block-bootstrap interval [-0.0083, 0.0598] included zero. | `results/robustness/shifted/paired_seed_effects.csv`, `results/robustness/shifted/robustness_inference.json`, `results/robustness/shifted/ann_vs_mps_block_bootstrap.csv`, `results/robustness/shifted/robustness_manifest.json` | Ready as secondary window-specific evidence. Report the sign reversal and do not call it a stable or architecture-wide MPS advantage. |
| C09 | The implementation is a classical MPS/tensor-network simulation and did not execute a quantum circuit or use quantum hardware. | `run_experiment.py`, `results/run_manifest.json`, `docs/EXPERIMENT_PROTOCOL.md` | Ready and required in the abstract/methods or limitations. |
| C10 | The RTX 5060 was available but CPU was retained after a matched encoder benchmark was faster. | `results/smoke/2026-07-27/device_benchmark.csv`, `REPRODUCIBILITY.md` | Ready as reproducibility context only; it is not a model-performance result. |
| C11 | Across the prespecified equal-length 2017--2018, 2019--2020, and 2021--2022 panels, mean MPS-minus-ANN Sharpe was +0.0862, +0.0925, and +0.1397; MPS was higher in 9/10, 7/10, and 6/10 seeds. The latter intervals [-0.0080, 0.1966] and [-0.0680, 0.3385] included zero. | `results/robustness/equal_windows/window_paired_metric_summary.csv`, `results/robustness/equal_windows/window_paired_seed_effects.csv`, `results/robustness/equal_windows/equal_window_manifest.json` | Ready as exploratory equal-length evidence. Report all windows and the negative five-year primary estimate; do not pool the cells or attribute them causally to market regimes. |

| C12 | Re-scoring the same frozen primary policies over cumulative one- through five-year prefixes gives paired mean MPS-minus-ANN Sharpe of +0.0048, -0.0626, -0.0821, -0.0284, and -0.0366; MPS is higher in 4/10, 2/10, 3/10, 3/10, and 3/10 seeds, and every interval includes zero. | `results/robustness/nested_horizons/horizon_paired_metric_summary.csv`, `results/robustness/nested_horizons/horizon_paired_seed_effects.csv`, `results/robustness/nested_horizons/nested_horizon_manifest.json` | Ready only as post-hoc nested evidence. Report all five dependent prefixes, disclose that the five-year primary outcome was known, and do not claim a causal horizon effect. |
| C13 | In the post-hoc benchmark-direction audit, MPS-minus-ANN annualized mean-return differences were -6.63, -0.56, and -2.15 percentage points on benchmark-down days and +8.56, +4.99, and +9.94 points on nonnegative days; every direction-state seed-bootstrap interval included zero. | `results/robustness/market_states/market_state_summary.csv`, `results/robustness/market_states/direction_contribution.csv`, `results/robustness/market_states/market_state_manifest.json` | Ready only as a bounded exploratory clue. State that the audit was declared after the equal-window outcomes, do not claim downside protection or a causal regime effect, and do not treat seed intervals as calendar-sample uncertainty. |

## Tables

| ID | Planned content | Source | Status |
|---|---|---|---|
| T01 | Data splits, state construction, costs, PPO budget, seeds, encoder controls, and corrected target boundaries | `results/final/run_manifest.json`, `results/final/run_status.json`, `docs/EXPERIMENT_PROTOCOL.md`, `docs/DECISIONS.md` | Ready |
| T02 | Bond-dimension validation, parameter, runtime, and descriptive trading sensitivity | `results/pilots/2026-07-28/dimension_summary.csv`, `results/pilots/2026-07-28/dimension_paired.csv` | Ready; retained as the manuscript's single capacity presentation |
| T03 | Ten-seed Base, ANN, and selected-MPS portfolio outcomes with paired differences | `results/final/condition_summary.csv`, `results/final/paired_seed_effects.csv` | Ready |
| T04 | Shifted-window Base, ANN, and MPS outcomes with paired differences | `results/robustness/shifted/condition_summary.csv`, `results/robustness/shifted/paired_seed_effects.csv`, `results/robustness/shifted/robustness_inference.json` | Ready |
| T05 | Equal-length two-year ANN/MPS Sharpe means, paired differences, seed wins, secondary risk/cost metrics, and signal quality | `results/robustness/equal_windows/window_condition_summary.csv`, `results/robustness/equal_windows/window_paired_metric_summary.csv`, `results/robustness/equal_windows/window_signal_quality.csv` | Ready; exploratory and unpooled |

| T06 | One- through five-year cumulative ANN/MPS Sharpe means, paired differences, wins, return, drawdown, turnover, and cost | `results/robustness/nested_horizons/horizon_condition_summary.csv`, `results/robustness/nested_horizons/horizon_paired_metric_summary.csv` | Ready as repository-only evidence; intentionally excluded from the short paper because the dependent post-hoc prefixes add limited value within the page budget |

## Figures

| ID | Planned content | Source | Status |
|---|---|---|---|
| F01 | MPS validation error and parameter count by bond dimension | `results/pilots/2026-07-28/dimension_summary.csv`, `results/figures/dimension_sensitivity.pdf` | Ready as a repository artifact; the manuscript retains Table T02 and omits this redundant figure |
| F02 | Corrected paired ten-seed MPS-minus-ANN primary effect with uncertainty | `results/final/paired_seed_effects.csv`, `results/final/primary_inference.json`, `results/figures/final_paired_effect.pdf` | Ready; PNG and vector PDF regenerated; structural validation complete, visual reinspection pending |
| F03 | Shifted-window paired seed-level effect, available if legible within the page budget | `results/robustness/shifted/paired_seed_effects.csv`, `results/robustness/shifted/robustness_inference.json`, `results/figures/shifted_paired_effect.pdf` | Ready; not currently included in the manuscript to preserve page budget |
| F04 | Equal-length seed-level Sharpe differences and descriptive intervals | `results/robustness/equal_windows/window_paired_seed_effects.csv`, `results/robustness/equal_windows/window_paired_metric_summary.csv`, `results/figures/equal_window_paired_effect.pdf` | Ready as a repository artifact; PNG/PDF structure verified, visual inspection pending, not included in the manuscript page budget |

| F05 | Same-policy paired Sharpe differences across every cumulative one- through five-year prefix | `results/robustness/nested_horizons/horizon_paired_seed_effects.csv`, `results/robustness/nested_horizons/horizon_paired_metric_summary.csv`, `results/figures/nested_horizon_paired_effect.pdf` | Ready as a repository-only artifact; intentionally excluded from the short paper |

## Primary references verified for drafting

- Schulman et al., *Proximal Policy Optimization Algorithms*,
  arXiv:1707.06347.
- Liu et al., *FinRL: A Deep Reinforcement Learning Library for Automated Stock
  Trading in Quantitative Finance*, arXiv:2011.09607.
- Liu et al., *FinRL-Meta: A Universe of Near-Real Market Environments for
  Data-Driven Deep Reinforcement Learning in Quantitative Finance*,
  arXiv:2112.06753.
- Henderson et al., *Deep Reinforcement Learning That Matters*, AAAI 2018,
  doi:10.1609/aaai.v32i1.11694.
- Stoudenmire and Schwab, *Supervised Learning with Quantum-Inspired Tensor
  Networks*, arXiv:1605.05775.
- Efthymiou et al., *TensorNetwork for Machine Learning*, arXiv:1906.06329.
- Biamonte et al., *Quantum Machine Learning*, Nature 549, 195-202 (2017),
  doi:10.1038/nature23474.
- Liu and Fang, *Quantum Tensor Networks for Variational Reinforcement
  Learning*, NeurIPS 2020 Workshop on Quantum Tensor Networks in Machine
  Learning.

These references establish methods and context. They do not provide evidence
for this repository's empirical results.
