# Saved Results

## Evidence hierarchy

1. [`final/`](final/) is the corrected primary ten-seed 2019--2023 evaluation
   used by the manuscript.
2. [`robustness/shifted/`](robustness/shifted/) is the prespecified secondary
   ten-seed 2017--2018 evaluation.
3. [`pilots/`](pilots/) contains the guarded PPO-budget and MPS-capacity
   selection evidence.
4. Root-level tables are the older three-seed reference study. They remain
   preserved for audit history but are not the final manuscript evidence.
5. [`smoke/`](smoke/) contains reduced engineering checks and cannot support
   performance claims.

## Corrected primary result

| Condition | Mean annual return | Mean Sharpe | Mean max drawdown |
|---|---:|---:|---:|
| Base FinRL | 15.36% | 0.689 | -41.96% |
| ANN signal | 17.96% | 0.821 | -34.07% |
| MPS signal | 17.04% | 0.784 | -31.97% |

The mean paired MPS-minus-ANN Sharpe difference is -0.0366 with paired-seed
bootstrap interval [-0.1129, 0.0448]. MPS is higher in 3/10 matched seeds.

## Shifted-window result

| Condition | Mean annual return | Mean Sharpe | Mean max drawdown |
|---|---:|---:|---:|
| Base FinRL | 10.73% | 0.652 | -21.50% |
| ANN signal | 9.36% | 0.627 | -23.04% |
| MPS signal | 11.61% | 0.714 | -23.73% |

The shifted mean paired Sharpe difference is +0.0862 and MPS is higher in 9/10
seeds, while the annualized-return block-bootstrap interval includes zero. The
sign reversal supports window sensitivity rather than stable MPS superiority.

## Paper-facing figures

- `figures/dimension_sensitivity.png` and `.pdf`: MPS validation MSE and
  parameter-count sensitivity.
- `figures/final_paired_effect.png` and `.pdf`: corrected primary paired Sharpe
  differences and seed-bootstrap interval.
- `figures/shifted_paired_effect.png` and `.pdf`: shifted paired Sharpe
  differences, available if the page budget permits.
- `figures/equity_curves.png`, `sharpe_by_condition.png`, and
  `encoder_validation_loss.png`: corrected primary diagnostics.

Every paper claim is mapped to its exact artifact in
[`../paper/CLAIM_TRACEABILITY.md`](../paper/CLAIM_TRACEABILITY.md).