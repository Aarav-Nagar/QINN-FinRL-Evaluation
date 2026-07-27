# PPO Training-Budget Pilot - 2026-07-27

This pilot compares 5,000, 10,000, and 20,000 PPO steps using matched seeds
0, 1, and 2 for the base, ANN-signal, and dimension-4 MPS-signal conditions.
Encoders use the reference 60-epoch maximum and early stopping configuration.

## Decision

Use 20,000 PPO steps for the expanded ten-seed evaluation.

This is not a claim that PPO converged. It is the largest prespecified tested
budget and had materially stronger final explained-variance diagnostics than
5,000 steps across all three conditions. The 10,000-to-20,000 comparison also
changed portfolio outcomes, so the smaller budget cannot be called stable.

## Budget sensitivity

| Steps | ANN Sharpe | MPS Sharpe | MPS-ANN paired Sharpe | Positive paired seeds |
|---:|---:|---:|---:|---:|
| 5,000 | 0.735 | 0.694 | -0.041 | 0/3 |
| 10,000 | 0.618 | 0.642 | +0.024 | 1/3 |
| 20,000 | 0.709 | 0.682 | -0.027 | 1/3 |

MPS-versus-ANN Sharpe changes sign across budgets and is not consistently
positive within seeds. The paired annual-return difference also changes from
-1.12 percentage points at 5,000 steps to +1.46 and +1.94 points at 10,000 and
20,000 steps.

Complementary moving-block-bootstrap intervals for the annualized daily-return
difference all included zero:

| Steps | MPS-ANN annualized daily-return difference | 95% block-bootstrap interval |
|---:|---:|---:|
| 5,000 | -0.92 pp | [-4.63, +2.63] pp |
| 10,000 | +1.91 pp | [-2.99, +6.45] pp |
| 20,000 | +2.90 pp | [-3.89, +9.05] pp |

The paired seed means and block-bootstrap estimands are related but not
identical; they must not be substituted for one another.

## Training diagnostics

Mean final PPO explained variance increased from 0.14-0.26 at 5,000 steps to
0.75-0.84 at 20,000 steps, depending on condition. This supports using the
larger tested budget, but final-update diagnostics do not prove global policy
convergence.

## Limitations

- Three seeds are sufficient for a budget pilot, not the final uncertainty
  analysis.
- Endpoint comparisons do not provide a dense learning curve.
- Wall-clock time was highly variable across identical nominal budgets and
  seeds.
- The budget choice was made across all conditions and diagnostics, not by
  selecting the endpoint most favorable to MPS.

Machine-readable aggregates:

- [`budget_summary.csv`](budget_summary.csv)
- [`budget_paired.csv`](budget_paired.csv)
