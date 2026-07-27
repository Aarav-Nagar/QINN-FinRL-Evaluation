# Experiment Decision Log

This append-only log records analysis decisions made after a prespecified pilot
and before the affected final experiment. It is separate from the protocol so
the original plan remains visible.

## 2026-07-27 - Final PPO training budget

Decision: use 20,000 PPO steps for the expanded ten-seed evaluation.

Evidence:

- Prespecified endpoints: 5,000, 10,000, and 20,000 steps.
- Matched pilot seeds: 0, 1, and 2.
- Conditions: base FinRL, ANN signal, and dimension-4 MPS signal.
- Mean final explained variance was materially higher at 20,000 than 5,000
  steps for every condition.
- Portfolio performance and paired ANN/MPS differences remained
  budget-sensitive between 10,000 and 20,000 steps.
- The MPS-minus-ANN Sharpe difference changed sign across endpoints and was
  positive in at most one of three paired seeds at either larger budget.

Rationale:

The protocol called for the smallest budget showing no material systematic
improvement at the next tested budget. The 10,000-step endpoint did not meet
that stopping rule because the 20,000-step diagnostics and portfolio outcomes
still changed materially. Therefore, 20,000 is selected as the largest tested
budget, not as proven convergence.

Guardrails:

- The choice was not based on which endpoint favored MPS.
- The final paper must state that convergence was not established.
- The final ten-seed run must use 20,000 steps for every condition.
- A later budget change requires a new dated decision and cannot replace this
  entry.

Supporting artifacts:

- `results/pilots/2026-07-27/README.md`
- `results/pilots/2026-07-27/budget_summary.csv`
- `results/pilots/2026-07-27/budget_paired.csv`
