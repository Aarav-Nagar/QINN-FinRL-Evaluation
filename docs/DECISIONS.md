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

## 2026-07-28 - Primary MPS bond dimension

Decision: use MPS bond dimension 2 for the expanded ten-seed evaluation.

Evidence:

- Prespecified dimensions: 2, 4, and 8.
- Matched pilot seeds: 0, 1, and 2 at 20,000 PPO steps.
- Validation MSE: 1.280290, 1.281044, and 1.275307, respectively.
- All dimensions were within 1% of the minimum validation MSE.
- Trainable parameters: 97, 369, and 1,441, respectively.

Rationale:

The frozen rule treats dimensions within 1% of the lowest validation MSE as
practically tied, then chooses the fewest parameters. Dimension 2 therefore
wins on parsimony. Its longer measured fit time did not invoke the final
tie-breaker because parameter counts differed.

Guardrails:

- Test-period signal metrics and pilot trading outcomes did not determine the
  selection.
- The pilot's paired trading differences remain visible and do not support a
  stable MPS advantage.
- The final comparison must use dimension 2, 20,000 PPO steps, and matched
  seeds 0 through 9 for every condition.
- The result is a classical MPS simulation, not quantum-hardware execution.

Supporting artifacts:

- `results/pilots/2026-07-28/README.md`
- `results/pilots/2026-07-28/dimension_summary.csv`
- `results/pilots/2026-07-28/dimension_paired.csv`
- `results/pilots/2026-07-28/dimension_manifest.json`
