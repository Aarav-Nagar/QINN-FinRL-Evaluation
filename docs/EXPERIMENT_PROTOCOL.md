# Experiment Protocol

Version: 0.1, drafted 2026-07-27

This protocol defines the analyses planned for the SecureFinAI short paper
before the expanded results are observed. Deviations must be dated, justified,
and retained in the daily log.

## Objective and hypotheses

The objective is to test whether appending a frozen learned return signal to a
standard FinRL PPO state changes out-of-sample portfolio performance.

- H1: ANN-signal PPO and MPS-signal PPO differ in paired out-of-sample
  risk-adjusted performance.
- H2: MPS bond dimension changes representation capacity, runtime, signal
  quality, and downstream trading performance.
- H3: conclusions from the fixed 2013-2018 training / 2019-2023 test split are
  directionally stable under at least one temporally shifted evaluation.

These are two-sided empirical questions. No directional MPS benefit is assumed.

## Fixed elements

- Assets: the existing 15-stock Nasdaq subset.
- Data: the repository's versioned 2013-2023 snapshot and recorded SHA-256
  checksums.
- Conditions: base FinRL state, base plus ANN signal, and base plus MPS signal.
- Transaction cost: 0.10% per executed trade.
- Initial portfolio value: USD 1,000,000.
- PPO seeds: matched across all conditions.
- Representation split: train through 2017-12-29 and validate during 2018.
- Primary fixed-split evaluation: 2019-01-02 through 2023-12-28.
- Signal models are frozen before PPO evaluation.
- No test-period results may be used to retrain an encoder.

## Staged computation

### Stage A: configuration smoke test

Run one seed on a reduced training budget for every newly exposed configuration
path. This stage validates execution and artifact provenance; it is not evidence
for the paper's performance claims.

### Stage B: PPO training-budget pilot

Evaluate 5,000, 10,000, and 20,000 PPO steps with matched pilot seeds. Prefer
seeds 0, 1, and 2 if compute permits. Choose the smallest budget whose endpoint
performance and training diagnostics show no material systematic improvement at
the next budget. If diagnostics do not support convergence, report the training
budget as a limitation rather than claiming convergence.

The budget decision must be based on recorded diagnostics across conditions,
not on whichever budget favors MPS.

### Stage C: MPS capacity sensitivity

Evaluate bond dimensions 2, 4, and 8. Dimension 16 is optional and may be added
only if runtime remains practical. Record:

- actual trainable parameter count;
- encoder training and inference runtime;
- validation and test signal metrics;
- downstream portfolio metrics under matched PPO settings.

Parameter equality with the reference ANN is not assumed outside the
dimension-4 comparison. Results must distinguish capacity sensitivity from the
strict parameter-matched comparison.

Choose a primary MPS dimension using representation-validation evidence and
computational practicality before examining its final ten-seed trading
comparison. The selection rule is:

1. Find the lowest MPS validation-period signal MSE among the prespecified
   dimensions.
2. Treat any dimension within 1% of that minimum validation MSE as practically
   tied.
3. Within that set, select the dimension with the fewest trainable parameters;
   use MPS fit time as the final tie-breaker.

Test-period signal metrics and pilot trading performance are descriptive
sensitivity evidence and must not determine this selection. Preserve all
sensitivity results.

### Stage D: final matched evaluation

Run seeds 0 through 9 for base, ANN-signal, and the prespecified primary MPS
configuration using the selected PPO budget. Each run must use identical dates,
features, costs, and environment settings. Failed seeds may be rerun with the
same configuration; exclusions require an explicit recorded reason.

### Stage E: temporal robustness

Run at least one temporally shifted or rolling/expanding-window evaluation.
Window boundaries must prevent representation training, PPO training, or
normalization from observing its evaluation period. If full PPO retraining is
computationally infeasible, label any reduced robustness analysis as secondary.

The exact pre-outcome shifted window, fixed seed set, settings, boundary
controls, estimand, and interpretation rules are frozen in
`docs/ROBUSTNESS_PROTOCOL.md`. That document governs Stage E where it is more
specific than this general protocol.

After the shifted result exposed an evaluation-length confound, the separately
versioned `docs/EQUAL_WINDOW_PROTOCOL.md` froze a three-cell extension before
its two new outcomes. It fixes four-year PPO training spans, three-year
encoder-fit spans, one-year validation spans, two-year evaluation spans, and
the same ten seeds and model controls. It governs that exploratory extension
without altering the primary or shifted estimands.

## Outcomes

### Primary outcome

- Paired MPS-minus-ANN difference in annualized Sharpe ratio across the ten PPO
  seeds in the fixed-split final evaluation.

### Secondary outcomes

- Annualized mean return.
- Annualized volatility.
- Maximum drawdown.
- Cumulative and annualized turnover.
- Transaction costs paid.
- Final account value.
- ANN-minus-base and MPS-minus-base paired differences.
- Signal MSE, MAE, directional accuracy, and information coefficient.
- Calendar-year portfolio metrics.
- Runtime and parameter count.

The equal-weight portfolio remains a non-PPO reference and is not assigned
artificial seeds.

## Statistical reporting

- Report every seed, not only aggregate values.
- Report mean, standard deviation, and paired seed differences.
- Use paired uncertainty intervals because conditions share PPO seeds.
- Retain the existing moving-block bootstrap over daily returns as a
  complementary time-series analysis, with block length and sample count
  recorded.
- Treat intervals as uncertainty descriptions, not proof of no effect.
- Do not perform or selectively report a large set of uncorrected hypothesis
  tests.
- Discuss economic magnitude as well as sign and uncertainty.

## Reproducibility and provenance

Every run directory must contain:

- full resolved configuration and code commit;
- start/end timestamps and elapsed runtime;
- environment/package versions;
- input checksums;
- per-seed metrics and equity curves;
- encoder metrics and parameter counts;
- failures or deviations.

Generated paper tables and figures must be reproducible from committed
machine-readable results. Intermediate smoke results must be visually and
structurally separated from final paper results.

## Stopping and deviations

Computation may stop for resource or deadline constraints, but not because an
intermediate result is unfavorable. Any change to seeds, budgets, dimensions,
windows, metrics, or exclusions after results are viewed must be added to
`docs/DAILY_LOG.md` with the date, reason, and affected claims.

## Known limitations retained from the reference study

- The MPS is a classical tensor-network simulation, not a quantum-hardware
  experiment.
- The asset universe is a small Nasdaq subset.
- Historical backtests do not establish deployable profitability.
- The cost model omits spread, slippage, market impact, taxes, liquidity limits,
  and borrow costs.
- Policy and market-period instability limit generalization.
