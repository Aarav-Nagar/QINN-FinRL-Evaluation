# Nested Evaluation-Horizon Evidence

This directory contains the separately versioned, post-hoc exploratory
analysis governed by `docs/NESTED_HORIZON_PROTOCOL.md`. It re-scores the
already-frozen 2019--2023 primary equity curves at every available cumulative
calendar-year endpoint. No encoder or PPO model was retrained.

## Design

Every prefix starts on 2019-01-02 and uses the same saved Base, ANN, and
dimension-2 MPS portfolio paths for matched PPO seeds 0--9. The prefixes end
after one, two, three, four, and five calendar years. The four- and five-year
results are included to prevent selective stopping after the user-requested
one- through three-year view.

The five-year endpoint exactly reproduces the frozen primary result. These
prefixes are nested and dependent, so they are not independent replications
and cannot identify a causal effect of evaluation length.

## Main result

| Cumulative horizon | ANN Sharpe | MPS Sharpe | Paired MPS-ANN | MPS wins | Seed-bootstrap 95% interval |
|---:|---:|---:|---:|---:|---:|
| 1 year | 1.549 | 1.553 | +0.005 | 4/10 | [-0.465, 0.509] |
| 2 years | 1.087 | 1.024 | -0.063 | 2/10 | [-0.156, 0.052] |
| 3 years | 1.168 | 1.086 | -0.082 | 3/10 | [-0.194, 0.036] |
| 4 years | 0.597 | 0.568 | -0.028 | 3/10 | [-0.109, 0.056] |
| 5 years | 0.821 | 0.784 | -0.037 | 3/10 | [-0.113, 0.045] |

The one-year means are essentially tied, but MPS is higher in only 4/10
matched seeds and the interval is very wide. ANN has the higher mean Sharpe
at every two- through five-year cutoff. Every interval includes zero.
Consequently, this analysis does not establish a statistically stable winner.

The result also helps interpret the positive equal-length window panel:
evaluation length by itself does not reproduce those positive MPS means when
the trained policies and 2019 start are held fixed. Differences in training
window, evaluation dates, policy refitting, or their interaction remain
possible explanations; this analysis cannot distinguish them causally.

## Secondary metrics

Paired MPS-minus-ANN mean annualized return is negative at all five cutoffs:
-0.37, -1.13, -1.83, -0.76, and -0.92 percentage points. None of the
descriptive seed-bootstrap intervals excludes zero.

MPS turnover and modeled cost are slightly higher at one year, then lower
from years two through five. At five years, its mean annualized turnover is
0.249 versus 0.275 for ANN, and its mean modeled cost is USD 1,313 versus USD
1,482. This lower trading intensity does not offset the lower mean return and
Sharpe. Mean MPS maximum drawdown is slightly worse at years two and three,
then less severe at years four and five; those intervals also include zero.

Date-level prediction outputs were not stored in the frozen primary artifact,
so prefix prediction quality was not invented or approximated. The existing
full-period signal-quality result remains separate.

## Artifact map

- `horizon_seed_metrics.csv`: all 150 condition/seed/prefix metric rows.
- `horizon_condition_summary.csv`: condition means and standard deviations.
- `horizon_paired_seed_effects.csv`: all 50 matched MPS-minus-ANN seed rows.
- `horizon_paired_metric_summary.csv`: paired means, dispersion, win counts,
  and deterministic intervals for five portfolio metrics.
- `nested_horizon_inference.json`: the bounded descriptive sign pattern.
- `nested_horizon_manifest.json`: source, protocol, table, script, and figure
  hashes.
- `../../figures/nested_horizon_paired_effect.{png,pdf}`: seed-level visual.

## Reproduction

```powershell
py -3.12 scripts\summarize_nested_horizons.py `
  --run-dir results\final `
  --freeze results\SCIENTIFIC_RESULTS_FREEZE.json `
  --protocol docs\NESTED_HORIZON_PROTOCOL.md `
  --output-dir .cache\reproduced-nested-horizons
py -3.12 scripts\plot_nested_horizons.py `
  --paired .cache\reproduced-nested-horizons\horizon_paired_seed_effects.csv `
  --summary .cache\reproduced-nested-horizons\horizon_paired_metric_summary.csv `
  --png-output .cache\reproduced-nested-horizons\nested_horizon.png `
  --pdf-output .cache\reproduced-nested-horizons\nested_horizon.pdf `
  --manifest .cache\reproduced-nested-horizons\nested_horizon_manifest.json
```

This analysis is a bounded historical diagnostic on one 15-stock universe,
not evidence of live profitability, quantum advantage, or a general
architecture ranking.
