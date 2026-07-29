# MPS Bond-Dimension Pilot

This pilot compares bond dimensions 2, 4, and 8 at the selected 20,000-step
PPO budget using matched seeds 0, 1, and 2. The guarded summary validated
completed manifests, fixed non-dimension settings, matched controls, required
schemas, and SHA-256 provenance before publishing these files.

## Prespecified selection

| Bond dimension | Parameters | Validation MSE | MPS fit seconds | Selected |
|---:|---:|---:|---:|---|
| 2 | 97 | 1.280290 | 48.679 | Yes |
| 4 | 369 | 1.281044 | 4.739 | No |
| 8 | 1,441 | 1.275307 | 9.304 | No |

All three validation MSE values were within 1% of the minimum. The frozen rule
therefore selects dimension 2 because it has the fewest parameters among the
practically tied results. Test-period and trading metrics were not used for
selection.

## Descriptive trading result

Mean MPS-minus-ANN Sharpe differences were -0.045, -0.027, and -0.046 for
dimensions 2, 4, and 8. Each dimension exceeded ANN in one of three seeds.
These outcomes are sensitivity evidence, not the final ten-seed estimate, and
do not support a stable MPS advantage.

## Files

- `dimension_summary.csv`: validation, parameter, runtime, and aggregate
  descriptive results.
- `dimension_paired.csv`: seed-level MPS-minus-ANN differences.
- `dimension_manifest.json`: frozen selection and hashes of every source
  manifest/result table and generated CSV.

