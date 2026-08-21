# ATL residual-MPS ensemble frozen protocol

## Objective

Test whether a higher-capacity, rank-aware residual MPS improves fresh-window
forecasting and cost-adjusted decisions relative to an exactly parameter-matched
ANN. Architecture and policy choices were fixed before inspecting the May-June
2026 test results.

## Architecture

Each MPS member contains:

- a 13-site classical MPS with bond dimension 5;
- a linear residual path over the same standardized inputs; and
- learned MPS and residual mixing coefficients.

Each member has exactly 586 trainable parameters. The ANN control is a
13-39-1 tanh network with exactly 586 parameters. Both use the same inputs,
rows, normalization, robust-regression loss, ranking loss, optimizer, dates,
costs, early stopping, and paired seeds.

The deployed model is a five-member residual-MPS ensemble using seeds 0-4.
Its decision score is `ensemble mean - one ensemble standard deviation`.
Confidence combines validation residual error and member disagreement.

## Objective function

Training minimizes smooth-L1 next-hour return loss with beta 0.25 plus a 0.10
weighted pairwise ranking loss within each timestamp. The ranking term rewards
correct cross-sectional ordering rather than only point prediction.

## Frozen periods

- Training: January 5-April 10, 2026.
- Validation, thresholding, and abstention decision: April 15-May 15, 2026.
- Fresh one-time test: May 18-June 30, 2026.

Rows whose target crosses a split, overnight, or weekend boundary are removed.
The current pipeline also resets lagged-price history whenever observations are more than two
hours apart, so Friday-to-Monday changes cannot be mislabeled as hourly inputs.
Normalization is fit on training rows only.

## Comparisons and endpoints

- Ten paired residual-MPS and ANN members, seeds 0-9.
- Matched five-member MPS and ANN ensembles, seeds 0-4.
- Ten basis points of modeled cost per unit portfolio turnover.
- Prediction: MSE, MAE, directional accuracy, and rank correlation.
- Portfolio: total return, drawdown, turnover, modeled cost, and activity.
- Paired-seed bootstrap interval for MPS-minus-ANN total return.

The forecasting model is called better only if the fresh prediction evidence
supports that statement. A system-level abstention outcome is reported
separately and is not relabeled as superior prediction.
