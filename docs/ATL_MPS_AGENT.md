# Aarav MPS Signal Agent for Agentic Trading Lab

This directory contains a deployable research prototype that connects a
classical matrix-product-state (MPS) signal model to Agentic Trading Lab (ATL).
It is not a prompt renamed as an MPS agent, and it does not use quantum
hardware.

## What is shared with the paper experiment

- The model is a classical MPS regressor with the same cosine/sine local
  feature map and sequential tensor contraction used in `run_experiment.py`.
- It uses 13 dimensionless market-only inputs. Portfolio state is confined to
  execution and risk controls.
- Bond dimension 4 gives 369 trainable parameters.

## What is different

ATL provides hourly DJIA snapshots, while the paper experiment used daily data,
a different stock universe, frozen encoder signals, and a FinRL PPO policy.
The ATL model is therefore trained separately on earlier ATL snapshots and uses
a documented risk-bounded ranking policy. Results from one system must not be
presented as results from the other.

## Reproducible workflow

```powershell
py -3.12 -m atl_mps_agent.cli collect `
  --start 2026-01-05 --end 2026-02-01 `
  --output .atl/v2_train_january.json

# Repeat collection for February and March-April as specified in the v2 protocol.

py -3.12 -m atl_mps_agent.cli train `
  --snapshots .atl/v2_train_january.json .atl/v2_train_february.json `
    .atl/v2_train_march_validation.json `
  --train-end 2026-03-31T23:59:59 `
  --validation-end 2026-04-10T23:59:59 `
  --artifact atl_mps_agent/artifacts/atl_mps_v2.pt

py -3.12 -m atl_mps_agent.cli register `
  --credentials .atl/credentials.json `
  --source-repo https://github.com/Aarav-Nagar/QINN-FinRL-Evaluation

py -3.12 -m atl_mps_agent.cli run `
  --start 2026-04-15 --end 2026-04-16 `
  --artifact atl_mps_agent/artifacts/atl_mps_v2.pt `
  --credentials .atl/credentials.json `
  --result .atl/held_out_result.json
```

The training window ends before the held-out evaluation window. The live ATL
API key is written only to `.atl/credentials.json`, which is gitignored.

## Interpretation boundary

This is a historical-simulation research agent. Its offline comparison uses the
same rows, costs, parameters, splits, and seeds for the MPS and matched ANN.
The hosted ATL run separately verifies the integer-order integration trace.
Neither is evidence of production readiness or safety for real capital.
