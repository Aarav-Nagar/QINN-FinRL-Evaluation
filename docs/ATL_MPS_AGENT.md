# Aarav MPS Signal Agent for Agentic Trading Lab

This directory contains a deployable research prototype that connects a
classical matrix-product-state (MPS) signal model to Agentic Trading Lab (ATL).
It is not a prompt renamed as an MPS agent, and it does not use quantum
hardware.

## What is shared with the paper experiment

- The model is a classical MPS regressor with the same cosine/sine local
  feature map and sequential tensor contraction used in `run_experiment.py`.
- It uses 13 dimensionless market and portfolio inputs.
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
  --start 2026-04-01 --end 2026-04-11 `
  --output .atl/training_snapshots.json

py -3.12 -m atl_mps_agent.cli train `
  --snapshots .atl/training_snapshots.json `
  --artifact atl_mps_agent/artifacts/atl_mps_bond4.pt

py -3.12 -m atl_mps_agent.cli register `
  --credentials .atl/credentials.json `
  --source-repo https://github.com/Aarav-Nagar/QINN-FinRL-Evaluation

py -3.12 -m atl_mps_agent.cli run `
  --start 2026-04-15 --end 2026-04-16 `
  --artifact atl_mps_agent/artifacts/atl_mps_bond4.pt `
  --credentials .atl/credentials.json `
  --result .atl/held_out_result.json
```

The training window ends before the held-out evaluation window. The live ATL
API key is written only to `.atl/credentials.json`, which is gitignored.

## Interpretation boundary

This is a historical-simulation research agent. A short held-out run verifies
the integration and decision trace, not trading superiority, production
readiness, or safety for real capital.
