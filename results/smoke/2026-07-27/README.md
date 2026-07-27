# Configuration Smoke Matrix - 2026-07-27

This smoke matrix verifies that MPS bond dimensions 2, 4, and 8 complete the
same end-to-end encoder, FinRL PPO, evaluation, and artifact-writing path.

## Configuration

- PPO seed: 0
- PPO steps: 512
- Encoder epochs: 1
- Encoder patience: 1
- Encoder device: CPU
- Conditions: base FinRL, ANN signal, and MPS signal

## Verified behavior

- All three dimensions completed without configuration or state-shape errors.
- The MPS parameter counts changed as expected: 97, 369, and 1,441.
- Dimension 4 remained exactly parameter matched to the 369-parameter ANN.
- Each run wrote a completed runtime record and took approximately 20 seconds.
- Base and ANN results were identical across runs, confirming that changing MPS
  capacity did not alter those control paths.

The machine-readable summary is in [`config_matrix.csv`](config_matrix.csv).

## Interpretation boundary

These runs use one seed, one encoder epoch, and 512 nominal PPO steps. They
validate execution only. Their Sharpe ratios must not be cited as sensitivity
evidence, used to select a favorable bond dimension, or included as paper
results. The planned capacity study uses the prespecified protocol in
[`docs/EXPERIMENT_PROTOCOL.md`](../../../docs/EXPERIMENT_PROTOCOL.md).

## Hardware finding

The host exposed an NVIDIA GeForce RTX 5060 Laptop GPU with 8,151 MiB reported
VRAM, but the active environment contained PyTorch `2.10.0+cpu`. CUDA was
therefore unavailable to PyTorch and these runs correctly recorded CPU encoder
execution. No GPU speedup is claimed.

An isolated, free PyTorch `2.10.0+cu130` installation subsequently verified
CUDA execution on the RTX 5060 without changing the reference environment.
Matched 10-epoch device checks are recorded in
[`device_benchmark.csv`](device_benchmark.csv). CUDA was slower for these small
encoders:

| Phase | CPU | CUDA |
|---|---:|---:|
| ANN fit | 1.63 s | 4.10 s |
| MPS fit | 2.45 s | 8.07 s |
| Total signal pipeline | 5.50 s | 13.60 s |
| End-to-end run | 22.09 s | 30.91 s |

The final experiment should therefore retain CPU encoder execution unless a
larger workload is separately shown to reverse this result. GPU presence alone
is not evidence of a speedup.
