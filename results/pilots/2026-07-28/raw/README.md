# Capacity-pilot source evidence

This directory contains the compact source evidence required to regenerate the
published MPS bond-dimension summary without retraining PPO. Each job directory
retains:

- `run_manifest.json`: frozen configuration, runtime, parameter count, and
  provenance;
- `ppo_backtest_metrics.csv`: Base, ANN, and MPS portfolio metrics for matched
  seeds 0--2; and
- `signal_metrics.csv`: validation and test prediction metrics.

The three directories cover the prespecified bond dimensions 2, 4, and 8 at
20,000 PPO steps. `../dimension_manifest.json` records canonical SHA-256
digests for all nine files. Text is normalized to LF before hashing so the
same committed evidence verifies on Windows, macOS, and Linux.

From the repository root, regenerate the guarded artifacts into a disposable
cache directory:

```powershell
python scripts\summarize_dimension_pilot.py `
  results\pilots\2026-07-28\raw\dimension-pilot_steps20000_bd2_seeds0-1-2_epochs60_batch512_cpu `
  results\pilots\2026-07-28\raw\dimension-pilot_steps20000_bd4_seeds0-1-2_epochs60_batch512_cpu `
  results\pilots\2026-07-28\raw\dimension-pilot_steps20000_bd8_seeds0-1-2_epochs60_batch512_cpu `
  --summary-output .cache\reproduced-capacity\dimension_summary.csv `
  --paired-output .cache\reproduced-capacity\dimension_paired.csv `
  --manifest-output .cache\reproduced-capacity\dimension_manifest.json
```

The raw training curves and diagnostic figures are intentionally omitted
because the guarded summarizer does not consume them. The original run
manifests identify the source commit and completed UTC time.
