# Daily Research Log

This append-only log records completed milestones, verification, deviations
from the experiment protocol, and the next planned task. Performance claims
belong in generated result files, not in undocumented prose here.

## 2026-07-27 - Submission audit and protocol

Completed:

- Verified the current SecureFinAI special-track scope, IEEE-proceedings
  statement, EasyChair destination, and published dates.
- Verified SmartCom's short-paper page limits and formatting requirements.
- Documented the conflict between the August 25 special-track deadline and the
  August 1 general SmartCom deadline.
- Created the paper deliverables tracker and a staged experiment protocol.
- Preserved the existing negative/inconclusive result as an acceptable outcome.

Evidence:

- `docs/PAPER_DELIVERABLES.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `results/run_manifest.json`
- Open GitHub issues 3, 4, and 5

Protocol deviations: none; this is the initial expanded-study protocol.

Author action:

- Confirm in EasyChair or with the organizers that August 25 is the applicable
  SecureFinAI deadline.
- Review the primary outcome and the staged budget/dimension selection rules
  before expanded results are generated.

Next:

- Add CLI/config validation and manifest provenance for PPO budgets, matched
  seeds, and MPS bond dimensions.
- Add tests and run a reduced smoke matrix without treating it as paper
  evidence.

## 2026-07-27 - Configuration and smoke-matrix continuation

Completed:

1. Audited the host GPU, driver visibility, VRAM, PyTorch build, and CUDA
   availability.
2. Added pre-training validation for seed sets, PPO budgets, MPS dimensions,
   encoder controls, costs, and devices.
3. Exposed MPS bond dimension through the command line.
4. Exposed encoder epoch, patience, and batch-size controls for bounded pilots.
5. Allowed non-parameter-matched MPS dimensions while retaining the strict
   dimension-4 equality assertion.
6. Verified MPS parameter counts at dimensions 2, 4, and 8.
7. Added explicit `auto`, `cpu`, and `cuda` encoder-device resolution.
8. Added actual device placement for encoder training and signal inference.
9. Added start/completion status, source commit, environment, device, UTC, and
   elapsed-time provenance to run artifacts.
10. Expanded focused verification from 6 to 23 passing tests.
11. Completed end-to-end smoke runs for dimensions 2, 4, and 8.
12. Added a guarded smoke-matrix summarizer that rejects incomparable runs.
13. Published a machine-readable smoke matrix explicitly marked as unsuitable
   for paper evidence.
14. Updated reproduction commands, hardware boundaries, and integrity-test
   documentation.

Verification:

- `py -3.12 -m pytest -q test_experiment.py test_smoke_matrix.py`: 23 passed.
- Bond dimensions 2, 4, and 8 each completed one seed, 512 PPO steps, and one
  encoder epoch across all three PPO conditions.
- Recorded MPS parameter counts: 97, 369, and 1,441.
- Smoke runtimes: 20.15, 20.18, and 20.60 seconds on CPU.
- Base and ANN control Sharpe values were identical across the three smoke
  runs, as expected.

Hardware:

- Windows/NVIDIA driver exposed an RTX 5060 Laptop GPU with 8,151 MiB reported
  VRAM.
- Active PyTorch was `2.10.0+cpu`; `torch.cuda.is_available()` was false.
- Smoke runs therefore used CPU, and no GPU acceleration is claimed.

Interpretation:

- Smoke Sharpe values are execution checks only. They are not capacity evidence
  and must not influence selection of a paper configuration.

Protocol deviations: none.

Next:

- Define and validate resumable matrix-run orchestration.
- Evaluate a free CUDA-enabled environment separately without changing the
  reference lock file.
- Begin the prespecified PPO training-budget pilot.
