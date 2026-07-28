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

## 2026-07-27 - Resumable orchestration and training-budget pilot

Completed:

1. Added deterministic experiment-matrix planning across budgets and MPS
   dimensions.
2. Added machine-readable matrix plans containing every resolved job setting.
3. Added complete command construction for seeds, encoder controls, devices,
   data, FinRL, and output paths.
4. Added dry-run support for reviewing a matrix before spending compute.
5. Added completed-job detection and automatic skipping.
6. Added stale-configuration detection to prevent silent result overwrite.
7. Added partial policy-result loading by condition and seed.
8. Added checkpoint consistency validation across metrics and equity curves.
9. Bound every partial checkpoint to its exact experiment configuration.
10. Normalized tuple-valued seed configurations for stable JSON comparison.
11. Added reproducible NVIDIA/PyTorch acceleration diagnostics.
12. Installed the exact-version CUDA wheel in an isolated free local directory
    without changing the reference dependency lock.
13. Verified an actual CUDA matrix multiplication on the RTX 5060.
14. Added per-encoder, inference, total signal-pipeline, and per-policy runtime
    measurements.
15. Ran matched CPU/CUDA encoder benchmarks and retained CPU as the faster
    measured device.
16. Added final PPO training diagnostics including explained variance, KL,
    clipping, entropy, and loss.
17. Added exclusive output-directory locks to prevent concurrent writers.
18. Added stale-lock recovery for force-stopped experiment processes.
19. Fixed checkpoint date restoration, including mixed CSV timestamp formats.
20. Completed matched three-seed evaluations at 5,000 PPO steps.
21. Completed matched three-seed evaluations at 10,000 PPO steps.
22. Completed matched three-seed evaluations at 20,000 PPO steps.
23. Added a guarded budget-pilot summarizer with paired seed comparisons.
24. Published machine-readable endpoint summaries and an interpretation note.
25. Selected 20,000 steps using the protocol's decision rule and recorded the
    decision without claiming convergence.

Verification:

- Full focused suite: 37 tests passed before the later checkpoint additions.
- Experiment tests after locking/date fixes: 29 passed.
- Matrix tests: 7 passed.
- Budget-summary tests: 3 passed.
- Acceleration tests: 2 passed.
- All 27 PPO condition/seed endpoint evaluations completed: three budgets,
  three conditions, and three matched seeds.

Key results:

- MPS-minus-ANN paired mean Sharpe: -0.041 at 5k, +0.024 at 10k, and -0.027 at
  20k steps.
- MPS had higher Sharpe in 0/3, 1/3, and 1/3 paired seeds, respectively.
- All three moving-block-bootstrap intervals for the annualized return
  difference included zero.
- Mean final PPO explained variance increased materially by 20k steps.
- The result is budget-sensitive and does not support a stable MPS advantage.

Incident and resolution:

- A shell timeout left a child experiment process alive. A resume attempt then
  created two writers for one directory. Both exact process trees were stopped,
  the six saved condition/seed pairs were verified against their curves, and
  concurrent-writer locking was added before continuing.
- Post-processing of resumed curves exposed mixed date encodings. Typed mixed
  date restoration was added and tested; all nine 20k evaluations were then
  summarized without retraining.

Protocol decisions:

- Use 20,000 PPO steps for the final ten-seed evaluation.
- Retain CPU encoder execution because the matched signal pipeline took 5.50
  seconds on CPU versus 13.60 seconds on CUDA.

Next:

- Run the prespecified MPS bond-dimension sensitivity at 20,000 PPO steps.
- Select the primary MPS dimension using validation evidence and computational
  practicality before the final ten-seed comparison.

## 2026-07-28 - Capacity-analysis safeguards and pilot launch

Completed:

1. Confirmed the repository was clean and synchronized before work.
2. Rechecked open issues 3, 4, and 5 against the deliverables tracker.
3. Verified the earlier submission audit, configuration work, and budget pilot
   were already complete, avoiding duplicate work.
4. Reverified the RTX 5060, driver, reference Python environment, and actual
   CPU-only PyTorch device.
5. Ran the pre-change full suite: 43 tests passed.
6. Prespecified the MPS dimension-selection rule before viewing capacity
   results.
7. Implemented a guarded capacity-pilot summarizer.
8. Added completed-run validation.
9. Added matched-seed and expected-condition validation.
10. Added fixed non-dimension configuration validation.
11. Added signal- and portfolio-schema validation.
12. Added Base/ANN control-stability validation across dimensions.
13. Added seed-level MPS-minus-ANN paired outputs.
14. Added validation, parameter, runtime, and descriptive portfolio summaries.
15. Required the complete prespecified dimension set before publication.
16. Added source and output SHA-256 provenance to the generated pilot manifest.
17. Added eight focused capacity-analysis tests.
18. Ran the post-change full suite: 51 tests passed.
19. Documented exact matrix and summarization commands.
20. Separated the 5k reference configuration from the selected 20k expanded
    budget in the reproducibility guide.
21. Created a paper evidence/claim registry with ready and pending claims.
22. Created a paper-source policy that prohibits estimated or favorable
    placeholders.
23. Added five verified primary scholarly references for drafting.
24. Dry-ran and inspected the exact 20k-step, dimensions 2/4/8, seeds 0/1/2
    matrix plan.
25. Launched the resumable capacity matrix on the measured CPU device.

In-progress experiment:

- Raw root:
  `work/dimension_pilot_2026-07-28` relative to the parent project directory.
- Matrix: 20,000 PPO steps; dimensions 2, 4, and 8; matched seeds 0, 1, and 2;
  60 encoder epochs maximum; patience 10; batch size 512; CPU encoder.
- The first dimension-2 Base/seed-0 checkpoint was preserved.
- A foreground tool timeout left the original process attached to an abandoned
  output pipe. After verifying the exact three-process tree and saved
  checkpoint, that tree was stopped and the same matrix was resumed with
  stdout/stderr redirected to `matrix.stdout.log` and `matrix.stderr.log`.
- The resumed process has one writer, an exclusive output lock, increasing CPU
  time, and an empty stderr log at this checkpoint.

Evidence:

- `docs/EXPERIMENT_PROTOCOL.md`
- `scripts/summarize_dimension_pilot.py`
- `test_dimension_pilot.py`
- `REPRODUCIBILITY.md`
- `paper/CLAIM_TRACEABILITY.md`
- `paper/references.bib`

Hardware:

- NVIDIA reports an RTX 5060 Laptop GPU with 8,151 MiB VRAM.
- The reference PyTorch build is `2.10.0+cpu`; the capacity run uses CPU.
- Prior matched engineering evidence retained CPU because this small signal
  pipeline was faster there. No GPU acceleration is claimed.

Protocol deviations:

- None. The experiment settings and selection rule match the frozen protocol.
- The host sleep/output-pipe delay affects wall-clock timing interpretation but
  not the saved configuration or endpoint metrics. The completed raw manifests
  will determine which runtime fields are safe to report.

Next:

- Allow the resumable dimension matrix to complete.
- Run the guarded summarizer and inspect all provenance hashes.
- Record the validation-based primary-dimension decision before launching the
  ten-seed final evaluation.
