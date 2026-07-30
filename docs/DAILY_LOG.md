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

## 2026-07-28 - Capacity decision and final-evaluation launch

Completed:

1. Confirmed all three 20k-step capacity jobs completed with empty experiment
   stderr.
2. Fixed the capacity guard to exclude elapsed runtime from deterministic
   Base/ANN control comparisons.
3. Added a regression test that permits runtime variation while retaining
   metric-drift detection.
4. Generated the guarded dimension summary, seed-level paired effects, and
   SHA-256 provenance manifest.
5. Applied the frozen selection rule without consulting trading outcomes.
6. Selected MPS bond dimension 2 because all validation MSE values were within
   1% and dimension 2 had the fewest parameters.
7. Preserved descriptive pilot trading results showing no stable MPS advantage.
8. Updated the experiment decision log and paper claim registry.
9. Added a paper-ready capacity figure in raster and vector formats.
10. Visually inspected the capacity figure for labels, clipping, and selection
    wording.
11. Added tested final-run validation for the exact budget, dimension, seed
    set, conditions, and result schema.
12. Added deterministic paired-seed bootstrap and exact sign-test reporting.
13. Added source/output hashing for the final analysis package.
14. Added a tested paired-effect figure pipeline for the final result.
15. Drafted an IEEE-style short-paper source containing verified methods and
    capacity evidence.
16. Kept the final-effect and robustness sections explicitly evidence-gated.
17. Dry-ran and inspected the exact ten-seed matrix plan.
18. Launched the dimension-2, 20k-step, seeds 0--9 final evaluation on CPU.
19. Diagnosed a missing-directory failure at the first partial checkpoint.
20. Hardened checkpoint writes to recreate the exact run directory and replace
    metrics, curves, and configuration files atomically.
21. Added a regression test for checkpoint recovery after directory loss.
22. Resumed the same frozen matrix with separate retained failure/resume logs.
23. Ran the complete post-change suite: 66 tests passed.
24. Committed and pushed each independently reviewable checkpoint to `main`.

Capacity evidence:

- Validation MSE for dimensions 2/4/8: 1.280290, 1.281044, and 1.275307.
- Parameter counts: 97, 369, and 1,441.
- Mean MPS-minus-ANN pilot Sharpe differences: -0.045, -0.027, and -0.046.
- MPS exceeded ANN in one of three seeds at every tested dimension.
- These pilot trading values were not used for dimension selection.

Final evaluation:

- Raw root:
  `work/final_evaluation_2026-07-28` relative to the parent project directory.
- Configuration: 20,000 PPO steps; MPS dimension 2; seeds 0 through 9; 60
  encoder epochs maximum; patience 10; batch size 512; CPU encoder.
- `matrix.stderr.log` preserves the first checkpoint failure.
- `resume.stdout.log` and `resume.stderr.log` record the active resumed run.
- No final performance value is available or claimed at this checkpoint.

Verification:

- Capacity-analysis regression: 9 tests passed.
- Final-analysis validation: 7 tests passed.
- Capacity-figure validation: 3 tests passed.
- Final-figure validation: 3 tests passed.
- Checkpoint/locking focus: 6 tests passed.
- Complete suite: 66 tests passed.
- LaTeX citation keys all resolve; source braces are balanced.
- No local TeX compiler is available, so the manuscript PDF is not yet
  compile- or render-verified.

Hardware:

- NVIDIA reports an RTX 5060 Laptop GPU with 8,151 MiB VRAM.
- Reference PyTorch is `2.10.0+cpu`; CUDA is unavailable in that environment.
- The resumed final evaluation uses CPU, matching the prior measured device
  decision. No GPU acceleration is claimed.

Protocol deviations:

- None. The selected dimension, budget, seeds, and encoder settings match the
  frozen protocol and decision log.
- The checkpoint incident affects runtime interpretation, not the experiment
  configuration. Any completed manifest will retain actual runtime metadata.

Next:

- Allow the resumed ten-seed evaluation to complete and validate all 30
  condition/seed endpoints.
- Run the final summarizer and paired-effect figure pipeline.
- Begin the prespecified shifted-period robustness implementation without
  changing the fixed-split primary estimand.

## 2026-07-29 - Durable final-evaluation recovery

Completed:

1. Reconciled the prior automation handoff with the clean, synchronized
   repository before making changes.
2. Rechecked open issues 3, 4, and 5 and retained the final ten-seed evaluation
   as the highest-value unfinished deliverable.
3. Verified that neither retained final-evaluation attempt was active.
4. Confirmed both prior attempts stopped before producing any endpoint metric.
5. Preserved both failure logs rather than treating the attempts as results.
6. Reverified the RTX 5060 visibility and 8,151 MiB reported VRAM.
7. Reverified that the reference PyTorch 2.10.0 build is CPU-only.
8. Ran the unchanged pre-fix suite with the required dependency path: 66 tests
   passed.
9. Moved active raw-checkpoint guidance to a durable, repository-local,
   gitignored root.
10. Reproduced the remaining Windows failure on the live workload.
11. Measured the failing temporary checkpoint path at 264 characters.
12. Shortened the atomic checkpoint basenames and temporary basenames.
13. Retained read compatibility with legacy checkpoint basenames.
14. Extended the checkpoint regression to assert the compact artifact set.
15. Ran 30 focused experiment tests successfully.
16. Ran the complete post-fix suite successfully: 66 tests passed.
17. Restarted the frozen 20,000-step, dimension-2, seeds-0-through-9 matrix
    without changing any experimental setting.
18. Verified the real workload wrote the first atomic checkpoint with empty
    stderr.
19. Verified continued progress through five completed Base endpoints.
20. Documented the durable checkpoint policy and updated the deliverables
    tracker.

Active run:

- Raw root: `local_runs/final_evaluation_2026-07-29`.
- Configuration: 20,000 PPO steps; MPS dimension 2; seeds 0 through 9; 60
  encoder epochs maximum; patience 10; batch size 512; CPU encoder.
- Checkpoints verified at this log point: Base seeds 0 through 4.
- The partial endpoint values are progress evidence only and are not paper
  results until all 30 condition/seed endpoints pass the guarded final
  summarizer.
- Resume stdout and stderr are retained at the raw root.

Incident:

- The earlier work-directory failures and the first repository-local restart
  shared one immediate symptom but had different path contexts. The live
  repository-local reproduction isolated the actionable Windows cause: the
  descriptive job directory plus `ppo_backtest_metrics.partial.csv.tmp`
  produced a 264-character path. The compact atomic filenames keep checkpoint
  paths below that boundary while legacy reads remain supported.

Hardware:

- NVIDIA reports an RTX 5060 Laptop GPU with 8,151 MiB VRAM.
- Reference Python is 3.12.10 with PyTorch 2.10.0+cpu; CUDA availability is
  false and device count is zero.
- The active final matrix uses CPU, consistent with the frozen protocol and
  prior measured device decision. No GPU acceleration is claimed.

Next:

- Allow the resumable final matrix to complete.
- Validate all 30 endpoints with `scripts/summarize_final_evaluation.py`.
- Generate and visually inspect the paired-effect figure.
- Publish only validated, hash-linked compact artifacts.
- Then begin the prespecified temporally shifted robustness evaluation.

## 2026-07-29 - Literature positioning and contribution revision

Completed:

1. Reframed the research question around the incremental decision value of a
   frozen MPS signal under matched PPO conditions.
2. Made the Base, ANN-signal, and MPS-signal comparison explicit in the
   question.
3. Replaced the broad contribution paragraph with three bounded contributions:
   the FinRL integration, prespecified matched evaluation, and
   claim-to-artifact workflow.
4. Expanded related work into financial DRL, supervised tensor-network
   learning, and tensor-network RL.
5. Distinguished FinRL infrastructure from evidence that a supervised signal
   improves a trading policy.
6. Added deep-RL seed and implementation variability as motivation for paired
   policy-seed reporting.
7. Distinguished supervised MPS prediction from sequential portfolio
   evaluation.
8. Distinguished this frozen-signal study from Liu and Fang's MPS policy and
   Hamiltonian formulation.
9. Added three primary references and registered all drafting sources in the
   claim-traceability file.
10. Verified that every citation key resolves, bibliography keys are unique,
    TeX/BibTeX braces balance, and the complete 66-test suite passes.

Guardrails:

- No final-evaluation or robustness value was added.
- The revised text does not claim a quantum circuit, quantum hardware,
  quantum speedup, or reproduction of the prior variational RL algorithm.
- A PDF was not produced because no local TeX compiler is available; page
  count and visual layout remain pending.

## 2026-07-29 - Guarded final result and robustness prespecification

Completed:

1. Confirmed the repository was clean and synchronized before work.
2. Rechecked open issues 3, 4, and 5 against current artifacts.
3. Verified all 30 configured PPO endpoints plus the equal-weight reference
   were present in the interrupted final run.
4. Confirmed the missing manifest and running status were caused by a
   post-training figure-path failure, not incomplete policy training.
5. Reverified the RTX 5060 and 8,151 MiB VRAM.
6. Reverified that reference PyTorch 2.10.0+cpu exposes no CUDA device.
7. Ran the pre-change full suite: 66 tests passed.
8. Added Windows-safe compact plot-path selection.
9. Added a recovery path that finalizes complete tabular artifacts without
   retraining.
10. Required every configured condition/seed key in metrics and curves before
    recovery.
11. Required all final tables, one bootstrap row, a running status, and no
    preexisting manifest.
12. Preserved the original training commit and recorded the finalization
    commit separately.
13. Derived recovery completion time from the latest required tabular artifact
    timestamp and disclosed that basis.
14. Explicitly marked unavailable encoder timing instead of estimating it.
15. Corrected manifest limitations so seed count and parameter matching are
    configuration-aware.
16. Added focused tests for compact paths and recovered finalization.
17. Ran 32 focused experiment tests and the expanded 68-test full suite.
18. Finalized the existing ten-seed run without retraining.
19. Generated the guarded condition summary, every paired seed effect, primary
    inference JSON, and SHA-256 provenance manifest.
20. Independently regenerated all four guarded outputs and obtained identical
    SHA-256 hashes.
21. Copied the source manifest, all per-seed metrics, full equity curves,
    signal metrics, and encoder history into the committed final package.
22. Generated raster and vector paired-effect figures and visually inspected
    the raster output.
23. Integrated the fixed-split final result into the abstract, Results,
    conclusion, table, figure, claim registry, README, and deliverables tracker.
24. Preserved the negative/inconclusive finding: no MPS advantage is claimed.
25. Closed GitHub issues 3 and 4 with completion evidence; issue 5 remains
    open.
26. Froze a non-overlapping 2017-2018 temporal-robustness protocol before
    viewing shifted-window outcomes.
27. Prespecified robustness dates, ten matched seeds, 20,000 PPO steps,
    dimension 2, boundary leakage controls, outcomes, and interpretation.
28. Recorded that temporal robustness remains pending and cannot be inferred
    from annual slices of the primary trajectory.

Final fixed-split evidence:

- ANN mean Sharpe: 0.799767.
- MPS mean Sharpe: 0.761550.
- Mean paired MPS-minus-ANN Sharpe: -0.038217.
- Paired-seed bootstrap 95% interval: [-0.089151, 0.015716].
- Positive/negative paired differences: 3/7.
- Exact two-sided sign-test p-value: 0.34375.
- Mean annual return: 0.170499 ANN and 0.167189 MPS.
- The MPS test prediction MSE was slightly lower (1.499920 versus 1.510125),
  but this did not translate into higher mean downstream Sharpe.

Interpretation:

- The fixed-split result does not establish an MPS advantage.
- The interval includes zero and is not evidence of equivalence.
- The interval describes ten policy-training seeds on one historical split,
  not population or causal uncertainty.
- This is a classically simulated MPS result; no quantum hardware or circuit
  was used.

Recovery disclosure:

- All scientific tables were complete before the original plot failure.
- Completion was recovered without retraining or changing any configuration.
- Encoder timing was not persisted before failure and remains explicitly
  unavailable.

Verification:

- Focused recovery/finalization tests: 32 passed.
- Complete suite: 68 passed.
- All guarded summary outputs regenerated byte-identically.
- Every source and output SHA-256 in `results/final/final_manifest.json`
  matches the committed file.
- Final paired-effect PNG visually inspected; vector PDF generated.
- No local TeX compiler is installed, so manuscript page count and rendered
  layout remain unverified.

Hardware:

- NVIDIA GeForce RTX 5060 Laptop GPU, 8,151 MiB VRAM, driver 610.47.
- Reference Python 3.12.10 and PyTorch 2.10.0+cpu.
- Final training used CPU. No GPU acceleration is claimed.

Next:

- Implement the exact dates and target-boundary guards frozen in
  `docs/ROBUSTNESS_PROTOCOL.md`.
- Add focused leakage and window-validation tests.
- Dry-run the shifted 2017-2018 matrix, then launch or resume all 30 endpoints.
- Compile and visually inspect the IEEE manuscript once a free local TeX
  toolchain is available.
