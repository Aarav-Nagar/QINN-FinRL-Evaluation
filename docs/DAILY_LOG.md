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

## 2026-07-30 — Shifted-window implementation and launch

Contributions:

1. Confirmed the repository was clean and synchronized at `42c0a38` before
   editing.
2. Rechecked GitHub issues and confirmed temporal robustness issue 5 was the
   only open issue.
3. Revalidated the final fixed-split artifacts and the append-only log before
   starting new work.
4. Verified the RTX 5060 Laptop GPU, 8,151 MiB VRAM, and driver 610.47.
5. Confirmed the reference PyTorch 2.10.0 build remains CPU-only.
6. Added named `primary` and `shifted` experiment windows.
7. Encoded the exact frozen 2013-2016 training and 2017-2018 evaluation dates.
8. Added validation that rejects overlapping, misordered, or protocol-drifted
   dates.
9. Combined the checksum-verified source periods before selecting the requested
   PPO window, allowing the earlier evaluation without downloading new data.
10. Required observed market-data boundaries to match the configured window.
11. Persisted each row's next trading date during feature engineering.
12. Excluded encoder fitting rows whose next-day target crosses the fitting
    boundary.
13. Excluded early-stopping rows whose next-day target crosses into the
    2017-2018 evaluation.
14. Made signal-metric split labels reflect the configured validation and test
    years.
15. Extended matrix plans, commands, job identifiers, and stale-result guards
    with the temporal window.
16. Preserved backward-compatible primary-window matrix identifiers and
    completion checks.
17. Added focused tests for frozen dates, date drift, target-boundary leakage,
    non-overlap, matrix commands, and distinct shifted job identifiers.
18. Completed a real shifted-window end-to-end smoke run: one seed, three PPO
    conditions, 64 steps, bond dimension 2, one encoder epoch, CPU.
19. Ran the complete suite after implementation: 73 tests passed in 6.74
    seconds.
20. Added exact dry-run and durable Windows launcher instructions to the
    reproducibility guide.
21. Generated and inspected the complete ten-seed shifted matrix plan before
    launch.
22. Detected an initial `git_commit: unknown` provenance failure before any PPO
    endpoint completed.
23. Anchored commit discovery to the runner repository and added a focused
    regression test; 37 focused tests passed.
24. Preserved the failed-start status under
    `failed_start_unknown_commit` rather than silently deleting it.
25. Added a durable PowerShell launcher with a persistent `matrix.log`.
26. Launched the unchanged ten-seed, 20,000-step, dimension-2 shifted matrix
    from committed revision `4c8cd2e`.
27. Verified the active status records every frozen date, CPU execution,
    PyTorch 2.10.0+cpu, and the full commit SHA.

Smoke evidence:

- Base FinRL Sharpe: 0.408021.
- ANN signal Sharpe: 0.318421.
- QINN-MPS signal Sharpe: 0.256108.
- These 64-step, one-seed values validate execution only and are not paper
  evidence or model-selection evidence.

Active run:

- Output root:
  `work/temporal_robustness_2026-07-30`.
- Job:
  `temporal-robustness_shifted_steps20000_bd2_seeds0-1-2-3-4-5-6-7-8-9_epochs60_batch512_cpu`.
- Launcher process at verification: PID 15708.
- Monitor:
  `Get-Content work/temporal_robustness_2026-07-30/matrix.log -Wait`.
- Resume by running `scripts/run_temporal_robustness.ps1` again; partial
  condition/seed checkpoints are validated before reuse.

Verification:

- Shifted-window smoke run completed all three configured PPO conditions.
- Full test suite: 73 passed.
- Provenance-focused suite: 37 passed.
- PowerShell launcher parsed without syntax errors.
- Dry-run plan contains one exact frozen matrix and all ten matched seeds.

Commits pushed:

- `72c5ed8` — implement guarded shifted-window evaluation.
- `bad8584` — anchor run provenance to repository.
- `4c8cd2e` — add durable robustness launcher.

Next:

- Monitor the active 30-endpoint robustness matrix.
- After completion, generate a guarded paired summary, inference artifact,
  hashes, and compact paper table or figure.
- Integrate the result without changing the fixed protocol or suppressing an
  unfavorable sign.

## 2026-07-31 — Corrected evidence and shifted-window paper integration

Completed:

1. Confirmed the paper repository was clean at `c5515ba` before integration.
2. Verified the corrected primary run completed all 30 Base/ANN/MPS endpoints
   for seeds 0--9 with no duplicate condition/seed keys.
3. Verified the prespecified shifted run completed the same 30-endpoint matched
   matrix with the frozen 2017--2018 evaluation window.
4. Imported final metrics, curves, signal evidence, training histories, status,
   manifests, annual analyses, and block-bootstrap results for both windows.
5. Extended the guarded final summarizer with a stable `--artifact-name`
   option so primary and secondary evidence cannot share a provenance label.
6. Added a focused test for distinct shifted-window provenance labeling.
7. Regenerated the corrected primary condition summary, paired seed effects,
   deterministic inference JSON, and hash manifest.
8. Generated the corresponding guarded shifted-window summary, paired effects,
   inference JSON, and distinct robustness manifest.
9. Verified every recorded input and output SHA-256 hash through published-
   evidence regression tests.
10. Regenerated corrected primary diagnostic figures and the paired-effect PNG
    and vector PDF.
11. Generated a separate shifted paired-effect PNG and vector PDF without
    displacing the primary figure.
12. Added published-evidence tests that load both completed manifests, require
    ten seeds per condition, validate hash provenance, and bind manuscript
    numbers to generated artifacts.
13. Updated the abstract with the corrected primary result and the bounded
    shifted-window sign reversal.
14. Documented the next-day target boundary guard explicitly in Methods.
15. Replaced every stale primary table and narrative value with corrected
    results: ANN Sharpe 0.821, MPS 0.784, paired difference -0.0366, and
    paired-seed interval [-0.1129, 0.0448].
16. Integrated the shifted result: MPS Sharpe 0.714 versus ANN 0.627, 9/10 MPS
    seed wins, paired difference +0.0862, and seed-bootstrap interval
    [0.0011, 0.1945].
17. Preserved the more cautious time-series result: annualized return difference
    +2.34 percentage points with 20-day block-bootstrap interval
    [-0.83, 5.98] percentage points.
18. Updated limitations and conclusion to describe evaluation-window
    sensitivity rather than stable MPS superiority.
19. Updated claim traceability, result indexes, repository presentation,
    reproduction commands, and the deliverables tracker.
20. Distinguished the older three-seed root artifacts from the corrected
    manuscript evidence to prevent accidental citation drift.
21. Ran the focused integration suite: 15 tests passed in 2.33 seconds.
22. Ran the full repository suite: 79 tests passed in 9.15 seconds.

Verified results:

- Corrected primary mean Sharpe: Base 0.689390, ANN 0.821005, MPS 0.784397.
- Corrected primary MPS-minus-ANN paired Sharpe: -0.036607; 3/10 positive
  seeds; seed-bootstrap interval [-0.112857, 0.044800].
- Shifted mean Sharpe: Base 0.652458, ANN 0.627464, MPS 0.713709.
- Shifted MPS-minus-ANN paired Sharpe: +0.086245; 9/10 positive seeds;
  seed-bootstrap interval [0.001119, 0.194500].
- Both runs record Python 3.12.10, PyTorch 2.10.0+cpu, CUDA unavailable, and
  actual CPU execution. No new model training was performed during integration.

QA boundary:

- PNG and PDF files were regenerated and pass automated structural tests.
- Direct visual reinspection was blocked by the current Windows ACL image-view
  path, so visual QA is not claimed.
- No TeX compiler is installed locally; page-count, overflow, font embedding,
  and rendered-PDF inspection remain pending.

Next:

- Compile the IEEE manuscript in a free TeX environment and inspect every page.
- Reconcile the draft to the five-page limit, if necessary.
- Run clean-clone reproduction and prepare the release-ready archive/checklist.
## 2026-07-31 - Methodology and clean-checkout reproducibility audit

Completed:

1. Reconciled the primary and shifted dataset boundaries against code and frozen manifests.
2. Verified the exact 15-stock universe in the implementation and both final manifests.
3. Verified the 10 FinRL indicators and 13 encoder features across code and evidence.
4. Confirmed the 0.10% executed buy/sell cost, initial cash, order cap, and reward scale.
5. Confirmed matched PPO seeds 0--9 and the frozen 20,000-step budget.
6. Corrected public wording that implied the selected 97-parameter MPS was parameter matched.
7. Documented dimension 4 as the 369-parameter matched sensitivity model.
8. Created a no-local-hardlinks clean clone and a fresh Python 3.12 virtual environment.
9. Installed every direct pinned dependency and obtained a clean `pip check`.
10. Downloaded and checksum-verified both source datasets from the clean clone.
11. Checked out the recorded FinRL commit and imported `StockTradingEnv`.
12. Audited real-data feature finiteness, sample counts, and next-day target boundaries.
13. Completed a reduced end-to-end shifted-window Base/ANN/MPS smoke run.
14. Reproduced primary and shifted summaries byte for byte from committed run bundles.
15. Packaged the nine compact source files required to regenerate the capacity pilot.
16. Made text hashing invariant to CRLF/LF checkout behavior.
17. Made summary CSV/JSON output use deterministic LF line endings.
18. Removed timestamp metadata from publication PDFs and forced headless figure rendering.
19. Added methodology-contract tests binding code, manifests, raw sources, and public docs.
20. Published a detailed methodology/reproducibility audit and portable clean-clone commands.

Clean-checkout evidence:

- Source rows: 22,635 train and 18,855 trade; all 15 tickers present.
- Primary encoder rows: 18,870 fit and 3,735 validation.
- Shifted encoder rows: 11,325 fit and 3,765 validation.
- Source SHA-256 checksums and FinRL commit matched the frozen manifests.
- Fresh environment: Python 3.12.10, PyTorch 2.10.0+cpu, CUDA unavailable.
- Reduced smoke Sharpe: Base 0.408021, ANN 0.318421, MPS 0.256108; execution evidence only.

Verification:

- Scientific freeze: 38 hashes, 60 PPO endpoints, 3 capacity dimensions, and 5 paper claim groups verified.
- Focused deterministic-artifact suite: 16 tests passed.
- Methodology/reproducibility contract suite: 3 tests passed.
- Full repository suite: 84 tests passed in 6.00 seconds.
- Regenerated primary, shifted, and capacity summary artifacts matched committed evidence byte for byte.

Hardware actually used:

- The audit, clean-checkout smoke run, summaries, and tests used CPU only.
- The installed fresh PyTorch build was 2.10.0+cpu; no GPU acceleration is claimed.

QA boundary:

- The full 60-endpoint PPO matrices were not retrained; their completed endpoint tables, curves, configurations, commits, and hashes remain frozen.
- Manuscript compilation, page-count/font checks, and rendered visual inspection remain separate submission QA tasks.

## 2026-07-31 - Equal-length temporal robustness extension

Completed:

1. Inspected the clean worktree, open temporal-robustness issue, completed
   evidence, recent commits, daily log, active processes, and runtime hardware.
2. Identified a genuine design gap before new outcomes: the primary and
   shifted comparisons changed evaluation length and training-history length
   together.
3. Froze `docs/EQUAL_WINDOW_PROTOCOL.md` before viewing either new outcome.
4. Prespecified non-overlapping 2017--2018, 2019--2020, and 2021--2022
   two-year evaluations.
5. Fixed every cell to a four-year PPO training span, three encoder-fit years,
   one validation year, 20,000 steps, dimension 2, and matched seeds 0--9.
6. Reused the exact completed 2017--2018 run rather than duplicating it.
7. Added guarded named configurations for the two new windows.
8. Extended orchestration, date-drift validation, period-label validation,
   leakage-boundary tests, and distinct job identifiers.
9. Generated and inspected both complete dry-run matrix plans before training.
10. Completed reduced Base/ANN/MPS smoke runs for both new windows; these are
    execution evidence only and do not enter the paper.
11. Committed and pushed the protocol and implementation before full outcomes.
12. Detected a primary-window-specific manifest filter near the start of the
    first full run, stopped only its exact process tree, and preserved three
    completed Base checkpoints.
13. Bound manifest signal evidence to each configured test window and added a
    focused regression test before resuming the unchanged matrix.
14. Resumed the first matrix from matching checkpoints with one durable writer.
15. Completed all 30 endpoints for 2019--2020 in 865.206 seconds.
16. Completed all 30 endpoints for 2021--2022 in 972.014 seconds.
17. Confirmed both manifests record Python 3.12.10, PyTorch 2.10.0+cpu, CUDA
    unavailable, and actual CPU execution.
18. Added a guarded cross-window summarizer that rejects incomplete runs,
    control/date drift, missing seeds, duplicate keys, and missing signal rows.
19. Reused the existing deterministic 10,000-resample seed-bootstrap procedure
    so the 2017--2018 interval remains byte-for-number consistent.
20. Generated condition, paired-seed, paired-metric, prediction-quality, and
    bounded-inference tables for every fixed window.
21. Generated a seed-level PNG and vector-PDF equal-window effect figure.
22. Independently recomputed all condition means, paired Sharpe differences,
    win counts, and input/output hashes from the raw run bundles.
23. Preserved both complete run bundles, matrix plans, manifests, curves,
    tables, figures, and derived provenance under `results/robustness/`.
24. Integrated the exploratory panel into the abstract, Methods, Results table,
    limitations, conclusion, scientific-freeze registry, and claim traceability.
25. Updated the repository overview, reproducibility commands, experiment
    protocol routing, deliverables tracker, and focused evidence tests.

Verified equal-window results:

- 2017--2018: ANN mean Sharpe 0.627464; MPS 0.713709; paired difference
  +0.086245; 9/10 positive seeds; interval [0.001119, 0.194500].
- 2019--2020: ANN mean Sharpe 0.870327; MPS 0.962777; paired difference
  +0.092451; 7/10 positive seeds; interval [-0.007979, 0.196643].
- 2021--2022: ANN mean Sharpe 0.037627; MPS 0.177289; paired difference
  +0.139662; 6/10 positive seeds; interval [-0.067971, 0.338489].
- Paired MPS-minus-ANN annual-return means were +0.022534, +0.030742, and
  +0.040504 across the three cells.
- Paired maximum-drawdown means were -0.006829, +0.005217, and +0.067607;
  turnover means were +0.022726, +0.003620, and -0.025992; modeled-cost means
  were +$41.23, +$1.98, and -$42.20.
- MPS test MSE was lower in the first two cells but higher in 2021--2022
  (1.525238 versus 1.506480 for ANN), despite positive mean Sharpe differences
  in every equal-length cell.

Verification:

- Focused equal-window configuration, analysis, figure, and evidence suite:
  10 tests passed.
- Full repository suite: 100 tests passed in 4.84 seconds.
- Original scientific freeze: 38 hashes, 60 primary/shifted PPO endpoints,
  three capacity dimensions, and five original paper claim groups verified.
- Independent evidence check: 90 equal-window condition/seed rows across all
  three cells, all manifests and derived hashes matched.

Interpretation boundary:

- Positive mean Sharpe in all three equal-length cells is useful exploratory
  evidence, but the two new intervals include zero and the five-year primary
  estimate remains negative.
- The result supports evaluation-horizon and training-window sensitivity, not
  stable MPS superiority or a causal market-regime explanation.
- No additional date window will be added in response to these outcomes.

QA boundary:

- The generated PNG is structurally valid at 2130 by 1163 pixels and the
  vector PDF has a valid PDF signature.
- Direct visual reinspection remains blocked by the current Windows ACL image
  path; visual QA is not claimed.
- IEEE manuscript compilation, page-budget inspection, font embedding, and
  rendered overflow checks remain pending because no TeX engine is installed.

## 2026-07-31 - Post-hoc nested evaluation-horizon diagnostic

Completed:

1. Converted the requested one-, two-, and three-year comparison into a
   same-policy cumulative-prefix design anchored on 2019-01-02.
2. Added mandatory four- and five-year prefixes to prevent selective stopping.
3. Documented that the negative five-year primary result was already known
   when the follow-up design was written.
4. Froze and pushed `docs/NESTED_HORIZON_PROTOCOL.md` before computing any new
   prefix outcome.
5. Kept both completed experiments and the original scientific freeze intact;
   the new evidence is separately versioned and explicitly post hoc.
6. Reused only the frozen primary daily equity curves; no encoder or PPO model
   was retrained.
7. Verified the frozen source hashes, primary configuration, 20,000-step
   budget, dimension 2, costs, seeds 0--9, and 2019--2023 boundaries.
8. Detected, validated, and explicitly excluded the saved equal-weight
   benchmark from the three PPO-condition comparison.
9. Required one identical daily date grid across Base, ANN, MPS, all ten seeds,
   and the benchmark.
10. Implemented metric-compatible prefix scoring for return, volatility,
    Sharpe, drawdown, turnover, traded notional, and modeled cost.
11. Generated all 150 condition/seed/horizon metric rows and all 50 matched
    MPS-minus-ANN seed rows.
12. Generated condition summaries, five-metric paired summaries, deterministic
    seed-bootstrap intervals, and bounded inference metadata.
13. Required the five-year prefix to reproduce every frozen primary condition,
    seed, and metric within numerical tolerance.
14. Generated a seed-level PNG and vector-PDF nested-horizon effect figure.
15. Bound the protocol, frozen sources, derived tables, plotting script, and
    both figure formats through SHA-256 provenance.
16. Added an evidence README with the complete result path, secondary metrics,
    artifact map, reproduction commands, and interpretation limits.
17. Integrated the post-hoc design and complete one- through five-year result
    into the abstract, Methods, Results table, limitations, and conclusion.
18. Added claim, table, and figure traceability plus repository overview,
    reproduction, scientific-freeze, and deliverables documentation.

Verified nested-horizon results:

- One year: ANN mean Sharpe 1.548594; MPS 1.553420; paired difference
  +0.004826; MPS higher in 4/10 seeds; interval [-0.464850, 0.509062].
- Two years: ANN mean Sharpe 1.086569; MPS 1.023986; paired difference
  -0.062583; MPS higher in 2/10 seeds; interval [-0.155890, 0.052248].
- Three years: ANN mean Sharpe 1.168339; MPS 1.086227; paired difference
  -0.082111; MPS higher in 3/10 seeds; interval [-0.194423, 0.035542].
- Four years: ANN mean Sharpe 0.596813; MPS 0.568364; paired difference
  -0.028449; MPS higher in 3/10 seeds; interval [-0.108988, 0.055582].
- Five years: ANN mean Sharpe 0.821005; MPS 0.784397; paired difference
  -0.036607; MPS higher in 3/10 seeds; interval [-0.112857, 0.044800].
- Paired annualized-return differences were -0.003744, -0.011347, -0.018265,
  -0.007644, and -0.009193 from one through five years.
- MPS annualized turnover and modeled cost were higher at one year and lower
  at every two- through five-year cutoff.
- MPS mean maximum drawdown was less severe at one year, more severe at years
  two and three, and less severe at years four and five.
- Every paired seed-bootstrap interval for every reported portfolio metric
  included zero.

Verification:

- Focused nested-horizon scoring, figure, and evidence suite: 14 tests passed.
- Full repository suite: 114 tests passed in 9.95 seconds.
- Original scientific freeze: 38 hashes, 60 primary/shifted PPO endpoints,
  three capacity dimensions, and five original paper claim groups verified.
- Five-year prefix metrics reproduce the frozen primary endpoint exactly.
- Evidence bundle contains 150 condition/seed/horizon rows, 50 paired seed
  rows, 25 paired metric summaries, and 15 condition summaries.

Hardware actually used:

- The diagnostic, tests, summaries, and figures used CPU only.
- No model training was performed, so GPU acceleration was neither needed nor
  claimed.

Interpretation boundary:

- The one-year mean is effectively tied and is positive despite only 4/10 MPS
  seed wins, showing substantial seed dispersion.
- ANN has higher mean Sharpe at every two- through five-year cutoff, but every
  interval includes zero; no stable winner is established.
- Evaluation length alone does not reproduce the positive equal-window panel
  when policies and the 2019 start are fixed.
- The nested prefixes are dependent and post hoc. They cannot identify whether
  training-window changes, calendar exposure, refitting, compounding, or their
  interaction causes the cross-design difference.

QA boundary:

- PNG/PDF signatures and manifest hashes validate automatically.
- Direct visual inspection and compiled IEEE page-budget/overflow checks remain
  pending because the current image-view path and local TeX toolchain are
  unavailable.

## 2026-08-02 - Post-hoc market-state and calendar trend audit

Completed:

1. Prespecified the market-state questions before calculating their outcomes.
2. Preserved the corrected primary, equal-window, nested-horizon, and original
   scientific-freeze artifacts without retraining or changing their claims.
3. Limited the audit to the three complete non-overlapping two-year windows.
4. Required the saved equal-weight benchmark to define every market state
   independently of ANN or MPS outcomes.
5. Defined complete direction, 20-session volatility, drawdown, return-tail,
   and calendar-year views with no outcome-driven alternate thresholds.
6. Validated completed manifests, exact dates, 20,000 PPO steps, dimension 2,
   and matched seeds 0--9.
7. Required identical ANN, MPS, and benchmark daily grids within every window.
8. Rejected missing, duplicate, unexpected, or non-finite curve rows.
9. Generated 270 window/state/seed conditional effect rows.
10. Generated all 27 prespecified window/state summary cells.
11. Added deterministic ten-seed bootstrap intervals for every state cell.
12. Calculated conditional mean-return and downside-deviation differences.
13. Generated a nine-row equal-weight cross-window sign audit.
14. Exactly decomposed each full-window mean daily difference into benchmark
    down-day and nonnegative-day contributions.
15. Recomputed the saved annual metrics from daily curves and required exact
    agreement before using them.
16. Generated 300 paired calendar-year/seed/metric rows and 30 summaries.
17. Reported annual return, Sharpe, drawdown, turnover, and modeled cost for
    all six calendar years.
18. Bound the protocol, script, three curve files, three annual tables, three
    manifests, and every generated output through SHA-256.
19. Added six focused tests for state partitions, matched scoring, complete
    cells, deterministic intervals, exact decomposition, and provenance.
20. Documented the full favorable, unfavorable, and null result pattern in the
    evidence bundle and repository indexes.

Verified findings:

- MPS-minus-ANN annualized mean-return differences on benchmark-down days were
  -6.63, -0.56, and -2.15 percentage points across 2017--2018, 2019--2020,
  and 2021--2022.
- The corresponding nonnegative-day differences were +8.56, +4.99, and +9.94
  points. Every direction-state seed-bootstrap interval included zero.
- Down days offset 53.7%, 8.0%, and 20.5% of the positive nonnegative-day
  contribution, leaving reconciled full-window annualized mean differences of
  +2.34, +2.68, and +4.05 points.
- Top-decile benchmark days favored MPS in all three windows, while
  bottom-decile signs were negative, negative, then positive with wide
  intervals.
- Both high- and low-volatility state means were positive in all three
  windows; no specifically high-volatility explanation is supported.
- Mean annual return and Sharpe differences were positive in five of six
  years. The only intervals excluding zero were positive in 2018 for both
  metrics and negative in 2017 for Sharpe.
- Downside-deviation, turnover, and modeled-cost differences changed signs
  across windows or years and do not provide a stable mechanism.

Verification:

- Focused market-state suite: 6 tests passed.
- Full repository suite: 120 tests passed in 21.38 seconds.
- All 27 market-state cells and six calendar years were present.
- All saved annual metrics reproduced directly from daily curves.
- Every generated and source hash matched the manifest.

Hardware actually used:

- The audit, bootstrap calculations, documentation, and tests used CPU only.
- No encoder or PPO training was performed, so the RTX 5060 was not used.

Interpretation boundary:

- The main new clue is upside capture rather than downside protection.
- The states overlap, only three windows are available, and intervals reflect
  seed dispersion rather than calendar-sample uncertainty.
- This evidence is post hoc and should remain secondary to the frozen primary
  and prespecified shifted results.
