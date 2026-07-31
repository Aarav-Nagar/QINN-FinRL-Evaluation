# Scientific Results Freeze

**Status:** Frozen for the SecureFinAI short-paper draft on 2026-07-31.

The scientific evidence entering this paper is now limited to the
prespecified capacity pilot, the corrected primary ten-seed evaluation, and
the prespecified shifted-window ten-seed evaluation. The exact values and
SHA-256 hashes are recorded in
`results/SCIENTIFIC_RESULTS_FREEZE.json`.

## Frozen scope

| Evidence | Frozen configuration | Role in paper |
|---|---|---|
| Capacity pilot | Bond dimensions 2, 4, and 8; PPO seeds 0--2; 20,000 steps | Select dimension 2 using validation MSE and parsimony |
| Primary evaluation | 2019--2023; Base, ANN, and MPS; PPO seeds 0--9 | Primary fixed-split result |
| Shifted evaluation | 2017--2018; Base, ANN, and MPS; PPO seeds 0--9 | Prespecified temporal sensitivity result |

## Frozen conclusions

- Primary: ANN mean Sharpe is 0.821005 and MPS mean Sharpe is 0.784397.
  The paired MPS-minus-ANN mean is -0.036607 with a paired-seed bootstrap
  interval of [-0.112857, 0.044800].
- Shifted: ANN mean Sharpe is 0.627464 and MPS mean Sharpe is 0.713709.
  The paired MPS-minus-ANN mean is +0.086245 with a paired-seed bootstrap
  interval of [0.001119, 0.194500].
- The shifted annualized-return difference is +2.342939 percentage points,
  with a 20-day moving-block-bootstrap interval of
  [-0.834107, 5.979104] percentage points.
- The sign reversal supports evaluation-window sensitivity. It does not
  establish a stable MPS advantage, quantum advantage, or a general result
  about tensor networks.

## Verification

Run:

```powershell
$env:PYTHONPATH='..\..\work\pydeps'
py -3.12 scripts\verify_scientific_results_freeze.py
```

The verifier:

1. checks every frozen SHA-256 digest;
2. independently recomputes condition summaries from raw PPO rows;
3. recomputes all paired seed effects, seed-bootstrap intervals, and sign
   tests;
4. recomputes both moving-block-bootstrap return intervals from daily equity
   curves;
5. reproduces the capacity-selection rule;
6. checks the dates, seeds, budget, costs, state sizes, and CPU-only runtime;
7. binds the rounded abstract, table, Results, robustness, and conclusion
   values to the exact frozen claims; and
8. confirms the current PNG/PDF figure files and plotting scripts match the
   freeze hashes.

## Change policy

Writing, formatting, references, and figure styling may still improve. A
scientific value or frozen evidence file may change only if a documented
error is found. A new experiment may enter this paper only after documenting
the methodological gap, freezing its protocol before inspecting outcomes,
and versioning it separately. Existing unfavorable or inconclusive evidence
must remain visible.

GitHub issue 5, a broader rolling or expanding-window study, remains useful
future work. It is not required to reinterpret or replace the two frozen
windows in this short paper.

## QA boundary

The numerical and file-integrity audit is complete. The figures have valid
PNG/PDF structure and their data sources and plotting scripts are frozen.
Direct visual inspection remains a separate presentation QA gate because the
current Windows image-view path is blocked. The IEEE PDF also still requires
compilation and page-by-page inspection.
