# Scientific Results Freeze

**Status:** Frozen for the SecureFinAI short-paper draft on 2026-07-31.

The scientific evidence entering this paper is now limited to the
prespecified capacity pilot, the corrected primary ten-seed evaluation, and
the prespecified shifted-window ten-seed evaluation. Those original values and
SHA-256 hashes remain unchanged in `results/SCIENTIFIC_RESULTS_FREEZE.json`.
A documented horizon-length confound then justified one separately versioned
exploratory extension: the three-window equal-length panel frozen before its
two new outcomes. Its hashes and reporting rule are recorded in
`results/robustness/equal_windows/equal_window_manifest.json`.

After those outcomes were known, the repository owner requested a cumulative
one-, two-, and three-year horizon view. Before computing those new prefixes,
`docs/NESTED_HORIZON_PROTOCOL.md` separately locked all available one- through
five-year prefixes of the frozen primary equity curves. The added four- and
five-year cutoffs prevent selective stopping. This post-hoc analysis cannot
change the role or conclusion of any frozen experiment.

## Frozen scope

| Evidence | Frozen configuration | Role in paper |
|---|---|---|
| Capacity pilot | Bond dimensions 2, 4, and 8; PPO seeds 0--2; 20,000 steps | Select dimension 2 using validation MSE and parsimony |
| Primary evaluation | 2019--2023; Base, ANN, and MPS; PPO seeds 0--9 | Primary fixed-split result |
| Shifted evaluation | 2017--2018; Base, ANN, and MPS; PPO seeds 0--9 | Prespecified temporal sensitivity result |
| Equal-length extension | 2017--2018, 2019--2020, and 2021--2022; four-year training and two-year evaluation spans; PPO seeds 0--9 | Exploratory horizon/training-window sensitivity |

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
- Equal-length panel: paired mean MPS-minus-ANN Sharpe is +0.086245,
  +0.092451, and +0.139662 across the three chronological cells. MPS is
  higher in 9/10, 7/10, and 6/10 seeds.
- The latter two seed-bootstrap intervals, [-0.007979, 0.196643] and
  [-0.067971, 0.338489], include zero. The original five-year primary estimate
  remains negative.
- The panel therefore supports horizon and training-window sensitivity, not a
  stable MPS advantage or a causal market-regime explanation.

## Verification

Run:

```powershell
$env:PYTHONPATH='..\..\work\pydeps'
py -3.12 scripts\verify_scientific_results_freeze.py
```

The verifier:

1. checks all 38 frozen SHA-256 digests, including the nine compact raw
   capacity-pilot source files, using cross-platform canonical text hashing;
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

The nested-horizon analysis is a deterministic re-scoring of the frozen
primary curves rather than a newly trained date-window experiment. It is
post hoc, reports every available cumulative cutoff, and is governed by
`docs/NESTED_HORIZON_PROTOCOL.md`. The equal-window extension is governed by
`docs/EQUAL_WINDOW_PROTOCOL.md`; it does not alter the original freeze
manifest.

No additional independently trained date window may be added in response to
the equal-panel results.
A different stock universe remains useful future work.

## QA boundary

The numerical and file-integrity audit is complete. The figures have valid
PNG/PDF structure and their data sources and plotting scripts are frozen.
Direct visual inspection remains a separate presentation QA gate because the
current Windows image-view path is blocked. The IEEE PDF also still requires
compilation and page-by-page inspection.
