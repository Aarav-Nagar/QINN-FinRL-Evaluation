# SecureFinAI Short-Paper Deliverables

Last verified: 2026-07-31

## Submission target

- Venue: SmartCom 2026 Special Track on Secure and Open Financial Intelligence
  (SecureFinAI).
- Track fit: the call explicitly includes FinRL evaluation and benchmarking,
  policy instability, and quantum-inspired RL.
- Format: IEEE conference template, PDF, two-column layout.
- Short-paper length: three complimentary pages and up to two additional pages,
  including figures, tables, and references. Extra pages are listed at USD 150
  per page.
- Working page budget: five pages total. Aim for four pages before references
  and use the fifth page for references and any unavoidable overflow.
- Internal artifact freeze: 2026-08-05.
- Special-track submission deadline: 2026-08-25.
- Authorship: Aarav Nagar, sole author. Acknowledge Dr. Xiao-Yang Liu for
  feedback; do not imply endorsement or authorship.

### Deadline discrepancy

The SecureFinAI special-track page lists August 25, 2026, while the SmartCom
general submission page lists August 1, 2026. The special-track date is the
working deadline because it is the more specific source, but the author should
confirm the applicable deadline in EasyChair or with the organizers before
submission. The August 5 internal freeze is intentionally earlier than the
special-track deadline.

Sources checked on 2026-07-27:

- https://cloud-conf.net/smartcom/2026/st_securefinai.html
- https://cloud-conf.net/smartcom/2026/submission.html
- https://easychair.org/conferences/?conf=smartcom20260

## Paper claim

Primary research question:

> Under matched PPO training conditions, does a parameter-controlled,
> quantum-inspired MPS return signal improve out-of-sample trading performance
> relative to an ANN signal or the standard FinRL state?

The paper is an empirical evaluation and possible failure analysis. A null or
negative result is publishable evidence and must not be reframed as a positive
quantum advantage. The implementation is a classical tensor-network simulation;
no quantum hardware is used.

## Required artifacts

| Artifact | Completion evidence | Status |
|---|---|---|
| Frozen experiment protocol | `docs/EXPERIMENT_PROTOCOL.md` and `docs/ROBUSTNESS_PROTOCOL.md` frozen before their outcomes | Complete |
| Configurable experiment runner | CLI and manifest record seeds, PPO budget, MPS dimension, temporal window, device, and runtime | Complete |
| Automated verification | Configuration, boundaries, state construction, summaries, provenance hashes, published evidence, figures, and smoke comparison | Complete: 84 tests passed on 2026-07-31 |
| Training-budget pilot | Matched 5k, 10k, and 20k results with PPO diagnostics | Complete: 20k selected |
| Bond-dimension study | Machine-readable results for dimensions 2, 4, and 8, including parameter count and runtime | Complete: dimension 2 selected by frozen rule |
| Matched final evaluation | Ten matched PPO seeds for base, ANN, and selected MPS configuration | Complete: corrected 30-endpoint package, status, manifest, curves, and per-seed metrics preserved |
| Temporal robustness | At least one non-overlapping refit/evaluation window with unchanged controls | Complete: shifted 2017--2018 30-endpoint evaluation imported with guarded provenance |
| Statistics | Paired seed differences, uncertainty intervals, drawdown, turnover, and costs | Complete for corrected primary and shifted window |
| Figures | Legible equity/performance figure and sensitivity/uncertainty figure | Corrected primary and shifted figures regenerated; structural checks pass, visual reinspection pending |
| Short-paper source | IEEE-template source with traceable result references | Corrected primary and shifted Methods, Results, limitations, and conclusion integrated |
| Submission PDF | Five pages or fewer; fonts embedded; no clipping or overflow | Not started |
| Reproducibility package | Environment lock, commands, checksums, manifests, raw capacity inputs, and result index | Complete: fresh environment, clean checkout, source acquisition, smoke run, and artifact regeneration verified |
| Repository presentation | README points to current paper, corrected evidence, protocol, commands, and limitations | Complete; final release link remains |
| Final archive | Tagged, checksummed, release-ready bundle | Not started |

## Dated execution plan

| Date | Milestone | Gate |
|---|---|---|
| Jul 28 | Requirements audit, deliverables tracker, frozen experiment protocol | Sources and unresolved deadline conflict documented |
| Jul 29 | Configuration validation and smoke matrix | Tests pass; each run records full configuration |
| Jul 30 | PPO training-budget convergence pilot | Selected final budget justified by recorded evidence |
| Jul 31 | MPS bond-dimension sensitivity | Selected dimension justified with accuracy, parameter, runtime, and trading evidence |
| Aug 1 | Ten-seed matched evaluation | All conditions finish with identical seeds and saved manifests |
| Aug 2 | Temporal robustness evaluation | Window definitions and leakage controls documented |
| Aug 3 | Final tables, figures, Methods, and Results | Every reported number maps to a generated artifact |
| Aug 4 | Complete IEEE short-paper draft and QA | Page, reference, claim, and accessibility checks pass |
| Aug 5 | Re-run critical checks and freeze release-ready package | Clean clone instructions, checksums, final checklist |

## Author review gates

The author must personally review and be able to explain:

1. Why the ANN and MPS comparison is controlled and where it is not.
2. Why the selected PPO budget and MPS bond dimension were chosen.
3. Why prediction quality can diverge from downstream trading performance.
4. What the uncertainty intervals do and do not establish.
5. Why the results do not demonstrate quantum advantage.
6. Every paper claim, figure, table, citation, limitation, and disclosure.

No automation may submit the manuscript, accept registration costs, email an
advisor, or change authorship.
