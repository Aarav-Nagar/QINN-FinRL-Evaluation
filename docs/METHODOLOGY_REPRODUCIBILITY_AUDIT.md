# Methodology and Reproducibility Audit

**Audit date:** 2026-07-31

**Scientific-result status:** Frozen; no reported value was changed
**Overall assessment:** Reproducible from a clean checkout, with the full
60-endpoint PPO retraining cost explicitly outside this audit

## Controls checked

| Item | Verified contract |
|---|---|
| Dataset | `train_data_2013_2018.csv` and `trade_data_2019_2023.csv`; SHA-256 values match `run_experiment.py` and both frozen run manifests |
| Stock universe | The same 15 tickers in code and both manifests: AAPL, AMD, AMGN, AMZN, COST, FANG, GILD, HON, INTC, MSFT, NFLX, NVDA, PEP, SBUX, XEL |
| FinRL state features | 10 indicators per asset: MACD, Bollinger bounds, RSI-30, CCI-30, DX-30, 30/60-day SMAs, VIX, and turbulence |
| Encoder features | 13 price, trend, momentum, volume, and volatility features; all engineered values were finite in the downloaded data |
| Transaction costs | 0.001, or 0.10%, on each executed buy and sell |
| Other trading controls | $1,000,000 initial cash, maximum 100 shares per order, reward scaling `1e-4`, PPO update epochs 3 |
| PPO seeds and budget | Seeds 0--9 and 20,000 steps for primary and shifted evaluations |
| MPS capacity | Dimensions 2, 4, and 8 tested; dimension 2 selected by the frozen validation/parsimony rule |
| Capacity comparison | ANN has 369 parameters; selected dimension-2 MPS has 97; dimension-4 MPS has 369 and is the exact parameter-matched sensitivity condition |
| Runtime | Frozen primary and shifted runs used Python 3.12.10, PyTorch 2.10.0+cpu, CPU PPO/encoders, and no CUDA |

## Temporal boundary audit

The downloaded source files contained 22,635 training rows from 2013-01-02
through 2018-12-28 and 18,855 trading rows from 2019-01-02 through
2023-12-28, with all 15 tickers present.

For the primary analysis, the encoder fit targets end on 2017-12-29,
validation targets end on 2018-12-28, PPO training ends on 2018-12-28, and
evaluation begins on 2019-01-02. For the shifted analysis, encoder fit targets
end on 2015-12-31, validation targets end on 2016-12-30, PPO training ends on
2016-12-30, and evaluation begins on 2017-01-03. The implementation excludes
rows whose next-day target crosses a fit or validation boundary.

Observed encoder sample counts were:

| Window | Fit rows | Validation rows |
|---|---:|---:|
| Primary | 18,870 | 3,735 |
| Shifted | 11,325 | 3,765 |

## Clean-checkout evidence

A no-local-hardlinks clone was created from the audited revision. In a new
Python 3.12 virtual environment:

- every exact direct version in `requirements-lock.txt` installed;
- `pip check` reported no broken requirements;
- the pipeline downloaded both source CSVs and verified their recorded
  checksums;
- the recorded FinRL commit
  `2334a5fe6d30629157f13c3b0319e1637e15e123` checked out and
  `StockTradingEnv` imported;
- the complete test suite passed after the cross-platform fixes;
- the scientific freeze verifier recomputed condition means, paired effects,
  bootstrap intervals, capacity selection, manuscript claims, and all frozen
  hashes; and
- regenerated primary, shifted, and capacity summaries matched the committed
  outputs byte for byte.

A reduced end-to-end smoke run used the shifted window, seed 0, 64 PPO steps,
dimension 2, and one encoder epoch. It completed Base, ANN, and MPS conditions
with Sharpe values 0.408021, 0.318421, and 0.256108 respectively. These values
are execution evidence only and are not scientific results.

## Defects found and resolved

1. Three capacity-pilot hashes had been calculated from Windows CRLF working
   bytes, so the freeze verifier failed after Git normalized them to LF in a
   clean checkout. Text hashes are now canonicalized to LF.
2. The compact source tables for the capacity summary were not in the
   repository. The nine required files are now packaged and frozen.
3. Matplotlib embedded the generation time in PDFs. Publication PDFs now omit
   creation/modification timestamps and reproduce byte for byte.
4. Figure scripts implicitly depended on the host's GUI backend. They now use
   the headless Agg backend.
5. pandas 3.0 could expose read-only NumPy views to PyTorch. The encoder arrays
   are now explicitly copied before tensor conversion.
6. Public README wording implied the selected final ANN and MPS were parameter
   matched. The documentation now distinguishes the 97-parameter selected MPS
   from the 369-parameter matched dimension-4 sensitivity model.
7. Several portable commands wrote to an unignored `work/` path. Clean-checkout
   commands now use Git-ignored `local_runs/` or disposable `.cache/` paths.

None of these fixes changes a metric, statistical result, selected dimension,
or paper conclusion.

## Reproduction boundary

This audit independently reproduced source acquisition, feature and date
construction, one complete reduced run, all guarded summaries, figures,
manuscript claims, and artifact hashes. It did not retrain the full primary
and shifted 60 PPO endpoints because those completed raw endpoint tables,
curves, configurations, source commits, and checksums are already frozen.
A full retraining remains possible using `REPRODUCIBILITY.md`, but stochastic
training may not be bitwise identical across hardware and library builds.

The remaining submission QA tasks are manuscript compilation, page-count and
font checks, and visual inspection of the rendered paper. Those are
presentation checks, not gaps in the frozen experimental backend.
