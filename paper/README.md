# SecureFinAI Short-Paper Source

This directory is the staging area for the five-page-or-shorter IEEE
SecureFinAI paper. The older long-form technical report remains under `docs/`;
it is background material, not the submission manuscript.

## Evidence policy

- Every numerical claim, table, and figure must be registered in
  `CLAIM_TRACEABILITY.md`.
- Paper-facing generated files must originate from committed machine-readable
  results.
- Smoke outputs may verify execution but cannot support performance claims.
- Pending experiments remain visibly pending; placeholders cannot be replaced
  with estimated or favorable numbers.
- The MPS is a classical tensor-network model. The manuscript must not call the
  experiment quantum-hardware execution or claim quantum advantage.
- Negative, null, unstable, or inconclusive outcomes remain reportable results.

## Planned source layout

The eventual IEEE source should contain:

1. Introduction and contribution statement.
2. Related work on FinRL, PPO, and quantum-inspired tensor networks.
3. Methods and prespecified evaluation protocol.
4. Results for capacity, matched seeds, and temporal robustness.
5. Limitations and conclusion.
6. Acknowledgment of Dr. Xiao-Yang Liu for feedback, without authorship or
   implied endorsement.

`main.tex` now contains an IEEE-style short-paper draft with verified methods
and capacity evidence. Its final-evaluation section remains explicitly gated;
numerical placeholders are prohibited. The local machine currently has no TeX
compiler, so a submission PDF must not be marked verified until the source is
compiled and visually inspected in a suitable free TeX environment.

Primary scholarly references already verified for drafting are stored in
`references.bib`. Repository or dataset citations must remain distinct from
evidence for empirical performance claims.
