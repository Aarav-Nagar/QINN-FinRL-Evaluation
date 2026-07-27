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
