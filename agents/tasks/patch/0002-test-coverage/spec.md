# Study Utils Test Coverage Lift — Spec

## Summary
We will raise the repository’s Python test coverage from 56% to a stable 100% by backfilling unit tests, introducing dependency seams, and enforcing a coverage gate so future contributions preserve the bar.

## Goals
- Reach 100% line coverage for the `study_utils` package using pytest-cov.
- Add reusable fixtures/mocks that isolate external services (OpenAI, WeasyPrint, Textual, Pydub) and filesystem state.
- Wire a coverage gate (`--cov-fail-under=100`) into the default test command so CI and local runs fail on regressions.
- Document the testing strategy (fixtures, mocks, coverage expectations) for contributors.

## Non-Goals
- Refactor or redesign core application logic beyond what is required for testability.
- Exercise real networked services or external binaries during tests.
- Replace existing tooling (pytest, pytest-cov, justfile targets).

## Behavior (BDD-ish)
- Given a fresh checkout, when `pytest --cov=study_utils` (or the `just test` shortcut) runs, then the suite completes without contacting external services and reports 100% coverage.
- Given CI runs on a pull request, when coverage dips below 100%, then the build fails with a clear coverage error message.
- Given new contributors review project docs, when they read the testing guide, then they understand how to use the provided mocks/fixtures to keep coverage at 100%.

## Constraints & Dependencies
- Constraints: Tests must run offline and deterministically; no reliance on API keys or system-installed GUI/pdf dependencies.
- Upstream/Downstream: Coordinate with CI configuration (existing GitHub Actions) to ensure the coverage gate is adopted; confirm no other repositories consume uncovered CLI modules in ways that would break with added seams.

## Security & Privacy
- Ensure mocks prevent accidental leakage of `OPENAI_API_KEY` or other secrets; do not record fixture data that contains PII.
- Validate that any temporary files created during tests are isolated within pytest tmp paths and cleaned up.

## Telemetry & Operability
- Coverage gate acts as the primary health signal; optionally log uncovered lines during local development via `coverage xml/json` artifacts for troubleshooting.
- No runtime telemetry changes are required.

## Rollout / Revert
- Rollout: land tests and helper seams incrementally, then enable the coverage gate once the suite is at 100%.
- Revert: to undo, remove or relax the coverage gate and delete the added fixture modules (not expected unless blockers arise).

## Definition of Done
- [ ] Behavior verified (100% coverage, gate green locally and in CI)
- [ ] Docs updated (testing/fixtures guidance)
- [ ] Tests added/updated (unit, CLI, utility seams)
- [ ] Flags defaulted per channel (n/a)
- [ ] Monitoring in place (coverage gate scripted)

## Ownership
- Owner: @matt
- Reviewers: @codex
- Stakeholders: @study-utils-team

## Links
- Related: coverage baseline (`pytest --cov=study_utils --cov-report=term-missing`)

## History
### 2024-05-23 00:00
**Summary** — Initial draft
**Changes**
- Captured plan to reach and enforce 100% coverage.
