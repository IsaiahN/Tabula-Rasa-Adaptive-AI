# Simplifications & Refactor Concepts

This document lists concrete simplification concepts to reduce complexity and improve maintainability.

## Goals

- Strongly preserve core mechanics: ARC3 API integration, frame capture/analysis, action send/receive, session lifecycle.
- Reduce duplication, decouple experimental systems, and make the system easier to test and run.

## Suggested simplifications (ranked)

1. Modularize the orchestrator
   - Break `continuous_learning_loop.py` into smaller classes: Orchestrator, GameRunner, Trainer, Integrator.
   - Benefits: easier testing, faster reasoning about responsibilities.
   - Cost: initial refactor effort.

2. Feature flags / plugin system
   - Introduce a small plugin/feature-flag loader so optional subsystems (NEAT, Bayesian, Graph) can be enabled/disabled via config.
   - Benefits: simpler default runtime, explicit toggles for experiments.

3. Standardize async boundaries
   - Make clear boundaries: initialization (sync), runtime loop (async). Convert helpers to `async` where they call async APIs.
   - Benefits: reduces race conditions and import-time side-effects.

4. Flatten overlapping learning components
   - Identify overlapping responsibilities (pattern discovery vs. real-time learner vs. strategy discovery) and define a single source-of-truth for pattern storage and retrieval.
   - Benefits: fewer inconsistencies, smaller surface area for bugs.

5. Replace bespoke components where sensible
   - E.g., use NetworkX for graph modeling (if acceptable), or an off-the-shelf Bayesian library for core parts.
   - Benefits: less custom code to maintain; may increase dependencies.

6. Lightweight vision shim for non-vision runs
   - Allow running without heavy vision dependencies by providing a "frame provider" interface and a noop/dummy implementation.
   - Benefits: CI and low-resource testing become feasible.

7. Improve initialization order & avoid circular imports
   - Lazy-import modules only when needed and pass concrete interfaces rather than importing heavy modules at top-level.

## Prioritized simplification tasks (concrete 8-step plan)

1) Extract Orchestrator Subsystems (Priority: High)
   - Break `continuous_learning_loop.py` into smaller modules: `orchestrator.py`, `session_manager.py`, `model_adapters.py`, `detector_adapters.py`, and `persistence.py`.
   - Benefit: single-responsibility, easier to test and review.
   - Estimated effort: 3-5 days.

2) Standardize Frame / Object Schema (Priority: High)
   - Add `src/vision/schema.py` to define `Frame`, `DetectedObject`, and a validator asserting fields: `bbox`, `label`, `score`.
   - Benefit: removes subtle runtime errors from detector mismatches.
   - Estimated effort: 1 day.

3) Centralize DBFacade and add basic migrations (Priority: High)
   - Ensure scripts and modules use `src/database/db_facade.py`. Add a `migrations` table and a small migration helper.
   - Benefit: prevents schema drift and eases upgrades.
   - Estimated effort: 2-3 days.

4) ModelAdapter interface and adapters (Priority: Medium)
   - Define a `ModelAdapter` interface with `train`, `predict`, `save`, `load` and convert NEAT/Bayesian/Graph into adapters.
   - Benefit: decouples models from orchestrator; allows experimentation without touching orchestrator.
   - Estimated effort: 3-7 days.

5) Test harness & smoke tests (Priority: High)
   - Add `/tests` with unit tests for `DBFacade`, `schema` validators, and `DummyDetector`. Add CI smoke test running `scripts/run_minimal.py --use-stub-api-for-ci`.
   - Benefit: prevents regressions and enforces stub opt-in behavior.
   - Estimated effort: 1-2 days.

6) Archive legacy scripts and cleanup (Priority: Low)
   - Move duplicates and unmaintained scripts to `scripts/archive/` and add a short status README.
   - Benefit: reduces accidental usage.
   - Estimated effort: 0.5 day.

7) Central configuration (Priority: Medium)
   - Add a `src/config.py` to centralize DB path, API host, and toggles (stub opt-in, minimal mode).
   - Benefit: easier reproducible runs and CI integration.
   - Estimated effort: 1 day.

8) Optional DI lightweight (Priority: Medium)
   - Make orchestrator accept adapters (db, api, detector, model) via constructor for testability.
   - Benefit: improved testability and reduced import cycles.
   - Estimated effort: 2-4 days.

## Quick wins (low risk)

- Add a small `config.yaml` or `pyproject` section to enable toggles.
- Extract DB integration helpers into a single `DatabaseIntegration` facade.
- Add a lightweight CLI mode: `--minimal` to run only core ARC3 + GameRunner.


## Long-term / higher effort

- Re-architect NEAT / Bayesian / Graph features into separate micro-packages or services.
- Create unit tests for the core runner and API integration with lightweight stubs/mocks.

*Next: create branch plans showing how to express these simplifications into concrete child branches.*
