# Branch Variant Plans (children of TB-Seed)

Goal: create four branches that "express" different subsets of TB-Seed's DNA. All branches share `seed.md` (contract & core) and diverge in which subsystems are enabled, refactored, or replaced.

Common seed (seed.md): defines core contracts: ARC3 integration, game state format, action format, DB schema expectations, and minimal runner API.

## 1) branch: tb-minimal

Objective: smallest runnable system that preserves ARC3 gameplay loop and frame pipeline.

Features expressed:
- ARC3 API integration
- GameRunner & session lifecycle
- Minimal persistence (use lightweight sqlite with required tables)
- Dummy vision (frame provider stub) or minimal frame analyzer

Removed/disabled:
- NEAT architect
- Bayesian/Graph systems
- Attention/Communication systems
- Real-time advanced learners (only basic learning engine)

Use case: CI, demos, debugging, reproducible baseline.

## 2) branch: tb-core-stable

Objective: refactor and stabilize the main systems into a clean, well-tested core.

Features expressed:
- ARC3 API
- Refactored orchestrator (Orchestrator, GameRunner, Trainer)
- Database integration facade
- Performance monitor and governor
- Vision pipeline (kept) but with clear interfaces

Removed/disabled:
- Experimental research modules moved behind feature flags

Use case: production-ready baseline for continued development.

## 3) branch: tb-performance

Objective: optimize runtime performance for large-scale training.

Features expressed:
- All core + performance-focused improvements
- Async-first loop, batch action submission where possible
- Profiling hooks, optimized DB writes (batched), in-memory caches
- Optional C-accelerated image processing (if available)

Trade-offs:
- More complex deployment, additional dependencies, reduced introspection.

Use case: large-scale training and experiments requiring throughput.

## 4) branch: tb-research-playground

Objective: playground for rapid experimentation and new algorithms.

Features expressed:
- All experimental modules enabled (NEAT, Bayesian, Graph traversal, attention/communication)
- Easy toggles, live-reload of components, and instrumentation for ablations
- Quick wiring to external notebooks and visualization tools

Trade-offs:
- Less focus on stability and performance; more complexity.

## Implementation plan for branching

- Create `seed.md` in repository root with contracts and minimal API.
- Create branch `tb-minimal` from `Tabula-Rasa-v4`, prune features and add minimal runner.
- Create branch `tb-core-stable` from `Tabula-Rasa-v4`, perform refactor into modules and add tests.
- Create branch `tb-performance` from `tb-core-stable`, apply profiling and performance improvements.
- Create branch `tb-research-playground` from `Tabula-Rasa-v4`, enable experimental modules and add developer tooling.

Mapping branches to simplifications and priorities
- tb-minimal: implement `SIMPLIFICATIONS` #2 (standard schema), #5 (tests), #6 (archive scripts), #7 (config). Add CI smoke test that explicitly uses stub API (flag required).
- tb-core-stable: implement `SIMPLIFICATIONS` #1 (orchestrator extraction), #3 (DBFacade migrations), #5 (tests). Prioritize integration tests.
- tb-performance: implement `SIMPLIFICATIONS` #3 (batched DB writes), #8 (DI for injected workers), #4 (model adapter optimizations where needed).
- tb-research-playground: implement `SIMPLIFICATIONS` #4 (ModelAdapters), #2 (schema), #1 (extraction), #5 (tests).

CI & policy notes
- Add a small CI pipeline that runs the smoke test `scripts/run_minimal.py --use-stub-api-for-ci` for tb-minimal. The CI must set the flag explicitly; if the flag is missing the job should fail.
- Require PRs targeting `Tabula-Rasa-v4` to include tests for DBFacade or orchestrator changes.

Next steps (implementation sequence)
- Create `tb-minimal` branch and land schema standardization + tests + config.
- Add CI smoke test that runs minimal mode with the stub API flag set.
- Then implement the `tb-core-stable` refactor incrementally (small PRs), starting with DBFacade hardening and orchestrator extraction.

Notes on migration and PR strategy
- Do small PRs: first add `seed.md` and feature flags in `Tabula-Rasa-v4`.
- Implement tb-minimal first to provide a runnable baseline.
- Use `tb-core-stable` to land refactors incrementally.
- Reuse platform CI to validate branches.

*Next steps: produce `seed.md` (contract) and mark analysis todo in-progress.*
