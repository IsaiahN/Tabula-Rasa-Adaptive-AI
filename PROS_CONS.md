# Pros / Cons Analysis (Initial)

This file lists pros, cons, maintenance cost, runtime cost, and coupling concerns for major TB-Seed features.

## ARC3 API integration
- Pros: Real-world grounding, ensures experiments are meaningful; single source of truth for game state.
- Cons: External dependency makes CI and offline development harder; API rate limits and flakiness can affect runs.
- Maintenance cost: Medium (keep integration up-to-date with API changes).
- Runtime cost: Low per-call but can accumulate; needs robust rate-limiting.
- Coupling concerns: Many components assume API behavior; breaking changes in API can cascade.

## Continuous learning orchestrator (`continuous_learning_loop.py`)
- Pros: Central orchestrator simplifies end-to-end experiments; many features and fallbacks make it robust.
- Cons: Very large single file, high cognitive load, hard to test and maintain.
- Maintenance cost: High
- Runtime cost: Moderate
- Coupling concerns: Strong coupling to many subsystems (vision, NEAT, Bayesian, graph traversal). Recommend refactor.

## Vision pipeline (frame analysis & detection)
- Pros: Core for decision-making; supports coordinate actions and advanced perception.
- Cons: Heavy dependencies (image processing libs), risk of import cycles; complex to test locally.
- Maintenance cost: Medium-high
- Runtime cost: High if using heavy models
- Coupling concerns: Multiple systems read frame data; define single FrameProvider interface to decouple.

## Real-time learning engine & strategy discovery
- Pros: Enables mid-game adaptation and pattern discovery; powerful for improving performance.
- Cons: Overlaps with other learning modules; complex state management; potential for noisy decisions.
- Maintenance cost: High
- Runtime cost: Moderate-high
- Coupling concerns: Needs clear ownership of pattern storage; avoid duplicate data stores.

## NEAT architect, Bayesian engine, Graph traversal (experimental modules)
- Pros: Powerful research tools for architecture search, probabilistic reasoning, and planning.
- Cons: High complexity and maintenance burden; should be optional behind feature flags.
- Maintenance cost: Very high
- Runtime cost: High
- Coupling concerns: Keep these decoupled, provide plugin interfaces.

## Attention & communication subsystems
- Pros: Useful for resource allocation and inter-module messaging.
- Cons: Adds complexity; may be overkill for minimal/production runs.
PROS / CONS ANALYSIS

This document lists major features in the repository and analyzes pros, cons, maintenance cost, runtime cost, and coupling concerns. Use this to drive simplification and branch choices.

1) Continuous Learning Orchestrator (`src/training/core/continuous_learning_loop.py`)
- Pros
	- Centralized control over training loop, multiple learning subsystems (NEAT, Bayesian, Graph, etc.).
	- Single place to implement experiments and high-level policies.
- Cons
	- Very large file: hard to navigate and brittle to small edits.
	- Tightly coupled components increase risk of regressions and import cycles.
	- Hard to unit test; stateful global behavior.
- Maintenance cost: High — needs refactor into smaller modules and clear interfaces.
- Runtime cost: Medium; orchestration overhead matters only at scale.
- Coupling concerns: Tight coupling to NEAT/Bayesian components and to persistence layer; refactor into interfaces.

2) Vision & Frame handling (`src/vision/*`, `scripts/run_minimal.py`)
- Pros
	- Frame-first design is appropriate for game-grid/visual tasks.
	- Pluggable FrameProvider and detectors make experiments easy.
- Cons
	- Real detectors aren't wired consistently; multiple ad-hoc shims exist.
	- Tests often simulate frames in code instead of using consistent provider patterns.
- Maintenance cost: Low-Medium if interfaces stabilized.
- Runtime cost: Varies with detector complexity; detection is often the bottleneck.
- Coupling concerns: Detector output schema needs to be standardized (`frame`, `objects` with bbox/label/score).

3) Database Persistence (`src/database/db_facade.py` and scripts interacting with DB)
- Pros
	- Centralized persistence makes experiment reproducibility and auditing easier.
	- SQLite-backed DB is simple and portable for local research runs.
- Cons
	- Current usage patterns sometimes bypass facade and write SQL in scripts, risking schema drift.
	- Concurrent access patterns not well tested; SQLite has locking semantics that must be respected.
- Maintenance cost: Low if API is used consistently; otherwise rising.
- Runtime cost: Low for local runs; could be higher for frequent writes under concurrency.
- Coupling concerns: Tightly coupled to session/game/action schemas; better to version the schema and add migrations.

4) ARC3 API integration (`src/api/*`)
- Pros
	- External API encapsulation keeps game-specific logic clean.
	- Allows real environment runs for evaluation.
- Cons
	- Mixing real API code and stub behavior in code paths can cause accidental production simulation.
	- Requires careful opt-in flags and documentation.
- Maintenance cost: Medium (API changes need updates); testing requires opt-in stubs.
- Runtime cost: Network I/O overhead for real API; stub is fast.
- Coupling concerns: Keep APIManager behind a clear interface and ensure CI uses the stub only when explicitly requested.

5) Training utilities, NEAT, Bayesian, and Graph models
- Pros
	- Multiple learning modalities support experimentation.
	- Reuse of older code for experiments convenience.
- Cons
	- Many legacy or half-complete implementations increase maintenance burden.
	- Lack of clear contracts between model outputs and orchestrator expectations.
- Maintenance cost: High unless culled or modularized.
- Runtime cost: Varies; some implementations may be slow or memory-heavy.
- Coupling concerns: Each model currently expects orchestrator internals; define adapter layers.

6) Scripts & Dev tooling (`scripts/`, `tools/`)
- Pros
	- Handy utilities for setup, DB checks, and quick experiments.
- Cons
	- Some scripts are duplicated or unmaintained (e.g., cleanup_database.py and cleanup_database.py.bak).
	- Mixed style and lack of standard CLI interface.
- Maintenance cost: Medium; regular cleanup recommended.
- Runtime cost: Low.
- Coupling concerns: Ensure scripts use DB facade and shared helpers rather than duplicating SQL.

Recommendations from Pros/Cons
- Refactor `continuous_learning_loop.py` into smaller modules with clear interfaces: `Orchestrator`, `SessionManager`, `ModelAdapter`, `DetectorAdapter`, `Persistence`.
- Stabilize the `frame` + `objects` detection schema and add a small validator.
- Make `DBFacade` the single source of truth for DB access; update scripts to use it.
- Keep the stub API strictly opt-in and add a command-line flag guarding its use; add CI job that explicitly sets the flag for smoke tests.
- Add unit tests for DBFacade and Detector adapters; add a smoke test that runs `scripts/run_minimal.py --use-stub-api-for-ci`.
