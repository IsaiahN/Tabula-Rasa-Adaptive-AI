Branch overview for TB-Seed and child branches

This document describes the branch structure created from `TB-Seed`, the role of each child branch, how the pieces interact, how to run focused tests, and an LLM runbook that specifies what an LLM agent should do to run or validate the system.

## Branch topology

- TB-Seed (root)
  - The canonical seed baseline. Minimal, self-contained project that defines the `frame`-first contract and DB-backed persistence policy. This branch is intentionally stable and kept as the origin for producing children.

- tb-minimal (child)
  - Purpose: a minimal runtime that is fast to run in CI and easy to use for reproducing core behaviors.
  - Key artifacts: `scripts/run_minimal.py`, `src/training/game_runner.py`, lightweight `DummyDetector` and `FrameProvider`, `src/database/db_facade.py` (lightweight facade), and a stub API (`src/api/stub_api_manager.py`) that is opt-in via `--use-stub-api-for-ci`.
  - Intended usage: quick smoke tests, CI checks that do not rely on the external ARC-AGI-3 API.

- tb-core-stable (child)
  - Purpose: stabilize and refactor core orchestration logic. Incrementally extract initialization responsibilities into `src/training/core/orchestrator.py` while keeping behavior backward-compatible.
  - Key artifacts: `ContinuousLearningLoop` delegating to `Orchestrator`, migration-safe initializers, and tests verifying delegation.
  - Intended usage: base branch for system-level refactors that should be merged into other children.

- tb-performance (child)
  - Purpose: contain performance-oriented scaffolding: benchmarking harnesses, profiling tools, and performance-focused CI jobs.
  - Key artifacts (planned): `bench/` harness, performance CI jobs, and measurement dashboards.
  - Intended usage: experiments to measure runtime & memory impact of refactors.

- tb-research-playground (child)
  - Purpose: sandbox for experiments, notebooks, and quick prototypes.
  - Key artifacts (planned): `notebooks/`, research utilities, experimental models and adapters.

## High-level interaction

- Core invariants:
  - The `frame` object is the canonical input for perception and action-selection. All modules consuming visual data must accept `frame` and validated detection outputs.
  - Persistence: non-CI runs must use the DB facade to persist patterns, sequences, and session data. The stub API is explicitly opt-in for CI and testing.
  - Orchestrator migration: we moved initialization out of the monolithic `ContinuousLearningLoop` one small initializer at a time into the `Orchestrator` facade. The legacy core will delegate to the Orchestrator when present.

- Branch relationships:
  - `tb-core-stable` contains the canonical, tested core refactors. It is merged into children so they inherit the stable core and then implement branch-specific features.
  - `tb-minimal` stays small and CI-friendly; it contains the GameRunner and the stub API for fast smoke runs.
  - `tb-performance` and `tb-research-playground` branch from `tb-minimal` plus `tb-core-stable` merges so they have the minimal hooks + stable core.

## How to run and test locally

Assumptions: you have Python 3.11+ installed and a working virtualenv. Run these commands from the repository root.

1) Run focused tests (fast):

```pwsh
# from repo root
python -m pytest -q tests
```

2) Run minimal smoke script using stub API (opt-in flag required):

```pwsh
$env:PYTHONPATH = (Resolve-Path .).Path
python scripts/run_minimal.py --use-stub-api-for-ci --max-actions 10
```

3) Run the full continuous training (CAUTION: interacts with external APIs):

```pwsh
python train.py
```

Note: The stub API is for CI/testing only and must be enabled explicitly. For local development against a real API, configure credentials and the upstream API manager.

## LLM runbook — how an LLM agent should run and validate everything

This is a short, prescriptive runbook for an LLM-based automation agent (or a human following steps) to run and validate the codebase. The runbook assumes the agent has a shell on a developer machine where `python` and `gh` are available.

1) Prepare environment
   - Create and activate a virtualenv and install the repo requirements (`pip install -r requirements.txt`).
   - Ensure `PYTHONPATH` is set to the repo root when running scripts that import `src.*` modules.

2) Run focused tests
   - `python -m pytest tests` — confirm all focused tests pass. If there are failures, collect the failing trace and stop.

3) Run the minimal smoke script
   - `python scripts/run_minimal.py --use-stub-api-for-ci --max-actions 5`
   - Expect deterministic, fast output showing the runner completed (e.g., `Result: {'game_id': 'test_game', 'score': X, 'actions_taken': Y, 'win': False}`)

4) Inspect branches & PRs
   - Use `git` and `gh` to list branches and open PRs.
   - For each child branch (tb-minimal, tb-performance, tb-research-playground) verify it contains `BRANCH_FEATURE.md` and that CI passes.

5) Validate Orchestrator behavior (basic check)
   - Run the lightweight delegation test `tests/test_orchestrator_delegation.py` to ensure orchestrator initializers set flags appropriately.

6) If everything passes, mark PRs ready for review and request reviewers.

## Notes and best practices

- Keep `TB-Seed` immutable unless a critical baseline change is required. Use it as an origin for new child branches.
- Make small, easily-reviewable PRs on child branches. Merge `tb-core-stable` into children to deliver core improvements.
- Use `tb-minimal` for CI-friendly smoke runs and `tb-performance` for heavier benchmarks and profiling-based PRs.

---

File created on branch `tb-core-stable`.

If you want this file added to the child branches as well, tell me and I will merge it into them (I can do that next).