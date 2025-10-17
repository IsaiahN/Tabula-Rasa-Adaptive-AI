TB-MINIMAL Manifest

This branch expresses the minimal TB-Seed DNA. Changes here are intentionally small and conservative.

Key goals:
- Provide a tiny, well-tested runtime useful for CI and quick iteration.
- Enforce seed constraints: `frame` schema, DB-backed persistence, and explicit stub opt-in for CI.

Included changes:
- `scripts/run_minimal.py` (minimal runner)
- `src/training/game_runner.py` (lightweight runner, uses DBFacade)
- `src/vision/schema.py` (detection validator)
- `tests/*` focused tests and smoke test
- CI job that runs the smoke test with `USE_STUB_API_FOR_CI=1`

Notes:
- Keep experimental models disabled by default and behind feature flags.
- Incrementally extract orchestrator pieces to `src/training/core/orchestrator.py` and `session_manager.py` (thin wrappers).