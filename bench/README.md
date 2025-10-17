# Bench

Lightweight benchmarking helpers for the project. The bench harness is intentionally minimal and designed to be CI-friendly.

Usage
------

Run a target locally (recommended to run as module so package imports resolve):

```powershell
python -m bench.runner --module bench.targets.game_runner_target --iters 3
```

Write JSON results to a file (useful for CI):

```powershell
python -m bench.runner --module bench.targets.game_runner_target --iters 3 --out bench/results.json
```

Notes
------
- Bench targets are under `bench/targets/`. Each target should expose a `main(iterations=...)` function or a callable `main()`.
- Running as a module (`python -m bench.runner`) ensures `bench` is on sys.path and relative imports work.
- Targets should use the project's `StubAPIManager` for deterministic runs in CI.

CI integration
--------------
The repository includes a GitHub Actions workflow `.github/workflows/bench.yml` that runs this harness on `workflow_dispatch` and on pushes to the `tb-performance` branch. The workflow writes the runner stdout to `bench_output.json` and uploads it as an artifact named `bench-results`.

Next steps
-----------
- Add more targets for microbenchmarks (DB, vision preprocessing, etc.).
- Consider adding a small history store (CSV/JSON) to track performance over time.
Performance harness README

This folder contains simple benchmarking utilities to measure runtime and memory for key operations.

Usage:

- `python -m bench.runner --help` for help.

The harness is intentionally minimal and designed to run quickly in CI with small inputs.