"""Benchmark target: Run a minimal GameRunner invocation once.

This module exposes a `main()` function that the bench runner imports to measure.
It uses the project's stub API manager and GameRunner where available.

This target is intentionally lightweight and deterministic for CI.
"""
from __future__ import annotations

import time
from typing import Dict, Any
import asyncio

# Import project modules with defensive fallbacks in case bench is run outside of PYTHONPATH
try:
    from src.api.stub_api_manager import StubAPIManager
    from src.training.game_runner import GameRunner
except Exception:
    # Try relative imports if bench is run as part of package
    try:
        from ..src.api.stub_api_manager import StubAPIManager  # type: ignore
        from ..src.training.game_runner import GameRunner  # type: ignore
    except Exception as e:  # pragma: no cover - best-effort import
        raise ImportError(
            "Could not import project GameRunner or StubAPIManager. Ensure bench is run with repository root on PYTHONPATH"
        ) from e


def _run_once(api: Any) -> Dict[str, Any]:
    """Helper to run the async GameRunner.run_game synchronously via asyncio.run."""
    runner = GameRunner(api_manager=api)
    # GameRunner exposes async run_game(game_id, max_actions)
    return asyncio.run(runner.run_game("bench_game", max_actions=10))


def main(iterations: int = 1) -> Dict[str, Any]:
    """Run GameRunner `iterations` times and return timing summary.

    Returns a dict with keys: iterations, total_time, avg_time, sample_result
    """
    api = StubAPIManager()
    results = []
    start = time.perf_counter()
    sample_result = None
    for _ in range(iterations):
        res = _run_once(api)
        sample_result = res
        results.append(res)
    end = time.perf_counter()
    total = end - start
    avg = total / iterations if iterations else 0
    return {
        "iterations": iterations,
        "total_time": total,
        "avg_time": avg,
        "sample_result": sample_result,
    }


if __name__ == "__main__":
    import json

    out = main(iterations=1)
    print(json.dumps(out, indent=2))
