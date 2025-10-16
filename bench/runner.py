"""Simple benchmark runner for the project.

This runner provides a tiny harness to measure execution time for a provided target function.
It is intentionally minimal and includes an option to write JSON results for CI.
"""
import time
import argparse
import json
from importlib import import_module
from typing import Dict, Any, List


def time_function(module_path: str, func_name: str = "main", iterations: int = 10) -> List[float]:
    module = import_module(module_path)
    fn = getattr(module, func_name)
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        # allow target to accept iterations via signature, but call without args by default
        fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return times


def summarize(times: List[float]) -> Dict[str, Any]:
    if not times:
        return {"runs": 0, "avg": None, "min": None, "max": None}
    return {
        "runs": len(times),
        "avg": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True, help="Module path to benchmark, e.g., src.training.game_runner_demo")
    parser.add_argument("--func", default="main", help="Function name to call inside module")
    parser.add_argument("--iters", type=int, default=3, help="Iterations to run")
    parser.add_argument("--out", help="Path to write JSON output (also prints to stdout)")
    args = parser.parse_args()

    times = time_function(args.module, args.func, args.iters)
    stats = summarize(times)

    # Human-friendly output
    if stats["runs"]:
        print(f"Runs: {stats['runs']}, avg: {stats['avg']:.6f}s, min: {stats['min']:.6f}s, max: {stats['max']:.6f}s")
    else:
        print("No runs executed")

    # Write JSON results if requested
    if args.out:
        payload = {"module": args.module, "func": args.func, "iterations": args.iters, "stats": stats}
        try:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            print(f"Wrote results to {args.out}")
        except Exception as e:
            print(f"Failed to write output file {args.out}: {e}")


if __name__ == "__main__":
    main()
