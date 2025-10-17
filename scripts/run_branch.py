"""Run branch-specific smoke/demo commands.

Usage:
  python scripts/run_branch.py --branch tb-performance --iters 1
  python scripts/run_branch.py --all --dry-run

The script runs short, safe commands per branch and defaults to using the stub API where applicable.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


BRANCH_COMMANDS = {
    "tb-minimal": {
        "desc": "Run minimal runner (stub API)",
        "cmd": [PYTHON, str(ROOT / "scripts" / "run_minimal.py"), "--game-id", "smoke_minimal", "--max-actions", "10", "--use-stub-api-for-ci"]
    },
    "tb-core-stable": {
        "desc": "Run core smoke (GameRunner with stub)",
        "cmd": [PYTHON, str(ROOT / "scripts" / "run_minimal.py"), "--game-id", "core_smoke", "--max-actions", "20", "--use-stub-api-for-ci"]
    },
    "tb-performance": {
        "desc": "Run bench target once",
        "cmd": [PYTHON, "-m", "bench.runner", "--module", "bench.targets.game_runner_target", "--iters", "1", "--out", str(ROOT / "bench" / "results.json")]  # runs as module
    },
    "tb-research-playground": {
        "desc": "Run a short notebook-free demo using stub",
        "cmd": [PYTHON, str(ROOT / "scripts" / "run_minimal.py"), "--game-id", "research_smoke", "--max-actions", "5", "--use-stub-api-for-ci"]
    }
}


def run_command(cmd, dry_run=False):
    print(" ")
    print("Running:", " ".join(map(str, cmd)))
    if dry_run:
        return 0
    try:
        proc = subprocess.run(cmd, check=True, capture_output=False)
        return proc.returncode
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit {e.returncode}")
        return e.returncode


def git_checkout(branch: str, dry_run: bool = False):
    print(f"Checking out branch {branch}")
    if dry_run:
        return 0
    try:
        subprocess.run(["git", "checkout", branch], check=True)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Git checkout failed: {e}")
        return e.returncode


def append_history(results_path: Path, history_path: Path):
    import json
    now = __import__("datetime").datetime.utcnow().isoformat()
    try:
        with open(results_path, "r", encoding="utf-8") as fh:
            latest = fh.read()
        # Try parse json; otherwise wrap as text
        try:
            data = json.loads(latest)
        except Exception:
            data = {"raw": latest}
        entry = {"timestamp": now, "results": data}
        hist = []
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as fh:
                try:
                    hist = json.load(fh)
                except Exception:
                    hist = []
        hist.append(entry)
        with open(history_path, "w", encoding="utf-8") as fh:
            json.dump(hist, fh, indent=2)
        print(f"Appended results to {history_path}")
        return True
    except Exception as e:
        print(f"Failed to append history: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", help="Branch to run (tb-minimal|tb-core-stable|tb-performance|tb-research-playground)")
    parser.add_argument("--all", action="store_true", help="Run all branch smoke commands")
    parser.add_argument("--dry-run", action="store_true", help="Print commands but don't execute")
    parser.add_argument("--iters", type=int, default=1, help="Iterations for bench where applicable")
    parser.add_argument("--checkout", action="store_true", help="Checkout the branch before running the command")
    parser.add_argument("--append-history", action="store_true", help="Append bench results to bench/history.json")
    parser.add_argument("--commit-history", action="store_true", help="Commit and push history.json after appending")
    args = parser.parse_args()

    to_run = []
    if args.all:
        to_run = list(BRANCH_COMMANDS.items())
    elif args.branch:
        if args.branch not in BRANCH_COMMANDS:
            print(f"Unknown branch: {args.branch}")
            sys.exit(2)
        to_run = [(args.branch, BRANCH_COMMANDS[args.branch])]
    else:
        parser.print_help()
        sys.exit(1)

    for name, info in to_run:
        cmd = info["cmd"].copy()
        # if bench and iters provided, replace --iters arg
        if name == "tb-performance":
            # find --iters or --iters position and update
            if "--iters" in cmd:
                idx = cmd.index("--iters")
                cmd[idx+1] = str(args.iters)
        print(f"--- {name}: {info['desc']} ---")
        # optionally checkout branch first
        if getattr(args, 'checkout', False):
            co_rc = git_checkout(name, dry_run=args.dry_run)
            if co_rc != 0:
                print(f"Skipping run for {name} due to checkout failure")
                continue

        rc = run_command(cmd, dry_run=args.dry_run)
        if rc != 0:
            print(f"Branch {name} command failed (rc={rc})")
            # continue to next but record failure
        else:
            # If requested, append bench results to history
            if name == 'tb-performance' and getattr(args, 'append_history', False):
                results_path = ROOT / 'bench' / 'results.json'
                history_path = ROOT / 'bench' / 'history.json'
                ok = append_history(results_path, history_path)
                if ok and getattr(args, 'commit_history', False):
                    try:
                        subprocess.run(["git", "add", str(history_path)], check=True)
                        subprocess.run(["git", "commit", "-m", "chore(bench): append benchmark history"], check=True)
                        subprocess.run(["git", "push"], check=True)
                        print("Committed and pushed history.json")
                    except subprocess.CalledProcessError as e:
                        print(f"Failed to commit/push history: {e}")

    print("All requested branch runs completed.")


if __name__ == '__main__':
    main()
