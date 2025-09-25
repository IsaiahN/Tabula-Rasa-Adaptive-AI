#!/usr/bin/env python3
"""
Code Formatting Script

Automated code formatting and linting for the Tabula Rasa project.
This script provides a unified interface for all code quality tools.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f" {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f" {description} failed:")
        print(e.stdout)
        print(e.stderr)
        return False


def format_code(files: Optional[List[str]] = None) -> bool:
    """Format code using black and isort."""
    success = True
    
    # Black formatting
    cmd = ["black", "--line-length=120"]
    if files:
        cmd.extend(files)
    else:
        cmd.extend(["src/", "tests/"])
    
    success &= run_command(cmd, "Black formatting")
    
    # isort import sorting
    cmd = ["isort", "--profile=black", "--line-length=120"]
    if files:
        cmd.extend(files)
    else:
        cmd.extend(["src/", "tests/"])
    
    success &= run_command(cmd, "isort import sorting")
    
    return success


def check_formatting(files: Optional[List[str]] = None) -> bool:
    """Check if code is properly formatted."""
    success = True
    
    # Black check
    cmd = ["black", "--check", "--line-length=120"]
    if files:
        cmd.extend(files)
    else:
        cmd.extend(["src/", "tests/"])
    
    success &= run_command(cmd, "Black format check")
    
    # isort check
    cmd = ["isort", "--check-only", "--profile=black", "--line-length=120"]
    if files:
        cmd.extend(files)
    else:
        cmd.extend(["src/", "tests/"])
    
    success &= run_command(cmd, "isort format check")
    
    return success


def run_linting(files: Optional[List[str]] = None) -> bool:
    """Run linting checks."""
    success = True
    
    # Flake8 linting
    cmd = ["flake8", "--max-line-length=120"]
    if files:
        cmd.extend(files)
    else:
        cmd.extend(["src/", "tests/"])
    
    success &= run_command(cmd, "Flake8 linting")
    
    # MyPy type checking
    cmd = ["mypy", "--ignore-missing-imports"]
    if files:
        cmd.extend(files)
    else:
        cmd.extend(["src/"])
    
    success &= run_command(cmd, "MyPy type checking")
    
    return success


def run_tests(test_type: str = "all") -> bool:
    """Run tests."""
    cmd = ["pytest", "-v"]
    
    if test_type == "unit":
        cmd.extend(["tests/unit/"])
    elif test_type == "integration":
        cmd.extend(["tests/integration/"])
    elif test_type == "performance":
        cmd.extend(["-k", "performance"])
    else:
        cmd.extend(["tests/"])
    
    return run_command(cmd, f"Running {test_type} tests")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Code formatting and quality tools")
    parser.add_argument(
        "action",
        choices=["format", "check", "lint", "test", "all"],
        help="Action to perform"
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Specific files to process"
    )
    parser.add_argument(
        "--test-type",
        choices=["all", "unit", "integration", "performance"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Fix issues where possible"
    )
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    import os
    os.chdir(project_root)
    
    success = True
    
    if args.action == "format":
        success = format_code(args.files)
    elif args.action == "check":
        success = check_formatting(args.files)
    elif args.action == "lint":
        success = run_linting(args.files)
    elif args.action == "test":
        success = run_tests(args.test_type)
    elif args.action == "all":
        success = (
            format_code(args.files) and
            run_linting(args.files) and
            run_tests(args.test_type)
        )
    
    if success:
        print("\n All operations completed successfully!")
        sys.exit(0)
    else:
        print("\n Some operations failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
