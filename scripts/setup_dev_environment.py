#!/usr/bin/env python3
"""
Development Environment Setup Script

Sets up the complete development environment for the Tabula Rasa project.
This includes installing dependencies, setting up pre-commit hooks, and
validating the environment.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def run_command(cmd: List[str], description: str, check: bool = True) -> Tuple[bool, str]:
    """Run a command and return success status and output."""
    print(f"Running {description}...")
    try:
        result = subprocess.run(
            cmd, 
            check=check, 
            capture_output=True, 
            text=True,
            cwd=Path(__file__).parent.parent
        )
        print(f" {description} completed successfully")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f" {description} failed:")
        print(e.stdout)
        print(e.stderr)
        return False, e.stderr


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    print("Checking Python version...")
    if sys.version_info < (3, 8):
        print(f" Python 3.8+ required, found {sys.version}")
        return False
    print(f" Python {sys.version.split()[0]} is compatible")
    return True


def install_dependencies() -> bool:
    """Install project dependencies."""
    success = True
    
    # Install base dependencies
    success &= run_command(
        ["pip", "install", "-r", "requirements.txt"],
        "Installing base dependencies"
    )[0]
    
    # Install development dependencies
    success &= run_command(
        ["pip", "install", "-e", ".[dev]"],
        "Installing development dependencies"
    )[0]
    
    return success


def setup_pre_commit() -> bool:
    """Set up pre-commit hooks."""
    success = True
    
    # Install pre-commit
    success &= run_command(
        ["pip", "install", "pre-commit"],
        "Installing pre-commit"
    )[0]
    
    # Install pre-commit hooks
    success &= run_command(
        ["pre-commit", "install"],
        "Installing pre-commit hooks"
    )[0]
    
    return success


def validate_environment() -> bool:
    """Validate the development environment."""
    success = True
    
    # Test imports
    success &= run_command(
        ["python", "-c", "from src.training import ContinuousLearningLoop, MasterARCTrainer; print(' Core imports working')"],
        "Testing core imports"
    )[0]
    
    # Test formatting tools
    success &= run_command(
        ["black", "--version"],
        "Testing Black formatter"
    )[0]
    
    success &= run_command(
        ["isort", "--version"],
        "Testing isort"
    )[0]
    
    success &= run_command(
        ["flake8", "--version"],
        "Testing flake8"
    )[0]
    
    success &= run_command(
        ["mypy", "--version"],
        "Testing mypy"
    )[0]
    
    return success


def run_initial_checks() -> bool:
    """Run initial code quality checks."""
    success = True
    
    # Format code
    success &= run_command(
        ["python", "scripts/format_code.py", "format"],
        "Running initial code formatting"
    )[0]
    
    # Run linting
    success &= run_command(
        ["python", "scripts/format_code.py", "lint"],
        "Running initial linting"
    )[0]
    
    return success


def create_git_hooks() -> bool:
    """Create additional git hooks if needed."""
    git_hooks_dir = Path(".git/hooks")
    if not git_hooks_dir.exists():
        print("  Git repository not found, skipping git hooks")
        return True
    
    # Create pre-push hook
    pre_push_hook = git_hooks_dir / "pre-push"
    pre_push_content = """#!/bin/bash
# Pre-push hook for Tabula Rasa
echo "Running pre-push checks..."

# Run tests
python scripts/format_code.py test

# Run linting
python scripts/format_code.py lint

echo "Pre-push checks completed"
"""
    
    try:
        with open(pre_push_hook, "w") as f:
            f.write(pre_push_content)
        pre_push_hook.chmod(0o755)
        print(" Created pre-push git hook")
        return True
    except Exception as e:
        print(f"  Could not create git hooks: {e}")
        return True  # Not critical


def main():
    """Main setup function."""
    print(" Setting up Tabula Rasa development environment...")
    print("=" * 60)
    
    success = True
    
    # Check Python version
    success &= check_python_version()
    
    if not success:
        print("\n Setup failed due to Python version incompatibility")
        sys.exit(1)
    
    # Install dependencies
    success &= install_dependencies()
    
    if not success:
        print("\n Setup failed during dependency installation")
        sys.exit(1)
    
    # Set up pre-commit
    success &= setup_pre_commit()
    
    # Validate environment
    success &= validate_environment()
    
    # Run initial checks
    success &= run_initial_checks()
    
    # Create git hooks
    success &= create_git_hooks()
    
    print("\n" + "=" * 60)
    
    if success:
        print(" Development environment setup completed successfully!")
        print("\nNext steps:")
        print("1. Run 'make check' to verify everything is working")
        print("2. Run 'make test' to run the test suite")
        print("3. Start developing!")
        print("\nUseful commands:")
        print("- make format     : Format code")
        print("- make lint       : Run linting")
        print("- make test       : Run tests")
        print("- make check      : Run all checks")
        print("- make clean      : Clean up temporary files")
    else:
        print(" Setup completed with some issues")
        print("Please check the output above for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
