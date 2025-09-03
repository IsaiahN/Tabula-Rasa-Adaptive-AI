#!/usr/bin/env python3
"""
Integration test runner that properly sets up Python path for the project.
"""
import sys
import os
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

# Now run the integration test
if __name__ == "__main__":
    # Import and run the test
    from tests.integration.test_enhanced_training_integration import main as test_main
    test_main()
