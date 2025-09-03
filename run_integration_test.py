#!/usr/bin/env python3
"""
Integration test runner for the project.
"""
import sys
import os
from pathlib import Path

# Now run the integration test
if __name__ == "__main__":
    # Import and run the test
    from tests.integration.test_enhanced_training_integration import main as test_main
    test_main()
