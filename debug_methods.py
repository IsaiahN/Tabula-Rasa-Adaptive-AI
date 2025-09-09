#!/usr/bin/env python3
"""
Debug script to check method indentation
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Read the file and check the exact indentation
with open('src/arc_integration/continuous_learning_loop.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the _take_action method
for i, line in enumerate(lines):
    if '_take_action' in line:
        print(f"Line {i+1}: {repr(line)}")
        # Show context
        for j in range(max(0, i-5), min(len(lines), i+5)):
            print(f"  {j+1:4d}: {repr(lines[j])}")
        break
