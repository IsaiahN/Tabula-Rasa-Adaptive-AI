#!/usr/bin/env python3
"""
Quick script to fix SalienceMode references in test files
"""

import os
import re

def fix_salience_modes_in_file(filepath):
    """Fix SalienceMode references in a single file."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace invalid SalienceMode values with valid ones
    replacements = [
        ('SalienceMode.DECAY', 'SalienceMode.LOSSLESS'),
        ('SalienceMode.MINIMAL', 'SalienceMode.DECAY_COMPRESSION'),
        # Fix lists/arrays that still have the old values
        ('[SalienceMode.LOSSLESS, SalienceMode.LOSSLESS, SalienceMode.DECAY_COMPRESSION]', '[SalienceMode.LOSSLESS, SalienceMode.DECAY_COMPRESSION]'),
    ]
    
    changed = False
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            changed = True
            print(f"  Replaced: {old} -> {new}")
    
    if changed:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Fixed: {filepath}")
    else:
        print(f"✅ No changes needed: {filepath}")

def main():
    """Fix all test files."""
    test_files = [
        'tests/unit/test_train_arc_agent.py',
        'tests/integration/test_arc_training_pipeline.py'
    ]
    
    for test_file in test_files:
        print(f"\n🔧 Fixing {test_file}...")
        fix_salience_modes_in_file(test_file)

if __name__ == '__main__':
    main()
