#!/usr/bin/env python3
"""
Script to remove all emojis from Python source files in the codebase.
"""

import os
import re
import sys
from pathlib import Path

# Comprehensive emoji pattern that covers most Unicode emoji ranges
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"  # Various symbols
    "]+",
    flags=re.UNICODE
)

def remove_emojis_from_text(text: str) -> str:
    """Remove all emojis from text."""
    return EMOJI_PATTERN.sub('', text)

def process_file(file_path: Path) -> bool:
    """Process a single Python file to remove emojis."""
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            original_content = f.read()

        # Remove emojis
        cleaned_content = remove_emojis_from_text(original_content)

        # Check if there were changes
        if original_content != cleaned_content:
            # Write back the cleaned content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)

            print(f" Cleaned: {file_path}")
            return True
        else:
            return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def find_python_files(root_dir: Path) -> list:
    """Find all Python files in the directory."""
    python_files = []

    for file_path in root_dir.rglob("*.py"):
        # Skip some directories that might cause issues
        skip_dirs = {'.git', '__pycache__', '.venv', 'venv', 'node_modules'}
        if not any(skip_dir in str(file_path) for skip_dir in skip_dirs):
            python_files.append(file_path)

    return python_files

def main():
    """Main function."""
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    print(f"Scanning for Python files in: {project_root}")

    # Find all Python files
    python_files = find_python_files(project_root)

    print(f"Found {len(python_files)} Python files")
    print("Removing emojis...")

    files_changed = 0

    # Process each file
    for file_path in python_files:
        if process_file(file_path):
            files_changed += 1

    print(f"\nCompleted!")
    print(f"Files processed: {len(python_files)}")
    print(f"Files changed: {files_changed}")

    if files_changed > 0:
        print(f"Successfully removed emojis from {files_changed} files")
    else:
        print("No emojis found to remove")

if __name__ == "__main__":
    main()