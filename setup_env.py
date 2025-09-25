#!/usr/bin/env python3
"""
Environment Setup Script for Tabula Rasa

This script helps users set up their .env file for the Tabula Rasa training system.
It will create a .env file based on the .env.template if one doesn't exist.
"""

import os
import shutil
from pathlib import Path

def setup_environment():
    """Set up the .env file for the user."""
    env_file = Path('.env')
    template_file = Path('.env.template')
    
    print(" Tabula Rasa Environment Setup")
    print("=" * 40)
    
    # Check if .env already exists
    if env_file.exists():
        print(" .env file already exists")
        print(f"   Location: {env_file.absolute()}")
        
        # Ask if user wants to update it
        response = input("\n Do you want to update it with the latest template? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("   Keeping existing .env file")
            return
    
    # Check if template exists
    if not template_file.exists():
        print(" .env.template file not found")
        print("   Please ensure .env.template exists in the project root")
        return
    
    try:
        # Copy template to .env
        shutil.copy2(template_file, env_file)
        print(f" Created .env file from template")
        print(f"   Location: {env_file.absolute()}")
        
        print("\n Next steps:")
        print("   1. Edit .env file and add your ARC-3 API key")
        print("   2. Update any paths if needed")
        print("   3. Run the training system")
        
        print(f"\n Get your API key from: https://three.arcprize.org")
        
    except Exception as e:
        print(f" Error creating .env file: {e}")

if __name__ == "__main__":
    setup_environment()