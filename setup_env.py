#!/usr/bin/env python3
"""
Environment Setup Script for Tabula Rasa ARC-AGI-3 Training

This script helps you set up the required environment variables for running
the Tabula Rasa training system with ARC-AGI-3 integration.
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create a .env file with the required environment variables."""
    env_content = """# ARC-AGI-3 API Configuration
# Get your API key from https://three.arcprize.org
ARC_API_KEY=your_arc_api_key_here

# Optional: Path to ARC-AGI-3-Agents repository
ARC_AGENTS_PATH=C:\\Users\\Admin\\Documents\\GitHub\\ARC-AGI-3-Agents

# Training Configuration
TARGET_WIN_RATE=0.90
TARGET_AVG_SCORE=85.0
MAX_EPISODES_PER_GAME=50

# ARC-AGI-3-Agents Server Configuration
DEBUG=False
RECORDINGS_DIR=recordings
SCHEME=https
HOST=three.arcprize.org
PORT=443
WDM_LOG=0
"""
    
    env_file = Path('.env')
    if env_file.exists():
        print(f"⚠️  .env file already exists at {env_file.absolute()}")
        response = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if response != 'y':
            print("Skipping .env file creation.")
            return False
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"✅ Created .env file at {env_file.absolute()}")
        print("📝 Please edit the .env file and add your actual ARC_API_KEY")
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def check_environment():
    """Check if the required environment variables are set."""
    print("🔍 Checking environment variables...")
    
    arc_api_key = os.getenv('ARC_API_KEY')
    if arc_api_key:
        print(f"✅ ARC_API_KEY is set: {arc_api_key[:8]}...")
        return True
    else:
        print("❌ ARC_API_KEY is not set")
        return False

def print_setup_instructions():
    """Print instructions for setting up environment variables."""
    print("\n📋 Environment Setup Instructions:")
    print("=" * 50)
    print("1. Get your ARC API key from: https://three.arcprize.org")
    print("2. Set the environment variable using one of these methods:")
    print()
    print("   Windows Command Prompt:")
    print("   set ARC_API_KEY=your_api_key_here")
    print()
    print("   Windows PowerShell:")
    print("   $env:ARC_API_KEY=\"your_api_key_here\"")
    print()
    print("   Linux/Mac:")
    print("   export ARC_API_KEY=your_api_key_here")
    print()
    print("   Or create a .env file with:")
    print("   ARC_API_KEY=your_api_key_here")
    print()
    print("3. Run the training script:")
    print("   python master_arc_trainer.py")

def main():
    """Main function."""
    print("🚀 Tabula Rasa Environment Setup")
    print("=" * 40)
    
    # Check if environment is already set up
    if check_environment():
        print("\n✅ Environment is already configured!")
        return
    
    print("\n🔧 Setting up environment...")
    
    # Try to create .env file
    if create_env_file():
        print("\n📝 Next steps:")
        print("1. Edit the .env file and replace 'your_arc_api_key_here' with your actual API key")
        print("2. Run: python master_arc_trainer.py")
    else:
        print_setup_instructions()

if __name__ == "__main__":
    main()

