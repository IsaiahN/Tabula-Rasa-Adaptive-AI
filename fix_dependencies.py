#!/usr/bin/env python3
"""
Fix missing dependencies and test GAN system
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install missing dependencies."""
    print("Installing missing dependencies...")
    
    try:
        # Install psutil
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil>=5.9.0"])
        print("✅ psutil installed successfully")
        
        # Install other potential missing dependencies
        dependencies = [
            "torch>=2.0.0",
            "numpy>=1.24.0",
            "aiohttp>=3.8.0",
            "python-dotenv>=1.0.0"
        ]
        
        for dep in dependencies:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                print(f"✅ {dep} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"⚠️  Warning: Failed to install {dep}: {e}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def test_imports():
    """Test that imports work after installing dependencies."""
    print("\nTesting imports...")
    
    try:
        import psutil
        print("✅ psutil import successful")
        
        import torch
        print("✅ torch import successful")
        
        import numpy as np
        print("✅ numpy import successful")
        
        import aiohttp
        print("✅ aiohttp import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def main():
    """Main function."""
    print("Fixing dependencies for GAN system...\n")
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return False
    
    # Test imports
    if not test_imports():
        print("❌ Import test failed")
        return False
    
    print("\n🎉 Dependencies fixed successfully!")
    print("You can now run the training script: ./run_9hour_simple_training.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
