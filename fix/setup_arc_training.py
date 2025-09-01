#!/usr/bin/env python3
"""
ARC-AGI-3 Training Setup Script

This script helps set up the environment for training the Adaptive Learning Agent
on ARC-AGI-3 tasks by installing dependencies and configuring the environment.
"""

import subprocess
import sys
import os
from pathlib import Path
import shutil
import logging

def setup_logging():
    """Set up basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def check_python_version():
    """Check if Python version is compatible."""
    logger = logging.getLogger(__name__)
    
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    
    logger.info(f"Python version: {sys.version}")
    return True

def install_uv():
    """Install uv package manager if not available."""
    logger = logging.getLogger(__name__)
    
    # Check if uv is already installed
    if shutil.which("uv"):
        logger.info("uv is already installed")
        return True
    
    logger.info("Installing uv package manager...")
    
    try:
        if os.name == 'nt':  # Windows
            # Use pip to install uv on Windows
            subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
        else:  # Unix-like systems
            # Use the official installer
            subprocess.run([
                "curl", "-LsSf", "https://astral.sh/uv/install.sh"
            ], stdout=subprocess.PIPE, check=True)
            subprocess.run(["sh"], input=b"curl -LsSf https://astral.sh/uv/install.sh | sh", check=True)
        
        logger.info("uv installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install uv: {e}")
        return False

def setup_arc_agents_repo(target_dir: str = "../ARC-AGI-3-Agents"):
    """Set up the ARC-AGI-3-Agents repository."""
    logger = logging.getLogger(__name__)
    
    target_path = Path(target_dir).resolve()
    
    if target_path.exists():
        logger.info(f"ARC-AGI-3-Agents already exists at {target_path}")
        return str(target_path)
    
    logger.info("Cloning ARC-AGI-3-Agents repository...")
    
    try:
        # Clone the repository
        subprocess.run([
            "git", "clone", 
            "https://github.com/arcprize/ARC-AGI-3-Agents.git",
            str(target_path)
        ], check=True)
        
        logger.info(f"Repository cloned to {target_path}")
        
        # Install dependencies using uv
        logger.info("Installing ARC-AGI-3-Agents dependencies...")
        subprocess.run(["uv", "sync"], cwd=str(target_path), check=True)
        
        logger.info("Dependencies installed successfully")
        return str(target_path)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to set up ARC-AGI-3-Agents: {e}")
        return None

def install_tabula_rasa_dependencies():
    """Install additional dependencies for tabula-rasa ARC integration."""
    logger = logging.getLogger(__name__)
    
    logger.info("Installing tabula-rasa dependencies...")
    
    additional_deps = [
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "pydantic>=1.8.0",
        "requests>=2.25.0",
        "asyncio-mqtt",  # For potential future MQTT integration
    ]
    
    try:
        for dep in additional_deps:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
            
        logger.info("All dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def create_env_file(arc_agents_path: str, api_key: str = None):
    """Create .env file for ARC-AGI-3-Agents."""
    logger = logging.getLogger(__name__)
    
    env_file = Path(arc_agents_path) / ".env"
    env_example = Path(arc_agents_path) / ".env.example"
    
    if env_file.exists():
        logger.info(".env file already exists")
        return True
    
    # Copy from example if it exists
    if env_example.exists():
        shutil.copy(env_example, env_file)
        logger.info("Created .env file from .env.example")
    else:
        # Create basic .env file
        with open(env_file, 'w') as f:
            f.write("# ARC-AGI-3 Configuration\n")
            f.write("ARC_API_KEY=your_api_key_here\n")
            f.write("DEBUG=False\n")
        logger.info("Created basic .env file")
    
    # Update with API key if provided
    if api_key:
        with open(env_file, 'r') as f:
            content = f.read()
        
        content = content.replace("your_api_key_here", api_key)
        content = content.replace("ARC_API_KEY=", f"ARC_API_KEY={api_key}")
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        logger.info("Updated .env file with provided API key")
    
    return True

def verify_setup(arc_agents_path: str):
    """Verify that the setup is working correctly."""
    logger = logging.getLogger(__name__)
    
    logger.info("Verifying setup...")
    
    # Check that our agent is properly integrated
    agents_init = Path(arc_agents_path) / "agents" / "__init__.py"
    if not agents_init.exists():
        logger.error("agents/__init__.py not found")
        return False
    
    with open(agents_init, 'r') as f:
        content = f.read()
        if "AdaptiveLearning" not in content:
            logger.error("AdaptiveLearning agent not found in agents/__init__.py")
            return False
    
    # Check that our agent file exists
    agent_file = Path(arc_agents_path) / "agents" / "templates" / "adaptive_learning_agent.py"
    if not agent_file.exists():
        logger.error("adaptive_learning_agent.py not found")
        return False
    
    logger.info("Setup verification passed!")
    return True

def main():
    """Main setup function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting ARC-AGI-3 training setup...")
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install uv
    if not install_uv():
        logger.error("Failed to install uv package manager")
        return 1
    
    # Set up ARC-AGI-3-Agents repository
    arc_agents_path = setup_arc_agents_repo()
    if not arc_agents_path:
        logger.error("Failed to set up ARC-AGI-3-Agents repository")
        return 1
    
    # Install tabula-rasa dependencies
    if not install_tabula_rasa_dependencies():
        logger.error("Failed to install tabula-rasa dependencies")
        return 1
    
    # Create .env file
    api_key = input("Enter your ARC-AGI-3 API key (or press Enter to skip): ").strip()
    if not create_env_file(arc_agents_path, api_key if api_key else None):
        logger.error("Failed to create .env file")
        return 1
    
    # Verify setup
    if not verify_setup(arc_agents_path):
        logger.error("Setup verification failed")
        return 1
    
    logger.info("Setup completed successfully!")
    logger.info(f"ARC-AGI-3-Agents installed at: {arc_agents_path}")
    
    if not api_key:
        logger.info("Don't forget to add your ARC-AGI-3 API key to the .env file!")
        logger.info("Get your API key from: https://three.arcprize.org")
    
    logger.info("\nNext steps:")
    logger.info("1. Get your API key from https://three.arcprize.org if you haven't already")
    logger.info("2. Update the .env file with your API key")
    logger.info("3. Run: python train_arc_agent.py --api-key YOUR_API_KEY")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
