#!/usr/bin/env python3
"""
ARC-3 Continuous Learning System - Simple Wrapper

A lightweight wrapper script that uses the existing ContinuousLearningLoop class
from the main codebase instead of duplicating logic.

Usage:
    python run_continuous_learning.py --mode demo          # Quick demonstration
    python run_continuous_learning.py --mode full_training # Run until all levels mastered
    python run_continuous_learning.py --mode comparison    # Compare salience modes
"""

import asyncio
import logging
import argparse
import sys
import os
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import the actual implementation from the main codebase
from src.arc_integration.continuous_learning_loop import ContinuousLearningLoop

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_arc_agents_path() -> str:
    """Find ARC-AGI-3-Agents repository path."""
    arc_agents_path = os.getenv('ARC_AGENTS_PATH')
    if (arc_agents_path and Path(arc_agents_path).exists()):
        return arc_agents_path
        
    # Search common locations
    possible_paths = [
        Path.cwd().parent / "ARC-AGI-3-Agents",
        Path.cwd() / "ARC-AGI-3-Agents", 
        Path.home() / "ARC-AGI-3-Agents",
        Path("C:/Users/Admin/Documents/GitHub/ARC-AGI-3-Agents"),
        Path("C:/ARC-AGI-3-Agents"),
        Path("/opt/ARC-AGI-3-Agents")
    ]
    
    for path in possible_paths:
        if path.exists() and (path / "main.py").exists():
            return str(path)
            
    raise ValueError(
        "ARC-AGI-3-Agents repository not found. Please ensure it's available at one of these locations:\n" +
        "\n".join(f"- {path}" for path in possible_paths) +
        "\nOr set ARC_AGENTS_PATH in your .env file"
    )

def main():
    """Main function - simple wrapper that delegates to the actual implementation."""
    parser = argparse.ArgumentParser(description='ARC-3 Continuous Learning System')
    parser.add_argument('--mode', choices=['demo', 'full_training', 'comparison'], 
                        default='demo', help='Operation mode')
    
    args = parser.parse_args()
    
    try:
        # Get required paths and API key
        arc_agents_path = find_arc_agents_path()
        tabula_rasa_path = str(Path.cwd())
        api_key = os.getenv('ARC_API_KEY')
        
        if not api_key:
            print("❌ ARC_API_KEY not found in environment")
            print("💡 Please:")
            print("   1. Register at https://three.arcprize.org")
            print("   2. Get your API key from your profile")
            print("   3. Add it to your .env file: ARC_API_KEY=your_key_here")
            sys.exit(1)
        
        # Create the continuous learning loop with correct parameters
        learning_loop = ContinuousLearningLoop(
            arc_agents_path=arc_agents_path,
            tabula_rasa_path=tabula_rasa_path,
            api_key=api_key
        )
        
        # Run the appropriate mode
        if args.mode == "demo":
            results = asyncio.run(learning_loop.run_demo_mode())
        elif args.mode == "full_training":
            results = asyncio.run(learning_loop.run_full_training_mode())
        elif args.mode == "comparison":
            results = asyncio.run(learning_loop.run_comparison_mode())
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
            
        logger.info(f"Training completed successfully in {args.mode} mode")
        return results
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure the main codebase is properly installed")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()