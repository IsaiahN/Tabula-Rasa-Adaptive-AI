#!/usr/bin/env python3
"""
Demo script to restart ARC training with enhanced frame analyzer
and interaction logging/hypothesis generation system.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

async def main():
    """Start enhanced ARC training with visual intelligence."""
    
    print("🚀 STARTING ENHANCED ARC TRAINING")
    print("=" * 60)
    print("🔍 Enhanced Features:")
    print("   • Full-frame scanning for ALL color objects")
    print("   • Coordinate avoidance system (no more loops!)")
    print("   • Movement tracking and pattern analysis") 
    print("   • ACTION6 interaction logging with visual analysis")
    print("   • Sleep-time hypothesis generation")
    print("   • Actionable recommendations for targeting")
    print("=" * 60)
    
    # Check if the continuous learning loop exists
    if not os.path.exists('src/arc_integration/continuous_learning_loop.py'):
        print("❌ continuous_learning_loop.py not found in src/arc_integration/")
        return
    
    # Import and run the continuous learning system
    try:
        from arc_integration.continuous_learning_loop import ContinuousLearningLoop
        
        # Initialize the enhanced system
        system = ContinuousLearningLoop()
        
        print("\n🎯 System Initialization Complete")
        print("   • Frame Analyzer: Enhanced visual intelligence loaded")
        print("   • Sleep System: Hypothesis generation enabled")
        print("   • Coordinate Intelligence: Avoidance system active")
        
        # Start continuous learning
        print("\n🔄 Starting Continuous Learning Loop...")
        print("   Monitor for visual intelligence insights during sleep cycles!")
        print("   Watch for ACTION6 interaction logging and pattern discovery...")
        
        # Run the learning system
        await system.run_continuous_learning()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Make sure all dependencies are installed")
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the enhanced training
    asyncio.run(main())
