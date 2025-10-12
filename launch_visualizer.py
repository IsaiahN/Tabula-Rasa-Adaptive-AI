#!/usr/bin/env python3
"""
Launch the ARC Game Visualizer

This script starts the visualization widget that shows real-time gameplay
and allows replay of past game sessions.
"""

import sys
import os
from pathlib import Path

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Launch the visualizer."""
    try:
        from src.visualization.game_visualizer import GameVisualizer

        # Default database path
        db_path = "tabula_rasa.db"

        # Check if database exists
        if not os.path.exists(db_path):
            print(f"Database {db_path} not found.")
            print("Make sure you have run some training sessions first.")
            print("Or specify a different database path as an argument.")
            return

        print("=" * 60)
        print("ARC GAME VISUALIZER")
        print("=" * 60)
        print(f"Database: {db_path}")
        print()
        print("Features:")
        print("- Real-time visualization during gameplay")
        print("- Replay of past game sessions")
        print("- Frame-by-frame analysis")
        print("- Action highlighting and statistics")
        print()
        print("Usage:")
        print("1. Select 'Live' mode to watch current games")
        print("2. Select 'Replay' mode to analyze past sessions")
        print("3. Use playback controls to navigate frames")
        print("=" * 60)
        print()

        # Start visualizer
        visualizer = GameVisualizer(db_path)
        visualizer.run()

    except ImportError as e:
        print(f"Error importing visualizer: {e}")
        print("Make sure all dependencies are installed.")
    except KeyboardInterrupt:
        print("\nVisualizer closed.")
    except Exception as e:
        print(f"Error starting visualizer: {e}")

if __name__ == "__main__":
    main()