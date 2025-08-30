#!/usr/bin/env python3
"""
ARC-3 Continuous Learning System - Simple Wrapper

A lightweight wrapper script that uses the existing ContinuousLearningLoop class
from the main codebase instead of duplicating logic.

Usage:
    python -m run_continuous_learning --mode demo          # Quick demonstration
    python -m run_continuous_learning --mode full_training # Run until all levels mastered
    python -m run_continuous_learning --mode comparison    # Compare salience modes
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

# Fix imports for when run as module - use absolute paths
current_dir = Path(__file__).parent.resolve()
src_dir = current_dir / "src"

# Always add src to path
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Now import with the correct path setup - use direct imports to avoid relative import issues
try:
    from arc_integration.continuous_learning_loop import ContinuousLearningLoop
    print("✅ Successfully imported ContinuousLearningLoop")
except ImportError as e:
    print(f"❌ Failed to import ContinuousLearningLoop: {e}")
    # Try alternative import
    try:
        sys.path.insert(0, str(current_dir / "src" / "arc_integration"))
        from continuous_learning_loop import ContinuousLearningLoop
        print("✅ Successfully imported ContinuousLearningLoop via fallback")
    except ImportError as e2:
        print(f"❌ All import attempts failed: {e2}")
        print("Available modules:")
        import pkgutil
        for importer, modname, ispkg in pkgutil.iter_modules():
            print(f"  - {modname}")
        sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def get_available_games(api_key: str, arc_agents_path: str) -> list:
    """Get list of actually available games from ARC-3 API."""
    import subprocess
    import re
    
    try:
        print("🎮 Fetching available games from ARC-3 API...")
        
        # Set up environment
        env = os.environ.copy()
        env['ARC_API_KEY'] = api_key
        
        # Run without specifying a game to get the game list
        cmd = ['uv', 'run', 'main.py', '--agent=random']
        
        result = subprocess.run(
            cmd,
            cwd=arc_agents_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Parse game list from output
        output = result.stdout + result.stderr
        print(f"DEBUG - API Response: {output[:200]}...")
        
        # Look for game list in various formats
        game_patterns = [
            r'Game list:\s*\[(.*?)\]',                    # Game list: [...]
            r'Available games:\s*\[(.*?)\]',              # Available games: [...]
            r'games?[:\s]*\[(.*?)\]',                     # games: [...]
            r'tasks?[:\s]*\[(.*?)\]'                      # tasks: [...]
        ]
        
        games = []
        for pattern in game_patterns:
            match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
            if match:
                games_str = match.group(1)
                print(f"DEBUG - Found games string: {games_str}")
                
                # Parse individual game IDs
                game_ids = re.findall(r'["\']([a-f0-9-]{8,})["\']', games_str)
                if game_ids:
                    games.extend(game_ids)
                    break
                    
                # Try parsing without quotes
                game_ids = re.findall(r'([a-f0-9-]{8,})', games_str)
                if game_ids:
                    games.extend(game_ids)
                    break
        
        # Remove duplicates and filter valid-looking game IDs
        games = list(set([g for g in games if len(g) >= 8 and re.match(r'^[a-f0-9-]+$', g)]))
        
        if games:
            print(f"✅ Found {len(games)} available games: {games[:3]}...")
            return games
        else:
            print("⚠️ No games found in API response, using test mode")
            print(f"Full API response: {output}")
            return []
            
    except Exception as e:
        print(f"❌ Error fetching games: {e}")
        return []

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
            print("ARC_API_KEY not found in environment")
            print("Please:")
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
        print(f"Configuration Error: {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Make sure the main codebase is properly installed")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()