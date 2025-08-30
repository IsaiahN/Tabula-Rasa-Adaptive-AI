#!/usr/bin/env python3
"""
ARC-AGI-3 Training Script for Adaptive Learning Agent

This script sets up and runs continuous learning sessions for the Adaptive Learning Agent
on ARC-AGI-3 abstract reasoning tasks.
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path
from typing import List, Optional
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from arc_integration.continuous_learning_loop import ContinuousLearningLoop


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Set up logging configuration."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def validate_setup(arc_agents_path: str, api_key: str) -> bool:
    """Validate that the setup is correct."""
    logger = logging.getLogger(__name__)
    
    # Check ARC-AGI-3-Agents path
    arc_path = Path(arc_agents_path)
    if not arc_path.exists():
        logger.error(f"ARC-AGI-3-Agents path does not exist: {arc_path}")
        return False
        
    main_py = arc_path / "main.py"
    if not main_py.exists():
        logger.error(f"main.py not found in ARC-AGI-3-Agents directory: {main_py}")
        return False
        
    # Check API key
    if not api_key or len(api_key) < 10:
        logger.error("Invalid or missing ARC_API_KEY")
        return False
        
    # Check that our agent is registered
    agents_init = arc_path / "agents" / "__init__.py"
    if agents_init.exists():
        with open(agents_init, 'r') as f:
            content = f.read()
            if "AdaptiveLearning" not in content:
                logger.warning("AdaptiveLearning agent may not be properly registered")
                
    logger.info("Setup validation passed")
    return True


async def run_training_session(
    arc_agents_path: str,
    tabula_rasa_path: str,
    api_key: str,
    games: List[str],
    max_episodes: int,
    target_win_rate: float,
    target_score: float,
    save_dir: str
) -> dict:
    """Run a complete training session."""
    logger = logging.getLogger(__name__)
    
    # Initialize continuous learning loop
    learning_loop = ContinuousLearningLoop(
        arc_agents_path=arc_agents_path,
        tabula_rasa_path=tabula_rasa_path,
        api_key=api_key,
        save_directory=save_dir
    )
    
    # Start training session
    session_id = learning_loop.start_training_session(
        games=games,
        max_episodes_per_game=max_episodes,
        target_win_rate=target_win_rate,
        target_avg_score=target_score
    )
    
    logger.info(f"Started training session: {session_id}")
    
    # Run continuous learning
    results = await learning_loop.run_continuous_learning(session_id)
    
    # Print summary
    logger.info("Training session completed!")
    logger.info(f"Session ID: {session_id}")
    logger.info(f"Games trained: {results['overall_performance'].get('games_trained', 0)}")
    logger.info(f"Total episodes: {results['overall_performance'].get('total_episodes', 0)}")
    logger.info(f"Overall win rate: {results['overall_performance'].get('overall_win_rate', 0):.2%}")
    logger.info(f"Average score: {results['overall_performance'].get('overall_average_score', 0):.1f}")
    
    return results


def get_available_games() -> List[str]:
    """Get a list of recommended ARC games for training."""
    # These are example game IDs - replace with actual available games
    return [
        "ls20",  # Example game from the quickstart
        "basic_pattern_1",
        "basic_pattern_2",
        "spatial_reasoning_1",
        "logical_sequence_1"
    ]


async def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train Adaptive Learning Agent on ARC-AGI-3 tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training on default games
  python train_arc_agent.py --api-key YOUR_API_KEY
  
  # Train on specific games with custom parameters
  python train_arc_agent.py --api-key YOUR_API_KEY --games ls20,pattern1 --episodes 30
  
  # Resume training with higher targets
  python train_arc_agent.py --api-key YOUR_API_KEY --target-win-rate 0.5 --target-score 75
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--api-key",
        required=True,
        help="ARC-AGI-3 API key (get from https://three.arcprize.org)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--arc-agents-path",
        default="../ARC-AGI-3-Agents",
        help="Path to ARC-AGI-3-Agents repository (default: ../ARC-AGI-3-Agents)"
    )
    
    parser.add_argument(
        "--games",
        help="Comma-separated list of game IDs to train on (default: recommended games)"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=25,
        help="Maximum episodes per game (default: 25)"
    )
    
    parser.add_argument(
        "--target-win-rate",
        type=float,
        default=0.3,
        help="Target win rate to achieve (default: 0.3)"
    )
    
    parser.add_argument(
        "--target-score",
        type=float,
        default=50.0,
        help="Target average score to achieve (default: 50.0)"
    )
    
    parser.add_argument(
        "--save-dir",
        default="arc_training_data",
        help="Directory to save training data (default: arc_training_data)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        help="Log file path (default: console only)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without running training"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    # Resolve paths
    tabula_rasa_path = Path(__file__).parent.absolute()
    arc_agents_path = Path(args.arc_agents_path).resolve()
    
    logger.info(f"Tabula Rasa path: {tabula_rasa_path}")
    logger.info(f"ARC-AGI-3-Agents path: {arc_agents_path}")
    
    # Validate setup
    if not validate_setup(str(arc_agents_path), args.api_key):
        logger.error("Setup validation failed")
        return 1
        
    if args.dry_run:
        logger.info("Dry run completed successfully")
        return 0
        
    # Determine games to train on
    if args.games:
        games = [g.strip() for g in args.games.split(",")]
    else:
        games = get_available_games()
        logger.info(f"Using recommended games: {games}")
        
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    try:
        # Run training
        results = await run_training_session(
            arc_agents_path=str(arc_agents_path),
            tabula_rasa_path=str(tabula_rasa_path),
            api_key=args.api_key,
            games=games,
            max_episodes=args.episodes,
            target_win_rate=args.target_win_rate,
            target_score=args.target_score,
            save_dir=str(save_dir)
        )
        
        # Save final results summary
        summary_file = save_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"Training completed successfully! Results saved to {summary_file}")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
