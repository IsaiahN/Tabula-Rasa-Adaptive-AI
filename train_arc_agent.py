#!/usr/bin/env python3
"""
ARC-AGI-3 Training System - UNIFIED

Single script with mode parameters to eliminate bloat:
  python train_arc_agent.py --mode sequential --salience decay --verbose
  python train_arc_agent.py --mode swarm --salience lossless
  python train_arc_agent.py --help

No multiple scripts, just one with parameters.
Shows moves, memory operations, decay, consolidation with error alerts.
"""

import asyncio
import sys
import time
import os
import argparse
import logging
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from arc_integration.continuous_learning_loop import ContinuousLearningLoop
    from core.salience_system import SalienceMode
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    print(f"❌ Make sure you're in the tabula-rasa directory and src/ exists")
    sys.exit(1)

class UnifiedTrainer:
    """Single trainer with all modes accessible via parameters."""
    
    def __init__(self, args):
        self.mode = args.mode
        self.salience = args.salience
        self.verbose = args.verbose
        self.episodes = args.episodes
        self.games = args.games
        self.target_win_rate = args.target_win_rate
        self.target_score = args.target_score
        self.max_iterations = args.max_iterations
        
        self.continuous_loop = None
        self.training_iterations = 0
        self.best_performance = {'win_rate': 0.0, 'avg_score': 0.0}
        self.scorecard_id = None
        
    def get_salience_mode(self) -> SalienceMode:
        """Convert string to SalienceMode enum."""
        if self.salience == 'lossless':
            return SalienceMode.LOSSLESS
        elif self.salience == 'decay':
            return SalienceMode.DECAY_COMPRESSION  
        elif self.salience == 'adaptive':
            return SalienceMode.ADAPTIVE_COMPRESSION
        else:
            print(f"❌ ERROR: Unknown salience mode: {self.salience}")
            print(f"   Valid options: lossless, decay, adaptive")
            sys.exit(1)
    
    def display_config(self):
        """Display current training configuration."""
        print("🎯 UNIFIED TRAINING CONFIGURATION")
        print("="*50)
        print(f"Mode: {self.mode.upper()}")
        print(f"Salience: {self.salience.upper()}")
        print(f"Target Win Rate: {self.target_win_rate:.1%}")
        print(f"Target Score: {self.target_score}")
        print(f"Episodes per Game: {self.episodes}")
        print(f"Games: {self.games}")
        print(f"Max Iterations: {self.max_iterations}")
        print(f"Verbose: {'YES' if self.verbose else 'NO'}")
        print()
        
    async def initialize_with_error_handling(self):
        """Initialize with comprehensive error handling and user alerts."""
        try:
            print("🔧 INITIALIZING TRAINING SYSTEM...")
            
            self.continuous_loop = ContinuousLearningLoop(
                arc_agents_path="C:/Users/Admin/Documents/GitHub/ARC-AGI-3-Agents",
                tabula_rasa_path="C:/Users/Admin/Documents/GitHub/tabula-rasa"
            )
            
            print("🔍 VERIFYING CONNECTIONS...")
            verification = await self.continuous_loop.verify_api_connection()
            
            if not verification['api_accessible']:
                print(f"❌ API CONNECTION ERROR")
                print(f"   Cannot connect to ARC-AGI-3 API")
                print(f"   Check your internet connection and API key")
                return False
            
            if not verification['arc_agents_available']:
                print(f"❌ ARC-AGENTS ERROR")
                print(f"   ARC-AGI-3-Agents not found")
                print(f"   Please clone: https://github.com/neoneye/ARC-AGI-3-Agents")
                return False
            
            print("✅ INITIALIZATION SUCCESSFUL")
            print(f"   API Games Available: {verification['total_games_available']}")
            return True
            
        except Exception as e:
            print(f"❌ INITIALIZATION ERROR: {e}")
            print(f"   Error Type: {type(e).__name__}")
            print(f"   Check your setup and try again")
            return False
    
    async def run_unified_training(self) -> dict:
        """Run training with unified error handling and verbose monitoring."""
        
        self.display_config()
        
        try:
            # Get real games from API
            print(f"🎯 GETTING {self.games} REAL GAMES FROM API...")
            training_games = await self.continuous_loop.select_training_games(
                count=self.games,
                difficulty_preference='mixed'
            )
            
            if not training_games:
                print(f"❌ GAME SELECTION ERROR")
                print(f"   Failed to get games from ARC-AGI-3 API")
                return {'success': False, 'error': 'No games available'}
            
            print(f"✅ Retrieved {len(training_games)} games:")
            for i, game_id in enumerate(training_games, 1):
                print(f"   {i}. {game_id}")
            
            # Create scorecard if possible
            print(f"📊 CREATING SCORECARD...")
            scorecard_id = await self.continuous_loop.create_real_scorecard(training_games)
            if scorecard_id:
                self.scorecard_id = scorecard_id
                print(f"✅ Scorecard: https://three.arcprize.org/scorecard/{scorecard_id}")
            else:
                print(f"⚠️  Scorecard creation failed - continuing without")
            
            overall_start_time = time.time()
            salience_mode = self.get_salience_mode()
            
            print(f"\n🚀 STARTING {self.mode.upper()} TRAINING")
            print(f"Memory Mode: {salience_mode.value}")
            print("="*60)
            
            # Training iterations
            for iteration in range(1, self.max_iterations + 1):
                self.training_iterations = iteration
                
                print(f"\n🚀 ITERATION {iteration}/{self.max_iterations}")
                print("="*40)
                
                try:
                    # Start training session with error handling
                    session_id = self.continuous_loop.start_training_session(
                        games=training_games,
                        max_episodes_per_game=self.episodes,
                        target_win_rate=min(0.3 + (iteration * 0.05), self.target_win_rate),
                        target_avg_score=self.target_score,
                        salience_mode=salience_mode,
                        enable_salience_comparison=False,
                        swarm_enabled=(self.mode == 'swarm')
                    )
                    
                    print(f"Session: {session_id}")
                    print(f"Mode: {self.mode.upper()}")
                    print(f"Episodes per Game: {self.episodes}")
                    
                    if self.verbose:
                        print(f"🔬 VERBOSE MODE ENABLED - Detailed logging active")
                    
                    # Run training with comprehensive error handling
                    session_results = await self._run_with_error_handling(session_id)
                    
                    if not session_results:
                        print(f"❌ ITERATION {iteration} FAILED - SESSION ERROR")
                        continue
                    
                    # Process results
                    performance = session_results.get('overall_performance', {})
                    win_rate = performance.get('overall_win_rate', 0.0)
                    avg_score = performance.get('overall_average_score', 0.0)
                    
                    # Update best performance
                    if win_rate > self.best_performance['win_rate']:
                        self.best_performance['win_rate'] = win_rate
                    if avg_score > self.best_performance['avg_score']:
                        self.best_performance['avg_score'] = avg_score
                    
                    # Show results
                    print(f"\n📊 ITERATION {iteration} RESULTS:")
                    print(f"Win Rate: {win_rate:.1%} (Best: {self.best_performance['win_rate']:.1%})")
                    print(f"Avg Score: {avg_score:.1f} (Best: {self.best_performance['avg_score']:.1f})")
                    
                    # Show memory/system status
                    if self.verbose:
                        self._show_verbose_status()
                    
                    # Check if target reached
                    if win_rate >= self.target_win_rate and avg_score >= self.target_score:
                        total_duration = time.time() - overall_start_time
                        
                        print(f"\n🎉 TARGET ACHIEVED!")
                        print(f"Mode: {self.mode.upper()}")
                        print(f"Salience: {salience_mode.value}")
                        print(f"Final Win Rate: {win_rate:.1%}")
                        print(f"Final Score: {avg_score:.1f}")
                        print(f"Iterations: {iteration}")
                        print(f"Duration: {total_duration/3600:.1f} hours")
                        
                        return {'success': True, 'performance': performance}
                    
                    # Rest between iterations
                    if iteration < self.max_iterations:
                        print(f"😴 Rest 5s before iteration {iteration + 1}...")
                        await asyncio.sleep(5)
                
                except Exception as e:
                    print(f"❌ ITERATION {iteration} ERROR: {e}")
                    print(f"   Error Type: {type(e).__name__}")
                    print(f"   Continuing to next iteration...")
                    await asyncio.sleep(10)
                    continue
            
            # Max iterations reached
            print(f"⚠️  REACHED MAX ITERATIONS ({self.max_iterations})")
            return {'success': False, 'best_performance': self.best_performance}
            
        except Exception as e:
            print(f"❌ CRITICAL TRAINING ERROR: {e}")
            print(f"   Error Type: {type(e).__name__}")
            print(f"   Training cannot continue")
            return {'success': False, 'error': str(e)}
    
    async def _run_with_error_handling(self, session_id: str):
        """Run session with comprehensive error handling."""
        try:
            return await self.continuous_loop.run_continuous_learning(session_id)
        except KeyboardInterrupt:
            print(f"🛑 USER INTERRUPTED TRAINING")
            raise
        except Exception as e:
            print(f"❌ SESSION ERROR: {e}")
            print(f"   Error Type: {type(e).__name__}")
            print(f"   Session ID: {session_id}")
            return None
    
    def _show_verbose_status(self):
        """Show detailed verbose status information."""
        try:
            print(f"🔬 VERBOSE STATUS:")
            
            # Memory file counts
            memory_files = self._count_memory_files()
            print(f"   Memory Files: {memory_files}")
            
            # System status
            if hasattr(self.continuous_loop, 'get_sleep_and_memory_status'):
                status = self.continuous_loop.get_sleep_and_memory_status()
                sleep_info = status.get('sleep_status', {})
                memory_info = status.get('memory_consolidation_status', {})
                
                print(f"   Sleep Cycles: {sleep_info.get('sleep_cycles_this_session', 0)}")
                print(f"   Memory Consolidations: {memory_info.get('consolidation_operations_completed', 0)}")
                print(f"   Energy Level: {sleep_info.get('current_energy_level', 1.0):.2f}")
            
            # Check for recent file activity
            recent_files = self._check_recent_file_activity()
            if recent_files:
                print(f"   Recent Files: {len(recent_files)} created in last minute")
                for file in recent_files[:3]:  # Show first 3
                    print(f"     📄 {file}")
            
        except Exception as e:
            print(f"   Verbose status error: {e}")
    
    def _count_memory_files(self) -> int:
        """Count memory and checkpoint files."""
        try:
            memory_paths = [
                Path("checkpoints"),
                Path("meta_learning_data"),
                Path("continuous_learning_data"),
                Path("test_meta_learning_data")
            ]
            
            total_files = 0
            for path in memory_paths:
                if path.exists():
                    total_files += len(list(path.rglob("*")))
            
            return total_files
        except Exception:
            return 0
    
    def _check_recent_file_activity(self) -> list:
        """Check for recently created/modified files."""
        try:
            cutoff_time = time.time() - 60  # Last minute
            recent_files = []
            
            for path in [Path("checkpoints"), Path("meta_learning_data"), Path("continuous_learning_data")]:
                if path.exists():
                    for file in path.rglob("*"):
                        if file.is_file() and file.stat().st_mtime > cutoff_time:
                            recent_files.append(file.name)
            
            return recent_files
        except Exception:
            return []

def create_parser():
    """Create argument parser for unified training script."""
    parser = argparse.ArgumentParser(
        description="ARC-AGI-3 Unified Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sequential training with decay compression (recommended):
  python train_arc_agent.py --mode sequential --salience decay --verbose
  
  # SWARM training with lossless memory:
  python train_arc_agent.py --mode swarm --salience lossless --episodes 100
  
  # Quick test with minimal settings:
  python train_arc_agent.py --mode sequential --episodes 5 --games 1
        """
    )
    
    parser.add_argument('--mode', 
                       choices=['sequential', 'swarm'], 
                       default='sequential',
                       help='Training mode: sequential (one game at a time) or swarm (concurrent)')
    
    parser.add_argument('--salience',
                       choices=['lossless', 'decay', 'adaptive'],
                       default='decay', 
                       help='Memory salience mode: lossless, decay, or adaptive')
    
    parser.add_argument('--episodes', 
                       type=int, 
                       default=25,
                       help='Episodes per game (default: 25)')
    
    parser.add_argument('--games',
                       type=int,
                       default=6, 
                       help='Number of games to train on (default: 6)')
    
    parser.add_argument('--target-win-rate',
                       type=float,
                       default=0.85,
                       help='Target win rate (default: 0.85)')
    
    parser.add_argument('--target-score',
                       type=float, 
                       default=75.0,
                       help='Target average score (default: 75.0)')
    
    parser.add_argument('--max-iterations',
                       type=int,
                       default=20,
                       help='Maximum training iterations (default: 20)')
    
    parser.add_argument('--verbose', 
                       action='store_true',
                       help='Enable verbose logging (shows moves, memory, decay)')
    
    return parser

async def main():
    """Run unified training with comprehensive error handling."""
    
    parser = create_parser()
    args = parser.parse_args()
    
    print("🚀 ARC-AGI-3 UNIFIED TRAINING SYSTEM")
    print("="*50)
    print("Single script, multiple modes, proper error handling")
    print()
    
    trainer = UnifiedTrainer(args)
    
    try:
        # Initialize with error handling
        if not await trainer.initialize_with_error_handling():
            print("❌ INITIALIZATION FAILED - Cannot continue")
            return 1
        
        print("✅ INITIALIZATION SUCCESSFUL")
        print("🎯 STARTING UNIFIED TRAINING...")
        
        # Run training with error handling
        results = await trainer.run_unified_training()
        
        if results['success']:
            print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
            return 0
        else:
            print(f"\n⚠️  TRAINING INCOMPLETE")
            print(f"Best Performance: {results.get('best_performance', {})}")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n🛑 TRAINING INTERRUPTED BY USER")
        print(f"Iterations completed: {trainer.training_iterations}")
        if trainer.scorecard_id:
            print(f"Scorecard: https://three.arcprize.org/scorecard/{trainer.scorecard_id}")
        return 130  # Standard interrupt exit code
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR IN MAIN TRAINING")
        print(f"   Error: {e}")
        print(f"   Error Type: {type(e).__name__}")
        print(f"   Contact support if this persists")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
