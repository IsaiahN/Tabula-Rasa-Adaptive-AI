#!/usr/bin/env python3
"""
Configurable ARC-AGI-3 Training System - MINIMAL DEPENDENCIES

Run training with simple command-line options:
- SWARM mode vs Sequential mode  
- Lossless vs Decay vs Adaptive compression
- No external config files needed
"""

import asyncio
import sys
import time
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from arc_integration.continuous_learning_loop import ContinuousLearningLoop
    from core.salience_system import SalienceMode
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Make sure you're in the tabula-rasa directory and src/ exists")
    sys.exit(1)

class SimpleConfigurableTrainer:
    """Simple trainer with environment variable configuration."""
    
    def __init__(self):
        # Load configuration from environment variables with defaults
        self.swarm_enabled = os.getenv('SWARM_MODE', 'false').lower() == 'true'
        self.max_concurrent_games = int(os.getenv('MAX_CONCURRENT_GAMES', '3'))
        self.salience_mode_str = os.getenv('SALIENCE_MODE', 'decay_compression').lower()
        self.target_win_rate = float(os.getenv('TARGET_WIN_RATE', '0.85'))
        self.target_avg_score = float(os.getenv('TARGET_AVG_SCORE', '75.0'))
        self.max_episodes_per_game = int(os.getenv('MAX_EPISODES_PER_GAME', '50'))
        self.max_iterations = int(os.getenv('MAX_ITERATIONS', '20'))
        self.game_count = int(os.getenv('GAME_COUNT', '6'))
        
        self.continuous_loop = None
        self.training_iterations = 0
        self.best_performance = {'win_rate': 0.0, 'avg_score': 0.0}
        self.real_scorecard_id = None
        
    def get_salience_mode(self) -> SalienceMode:
        """Convert string to SalienceMode enum."""
        if self.salience_mode_str == 'lossless':
            return SalienceMode.LOSSLESS
        elif self.salience_mode_str == 'decay_compression':
            return SalienceMode.DECAY_COMPRESSION  
        elif self.salience_mode_str == 'adaptive_compression':
            return SalienceMode.ADAPTIVE_COMPRESSION
        else:
            print(f"⚠️  Unknown salience mode: {self.salience_mode_str}, using decay_compression")
            return SalienceMode.DECAY_COMPRESSION
    
    def display_config(self):
        """Display current training configuration."""
        print("🎯 TRAINING CONFIGURATION")
        print("="*50)
        print(f"Training Mode: {'SWARM' if self.swarm_enabled else 'SEQUENTIAL'}")
        if self.swarm_enabled:
            print(f"Max Concurrent Games: {self.max_concurrent_games}")
        print(f"Salience Mode: {self.salience_mode_str.upper()}")
        print(f"Target Win Rate: {self.target_win_rate:.1%}")
        print(f"Target Avg Score: {self.target_avg_score}")
        print(f"Max Episodes/Game: {self.max_episodes_per_game}")
        print(f"Game Count: {self.game_count}")
        print()
        
    async def initialize_and_verify(self):
        """Initialize and verify real API connection."""
        try:
            self.continuous_loop = ContinuousLearningLoop(
                arc_agents_path="C:/Users/Admin/Documents/GitHub/ARC-AGI-3-Agents",
                tabula_rasa_path="C:/Users/Admin/Documents/GitHub/tabula-rasa"
            )
            
            # Verify API connection
            verification = await self.continuous_loop.verify_api_connection()
            
            if not verification['api_accessible']:
                print(f"❌ Cannot connect to ARC-AGI-3 API")
                return False
            
            print("✅ Real API connection verified")
            print(f"   Available games: {verification['total_games_available']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            return False
    
    async def run_simple_training(self) -> dict:
        """Run training based on simple configuration."""
        
        self.display_config()
        
        # Get real training games from API
        print(f"🎯 GETTING {self.game_count} REAL GAMES FROM API")
        training_games = await self.continuous_loop.select_training_games(
            count=self.game_count,
            difficulty_preference='mixed'
        )
        
        if not training_games:
            return {'success': False, 'error': 'Failed to get real games'}
        
        # Create real scorecard
        print(f"📊 CREATING REAL SCORECARD...")
        scorecard_id = await self.continuous_loop.create_real_scorecard(training_games)
        if scorecard_id:
            self.real_scorecard_id = scorecard_id
            print(f"✅ Real scorecard: https://three.arcprize.org/scorecard/{scorecard_id}")
        
        overall_start_time = time.time()
        last_performance = {'overall_win_rate': 0.0, 'overall_average_score': 0.0}
        
        # Determine salience mode
        salience_mode = self.get_salience_mode()
        
        print(f"\n🚀 STARTING {'SWARM' if self.swarm_enabled else 'SEQUENTIAL'} TRAINING")
        print(f"Using {salience_mode.value.upper()} memory mode")
        print("="*60)
        
        # Continuous training loop
        for iteration in range(1, self.max_iterations + 1):
            self.training_iterations = iteration
            iteration_start_time = time.time()
            
            print(f"\n🚀 TRAINING ITERATION {iteration}")
            print("="*40)
            
            # Adaptive parameters
            episodes_per_game = self.get_adaptive_episodes(iteration, last_performance)
            iteration_target = min(0.3 + (iteration * 0.05), self.target_win_rate)
            
            print(f"Mode: {'SWARM' if self.swarm_enabled else 'SEQUENTIAL'}")
            print(f"Episodes per Game: {episodes_per_game}")
            print(f"Salience Mode: {salience_mode.value}")
            print(f"Target: {iteration_target:.1%}")
            
            try:
                # Start training session
                session_id = self.continuous_loop.start_training_session(
                    games=training_games,
                    max_episodes_per_game=episodes_per_game,
                    target_win_rate=iteration_target,
                    target_avg_score=self.target_avg_score,
                    salience_mode=salience_mode,
                    enable_salience_comparison=False,
                    swarm_enabled=self.swarm_enabled
                )
                
                # Run training
                if self.swarm_enabled:
                    print(f"🔥 SWARM MODE: {self.max_concurrent_games} concurrent games")
                else:
                    print(f"⚡ SEQUENTIAL MODE: One game at a time")
                
                session_results = await self.continuous_loop.run_continuous_learning(session_id)
                
                # Get performance results
                performance = session_results.get('overall_performance', {})
                current_win_rate = performance.get('overall_win_rate', 0.0)
                current_avg_score = performance.get('overall_average_score', 0.0)
                
                iteration_duration = time.time() - iteration_start_time
                
                # Update best performance
                if current_win_rate > self.best_performance['win_rate']:
                    self.best_performance['win_rate'] = current_win_rate
                if current_avg_score > self.best_performance['avg_score']:
                    self.best_performance['avg_score'] = current_avg_score
                
                # Display results
                print(f"\n📊 ITERATION {iteration} RESULTS:")
                print(f"Duration: {iteration_duration/60:.1f} minutes")
                print(f"Win Rate: {current_win_rate:.1%} (Best: {self.best_performance['win_rate']:.1%})")
                print(f"Avg Score: {current_avg_score:.1f} (Best: {self.best_performance['avg_score']:.1f})")
                
                # Show system status
                try:
                    system_status = self.continuous_loop.get_sleep_and_memory_status()
                    sleep_info = system_status['sleep_status']
                    memory_info = system_status['memory_consolidation_status']
                    
                    print(f"Sleep Cycles: {sleep_info['sleep_cycles_this_session']}")
                    print(f"Memory Consolidations: {memory_info['consolidation_operations_completed']}")
                except:
                    pass  # Skip if status not available
                
                # Check if target reached
                if (current_win_rate >= self.target_win_rate and 
                    current_avg_score >= self.target_avg_score):
                    
                    total_duration = time.time() - overall_start_time
                    
                    print(f"\n🎉 TARGET PERFORMANCE ACHIEVED!")
                    print("="*50)
                    print(f"Mode: {'SWARM' if self.swarm_enabled else 'SEQUENTIAL'}")
                    print(f"Salience: {salience_mode.value}")
                    print(f"Final Win Rate: {current_win_rate:.1%}")
                    print(f"Final Avg Score: {current_avg_score:.1f}")
                    print(f"Training Iterations: {iteration}")
                    print(f"Total Time: {total_duration/3600:.1f} hours")
                    
                    if self.real_scorecard_id:
                        print(f"🏆 Scorecard: https://three.arcprize.org/scorecard/{self.real_scorecard_id}")
                    
                    return {
                        'success': True,
                        'mode': 'swarm' if self.swarm_enabled else 'sequential',
                        'salience_mode': salience_mode.value,
                        'final_performance': performance,
                        'training_iterations': iteration,
                        'total_duration_hours': total_duration / 3600,
                        'real_scorecard_id': self.real_scorecard_id
                    }
                
                # Update for next iteration
                last_performance = performance
                
                # Brief rest between iterations
                if iteration < self.max_iterations:
                    rest_time = max(10, 60 - iteration_duration)
                    print(f"\n😴 Rest {rest_time:.0f}s before iteration {iteration + 1}...")
                    await asyncio.sleep(rest_time)
                
            except Exception as e:
                print(f"❌ Iteration {iteration} failed: {e}")
                await asyncio.sleep(30)
                continue
        
        # Max iterations reached
        total_duration = time.time() - overall_start_time
        print(f"\n⚠️  REACHED MAX ITERATIONS")
        print(f"Best: {self.best_performance['win_rate']:.1%} win rate")
        
        return {
            'success': False,
            'mode': 'swarm' if self.swarm_enabled else 'sequential',
            'salience_mode': salience_mode.value,
            'final_performance': last_performance,
            'training_iterations': self.max_iterations,
            'best_performance': self.best_performance
        }
    
    def get_adaptive_episodes(self, iteration: int, last_performance: dict) -> int:
        """Get adaptive episodes based on performance."""
        base_episodes = self.max_episodes_per_game
        win_rate = last_performance.get('overall_win_rate', 0.0)
        
        if win_rate < 0.1:
            return min(base_episodes + 25, 100)  # Need more exploration
        elif win_rate < 0.3:
            return min(base_episodes + 15, 80)   # Moderate training
        elif win_rate < 0.6:
            return min(base_episodes + 5, 60)    # Fine-tuning
        else:
            return base_episodes                 # Standard training

def show_usage():
    """Show usage and environment variable options."""
    print("🔧 SIMPLE CONFIGURATION OPTIONS")
    print("="*50)
    print("Set environment variables to configure training:")
    print()
    print("SWARM_MODE=false          # true = SWARM, false = SEQUENTIAL")
    print("SALIENCE_MODE=decay_compression  # lossless, decay_compression, adaptive_compression")
    print("TARGET_WIN_RATE=0.85      # 0.0-1.0 (85% target)")
    print("TARGET_AVG_SCORE=75.0     # Target average score")
    print("MAX_EPISODES_PER_GAME=50  # Episodes per game")
    print("GAME_COUNT=6              # Number of games")
    print("MAX_ITERATIONS=20         # Maximum training iterations")
    print()
    print("Examples:")
    print('  # Sequential training with decay compression (your request):')
    print('  python run_simple_training.py')
    print()
    print('  # SWARM training with lossless memory:')
    print('  set SWARM_MODE=true && set SALIENCE_MODE=lossless && python run_simple_training.py')
    print()

async def main():
    """Run simple configurable training."""
    
    print("🚀 SIMPLE ARC-AGI-3 TRAINING (No Extra Dependencies)")
    print("="*60)
    
    # Check if user wants help
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        show_usage()
        return
    
    trainer = SimpleConfigurableTrainer()
    
    # Show current configuration
    trainer.display_config()
    
    # Initialize with real API
    if not await trainer.initialize_and_verify():
        print("❌ Cannot proceed without API connection")
        return
    
    try:
        print("🎯 STARTING SIMPLE CONFIGURABLE TRAINING")
        print("Using REAL ARC-AGI-3 API data\n")
        
        results = await trainer.run_simple_training()
        
        if results['success']:
            print(f"\n✅ SUCCESS! Training completed in {results['mode'].upper()} mode")
            print(f"Memory mode: {results['salience_mode']}")
            if results.get('real_scorecard_id'):
                print(f"Scorecard: https://three.arcprize.org/scorecard/{results['real_scorecard_id']}")
        else:
            print(f"\n⚠️  Training incomplete")
            print(f"Best performance: {results['best_performance']['win_rate']:.1%}")
            
    except KeyboardInterrupt:
        print(f"\n🛑 Training stopped after {trainer.training_iterations} iterations")
        if trainer.real_scorecard_id:
            print(f"Scorecard: https://three.arcprize.org/scorecard/{trainer.real_scorecard_id}")

if __name__ == "__main__":
    # Show help message
    print("For options: python run_simple_training.py --help")
    print("Current settings: SEQUENTIAL mode, DECAY_COMPRESSION memory (as requested)\n")
    
    asyncio.run(main())
