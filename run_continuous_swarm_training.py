#!/usr/bin/env python3
"""
Continuous SWARM Training System

This system runs SWARM mode continuously until target win rates are achieved.
It combines:
- SWARM mode for concurrent training
- Sleep cycles with memory consolidation  
- Continuous training iterations
- Progress tracking until winning completion rates
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from arc_integration.continuous_learning_loop import ContinuousLearningLoop
    from core.salience_system import SalienceMode
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class ContinuousSWARMTrainer:
    """Continuous SWARM training system that runs until target performance."""
    
    def __init__(self, target_win_rate: float = 0.9, target_avg_score: float = 85.0):
        self.target_win_rate = target_win_rate
        self.target_avg_score = target_avg_score
        self.continuous_loop = None
        self.training_iterations = 0
        self.best_performance = {'win_rate': 0.0, 'avg_score': 0.0}
        
    async def initialize(self):
        """Initialize the continuous learning system."""
        try:
            self.continuous_loop = ContinuousLearningLoop(
                arc_agents_path="C:/Users/Admin/Documents/GitHub/ARC-AGI-3-Agents",
                tabula_rasa_path="C:/Users/Admin/Documents/GitHub/tabula-rasa"
            )
            print("✅ Continuous SWARM Trainer initialized")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            return False
    
    def has_reached_target(self, performance: dict) -> bool:
        """Check if target performance has been reached."""
        win_rate = performance.get('overall_win_rate', 0.0)
        avg_score = performance.get('overall_average_score', 0.0)
        
        target_reached = (win_rate >= self.target_win_rate and 
                         avg_score >= self.target_avg_score)
        
        # Update best performance
        if win_rate > self.best_performance['win_rate']:
            self.best_performance['win_rate'] = win_rate
        if avg_score > self.best_performance['avg_score']:
            self.best_performance['avg_score'] = avg_score
            
        return target_reached
    
    def get_adaptive_training_params(self, iteration: int, last_performance: dict) -> dict:
        """Get adaptive training parameters based on current performance."""
        base_episodes = 30
        
        # Increase episodes if performance is stuck
        win_rate = last_performance.get('overall_win_rate', 0.0)
        
        if win_rate < 0.1:
            episodes_per_game = base_episodes + 20  # More exploration needed
        elif win_rate < 0.3:
            episodes_per_game = base_episodes + 10  # Moderate training
        else:
            episodes_per_game = base_episodes  # Standard training
            
        # Adaptive salience mode
        if iteration <= 2:
            salience_mode = SalienceMode.LOSSLESS  # Keep everything early on
        else:
            salience_mode = SalienceMode.DECAY_COMPRESSION  # Compress later
            
        return {
            'max_episodes_per_game': episodes_per_game,
            'salience_mode': salience_mode,
            'target_win_rate': min(0.3 + (iteration * 0.1), self.target_win_rate),
            'enable_salience_comparison': iteration % 3 == 0  # Every 3rd iteration
        }
    
    async def run_continuous_training(self, games: list) -> dict:
        """Run continuous training until target performance is achieved."""
        
        print("🔥 CONTINUOUS SWARM TRAINING SYSTEM")
        print("="*60)
        print(f"Target Win Rate: {self.target_win_rate:.1%}")
        print(f"Target Average Score: {self.target_avg_score}")
        print(f"Training Games: {len(games)}")
        print(f"SWARM Mode: {'ENABLED' if len(games) > 2 else 'DISABLED'}")
        
        overall_start_time = time.time()
        last_performance = {'overall_win_rate': 0.0, 'overall_average_score': 0.0}
        
        # Training loop - continues until target reached
        while True:
            self.training_iterations += 1
            iteration_start_time = time.time()
            
            print(f"\n🚀 TRAINING ITERATION {self.training_iterations}")
            print("="*40)
            
            # Get adaptive parameters
            params = self.get_adaptive_training_params(self.training_iterations, last_performance)
            
            print(f"Episodes per Game: {params['max_episodes_per_game']}")
            print(f"Salience Mode: {params['salience_mode'].value}")
            print(f"Iteration Target Win Rate: {params['target_win_rate']:.1%}")
            
            # Start training session
            try:
                session_id = self.continuous_loop.start_training_session(
                    games=games,
                    max_episodes_per_game=params['max_episodes_per_game'],
                    target_win_rate=params['target_win_rate'],
                    target_avg_score=50.0,
                    salience_mode=params['salience_mode'],
                    enable_salience_comparison=params['enable_salience_comparison']
                )
                
                # Execute training with SWARM mode + continuous learning
                session_results = await self.continuous_loop.run_continuous_learning(session_id)
                
                # Get performance results
                performance = session_results.get('overall_performance', {})
                current_win_rate = performance.get('overall_win_rate', 0.0)
                current_avg_score = performance.get('overall_average_score', 0.0)
                
                iteration_duration = time.time() - iteration_start_time
                
                # Display iteration results
                print(f"\n📊 ITERATION {self.training_iterations} RESULTS:")
                print(f"Duration: {iteration_duration/60:.1f} minutes")
                print(f"Win Rate: {current_win_rate:.1%} (Best: {self.best_performance['win_rate']:.1%})")
                print(f"Avg Score: {current_avg_score:.1f} (Best: {self.best_performance['avg_score']:.1f})")
                
                # Show system status
                system_status = self.continuous_loop.get_sleep_and_memory_status()
                sleep_info = system_status['sleep_status']
                memory_info = system_status['memory_consolidation_status']
                reset_info = system_status['game_reset_status']
                
                print(f"Sleep Cycles: {sleep_info['sleep_cycles_this_session']}")
                print(f"Memory Consolidations: {memory_info['consolidation_operations_completed']}")
                print(f"High-Sal Strengthened: {memory_info['high_salience_memories_strengthened']}")
                print(f"Reset Decisions: {reset_info['total_reset_decisions']}")
                
                # Check if target reached
                if self.has_reached_target(performance):
                    total_duration = time.time() - overall_start_time
                    
                    print(f"\n🎉 TARGET PERFORMANCE REACHED!")
                    print("="*60)
                    print(f"Final Win Rate: {current_win_rate:.1%} (Target: {self.target_win_rate:.1%})")
                    print(f"Final Avg Score: {current_avg_score:.1f} (Target: {self.target_avg_score})")
                    print(f"Training Iterations: {self.training_iterations}")
                    print(f"Total Training Time: {total_duration/3600:.1f} hours")
                    print(f"🏆 SUBMIT TO LEADERBOARD - TARGET ACHIEVED!")
                    
                    return {
                        'success': True,
                        'final_performance': performance,
                        'training_iterations': self.training_iterations,
                        'total_duration_hours': total_duration / 3600,
                        'system_status': system_status
                    }
                
                # Update for next iteration
                last_performance = performance
                
                # Progress check - stop if no improvement after many iterations
                if (self.training_iterations > 10 and 
                    current_win_rate < 0.05 and 
                    self.best_performance['win_rate'] < 0.1):
                    
                    print(f"\n⚠️  STOPPING: No significant progress after {self.training_iterations} iterations")
                    print(f"Best achieved: {self.best_performance['win_rate']:.1%} win rate")
                    break
                
                # Adaptive sleep between iterations
                sleep_time = max(5, 30 - iteration_duration)  # Short break between iterations
                print(f"\n😴 Resting {sleep_time:.0f}s before next iteration...")
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                print(f"❌ Iteration {self.training_iterations} failed: {e}")
                # Continue with next iteration after short delay
                await asyncio.sleep(10)
                continue
        
        # If we exit the loop without reaching target
        total_duration = time.time() - overall_start_time
        return {
            'success': False,
            'final_performance': last_performance,
            'training_iterations': self.training_iterations,
            'total_duration_hours': total_duration / 3600,
            'best_performance': self.best_performance
        }

async def main():
    """Main entry point for continuous SWARM training with real ARC-AGI-3 games."""
    
    # Create and initialize continuous trainer
    trainer = ContinuousSWARMTrainer(
        target_win_rate=0.85,  # 85% win rate target
        target_avg_score=75.0   # 75 average score target
    )
    
    if not await trainer.initialize():
        return
    
    try:
        print("🎯 GETTING REAL GAMES FROM ARC-AGI-3 API")
        
        # Get real games from ARC-AGI-3 API instead of hardcoded list
        training_games = await trainer.continuous_loop.select_training_games(
            count=8,
            difficulty_preference="mixed",
            include_pattern=None  # No filter - get variety of games
        )
        
        if not training_games:
            print("❌ Failed to get games from API")
            return
        
        print(f"✅ Retrieved {len(training_games)} real games from API")
        
        print("🎯 STARTING CONTINUOUS SWARM TRAINING")
        print("This will run until 85% win rate is achieved!")
        print("Press Ctrl+C to stop training\n")
        
        results = await trainer.run_continuous_training(training_games)
        
        if results['success']:
            print(f"\n✅ TRAINING SUCCEEDED!")
            print(f"Achieved target in {results['training_iterations']} iterations")
            print(f"Total time: {results['total_duration_hours']:.1f} hours")
        else:
            print(f"\n⚠️  TRAINING INCOMPLETE")
            print(f"Best performance: {results['best_performance']}")
            
    except KeyboardInterrupt:
        print(f"\n🛑 Training stopped by user after {trainer.training_iterations} iterations")
        print(f"Best achieved: {trainer.best_performance}")

if __name__ == "__main__":
    asyncio.run(main())
