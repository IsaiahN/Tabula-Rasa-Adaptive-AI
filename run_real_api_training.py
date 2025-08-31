#!/usr/bin/env python3
"""
REAL API-Based Continuous SWARM Training

This system:
1. Gets real games from ARC-AGI-3 API (/api/games)
2. Creates real scorecards using the API
3. Runs continuous SWARM training until winning completion rates
4. Uses real API responses, not simulated results
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
    print(f"Make sure you're in the tabula-rasa directory and src/ exists")
    sys.exit(1)

class RealAPIContinuousTrainer:
    """Continuous trainer that uses real ARC-AGI-3 API for everything."""
    
    def __init__(self, target_win_rate: float = 0.85, target_avg_score: float = 75.0):
        self.target_win_rate = target_win_rate
        self.target_avg_score = target_avg_score
        self.continuous_loop = None
        self.training_iterations = 0
        self.best_performance = {'win_rate': 0.0, 'avg_score': 0.0}
        self.real_scorecard_id = None
        
    async def initialize_and_verify(self):
        """Initialize and verify real API connection."""
        try:
            self.continuous_loop = ContinuousLearningLoop(
                arc_agents_path="C:/Users/Admin/Documents/GitHub/ARC-AGI-3-Agents",
                tabula_rasa_path="C:/Users/Admin/Documents/GitHub/tabula-rasa"
            )
            
            # CRITICAL: Verify we can actually connect to the real API
            verification = await self.continuous_loop.verify_api_connection()
            
            if not verification['api_accessible']:
                print(f"❌ Cannot connect to ARC-AGI-3 API")
                print(f"   Check API key and internet connection")
                return False
            
            if not verification['arc_agents_available']:
                print(f"❌ ARC-AGI-3-Agents not found")
                print(f"   Please clone: https://github.com/neoneye/ARC-AGI-3-Agents")
                return False
            
            print("✅ Real API connection verified")
            print(f"   Available games: {verification['total_games_available']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            return False
    
    async def get_real_training_games(self, count: int = 6) -> list:
        """Get real games from ARC-AGI-3 API."""
        print(f"\n🎯 GETTING {count} REAL GAMES FROM ARC-AGI-3 API")
        
        # Get real games from the API
        games = await self.continuous_loop.select_training_games(
            count=count,
            difficulty_preference="mixed"
        )
        
        if not games:
            print("❌ Failed to get real games from API")
            return []
        
        print(f"✅ Retrieved {len(games)} real games")
        
        # Create a real scorecard for these games
        print(f"📊 CREATING REAL SCORECARD...")
        scorecard_id = await self.continuous_loop.create_real_scorecard(games)
        
        if scorecard_id:
            self.real_scorecard_id = scorecard_id
            scorecard_url = f"https://three.arcprize.org/scorecard/{scorecard_id}"
            print(f"✅ Real scorecard created: {scorecard_url}")
        else:
            print(f"⚠️  Scorecard creation failed - continuing without scorecard")
        
        return games
    
    async def run_until_winning(self) -> dict:
        """Run continuous training until winning completion rates achieved."""
        
        print("🚀 REAL API CONTINUOUS SWARM TRAINING")
        print("="*60)
        print(f"Target Win Rate: {self.target_win_rate:.1%}")
        print(f"Target Average Score: {self.target_avg_score}")
        print("Using REAL ARC-AGI-3 API data (not simulated)")
        
        # Get real training games from API
        training_games = await self.get_real_training_games(count=6)
        if not training_games:
            return {'success': False, 'error': 'Failed to get real games'}
        
        overall_start_time = time.time()
        last_performance = {'overall_win_rate': 0.0, 'overall_average_score': 0.0}
        
        # Continuous training loop
        max_iterations = 20  # Reasonable limit
        
        for iteration in range(1, max_iterations + 1):
            self.training_iterations = iteration
            iteration_start_time = time.time()
            
            print(f"\n🚀 REAL TRAINING ITERATION {iteration}")
            print("="*50)
            
            # Adaptive parameters based on performance
            episodes_per_game = self.get_adaptive_episodes(iteration, last_performance)
            salience_mode = SalienceMode.LOSSLESS if iteration <= 3 else SalienceMode.DECAY_COMPRESSION
            iteration_target = min(0.3 + (iteration * 0.05), self.target_win_rate)
            
            print(f"Episodes per Game: {episodes_per_game}")
            print(f"Salience Mode: {salience_mode.value}")
            print(f"Iteration Target: {iteration_target:.1%}")
            print(f"SWARM Mode: ENABLED ({len(training_games)} games)")
            
            try:
                # Start real training session
                session_id = self.continuous_loop.start_training_session(
                    games=training_games,
                    max_episodes_per_game=episodes_per_game,
                    target_win_rate=iteration_target,
                    target_avg_score=50.0,
                    salience_mode=salience_mode,
                    enable_salience_comparison=False
                )
                
                # Execute REAL training with API calls
                session_results = await self.continuous_loop.run_continuous_learning(session_id)
                
                # Get REAL performance results
                performance = session_results.get('overall_performance', {})
                current_win_rate = performance.get('overall_win_rate', 0.0)
                current_avg_score = performance.get('overall_average_score', 0.0)
                
                iteration_duration = time.time() - iteration_start_time
                
                # Update best performance
                if current_win_rate > self.best_performance['win_rate']:
                    self.best_performance['win_rate'] = current_win_rate
                if current_avg_score > self.best_performance['avg_score']:
                    self.best_performance['avg_score'] = current_avg_score
                
                # Display real results
                print(f"\n📊 ITERATION {iteration} REAL RESULTS:")
                print(f"Duration: {iteration_duration/60:.1f} minutes")
                print(f"Win Rate: {current_win_rate:.1%} (Best: {self.best_performance['win_rate']:.1%})")
                print(f"Avg Score: {current_avg_score:.1f} (Best: {self.best_performance['avg_score']:.1f})")
                
                # Show real system status
                system_status = self.continuous_loop.get_sleep_and_memory_status()
                sleep_info = system_status['sleep_status']
                memory_info = system_status['memory_consolidation_status']
                reset_info = system_status['game_reset_status']
                
                print(f"Sleep Cycles: {sleep_info['sleep_cycles_this_session']}")
                print(f"Memory Consolidations: {memory_info['consolidation_operations_completed']}")
                print(f"Reset Decisions: {reset_info['total_reset_decisions']}")
                
                # Check if target reached with REAL performance
                if (current_win_rate >= self.target_win_rate and 
                    current_avg_score >= self.target_avg_score):
                    
                    total_duration = time.time() - overall_start_time
                    
                    print(f"\n🎉 TARGET PERFORMANCE ACHIEVED WITH REAL API!")
                    print("="*60)
                    print(f"Final Win Rate: {current_win_rate:.1%}")
                    print(f"Final Avg Score: {current_avg_score:.1f}")
                    print(f"Training Iterations: {iteration}")
                    print(f"Total Training Time: {total_duration/3600:.1f} hours")
                    
                    if self.real_scorecard_id:
                        print(f"🏆 Real Scorecard: https://three.arcprize.org/scorecard/{self.real_scorecard_id}")
                    
                    return {
                        'success': True,
                        'final_performance': performance,
                        'training_iterations': iteration,
                        'total_duration_hours': total_duration / 3600,
                        'real_scorecard_id': self.real_scorecard_id,
                        'system_status': system_status
                    }
                
                # Update for next iteration
                last_performance = performance
                
                # Brief rest between iterations
                if iteration < max_iterations:
                    rest_time = max(10, 60 - iteration_duration)
                    print(f"\n😴 Resting {rest_time:.0f}s before iteration {iteration + 1}...")
                    await asyncio.sleep(rest_time)
                
            except Exception as e:
                print(f"❌ Iteration {iteration} failed: {e}")
                await asyncio.sleep(30)  # Longer delay on errors
                continue
        
        # If we reach max iterations without success
        total_duration = time.time() - overall_start_time
        print(f"\n⚠️  REACHED MAX ITERATIONS ({max_iterations})")
        print(f"Best Performance: {self.best_performance['win_rate']:.1%} win rate")
        
        return {
            'success': False,
            'final_performance': last_performance,
            'training_iterations': max_iterations,
            'total_duration_hours': total_duration / 3600,
            'best_performance': self.best_performance
        }
    
    def get_adaptive_episodes(self, iteration: int, last_performance: dict) -> int:
        """Get adaptive number of episodes based on performance."""
        base_episodes = 25
        win_rate = last_performance.get('overall_win_rate', 0.0)
        
        if win_rate < 0.1:
            return base_episodes + 25  # Need more exploration
        elif win_rate < 0.3:
            return base_episodes + 15  # Moderate training
        elif win_rate < 0.6:
            return base_episodes + 5   # Fine-tuning
        else:
            return base_episodes       # Standard training

async def main():
    """Run real API-based continuous training."""
    
    trainer = RealAPIContinuousTrainer(
        target_win_rate=0.85,  # 85% target
        target_avg_score=75.0   # 75 average score
    )
    
    # Initialize and verify real API connection
    if not await trainer.initialize_and_verify():
        print("❌ Cannot proceed without real API connection")
        return
    
    try:
        print("\n🎯 STARTING REAL API CONTINUOUS TRAINING")
        print("This uses REAL ARC-AGI-3 games and API calls!")
        print("Will continue until 85% win rate achieved\n")
        
        results = await trainer.run_until_winning()
        
        if results['success']:
            print(f"\n✅ SUCCESS! Real API training completed")
            print(f"Scorecard: https://three.arcprize.org/scorecard/{results.get('real_scorecard_id', 'N/A')}")
        else:
            print(f"\n⚠️  Training incomplete - best: {results['best_performance']}")
            
    except KeyboardInterrupt:
        print(f"\n🛑 Training stopped after {trainer.training_iterations} iterations")
        if trainer.real_scorecard_id:
            print(f"Scorecard: https://three.arcprize.org/scorecard/{trainer.real_scorecard_id}")

if __name__ == "__main__":
    asyncio.run(main())
