#!/usr/bin/env python3
"""
Configurable ARC-AGI-3 Training System

Run real API-based training with configurable options:
- SWARM mode vs Sequential mode  
- Lossless vs Decay vs Adaptive compression
- All parameters configurable via YAML
"""

import asyncio
import sys
import time
import yaml
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

class ConfigurableTrainer:
    """Configurable trainer that reads settings from YAML."""
    
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.continuous_loop = None
        self.training_iterations = 0
        self.best_performance = {'win_rate': 0.0, 'avg_score': 0.0}
        self.real_scorecard_id = None
        
    def load_config(self) -> dict:
        """Load training configuration from YAML."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Loaded config from {self.config_path}")
            return config
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            print(f"Using default configuration")
            return self.get_default_config()
    
    def get_default_config(self) -> dict:
        """Get default configuration if YAML loading fails."""
        return {
            'training_mode': {
                'swarm_enabled': False,
                'max_concurrent_games': 3
            },
            'memory_mode': {
                'salience_mode': 'decay_compression',
                'enable_salience_comparison': False
            },
            'training_params': {
                'target_win_rate': 0.85,
                'target_avg_score': 75.0,
                'max_episodes_per_game': 50,
                'max_training_iterations': 20,
                'game_count': 6
            }
        }
    
    def get_salience_mode(self) -> SalienceMode:
        """Convert config string to SalienceMode enum."""
        mode_str = self.config['memory_mode']['salience_mode'].lower()
        
        if mode_str == 'lossless':
            return SalienceMode.LOSSLESS
        elif mode_str == 'decay_compression':
            return SalienceMode.DECAY_COMPRESSION  
        elif mode_str == 'adaptive_compression':
            return SalienceMode.ADAPTIVE_COMPRESSION
        else:
            print(f"⚠️  Unknown salience mode: {mode_str}, using decay_compression")
            return SalienceMode.DECAY_COMPRESSION
    
    def display_config(self):
        """Display current training configuration."""
        training_mode = self.config['training_mode']
        memory_mode = self.config['memory_mode']
        training_params = self.config['training_params']
        
        print("🎯 TRAINING CONFIGURATION")
        print("="*50)
        print(f"Training Mode: {'SWARM' if training_mode['swarm_enabled'] else 'SEQUENTIAL'}")
        if training_mode['swarm_enabled']:
            print(f"Max Concurrent Games: {training_mode['max_concurrent_games']}")
        print(f"Salience Mode: {memory_mode['salience_mode'].upper()}")
        print(f"Target Win Rate: {training_params['target_win_rate']:.1%}")
        print(f"Target Avg Score: {training_params['target_avg_score']}")
        print(f"Max Episodes/Game: {training_params['max_episodes_per_game']}")
        print(f"Game Count: {training_params['game_count']}")
        print()
        
    async def initialize_and_verify(self):
        """Initialize and verify real API connection."""
        try:
            self.continuous_loop = ContinuousLearningLoop(
                arc_agents_path="C:/Users/Admin/Documents/GitHub/ARC-AGI-3-Agents",
                tabula_rasa_path="C:/Users/Admin/Documents/GitHub/tabula-rasa"
            )
            
            # Verify API connection if enabled
            if self.config.get('api_config', {}).get('verify_connection', True):
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
    
    async def run_configurable_training(self) -> dict:
        """Run training based on configuration settings."""
        
        self.display_config()
        
        training_params = self.config['training_params']
        training_mode = self.config['training_mode']
        memory_mode = self.config['memory_mode']
        
        # Get real training games from API
        print(f"🎯 GETTING {training_params['game_count']} REAL GAMES FROM API")
        training_games = await self.continuous_loop.select_training_games(
            count=training_params['game_count'],
            difficulty_preference=training_params.get('difficulty_preference', 'mixed')
        )
        
        if not training_games:
            return {'success': False, 'error': 'Failed to get real games'}
        
        # Create real scorecard if enabled
        if self.config.get('api_config', {}).get('create_real_scorecard', True):
            print(f"📊 CREATING REAL SCORECARD...")
            scorecard_id = await self.continuous_loop.create_real_scorecard(training_games)
            if scorecard_id:
                self.real_scorecard_id = scorecard_id
                print(f"✅ Real scorecard: https://three.arcprize.org/scorecard/{scorecard_id}")
        
        overall_start_time = time.time()
        last_performance = {'overall_win_rate': 0.0, 'overall_average_score': 0.0}
        
        # Determine salience mode
        salience_mode = self.get_salience_mode()
        
        print(f"\n🚀 STARTING {'SWARM' if training_mode['swarm_enabled'] else 'SEQUENTIAL'} TRAINING")
        print(f"Using {salience_mode.value.upper()} memory mode")
        print("="*60)
        
        # Continuous training loop
        for iteration in range(1, training_params['max_training_iterations'] + 1):
            self.training_iterations = iteration
            iteration_start_time = time.time()
            
            print(f"\n🚀 TRAINING ITERATION {iteration}")
            print("="*40)
            
            # Adaptive parameters
            episodes_per_game = self.get_adaptive_episodes(iteration, last_performance)
            iteration_target = min(0.3 + (iteration * 0.05), training_params['target_win_rate'])
            
            # Switch salience mode if configured
            if hasattr(self.config.get('adaptive_config', {}), 'switch_salience_mode_after_iterations'):
                switch_after = self.config['adaptive_config'].get('switch_salience_mode_after_iterations', 3)
                if iteration > switch_after and salience_mode == SalienceMode.LOSSLESS:
                    salience_mode = SalienceMode.DECAY_COMPRESSION
                    print(f"🔄 Switched to {salience_mode.value} mode")
            
            print(f"Mode: {'SWARM' if training_mode['swarm_enabled'] else 'SEQUENTIAL'}")
            print(f"Episodes per Game: {episodes_per_game}")
            print(f"Salience Mode: {salience_mode.value}")
            print(f"Target: {iteration_target:.1%}")
            
            try:
                # Start training session with current config
                session_id = self.continuous_loop.start_training_session(
                    games=training_games,
                    max_episodes_per_game=episodes_per_game,
                    target_win_rate=iteration_target,
                    target_avg_score=training_params['target_avg_score'],
                    salience_mode=salience_mode,
                    enable_salience_comparison=memory_mode['enable_salience_comparison']
                )
                
                # Run training based on mode
                if training_mode['swarm_enabled']:
                    print(f"🔥 SWARM MODE: {training_mode['max_concurrent_games']} concurrent games")
                    session_results = await self.continuous_loop.run_continuous_learning(session_id)
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
                if (current_win_rate >= training_params['target_win_rate'] and 
                    current_avg_score >= training_params['target_avg_score']):
                    
                    total_duration = time.time() - overall_start_time
                    
                    print(f"\n🎉 TARGET PERFORMANCE ACHIEVED!")
                    print("="*50)
                    print(f"Mode: {'SWARM' if training_mode['swarm_enabled'] else 'SEQUENTIAL'}")
                    print(f"Salience: {salience_mode.value}")
                    print(f"Final Win Rate: {current_win_rate:.1%}")
                    print(f"Final Avg Score: {current_avg_score:.1f}")
                    print(f"Training Iterations: {iteration}")
                    print(f"Total Time: {total_duration/3600:.1f} hours")
                    
                    if self.real_scorecard_id:
                        print(f"🏆 Scorecard: https://three.arcprize.org/scorecard/{self.real_scorecard_id}")
                    
                    return {
                        'success': True,
                        'mode': 'swarm' if training_mode['swarm_enabled'] else 'sequential',
                        'salience_mode': salience_mode.value,
                        'final_performance': performance,
                        'training_iterations': iteration,
                        'total_duration_hours': total_duration / 3600,
                        'real_scorecard_id': self.real_scorecard_id
                    }
                
                # Update for next iteration
                last_performance = performance
                
                # Brief rest between iterations
                if iteration < training_params['max_training_iterations']:
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
            'mode': 'swarm' if training_mode['swarm_enabled'] else 'sequential',
            'salience_mode': salience_mode.value,
            'final_performance': last_performance,
            'training_iterations': training_params['max_training_iterations'],
            'best_performance': self.best_performance
        }
    
    def get_adaptive_episodes(self, iteration: int, last_performance: dict) -> int:
        """Get adaptive episodes based on config and performance."""
        base_episodes = self.config['training_params']['max_episodes_per_game']
        
        if not self.config.get('adaptive_config', {}).get('adjust_episodes_based_on_performance', True):
            return base_episodes
        
        win_rate = last_performance.get('overall_win_rate', 0.0)
        
        if win_rate < 0.1:
            return min(base_episodes + 25, 100)  # Need more exploration
        elif win_rate < 0.3:
            return min(base_episodes + 15, 80)   # Moderate training
        elif win_rate < 0.6:
            return min(base_episodes + 5, 60)    # Fine-tuning
        else:
            return base_episodes                 # Standard training

def show_config_options():
    """Show available configuration options."""
    print("🔧 CONFIGURATION OPTIONS")
    print("="*50)
    print("Edit configs/training_config.yaml to change:")
    print()
    print("Training Mode:")
    print("  swarm_enabled: true/false")
    print("  max_concurrent_games: 1-10")
    print()
    print("Memory Mode:")
    print("  salience_mode: 'lossless', 'decay_compression', 'adaptive_compression'")
    print("  enable_salience_comparison: true/false")
    print()
    print("Training Parameters:")
    print("  target_win_rate: 0.0-1.0 (e.g., 0.85 = 85%)")
    print("  target_avg_score: 0-100")
    print("  max_episodes_per_game: 10-200")
    print("  game_count: 1-20")
    print()

async def main():
    """Run configurable training."""
    
    print("🚀 CONFIGURABLE ARC-AGI-3 TRAINING")
    print("="*50)
    
    # Check if user wants to see options
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        show_config_options()
        return
    
    trainer = ConfigurableTrainer()
    
    # Show current configuration
    trainer.display_config()
    
    # Initialize with real API
    if not await trainer.initialize_and_verify():
        print("❌ Cannot proceed without API connection")
        return
    
    try:
        print("🎯 STARTING CONFIGURABLE TRAINING")
        print("Using REAL ARC-AGI-3 API data\n")
        
        results = await trainer.run_configurable_training()
        
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
    print("To see configuration options: python run_configurable_training.py --help")
    print("To change settings: Edit configs/training_config.yaml\n")
    
    asyncio.run(main())
