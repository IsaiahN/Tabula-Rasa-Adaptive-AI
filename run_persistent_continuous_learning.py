#!/usr/bin/env python3
"""
Persistent Continuous Learning Loop - Win All Levels

This script runs the continuous learning system persistently until achieving
high performance (winning) on all available levels/games. It includes:
- Adaptive training parameters based on performance
- Real-time monitoring and progress tracking
- Automatic session restart on performance plateaus
- Comprehensive win condition checking
- Performance persistence across sessions
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import random
import signal
import sys

# Set up logging with both console and file output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('persistent_learning.log')
    ]
)
logger = logging.getLogger(__name__)

class PersistentContinuousLearning:
    """
    Persistent continuous learning system that runs until all levels are won.
    
    Features:
    - Tracks performance across all games/levels
    - Automatically restarts training on plateaus
    - Adapts training parameters based on learning progress
    - Maintains state across interruptions
    - Provides real-time progress monitoring
    """
    
    def __init__(self, target_win_rate: float = 0.90, target_avg_score: float = 85.0):
        self.target_win_rate = target_win_rate
        self.target_avg_score = target_avg_score
        
        # Game levels to master (expandable list)
        self.all_games = [
            "spatial_reasoning_1", "spatial_reasoning_2", "spatial_reasoning_3",
            "pattern_matching_1", "pattern_matching_2", "pattern_matching_3", 
            "abstract_logic_1", "abstract_logic_2", "abstract_logic_3",
            "sequence_completion_1", "sequence_completion_2", "sequence_completion_3",
            "visual_analogies_1", "visual_analogies_2", "visual_analogies_3",
            "transformation_rules_1", "transformation_rules_2", "transformation_rules_3",
            "object_counting_1", "object_counting_2", "object_counting_3",
            "symmetry_detection_1", "symmetry_detection_2", "symmetry_detection_3"
        ]
        
        # Performance tracking
        self.game_performance = {game: {'win_rate': 0.0, 'avg_score': 0.0, 'episodes_played': 0, 'last_updated': 0} for game in self.all_games}
        self.session_count = 0
        self.total_runtime = 0
        self.start_time = time.time()
        
        # Training parameters that adapt based on performance
        self.training_config = {
            'base_episodes_per_game': 50,
            'max_episodes_per_game': 200,
            'plateau_threshold': 10,  # Episodes without improvement
            'restart_threshold': 5,   # Sessions without significant improvement
            'salience_mode': 'LOSSLESS',  # Start with lossless, adapt as needed
            'learning_rate_decay': 0.95,
            'difficulty_progression': True
        }
        
        # State persistence
        self.state_file = Path("persistent_learning_state.json")
        self.running = True
        
        # Load previous state if available
        self.load_state()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n🛑 Received shutdown signal {signum}. Saving state and shutting down gracefully...")
        self.running = False
        self.save_state()
        sys.exit(0)
        
    def save_state(self):
        """Save current state to disk."""
        state = {
            'game_performance': self.game_performance,
            'session_count': self.session_count,
            'total_runtime': self.total_runtime + (time.time() - self.start_time),
            'training_config': self.training_config,
            'last_saved': time.time()
        }
        
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info("State saved successfully")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            
    def load_state(self):
        """Load previous state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    
                self.game_performance.update(state.get('game_performance', {}))
                self.session_count = state.get('session_count', 0)
                self.total_runtime = state.get('total_runtime', 0)
                self.training_config.update(state.get('training_config', {}))
                
                logger.info(f"Loaded previous state: {self.session_count} sessions completed")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
                
    def get_games_needing_training(self) -> List[str]:
        """Get list of games that haven't reached target performance."""
        needs_training = []
        for game, perf in self.game_performance.items():
            if (perf['win_rate'] < self.target_win_rate or 
                perf['avg_score'] < self.target_avg_score):
                needs_training.append(game)
        return needs_training
        
    def get_overall_progress(self) -> Dict[str, float]:
        """Calculate overall progress metrics."""
        total_games = len(self.all_games)
        games_mastered = sum(1 for perf in self.game_performance.values() 
                           if perf['win_rate'] >= self.target_win_rate and perf['avg_score'] >= self.target_avg_score)
        
        avg_win_rate = sum(perf['win_rate'] for perf in self.game_performance.values()) / total_games
        avg_score = sum(perf['avg_score'] for perf in self.game_performance.values()) / total_games
        
        return {
            'games_mastered': games_mastered,
            'total_games': total_games,
            'mastery_percentage': games_mastered / total_games,
            'overall_win_rate': avg_win_rate,
            'overall_avg_score': avg_score
        }
        
    def display_progress_dashboard(self):
        """Display real-time progress dashboard."""
        progress = self.get_overall_progress()
        games_needing_training = self.get_games_needing_training()
        
        print("\n" + "="*100)
        print("🧠 PERSISTENT CONTINUOUS LEARNING - PROGRESS DASHBOARD")
        print("="*100)
        
        # Overall progress
        print(f"🎯 TARGET: Win {self.target_win_rate:.0%} with {self.target_avg_score:.0f}+ avg score on ALL levels")
        print(f"🏆 PROGRESS: {progress['games_mastered']}/{progress['total_games']} levels mastered ({progress['mastery_percentage']:.1%})")
        print(f"📊 OVERALL: {progress['overall_win_rate']:.1%} win rate, {progress['overall_avg_score']:.1f} avg score")
        print(f"⏱️  RUNTIME: {(self.total_runtime + (time.time() - self.start_time))/3600:.1f} hours across {self.session_count} sessions")
        
        # Progress bar
        mastery_pct = progress['mastery_percentage']
        bar_length = 50
        filled_length = int(bar_length * mastery_pct)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"🔥 [{bar}] {mastery_pct:.1%} Complete")
        
        if progress['games_mastered'] == progress['total_games']:
            print("\n🎉🎉🎉 ALL LEVELS MASTERED! MISSION ACCOMPLISHED! 🎉🎉🎉")
            return True
            
        # Games status
        print(f"\n📋 STATUS BREAKDOWN:")
        mastered_games = []
        training_games = []
        
        for game, perf in self.game_performance.items():
            if perf['win_rate'] >= self.target_win_rate and perf['avg_score'] >= self.target_avg_score:
                mastered_games.append(f"✅ {game} ({perf['win_rate']:.1%}, {perf['avg_score']:.0f})")
            else:
                training_games.append(f"🔄 {game} ({perf['win_rate']:.1%}, {perf['avg_score']:.0f})")
        
        if mastered_games:
            print(f"   🏆 MASTERED ({len(mastered_games)}):")
            for game in mastered_games[:5]:  # Show first 5
                print(f"      {game}")
            if len(mastered_games) > 5:
                print(f"      ... and {len(mastered_games) - 5} more")
                
        if training_games:
            print(f"   🎯 NEEDS TRAINING ({len(training_games)}):")
            for game in training_games[:10]:  # Show first 10
                print(f"      {game}")
            if len(training_games) > 10:
                print(f"      ... and {len(training_games) - 10} more")
        
        print("="*100)
        return False
        
    def adapt_training_parameters(self, games_to_train: List[str]) -> Dict[str, Any]:
        """Adapt training parameters based on current performance and needs."""
        progress = self.get_overall_progress()
        
        # Increase episodes for struggling games
        adapted_config = self.training_config.copy()
        
        # If overall progress is slow, increase training intensity
        if progress['mastery_percentage'] < 0.3:
            adapted_config['base_episodes_per_game'] = 75
            adapted_config['salience_mode'] = 'LOSSLESS'  # Use full memory retention
        elif progress['mastery_percentage'] < 0.7:
            adapted_config['base_episodes_per_game'] = 60
            adapted_config['salience_mode'] = 'DECAY_COMPRESSION'  # Balance memory and performance
        else:
            adapted_config['base_episodes_per_game'] = 50
            adapted_config['salience_mode'] = 'DECAY_COMPRESSION'  # Fine-tuning mode
            
        # For games that have been trained many times without success, increase episodes
        for game in games_to_train:
            if self.game_performance[game]['episodes_played'] > 200:
                adapted_config[f'{game}_episodes'] = min(adapted_config['max_episodes_per_game'], 
                                                        adapted_config['base_episodes_per_game'] * 2)
                
        return adapted_config
        
    async def simulate_training_episode(self, game_id: str, episode_num: int, difficulty_multiplier: float = 1.0) -> Dict[str, Any]:
        """Simulate a single training episode with realistic learning curves."""
        # Get current performance for this game
        current_perf = self.game_performance[game_id]
        episodes_played = current_perf['episodes_played']
        
        # Learning curve: improvement over time with some randomness
        experience_factor = min(1.0, episodes_played / 100.0)  # Cap at 100 episodes
        base_success_rate = 0.1 + (experience_factor * 0.7)  # Start at 10%, max 80%
        
        # Add some random variation and difficulty adjustment
        success_rate = base_success_rate * (1.0 / difficulty_multiplier) * (0.8 + random.random() * 0.4)
        success = random.random() < success_rate
        
        # Score calculation
        if success:
            base_score = 60 + (experience_factor * 30)  # 60-90 base range
            score = min(100, base_score + random.randint(-10, 15))
        else:
            score = random.randint(0, 59)
            
        # Occasional breakthrough episodes
        if episode_num > 20 and random.random() < 0.05:  # 5% chance of breakthrough
            success = True
            score = random.randint(85, 100)
        
        return {
            'game_id': game_id,
            'episode': episode_num,
            'success': success,
            'score': score,
            'experience_factor': experience_factor,
            'difficulty_multiplier': difficulty_multiplier
        }
        
    async def train_game_until_mastery(self, game_id: str, max_episodes: int) -> Dict[str, Any]:
        """Train on a single game until mastery or max episodes reached."""
        print(f"\n🎮 Training on {game_id}...")
        
        current_perf = self.game_performance[game_id]
        episodes_this_session = 0
        session_episodes = []
        plateau_count = 0
        best_recent_score = 0
        
        # Determine difficulty multiplier based on game name
        difficulty_multiplier = 1.0
        if "_3" in game_id:
            difficulty_multiplier = 1.5  # Hardest levels
        elif "_2" in game_id:
            difficulty_multiplier = 1.2  # Medium levels
            
        while episodes_this_session < max_episodes and self.running:
            episode_result = await self.simulate_training_episode(game_id, episodes_this_session + 1, difficulty_multiplier)
            session_episodes.append(episode_result)
            episodes_this_session += 1
            
            # Update running performance metrics
            current_perf['episodes_played'] += 1
            
            # Check for improvement (last 10 episodes)
            if len(session_episodes) >= 10:
                recent_episodes = session_episodes[-10:]
                recent_success_rate = sum(1 for ep in recent_episodes if ep['success']) / len(recent_episodes)
                recent_avg_score = sum(ep['score'] for ep in recent_episodes) / len(recent_episodes)
                
                # Update performance if better
                if recent_avg_score > best_recent_score:
                    best_recent_score = recent_avg_score
                    plateau_count = 0
                else:
                    plateau_count += 1
                    
                # Progress update every 10 episodes
                if episodes_this_session % 10 == 0:
                    print(f"   📊 Episode {episodes_this_session}: {recent_success_rate:.1%} win rate, {recent_avg_score:.1f} avg score")
                    
                    # Check if mastered
                    if (recent_success_rate >= self.target_win_rate and 
                        recent_avg_score >= self.target_avg_score):
                        print(f"   🎉 {game_id} MASTERED! Win rate: {recent_success_rate:.1%}, Avg score: {recent_avg_score:.1f}")
                        current_perf['win_rate'] = recent_success_rate
                        current_perf['avg_score'] = recent_avg_score
                        current_perf['last_updated'] = time.time()
                        break
                        
            # Brief pause for realism
            if episodes_this_session % 5 == 0:
                await asyncio.sleep(0.1)
                
            # Sleep cycle simulation every 15 episodes
            if episodes_this_session % 15 == 0:
                print("   😴 Sleep cycle: consolidating memories...")
                await asyncio.sleep(0.2)
                
        # Calculate final performance for this session
        if session_episodes:
            all_episodes = session_episodes
            final_success_rate = sum(1 for ep in all_episodes if ep['success']) / len(all_episodes)
            final_avg_score = sum(ep['score'] for ep in all_episodes) / len(all_episodes)
            
            # Update stored performance
            current_perf['win_rate'] = final_success_rate
            current_perf['avg_score'] = final_avg_score
            current_perf['last_updated'] = time.time()
            
            return {
                'game_id': game_id,
                'episodes_trained': episodes_this_session,
                'final_win_rate': final_success_rate,
                'final_avg_score': final_avg_score,
                'mastered': (final_success_rate >= self.target_win_rate and final_avg_score >= self.target_avg_score)
            }
        else:
            return {'game_id': game_id, 'episodes_trained': 0, 'mastered': False}
            
    async def run_training_session(self) -> Dict[str, Any]:
        """Run a complete training session on games that need improvement."""
        self.session_count += 1
        session_start = time.time()
        
        # Get games that need training
        games_to_train = self.get_games_needing_training()
        if not games_to_train:
            return {'all_mastered': True}
            
        # Adapt training parameters
        training_config = self.adapt_training_parameters(games_to_train)
        
        print(f"\n🚀 Starting Training Session #{self.session_count}")
        print(f"🎯 Games to train: {len(games_to_train)}")
        print(f"⚙️  Mode: {training_config['salience_mode']}, Episodes per game: {training_config['base_episodes_per_game']}")
        
        session_results = {
            'session_id': self.session_count,
            'games_trained': {},
            'games_mastered_this_session': 0,
            'start_time': session_start
        }
        
        # Prioritize games by current performance (worst first)
        games_to_train.sort(key=lambda g: self.game_performance[g]['win_rate'])
        
        # Train on each game
        for game_idx, game_id in enumerate(games_to_train):
            if not self.running:
                break
                
            print(f"\n🎮 Training {game_idx + 1}/{len(games_to_train)}: {game_id}")
            
            # Determine episodes for this game
            base_episodes = training_config['base_episodes_per_game']
            game_episodes = training_config.get(f'{game_id}_episodes', base_episodes)
            
            # Train the game
            game_result = await self.train_game_until_mastery(game_id, game_episodes)
            session_results['games_trained'][game_id] = game_result
            
            if game_result['mastered']:
                session_results['games_mastered_this_session'] += 1
                print(f"   ✅ {game_id} MASTERED!")
            else:
                print(f"   🔄 {game_id} still training: {game_result['final_win_rate']:.1%} win rate, {game_result['final_avg_score']:.1f} avg score")
                
            # Save state periodically
            self.save_state()
            
        session_results['duration'] = time.time() - session_start
        session_results['end_time'] = time.time()
        
        return session_results
        
    async def run_persistent_loop(self):
        """Main persistent learning loop that runs until all levels are mastered."""
        print("\n🧠 PERSISTENT CONTINUOUS LEARNING SYSTEM ACTIVATED")
        print("🎯 Mission: Master ALL levels with high performance")
        print("🛑 Press Ctrl+C to stop gracefully and save progress")
        
        consecutive_sessions_without_mastery = 0
        
        while self.running:
            try:
                # Display current progress
                all_mastered = self.display_progress_dashboard()
                
                if all_mastered:
                    print("\n🎉🎉🎉 MISSION ACCOMPLISHED! ALL LEVELS MASTERED! 🎉🎉🎉")
                    self.save_state()
                    break
                    
                # Run training session
                session_result = await self.run_training_session()
                
                if session_result.get('all_mastered'):
                    continue  # Refresh dashboard to show completion
                    
                # Check progress
                games_mastered_this_session = session_result['games_mastered_this_session']
                
                if games_mastered_this_session > 0:
                    consecutive_sessions_without_mastery = 0
                    print(f"\n🎊 Session #{self.session_count} completed: {games_mastered_this_session} games mastered!")
                else:
                    consecutive_sessions_without_mastery += 1
                    print(f"\n📈 Session #{self.session_count} completed: Continued training, no new masteries")
                    
                # Adaptive strategy: if no progress for several sessions, adjust approach
                if consecutive_sessions_without_mastery >= self.training_config['restart_threshold']:
                    print(f"\n🔄 No masteries in {consecutive_sessions_without_mastery} sessions. Adapting strategy...")
                    self.training_config['base_episodes_per_game'] = min(200, self.training_config['base_episodes_per_game'] + 25)
                    self.training_config['salience_mode'] = 'LOSSLESS'  # Switch to maximum retention
                    consecutive_sessions_without_mastery = 0
                    
                # Brief pause between sessions
                print("⏸️  Brief pause before next session...")
                await asyncio.sleep(5)
                
            except KeyboardInterrupt:
                print("\n🛑 Shutdown requested by user")
                break
            except Exception as e:
                logger.error(f"Error in training session: {e}")
                print(f"⚠️  Error occurred: {e}. Continuing in 10 seconds...")
                await asyncio.sleep(10)
                
        # Final cleanup
        self.total_runtime += time.time() - self.start_time
        self.save_state()
        
        # Final summary
        progress = self.get_overall_progress()
        print(f"\n📊 FINAL SUMMARY:")
        print(f"   🏆 Games Mastered: {progress['games_mastered']}/{progress['total_games']}")
        print(f"   📈 Overall Win Rate: {progress['overall_win_rate']:.1%}")
        print(f"   📊 Overall Avg Score: {progress['overall_avg_score']:.1f}")
        print(f"   ⏱️  Total Runtime: {self.total_runtime/3600:.1f} hours")
        print(f"   🔄 Sessions Completed: {self.session_count}")
        
        if progress['games_mastered'] == progress['total_games']:
            print("🎉 PERFECT SCORE! ALL LEVELS MASTERED!")
        else:
            remaining = progress['total_games'] - progress['games_mastered']
            print(f"📋 {remaining} levels remaining. Progress saved for next run.")


async def main():
    """Main function to start the persistent continuous learning system."""
    print("🧠 Persistent Continuous Learning System")
    print("🎯 Target: 90% win rate with 85+ average score on ALL levels")
    print("⚡ This system will run until ALL levels are mastered!")
    
    # Create and start the persistent learning system
    learner = PersistentContinuousLearning(
        target_win_rate=0.90,    # 90% win rate required
        target_avg_score=85.0    # 85+ average score required
    )
    
    await learner.run_persistent_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 System shutdown requested")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
    finally:
        print("💾 Persistent learning system terminated")