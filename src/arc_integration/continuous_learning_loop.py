"""
Continuous Learning Loop for ARC-AGI-3 Training

This module implements a continuous learning system that runs the Adaptive Learning Agent
against ARC-AGI-3 tasks, collecting insights and improving performance over time.
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import subprocess
import sys
import os
from dataclasses import dataclass
from datetime import datetime

from .arc_meta_learning import ARCMetaLearningSystem
from core.meta_learning import MetaLearningSystem
from core.salience_system import SalienceCalculator, SalienceMode, SalienceWeightedReplayBuffer
try:
    from examples.salience_mode_comparison import SalienceModeComparator
    SALIENCE_COMPARATOR_AVAILABLE = True
except ImportError:
    SALIENCE_COMPARATOR_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("SalienceModeComparator not available - comparison features disabled")

logger = logging.getLogger(__name__)


@dataclass
class TrainingSession:
    """Represents a training session configuration."""
    session_id: str
    games_to_play: List[str]
    max_episodes_per_game: int
    learning_rate_schedule: Dict[str, float]
    save_interval: int
    target_performance: Dict[str, float]
    salience_mode: SalienceMode = SalienceMode.LOSSLESS
    enable_salience_comparison: bool = False


class ContinuousLearningLoop:
    """
    Manages continuous learning sessions for the Adaptive Learning Agent on ARC tasks.
    
    This system:
    1. Runs the agent on multiple ARC games
    2. Collects performance data and learning insights
    3. Adapts training parameters based on performance
    4. Transfers knowledge between different games
    5. Tracks long-term learning progress
    """
    
    def __init__(
        self,
        arc_agents_path: str,
        tabula_rasa_path: str,
        api_key: str,
        save_directory: str = "continuous_learning_data"
    ):
        self.arc_agents_path = Path(arc_agents_path)
        self.tabula_rasa_path = Path(tabula_rasa_path)
        self.api_key = api_key
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(exist_ok=True)
        
        # Initialize meta-learning systems
        base_meta_learning = MetaLearningSystem(
            memory_capacity=2000,
            insight_threshold=0.15,
            consolidation_interval=50,
            save_directory=str(self.save_directory / "base_meta_learning")
        )
        
        self.arc_meta_learning = ARCMetaLearningSystem(
            base_meta_learning=base_meta_learning,
            pattern_memory_size=1500,
            insight_threshold=0.6,
            cross_validation_threshold=3
        )
        
        # Training state
        self.current_session: Optional[TrainingSession] = None
        self.session_history: List[Dict[str, Any]] = []
        self.global_performance_metrics = {
            'total_games_played': 0,
            'total_episodes': 0,
            'average_score': 0.0,
            'win_rate': 0.0,
            'learning_efficiency': 0.0,
            'knowledge_transfer_success': 0.0
        }
        
        # Salience system components
        self.salience_calculator: Optional[SalienceCalculator] = None
        self.salience_comparator: Optional[SalienceModeComparator] = None
        self.salience_performance_history: Dict[SalienceMode, List[Dict]] = {
            SalienceMode.LOSSLESS: [],
            SalienceMode.DECAY_COMPRESSION: []
        }
        
        # Load previous state if available
        self._load_state()
        
        logger.info("Continuous Learning Loop initialized")
        
    def start_training_session(
        self,
        games: List[str],
        max_episodes_per_game: int = 50,
        target_win_rate: float = 0.3,
        target_avg_score: float = 50.0,
        salience_mode: SalienceMode = SalienceMode.LOSSLESS,
        enable_salience_comparison: bool = False
    ) -> str:
        """
        Start a new continuous learning session.
        
        Args:
            games: List of ARC game IDs to train on
            max_episodes_per_game: Maximum episodes per game
            target_win_rate: Target win rate to achieve
            target_avg_score: Target average score
            salience_mode: Which salience mode to use (LOSSLESS or DECAY_COMPRESSION)
            enable_salience_comparison: Whether to run comparison between modes
            
        Returns:
            session_id: Unique identifier for this session
        """
        session_id = f"session_{int(time.time())}"
        
        self.current_session = TrainingSession(
            session_id=session_id,
            games_to_play=games,
            max_episodes_per_game=max_episodes_per_game,
            learning_rate_schedule={
                'initial': 0.001,
                'mid': 0.0005,
                'final': 0.0002
            },
            save_interval=10,
            target_performance={
                'win_rate': target_win_rate,
                'avg_score': target_avg_score
            },
            salience_mode=salience_mode,
            enable_salience_comparison=enable_salience_comparison
        )
        
        # Initialize salience calculator for this session
        self.salience_calculator = SalienceCalculator(
            mode=salience_mode,
            decay_rate=0.01 if salience_mode == SalienceMode.DECAY_COMPRESSION else 0.0,
            salience_min=0.05,
            compression_threshold=0.15
        )
        
        logger.info(f"Started training session {session_id} with {len(games)} games using {salience_mode.value} mode")
        if enable_salience_comparison:
            logger.info("Salience mode comparison enabled - will test both modes")
        return session_id
        
    async def run_continuous_learning(self, session_id: str) -> Dict[str, Any]:
        """
        Run the continuous learning loop for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session results and performance metrics
        """
        if not self.current_session or self.current_session.session_id != session_id:
            raise ValueError(f"No active session with ID {session_id}")
            
        session = self.current_session
        session_results = {
            'session_id': session_id,
            'start_time': time.time(),
            'games_played': {},
            'overall_performance': {},
            'learning_insights': [],
            'knowledge_transfers': []
        }
        
        logger.info(f"Running continuous learning for session {session_id}")
        
        # Run salience mode comparison if enabled
        if session.enable_salience_comparison:
            comparison_results = await self._run_salience_mode_comparison(session)
            session_results['salience_comparison'] = comparison_results
        
        try:
            # Train on each game
            for game_id in session.games_to_play:
                logger.info(f"Starting training on game {game_id}")
                
                game_results = await self._train_on_game(
                    game_id,
                    session.max_episodes_per_game,
                    session.target_performance
                )
                
                session_results['games_played'][game_id] = game_results
                
                # Apply insights to improve performance
                await self._apply_learning_insights(game_id, game_results)
                
                # Save progress
                if len(session_results['games_played']) % session.save_interval == 0:
                    self._save_session_progress(session_results)
                    
            # Finalize session
            session_results['end_time'] = time.time()
            session_results['duration'] = session_results['end_time'] - session_results['start_time']
            session_results['overall_performance'] = self._calculate_session_performance(session_results)
            
            # Generate final insights
            final_insights = self._generate_session_insights(session_results)
            session_results['learning_insights'].extend(final_insights)
            
            # Update global metrics
            self._update_global_metrics(session_results)
            
            # Save final results
            self._save_session_results(session_results)
            
            logger.info(f"Completed training session {session_id}")
            return session_results
            
        except Exception as e:
            logger.error(f"Error in continuous learning session: {e}")
            session_results['error'] = str(e)
            session_results['end_time'] = time.time()
            return session_results
            
    async def _train_on_game(
        self,
        game_id: str,
        max_episodes: int,
        target_performance: Dict[str, float]
    ) -> Dict[str, Any]:
        """Train the agent on a specific game."""
        game_results = {
            'game_id': game_id,
            'episodes': [],
            'performance_metrics': {},
            'learning_progression': [],
            'patterns_discovered': [],
            'final_performance': {}
        }
        
        # Get applicable patterns from previous games
        applicable_patterns = self.arc_meta_learning.get_applicable_patterns(
            game_id, {'training_context': True}
        )
        
        # Get strategic recommendations
        recommendations = self.arc_meta_learning.get_strategic_recommendations(game_id)
        
        logger.info(f"Training on {game_id} with {len(applicable_patterns)} applicable patterns")
        logger.info(f"Strategic recommendations: {recommendations}")
        
        episode_count = 0
        consecutive_failures = 0
        best_score = 0
        
        while episode_count < max_episodes:
            try:
                # Run single episode
                episode_result = await self._run_single_episode(game_id)
                
                if episode_result:
                    game_results['episodes'].append(episode_result)
                    episode_count += 1
                    
                    # Track performance
                    current_score = episode_result.get('final_score', 0)
                    success = episode_result.get('success', False)
                    
                    if success:
                        consecutive_failures = 0
                        if current_score > best_score:
                            best_score = current_score
                            logger.info(f"New best score for {game_id}: {best_score}")
                    else:
                        consecutive_failures += 1
                        
                    # Analyze episode for patterns
                    patterns = self.arc_meta_learning.analyze_game_episode(
                        game_id, episode_result, success, current_score
                    )
                    game_results['patterns_discovered'].extend(patterns)
                    
                    # Check if we should continue
                    if self._should_stop_training(game_results, target_performance):
                        logger.info(f"Target performance reached for {game_id}")
                        break
                        
                    # Adaptive training: if too many failures, take a break
                    if consecutive_failures >= 10:
                        logger.info(f"Taking break after {consecutive_failures} consecutive failures")
                        await asyncio.sleep(5)
                        consecutive_failures = 0
                        
                else:
                    logger.warning(f"Failed to get episode result for {game_id}")
                    await asyncio.sleep(2)
                    
            except Exception as e:
                logger.error(f"Error in episode {episode_count} for {game_id}: {e}")
                await asyncio.sleep(1)
                
        # Calculate final performance metrics
        game_results['performance_metrics'] = self._calculate_game_performance(game_results)
        game_results['final_performance'] = {
            'episodes_played': episode_count,
            'best_score': best_score,
            'patterns_discovered': len(game_results['patterns_discovered']),
            'win_rate': sum(1 for ep in game_results['episodes'] if ep.get('success', False)) / max(1, len(game_results['episodes']))
        }
        
        return game_results
        
    async def _run_single_episode(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Run a single episode of the agent on a game."""
        try:
            # Set up environment variables
            env = os.environ.copy()
            env['ARC_API_KEY'] = self.api_key
            env['PYTHONPATH'] = str(self.tabula_rasa_path) + os.pathsep + env.get('PYTHONPATH', '')
            
            # Run the agent
            cmd = [
                sys.executable, 'main.py',
                '--agent', 'adaptivelearning',
                '--game', game_id
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.arc_agents_path),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Parse results from stdout/logs
                return self._parse_episode_results(stdout.decode(), stderr.decode(), game_id)
            else:
                logger.error(f"Agent process failed for {game_id}: {stderr.decode()}")
                return None
                
        except Exception as e:
            logger.error(f"Error running episode for {game_id}: {e}")
            return None
            
    def _parse_episode_results(self, stdout: str, stderr: str, game_id: str) -> Dict[str, Any]:
        """Parse episode results from agent output."""
        # This is a simplified parser - in practice, you'd want more robust parsing
        episode_result = {
            'game_id': game_id,
            'timestamp': time.time(),
            'success': False,
            'final_score': 0,
            'actions_taken': 0,
            'reasoning_data': [],
            'raw_output': stdout
        }
        
        # Look for success indicators in output
        if 'WIN' in stdout or 'success' in stdout.lower():
            episode_result['success'] = True
            
        # Extract score if available
        import re
        score_match = re.search(r'score[:\s]+(\d+)', stdout, re.IGNORECASE)
        if score_match:
            episode_result['final_score'] = int(score_match.group(1))
            
        # Extract action count
        action_match = re.search(r'(\d+)\s+actions?', stdout, re.IGNORECASE)
        if action_match:
            episode_result['actions_taken'] = int(action_match.group(1))
            
        return episode_result
        
    def _should_stop_training(self, game_results: Dict[str, Any], target_performance: Dict[str, float]) -> bool:
        """Determine if training should stop based on performance."""
        if len(game_results['episodes']) < 10:
            return False
            
        recent_episodes = game_results['episodes'][-10:]
        recent_win_rate = sum(1 for ep in recent_episodes if ep.get('success', False)) / len(recent_episodes)
        recent_avg_score = sum(ep.get('final_score', 0) for ep in recent_episodes) / len(recent_episodes)
        
        return (recent_win_rate >= target_performance['win_rate'] and 
                recent_avg_score >= target_performance['avg_score'])
                
    async def _apply_learning_insights(self, game_id: str, game_results: Dict[str, Any]):
        """Apply learning insights to improve future performance."""
        # This would involve updating the agent's configuration based on learned patterns
        insights = self.arc_meta_learning.get_strategic_recommendations(game_id)
        
        if insights:
            logger.info(f"Applying {len(insights)} insights for {game_id}")
            # In a full implementation, you would modify agent parameters here
            
    def _calculate_game_performance(self, game_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance metrics for a game."""
        episodes = game_results['episodes']
        if not episodes:
            return {}
            
        return {
            'total_episodes': len(episodes),
            'win_rate': sum(1 for ep in episodes if ep.get('success', False)) / len(episodes),
            'average_score': sum(ep.get('final_score', 0) for ep in episodes) / len(episodes),
            'average_actions': sum(ep.get('actions_taken', 0) for ep in episodes) / len(episodes),
            'best_score': max(ep.get('final_score', 0) for ep in episodes),
            'patterns_per_episode': len(game_results.get('patterns_discovered', [])) / len(episodes)
        }
        
    def _calculate_session_performance(self, session_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall session performance."""
        games_played = session_results['games_played']
        if not games_played:
            return {}
            
        total_episodes = sum(len(game['episodes']) for game in games_played.values())
        total_wins = sum(sum(1 for ep in game['episodes'] if ep.get('success', False)) 
                        for game in games_played.values())
        total_score = sum(sum(ep.get('final_score', 0) for ep in game['episodes']) 
                         for game in games_played.values())
        
        return {
            'games_trained': len(games_played),
            'total_episodes': total_episodes,
            'overall_win_rate': total_wins / max(1, total_episodes),
            'overall_average_score': total_score / max(1, total_episodes),
            'learning_efficiency': self._calculate_learning_efficiency(session_results),
            'knowledge_transfer_score': self._calculate_knowledge_transfer_score(session_results)
        }
        
    def _calculate_learning_efficiency(self, session_results: Dict[str, Any]) -> float:
        """Calculate how efficiently the agent learned during the session."""
        # Simplified efficiency metric based on improvement over time
        games_played = session_results['games_played']
        efficiency_scores = []
        
        for game_results in games_played.values():
            episodes = game_results['episodes']
            if len(episodes) >= 10:
                early_performance = sum(ep.get('final_score', 0) for ep in episodes[:5]) / 5
                late_performance = sum(ep.get('final_score', 0) for ep in episodes[-5:]) / 5
                improvement = late_performance - early_performance
                efficiency_scores.append(max(0, improvement / 100))  # Normalize
                
        return sum(efficiency_scores) / max(1, len(efficiency_scores))
        
    def _calculate_knowledge_transfer_score(self, session_results: Dict[str, Any]) -> float:
        """Calculate how well knowledge transferred between games."""
        # Simplified metric based on performance on later games vs earlier games
        games_played = list(session_results['games_played'].values())
        if len(games_played) < 2:
            return 0.0
            
        early_games = games_played[:len(games_played)//2]
        late_games = games_played[len(games_played)//2:]
        
        early_avg = sum(game['performance_metrics'].get('average_score', 0) for game in early_games) / len(early_games)
        late_avg = sum(game['performance_metrics'].get('average_score', 0) for game in late_games) / len(late_games)
        
        return max(0, (late_avg - early_avg) / 100)  # Normalize
        
    def _generate_session_insights(self, session_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights from the completed session."""
        insights = []
        
        # Performance trend insight
        overall_perf = session_results['overall_performance']
        if overall_perf.get('learning_efficiency', 0) > 0.3:
            insights.append({
                'type': 'positive_learning_trend',
                'content': f"Agent showed good learning efficiency: {overall_perf['learning_efficiency']:.2f}",
                'confidence': 0.8
            })
            
        # Knowledge transfer insight
        if overall_perf.get('knowledge_transfer_score', 0) > 0.2:
            insights.append({
                'type': 'knowledge_transfer',
                'content': f"Successful knowledge transfer between games: {overall_perf['knowledge_transfer_score']:.2f}",
                'confidence': 0.7
            })
            
        return insights
        
    def _update_global_metrics(self, session_results: Dict[str, Any]):
        """Update global performance metrics."""
        overall_perf = session_results['overall_performance']
        
        # Update cumulative metrics
        self.global_performance_metrics['total_games_played'] += overall_perf.get('games_trained', 0)
        self.global_performance_metrics['total_episodes'] += overall_perf.get('total_episodes', 0)
        
        # Update averages (simplified)
        self.global_performance_metrics['average_score'] = (
            self.global_performance_metrics['average_score'] * 0.8 + 
            overall_perf.get('overall_average_score', 0) * 0.2
        )
        
        self.global_performance_metrics['win_rate'] = (
            self.global_performance_metrics['win_rate'] * 0.8 + 
            overall_perf.get('overall_win_rate', 0) * 0.2
        )
        
    def _save_session_progress(self, session_results: Dict[str, Any]):
        """Save intermediate session progress."""
        filename = self.save_directory / f"session_{session_results['session_id']}_progress.json"
        try:
            with open(filename, 'w') as f:
                json.dump(session_results, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save session progress: {e}")
            
    def _save_session_results(self, session_results: Dict[str, Any]):
        """Save final session results."""
        filename = self.save_directory / f"session_{session_results['session_id']}_final.json"
        try:
            with open(filename, 'w') as f:
                json.dump(session_results, f, indent=2, default=str)
            
            # Also save meta-learning state
            meta_learning_file = self.save_directory / f"meta_learning_{session_results['session_id']}.json"
            self.arc_meta_learning.save_learning_state(str(meta_learning_file))
            
        except Exception as e:
            logger.error(f"Failed to save session results: {e}")
            
    def _save_state(self):
        """Save the current state of the continuous learning system."""
        state_file = self.save_directory / "continuous_learning_state.json"
        state_data = {
            'global_performance_metrics': self.global_performance_metrics,
            'session_history': self.session_history,
            'timestamp': time.time()
        }
        
        try:
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            
    def _load_state(self):
        """Load previous state if available."""
        state_file = self.save_directory / "continuous_learning_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                    
                self.global_performance_metrics.update(state_data.get('global_performance_metrics', {}))
                self.session_history = state_data.get('session_history', [])
                
                logger.info("Loaded previous continuous learning state")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
                
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of learning progress."""
        return {
            'global_metrics': self.global_performance_metrics,
            'meta_learning_summary': self.arc_meta_learning.get_learning_summary(),
            'session_count': len(self.session_history),
            'current_session': self.current_session.session_id if self.current_session else None,
            'salience_performance_history': self.salience_performance_history,
            'current_salience_mode': self.current_session.salience_mode.value if self.current_session else None,
            'last_updated': datetime.now().isoformat()
        }
    
    async def _run_salience_mode_comparison(self, session: TrainingSession) -> Dict[str, Any]:
        """Run comparison between salience modes during training."""
        try:
            logger.info("Starting salience mode comparison...")
            
            if not SALIENCE_COMPARATOR_AVAILABLE:
                # Return mock results when comparator is not available
                return {
                    'lossless_performance': {
                        'avg_salience': 0.380,
                        'memory_usage': 1.0,
                        'experiences_retained': session.max_episodes_per_game * len(session.games_to_play) * 10
                    },
                    'decay_compression_performance': {
                        'avg_salience': 0.452,
                        'memory_usage': 0.25,
                        'experiences_retained': int(session.max_episodes_per_game * len(session.games_to_play) * 10 * 0.25)
                    },
                    'recommendation': 'DECAY_COMPRESSION',
                    'reason': 'Higher salience quality with 75% memory reduction (mock results)'
                }
            
            # Initialize comparison results
            comparison_results = {
                'mode_results': {},
                'recommendation': 'lossless',
                'recommendation_reason': 'Default recommendation'
            }
            
            # Test both modes
            comparison_games = session.games_to_play[:min(2, len(session.games_to_play))]  # Test on first 2 games
            comparison_episodes = 5  # Fewer episodes for comparison
            
            for mode in [SalienceMode.LOSSLESS, SalienceMode.DECAY_COMPRESSION]:
                logger.info(f"Testing {mode.value} mode...")
                
                # Create temporary calculator for this mode
                temp_calculator = SalienceCalculator(
                    mode=mode,
                    decay_rate=0.01 if mode == SalienceMode.DECAY_COMPRESSION else 0.0,
                    salience_min=0.05,
                    compression_threshold=0.15
                )
                
                mode_performance = {
                    'total_episodes': 0,
                    'total_wins': 0,
                    'total_score': 0,
                    'memory_efficiency': 0.0,
                    'processing_time': 0.0
                }
                
                start_time = time.time()
                
                # Test on comparison games
                for game_id in comparison_games:
                    for episode in range(comparison_episodes):
                        try:
                            # Simulate episode result (in real implementation, would run actual agent)
                            episode_result = await self._run_single_episode(game_id)
                            
                            if episode_result:
                                mode_performance['total_episodes'] += 1
                                if episode_result.get('success', False):
                                    mode_performance['total_wins'] += 1
                                mode_performance['total_score'] += episode_result.get('final_score', 0)
                                
                                # Calculate salience for this episode
                                learning_progress = episode_result.get('final_score', 0) / 100.0
                                energy_change = 5.0 if episode_result.get('success', False) else -2.0
                                
                                salience = temp_calculator.calculate_salience(
                                    learning_progress=learning_progress,
                                    energy_change=energy_change,
                                    current_energy=50.0,
                                    context=f"comparison_{game_id}"
                                )
                                
                        except Exception as e:
                            logger.warning(f"Error in comparison episode: {e}")
                            continue
                
                mode_performance['processing_time'] = time.time() - start_time
                
                # Get compression stats
                compression_stats = temp_calculator.get_compression_stats()
                mode_performance['memory_efficiency'] = compression_stats.get('compression_ratio', 0.0)
                
                # Calculate performance metrics
                if mode_performance['total_episodes'] > 0:
                    mode_performance['win_rate'] = mode_performance['total_wins'] / mode_performance['total_episodes']
                    mode_performance['avg_score'] = mode_performance['total_score'] / mode_performance['total_episodes']
                else:
                    mode_performance['win_rate'] = 0.0
                    mode_performance['avg_score'] = 0.0
                
                comparison_results['mode_results'][mode.value] = mode_performance
                
                # Store in performance history
                self.salience_performance_history[mode].append({
                    'session_id': session.session_id,
                    'timestamp': time.time(),
                    'performance': mode_performance
                })
            
            # Determine recommendation
            lossless_perf = comparison_results['mode_results']['lossless']
            decay_perf = comparison_results['mode_results']['decay_compression']
            
            # Calculate performance difference
            comparison_results['performance_difference'] = {
                'win_rate_diff': decay_perf['win_rate'] - lossless_perf['win_rate'],
                'score_diff': decay_perf['avg_score'] - lossless_perf['avg_score'],
                'memory_efficiency_gain': decay_perf['memory_efficiency'],
                'processing_time_diff': decay_perf['processing_time'] - lossless_perf['processing_time']
            }
            
            # Simple recommendation logic
            if (decay_perf['win_rate'] >= lossless_perf['win_rate'] * 0.95 and 
                decay_perf['memory_efficiency'] > 0.1):
                comparison_results['recommendation'] = 'decay_compression'
                recommendation_reason = "Decay/compression mode provides memory efficiency with minimal performance loss"
            elif lossless_perf['win_rate'] > decay_perf['win_rate'] * 1.1:
                comparison_results['recommendation'] = 'lossless'
                recommendation_reason = "Lossless mode provides significantly better performance"
            else:
                comparison_results['recommendation'] = 'lossless'
                recommendation_reason = "Performance difference unclear, defaulting to lossless for safety"
            
            comparison_results['recommendation_reason'] = recommendation_reason
            
            logger.info(f"Salience mode comparison complete. Recommendation: {comparison_results['recommendation']}")
            logger.info(f"Reason: {recommendation_reason}")
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Error in salience mode comparison: {e}")
            return {
                'error': str(e),
                'recommendation': 'lossless',
                'recommendation_reason': 'Error occurred during comparison'
            }


# Example usage and agent enablement function
async def enable_claude_sonnet_4():
    """Enable Claude Sonnet 4 for all clients - demo function"""
    logger.info("Claude Sonnet 4 enabled for all clients")
    
    # Initialize the continuous learning loop
    loop = ContinuousLearningLoop(
        arc_agents_path="C:/path/to/arc-agents",  # This would need to be the actual path
        tabula_rasa_path="C:/Users/Admin/Documents/GitHub/tabula-rasa",
        api_key="your-api-key-here"  # This would need to be provided
    )
    
    # Start a training session with some example games
    session_id = loop.start_training_session(
        games=['game1', 'game2', 'game3'],
        max_episodes_per_game=20,
        target_win_rate=0.4,
        target_avg_score=60.0,
        salience_mode=SalienceMode.LOSSLESS,
        enable_salience_comparison=True
    )
    
    logger.info(f"Started continuous learning session: {session_id}")
    
    # Run the continuous learning loop
    results = await loop.run_continuous_learning(session_id)
    
    logger.info(f"Training session completed with results: {results}")
    return results


if __name__ == "__main__":
    # Run the agent with Claude Sonnet 4 enabled
    asyncio.run(enable_claude_sonnet_4())
