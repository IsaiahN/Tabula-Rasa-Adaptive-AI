"""
Coordinate-Aware ARC Integration
Enhanced integration layer that connects the coordinate system overhaul with existing training loop.
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from .continuous_learning_loop import ContinuousLearningLoop
from .arc_agent_adapter import AdaptiveLearningARCAgent
from api.enhanced_client import ArcAgiApiClient
from vision.frame_analyzer import FrameAnalyzer
from learning.pathway_system import PathwayLearningSystem

logger = logging.getLogger(__name__)


class CoordinateAwareTrainingManager:
    """
    Enhanced training manager that integrates coordinate-aware components
    with the existing continuous learning loop.
    """
    
    def __init__(self, api_key: str, arc_agents_path: str = None):
        """
        Initialize coordinate-aware training manager.
        
        Args:
            api_key: ARC-AGI-3 API key
            arc_agents_path: Path to arc-agents repository
        """
        self.api_key = api_key
        self.arc_agents_path = Path(arc_agents_path) if arc_agents_path else None
        
        # Initialize enhanced components
        self.api_client = ArcAgiApiClient(api_key)
        self.frame_analyzer = FrameAnalyzer()
        self.pathway_system = PathwayLearningSystem()
        
        # Initialize existing continuous learning loop with enhancements
        self.continuous_loop = ContinuousLearningLoop(
            api_key=api_key,
            tabula_rasa_path=str(Path(__file__).parent.parent.parent),
            arc_agents_path=arc_agents_path
        )
        
        # Track coordinate-aware agents
        self.coordinate_aware_agents = {}
        
    async def run_enhanced_training_session(
        self, 
        game_id: str,
        max_actions: int = 1000,
        use_coordinate_awareness: bool = True
    ) -> Dict[str, Any]:
        """
        Run enhanced training session with coordinate awareness.
        
        Args:
            game_id: Game to train on
            max_actions: Maximum actions per session
            use_coordinate_awareness: Enable coordinate-aware features
            
        Returns:
            Training results with coordinate system metrics
        """
        logger.info(f"🚀 Starting coordinate-aware training session for {game_id}")
        
        try:
            # Initialize coordinate-aware agent for this game
            if game_id not in self.coordinate_aware_agents:
                self.coordinate_aware_agents[game_id] = AdaptiveLearningARCAgent(game_id=game_id)
            
            agent = self.coordinate_aware_agents[game_id]
            
            # Start game session with enhanced API client
            session_result = self.api_client.start_game(game_id, f"coord_training_{game_id}")
            if not session_result.get('success', False):
                return {'error': f'Failed to start game: {session_result.get("error", "Unknown error")}'}
            
            # Training loop with coordinate awareness
            results = {
                'game_id': game_id,
                'actions_taken': 0,
                'coordinate_actions': 0,
                'coordinate_successes': 0,
                'pathway_learning_updates': 0,
                'score_progression': [],
                'coordinate_intelligence': {},
                'action_effectiveness': {}
            }
            
            current_score = 0
            for action_count in range(max_actions):
                # Get current game state
                game_state = self.api_client.get_game_state()
                if not game_state.get('success'):
                    break
                
                frame_data = game_state.get('frame_data')
                available_actions = game_state.get('available_actions', [])
                
                if not frame_data or not available_actions:
                    break
                
                # Use coordinate-aware action selection
                selected_action = None
                coordinates = None
                
                if use_coordinate_awareness:
                    # Enhanced action selection with coordinate intelligence
                    selected_action, coordinates = await self._select_coordinate_aware_action(
                        frame_data, available_actions, agent, game_id
                    )
                else:
                    # Fallback to existing system
                    selected_action = self.continuous_loop._select_intelligent_action_with_relevance(
                        available_actions, {'game_id': game_id, 'frame_data': frame_data}
                    )
                
                # Execute action
                if selected_action == 6 and coordinates:
                    action_result = self.api_client.execute_action(selected_action, coordinates)
                    results['coordinate_actions'] += 1
                else:
                    action_result = self.api_client.execute_action(selected_action)
                
                # Track results
                results['actions_taken'] += 1
                new_score = action_result.get('score', current_score)
                score_improvement = new_score - current_score
                current_score = new_score
                
                results['score_progression'].append({
                    'action': action_count + 1,
                    'score': current_score,
                    'improvement': score_improvement,
                    'action_type': selected_action,
                    'coordinates': coordinates
                })
                
                # Record action result for learning
                success = score_improvement > 0
                if coordinates:
                    results['coordinate_successes'] += success
                
                agent.record_action_result(
                    action=selected_action,
                    coordinates=coordinates,
                    success=success,
                    score_improvement=score_improvement
                )
                
                # Update pathway learning
                self.pathway_system.track_action(
                    action=selected_action,
                    action_data={'coordinates': coordinates} if coordinates else {},
                    score_before=current_score - score_improvement,
                    score_after=current_score,
                    win_score=game_state.get('win_score', 100),
                    game_id=game_id
                )
                results['pathway_learning_updates'] += 1
                
                # Check for game end
                if action_result.get('state') in ['WIN', 'GAME_OVER']:
                    break
            
            # Generate coordinate intelligence summary
            results['coordinate_intelligence'] = self._generate_coordinate_intelligence_report(game_id)
            results['action_effectiveness'] = self._calculate_action_effectiveness(results)
            
            logger.info(f"✅ Coordinate-aware training completed: {results['actions_taken']} actions, "
                       f"{results['coordinate_actions']} coordinate actions, "
                       f"{results['coordinate_successes']} coordinate successes")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in coordinate-aware training: {e}")
            return {'error': str(e)}
    
    async def _select_coordinate_aware_action(
        self,
        frame_data: Dict[str, Any],
        available_actions: List[int],
        agent: AdaptiveLearningARCAgent,
        game_id: str
    ) -> Tuple[int, Optional[Tuple[int, int]]]:
        """
        Select action using coordinate awareness and pathway learning.
        """
        # Get pathway recommendations
        pathway_recommendations_result = self.pathway_system.get_pathway_recommendations(
            available_actions=available_actions,
            current_score=frame_data.get('score', 0),
            game_id=game_id
        )
        pathway_recommendations = pathway_recommendations_result.get('action_weights', {})
        
        # Analyze frame for coordinate intelligence
        frame_analysis = self.frame_analyzer.analyze_frame(frame_data.get('frame', []))
        
        # Weight available actions based on pathway learning
        action_scores = {}
        for action in available_actions:
            base_score = pathway_recommendations.get(action, 1.0)
            
            # Boost ACTION6 if frame analysis suggests coordinate action would be effective
            if action == 6 and frame_analysis:
                if frame_analysis.get('movement_detected') or frame_analysis.get('agent_position'):
                    base_score *= 1.5  # Boost coordinate action when movement/position detected
            
            action_scores[action] = base_score
        
        # Select action based on weighted scores
        best_action = max(action_scores.keys(), key=lambda a: action_scores[a])
        
        # Get coordinates for ACTION6
        coordinates = None
        if best_action == 6:
            coordinates = agent.action_mapper._get_intelligent_coordinates(
                frame_data=frame_data,
                action_history=[]
            )
        
        return best_action, coordinates
    
    def _generate_coordinate_intelligence_report(self, game_id: str) -> Dict[str, Any]:
        """Generate intelligence report on coordinate system performance."""
        if game_id not in self.coordinate_aware_agents:
            return {}
        
        agent = self.coordinate_aware_agents[game_id]
        
        return {
            'pathway_recommendations': self.pathway_system.get_pathway_analysis(game_id),
            'coordinate_strategies_used': agent.action_mapper.coordinate_strategies,
            'current_strategy': agent.action_mapper.current_strategy,
            'frame_analysis_stats': getattr(self.frame_analyzer, 'analysis_stats', {}),
        }
    
    def _calculate_action_effectiveness(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate effectiveness metrics for different action types."""
        score_progression = results.get('score_progression', [])
        if not score_progression:
            return {}
        
        action_effectiveness = {}
        for entry in score_progression:
            action_type = entry['action_type']
            improvement = entry['improvement']
            
            if action_type not in action_effectiveness:
                action_effectiveness[action_type] = {
                    'total_uses': 0,
                    'successful_uses': 0,
                    'total_improvement': 0.0,
                    'average_improvement': 0.0
                }
            
            stats = action_effectiveness[action_type]
            stats['total_uses'] += 1
            if improvement > 0:
                stats['successful_uses'] += 1
                stats['total_improvement'] += improvement
            
            stats['average_improvement'] = stats['total_improvement'] / stats['total_uses']
        
        return action_effectiveness
    
    async def run_coordinate_system_benchmark(self, games: List[str]) -> Dict[str, Any]:
        """
        Run benchmark comparing coordinate-aware vs traditional training.
        """
        benchmark_results = {
            'games_tested': games,
            'coordinate_aware_results': {},
            'traditional_results': {},
            'performance_comparison': {}
        }
        
        for game_id in games:
            logger.info(f"🧪 Benchmarking coordinate system on {game_id}")
            
            # Test coordinate-aware approach
            coord_results = await self.run_enhanced_training_session(
                game_id=game_id,
                max_actions=500,
                use_coordinate_awareness=True
            )
            benchmark_results['coordinate_aware_results'][game_id] = coord_results
            
            # Test traditional approach
            traditional_results = await self.run_enhanced_training_session(
                game_id=game_id,
                max_actions=500,
                use_coordinate_awareness=False
            )
            benchmark_results['traditional_results'][game_id] = traditional_results
            
            # Compare performance
            coord_score = coord_results.get('score_progression', [])
            trad_score = traditional_results.get('score_progression', [])
            
            coord_final = coord_score[-1]['score'] if coord_score else 0
            trad_final = trad_score[-1]['score'] if trad_score else 0
            
            benchmark_results['performance_comparison'][game_id] = {
                'coordinate_aware_final_score': coord_final,
                'traditional_final_score': trad_final,
                'coordinate_improvement': coord_final - trad_final,
                'coordinate_actions_used': coord_results.get('coordinate_actions', 0),
                'coordinate_success_rate': (
                    coord_results.get('coordinate_successes', 0) / 
                    max(coord_results.get('coordinate_actions', 1), 1)
                )
            }
        
        return benchmark_results


# Convenience function for easy integration
async def run_coordinate_aware_training(
    api_key: str,
    games: List[str],
    arc_agents_path: str = None,
    benchmark_mode: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to run coordinate-aware training.
    
    Args:
        api_key: ARC-AGI-3 API key
        games: List of game IDs to train on
        arc_agents_path: Path to arc-agents repository
        benchmark_mode: Run benchmark comparison
        
    Returns:
        Training results
    """
    manager = CoordinateAwareTrainingManager(api_key, arc_agents_path)
    
    if benchmark_mode:
        return await manager.run_coordinate_system_benchmark(games)
    else:
        results = {}
        for game_id in games:
            results[game_id] = await manager.run_enhanced_training_session(game_id)
        return results
