"""
Intelligent Game Progression Analyzer

This module analyzes game progression patterns to make intelligent decisions about:
1. When to force exploration vs exploitation
2. How to detect and handle stagnation
3. When to prioritize action diversity
4. How to adapt based on game-specific progression metrics

Key Features:
- Progressive stage detection
- Stagnation risk assessment
- Action diversity analysis
- Game phase estimation
- End-game proximity detection
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import math
import statistics
from collections import defaultdict

logger = logging.getLogger(__name__)

class ProgressionAnalyzer:
    """Analyzes game progression to make intelligent strategy decisions."""
    
    def __init__(self, db_interface=None):
        """Initialize the progression analyzer.
        
        Args:
            db_interface: Optional database interface for storing/retrieving historical data
        """
        self.db_interface = db_interface
        self.stage_metrics = defaultdict(dict)
        self.history = defaultdict(list)
        self.action_frequencies = defaultdict(lambda: defaultdict(int))
        self.success_thresholds = {
            'early_game': 0.6,    # Higher success needed early to establish good patterns
            'mid_game': 0.5,      # Balance between exploration and exploitation
            'late_game': 0.4      # Allow more exploration if struggling late
        }
        
    def analyze_progression(self, game_id: str, current_step: int,
                          total_actions: int, score: float,
                          recent_actions: List[int],
                          success_rate: float = None) -> Dict[str, Any]:
        """Analyze current game progression and recommend strategy adjustments.
        
        Args:
            game_id: Unique game identifier
            current_step: Current step number in the game
            total_actions: Total actions taken so far
            score: Current game score
            recent_actions: List of recent action numbers (e.g., [6,6,3,6,2])
            success_rate: Optional success rate of recent actions
            
        Returns:
            Dict containing analysis results and recommendations
        """
        try:
            # Get historical data for this game type if available
            historical_data = self._get_historical_data(game_id)
            
            # Calculate progression metrics
            progression = self._calculate_progression_metrics(current_step, total_actions, score, historical_data)
            
            # Detect game phase
            game_phase = self._detect_game_phase(progression)
            
            # Analyze action patterns
            action_patterns = self._analyze_action_patterns(recent_actions, game_id)
            
            # Calculate stagnation risk
            stagnation_risk = self._calculate_stagnation_risk(
                action_patterns,
                progression,
                recent_actions
            )
            
            # Analyze diversity needs
            diversity_metrics = self._analyze_diversity_needs(
                action_patterns,
                game_phase,
                stagnation_risk
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                game_phase=game_phase,
                stagnation_risk=stagnation_risk,
                diversity_metrics=diversity_metrics,
                progression=progression,
                success_rate=success_rate
            )
            
            # Update history and metrics
            self._update_metrics(game_id, current_step, score, recommendations)
            
            return {
                'game_phase': game_phase,
                'stagnation_risk': stagnation_risk,
                'diversity_metrics': diversity_metrics,
                'recommendations': recommendations,
                'progression_metrics': progression
            }
            
        except Exception as e:
            logger.error(f"Error analyzing progression: {e}")
            return self._get_safe_fallback_analysis()
            
    def _calculate_progression_metrics(self, current_step: int, total_actions: int,
                                    score: float, historical_data: dict) -> Dict[str, float]:
        """Calculate detailed progression metrics."""
        metrics = {}
        
        # 1. Basic Progression
        metrics['completion_estimate'] = min(current_step / 100, 0.99)  # Assume 100 steps max
        
        # 2. Action Efficiency
        if total_actions > 0:
            metrics['action_efficiency'] = score / total_actions
        else:
            metrics['action_efficiency'] = 0.0
            
        # 3. Score Velocity (rate of score change)
        if historical_data and 'score_history' in historical_data:
            recent_scores = historical_data['score_history'][-5:]
            if len(recent_scores) >= 2:
                metrics['score_velocity'] = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
            else:
                metrics['score_velocity'] = 0.0
        else:
            metrics['score_velocity'] = 0.0
            
        # 4. Progress Rate
        if historical_data and 'avg_completion_steps' in historical_data:
            avg_steps = historical_data['avg_completion_steps']
            metrics['progress_rate'] = current_step / avg_steps if avg_steps > 0 else 0.5
        else:
            metrics['progress_rate'] = current_step / 50  # Assume 50 steps as baseline
            
        # 5. Performance Relative to History
        if historical_data and 'avg_score_progression' in historical_data:
            historical_score = historical_data['avg_score_progression'].get(str(current_step), 0)
            if historical_score > 0:
                metrics['historical_performance'] = score / historical_score
            else:
                metrics['historical_performance'] = 1.0
        else:
            metrics['historical_performance'] = 1.0
            
        # 6. Endgame Proximity Estimate
        metrics['endgame_proximity'] = self._estimate_endgame_proximity(
            current_step=current_step,
            score=score,
            historical_data=historical_data
        )
        
        return metrics
        
    def _detect_game_phase(self, progression: Dict[str, float]) -> str:
        """Detect current game phase based on progression metrics."""
        completion = progression['completion_estimate']
        efficiency = progression['action_efficiency']
        velocity = progression['score_velocity']
        
        # Early Game Indicators
        if completion < 0.25 and efficiency > 0.7:
            return 'early_game_strong'
        elif completion < 0.25:
            return 'early_game_weak'
            
        # Mid Game Indicators    
        elif completion < 0.7:
            if velocity > 0 and efficiency > 0.5:
                return 'mid_game_strong'
            else:
                return 'mid_game_struggling'
                
        # Late Game Indicators
        else:
            if velocity > 0 and progression['endgame_proximity'] > 0.8:
                return 'endgame_push'
            elif velocity <= 0:
                return 'endgame_recovery'
            else:
                return 'late_game'
                
    def _analyze_action_patterns(self, recent_actions: List[int], game_id: str) -> Dict[str, Any]:
        """Analyze patterns in recent action sequences."""
        if not recent_actions:
            return {'repetition': 0, 'variety': 0, 'staleness': 0}
            
        # Track action frequencies
        frequencies = defaultdict(int)
        for action in recent_actions:
            frequencies[action] += 1
            self.action_frequencies[game_id][action] += 1
            
        # Calculate pattern metrics
        total_actions = len(recent_actions)
        unique_actions = len(frequencies)
        
        # 1. Action Repetition
        most_common = max(frequencies.values())
        repetition = most_common / total_actions
        
        # 2. Action Variety
        variety = unique_actions / len(self.action_frequencies[game_id])
        
        # 3. Action Staleness (how long we've been using same actions)
        staleness = self._calculate_staleness(recent_actions)
        
        # 4. Sequence Patterns
        sequence_patterns = self._detect_sequences(recent_actions)
        
        return {
            'repetition': repetition,
            'variety': variety,
            'staleness': staleness,
            'sequence_patterns': sequence_patterns,
            'action_frequencies': dict(frequencies)
        }
        
    def _calculate_staleness(self, actions: List[int]) -> float:
        """Calculate how stale the action patterns have become."""
        if not actions:
            return 0.0
            
        unique_window_sizes = [
            len(set(actions[max(0, i-5):i]))
            for i in range(len(actions), max(0, len(actions)-15), -1)
        ]
        
        # More weight to recent windows
        weights = [1.0 / (i + 1) for i in range(len(unique_window_sizes))]
        weighted_sum = sum(w * v for w, v in zip(weights, unique_window_sizes))
        total_weight = sum(weights)
        
        avg_variety = weighted_sum / total_weight if total_weight > 0 else 0
        max_possible = min(5, len(set(actions)))  # Maximum possible variety in a 5-step window
        
        return 1.0 - (avg_variety / max_possible) if max_possible > 0 else 0.0
        
    def _detect_sequences(self, actions: List[int]) -> Dict[str, Any]:
        """Detect action sequences and patterns."""
        if len(actions) < 3:
            return {'repeated_sequences': [], 'cycle_length': 0}
            
        # Look for repeated sequences of various lengths
        sequences = []
        for length in range(2, min(len(actions) // 2 + 1, 6)):
            for i in range(len(actions) - length * 2 + 1):
                candidate = tuple(actions[i:i+length])
                next_seq = tuple(actions[i+length:i+length*2])
                if candidate == next_seq:
                    sequences.append({
                        'sequence': candidate,
                        'length': length,
                        'position': i
                    })
                    
        # Find the most common cycle length
        cycle_length = 0
        if sequences:
            lengths = [s['length'] for s in sequences]
            cycle_length = statistics.mode(lengths) if lengths else 0
            
        return {
            'repeated_sequences': sequences,
            'cycle_length': cycle_length
        }
        
    def _calculate_stagnation_risk(self, action_patterns: Dict[str, Any],
                                 progression: Dict[str, float],
                                 recent_actions: List[int]) -> Dict[str, float]:
        """Calculate detailed stagnation risk metrics."""
        # Base Stagnation Risk Factors
        risk_factors = {
            'action_repetition': action_patterns['repetition'],
            'low_variety': 1.0 - action_patterns['variety'],
            'pattern_staleness': action_patterns['staleness'],
            'score_stagnation': max(0, 1.0 - progression['score_velocity'])
        }
        
        # Sequence Analysis
        if action_patterns['sequence_patterns']['cycle_length'] > 0:
            risk_factors['cyclic_behavior'] = min(
                action_patterns['sequence_patterns']['cycle_length'] / 10.0,
                1.0
            )
        else:
            risk_factors['cyclic_behavior'] = 0.0
            
        # Progress Stagnation
        if progression['progress_rate'] < 0.1:
            risk_factors['progress_stagnation'] = 1.0
        else:
            risk_factors['progress_stagnation'] = max(0, 1.0 - progression['progress_rate'])
            
        # Historical Performance Decline
        if progression['historical_performance'] < 1.0:
            risk_factors['historical_decline'] = (
                1.0 - progression['historical_performance']
            )
        else:
            risk_factors['historical_decline'] = 0.0
            
        # Calculate Weighted Risk Score
        weights = {
            'action_repetition': 0.25,
            'low_variety': 0.15,
            'pattern_staleness': 0.2,
            'score_stagnation': 0.15,
            'cyclic_behavior': 0.1,
            'progress_stagnation': 0.1,
            'historical_decline': 0.05
        }
        
        total_risk = sum(
            risk * weights[factor]
            for factor, risk in risk_factors.items()
        )
        
        return {
            'total_risk': total_risk,
            'risk_factors': risk_factors
        }
        
    def _analyze_diversity_needs(self, action_patterns: Dict[str, Any],
                               game_phase: str,
                               stagnation_risk: Dict[str, float]) -> Dict[str, float]:
        """Analyze need for action diversity based on multiple factors."""
        diversity_metrics = {}
        
        # Base Diversity Need
        base_need = 0.5  # Start with moderate need
        
        # Phase-based Adjustments
        phase_multipliers = {
            'early_game_strong': 0.6,    # Conservative when doing well early
            'early_game_weak': 0.9,      # More aggressive if struggling early
            'mid_game_strong': 0.7,
            'mid_game_struggling': 1.0,
            'late_game': 0.8,
            'endgame_push': 0.5,         # Conservative in final push
            'endgame_recovery': 1.0      # Aggressive if need recovery
        }
        
        phase_multiplier = phase_multipliers.get(game_phase, 0.8)
        diversity_metrics['phase_adjusted_need'] = base_need * phase_multiplier
        
        # Stagnation-based Adjustments
        stagnation_factor = stagnation_risk['total_risk']
        diversity_metrics['stagnation_driven_need'] = (
            0.4 + (0.6 * stagnation_factor)  # Scale from 0.4 to 1.0
        )
        
        # Pattern-based Adjustments
        pattern_penalty = 0.0
        if action_patterns['sequence_patterns']['cycle_length'] > 0:
            pattern_penalty = min(
                action_patterns['sequence_patterns']['cycle_length'] * 0.1,
                0.3
            )
        
        variety_bonus = action_patterns['variety'] * 0.2  # Reward existing variety
        diversity_metrics['pattern_adjustment'] = pattern_penalty - variety_bonus
        
        # Calculate Final Diversity Need
        weights = {
            'phase_adjusted_need': 0.3,
            'stagnation_driven_need': 0.5,
            'pattern_adjustment': 0.2
        }
        
        total_need = sum(
            metric * weights[name]
            for name, metric in diversity_metrics.items()
        )
        
        diversity_metrics['total_need'] = max(0.0, min(1.0, total_need))
        
        return diversity_metrics
        
    def _generate_recommendations(self, game_phase: str,
                                stagnation_risk: Dict[str, float],
                                diversity_metrics: Dict[str, float],
                                progression: Dict[str, float],
                                success_rate: Optional[float] = None) -> Dict[str, Any]:
        """Generate strategic recommendations based on analysis."""
        recommendations = {
            'force_exploration': False,
            'exploration_type': None,
            'diversity_target': 0.0,
            'strategy_adjustments': [],
            'confidence': 0.0
        }
        
        # 1. Determine if we should force exploration
        stagnation_threshold = self._get_phase_stagnation_threshold(game_phase)
        current_success = success_rate if success_rate is not None else 0.5
        
        if stagnation_risk['total_risk'] > stagnation_threshold:
            recommendations['force_exploration'] = True
            if stagnation_risk['risk_factors']['action_repetition'] > 0.8:
                recommendations['exploration_type'] = 'action_diversity'
            elif stagnation_risk['risk_factors']['progress_stagnation'] > 0.8:
                recommendations['exploration_type'] = 'territory_expansion'
            else:
                recommendations['exploration_type'] = 'balanced'
                
        # 2. Set Diversity Targets
        diversity_target = diversity_metrics['total_need']
        # Adjust based on success rate
        if success_rate is not None:
            if success_rate > self.success_thresholds[self._get_base_phase(game_phase)]:
                diversity_target *= 0.7  # Reduce diversity if doing well
            else:
                diversity_target *= 1.3  # Increase diversity if struggling
                
        recommendations['diversity_target'] = max(0.2, min(1.0, diversity_target))
        
        # 3. Generate Strategy Adjustments
        adjustments = []
        
        if progression['score_velocity'] <= 0:
            adjustments.append({
                'type': 'score_focus',
                'priority': 'high',
                'reason': 'negative_score_velocity'
            })
            
        if stagnation_risk['risk_factors']['cyclic_behavior'] > 0.7:
            adjustments.append({
                'type': 'pattern_break',
                'priority': 'high',
                'reason': 'cyclic_behavior'
            })
            
        if diversity_metrics['total_need'] > 0.8:
            adjustments.append({
                'type': 'forced_diversity',
                'priority': 'medium',
                'reason': 'high_diversity_need'
            })
            
        # Add phase-specific adjustments
        phase_adjustment = self._get_phase_specific_adjustment(game_phase)
        if phase_adjustment:
            adjustments.append(phase_adjustment)
            
        recommendations['strategy_adjustments'] = adjustments
        
        # 4. Calculate Confidence
        confidence_factors = {
            'progression_data': 1.0 if progression['historical_performance'] > 0 else 0.5,
            'stagnation_clarity': 1.0 - (stagnation_risk['total_risk'] % 0.3),  # More confident when risk is clearly high/low
            'phase_certainty': self._calculate_phase_certainty(game_phase, progression)
        }
        
        recommendations['confidence'] = sum(confidence_factors.values()) / len(confidence_factors)
        
        return recommendations
        
    def _get_phase_stagnation_threshold(self, game_phase: str) -> float:
        """Get stagnation threshold based on game phase."""
        thresholds = {
            'early_game_strong': 0.8,    # Very tolerant early if doing well
            'early_game_weak': 0.6,      # Less tolerant if struggling early
            'mid_game_strong': 0.75,
            'mid_game_struggling': 0.5,
            'late_game': 0.7,
            'endgame_push': 0.85,        # Very tolerant during final push
            'endgame_recovery': 0.4      # Very aggressive if in recovery
        }
        return thresholds.get(game_phase, 0.7)
        
    def _get_base_phase(self, detailed_phase: str) -> str:
        """Convert detailed phase to base phase for threshold lookup."""
        if 'early' in detailed_phase:
            return 'early_game'
        elif 'mid' in detailed_phase:
            return 'mid_game'
        return 'late_game'
        
    def _get_phase_specific_adjustment(self, game_phase: str) -> Optional[Dict[str, str]]:
        """Get phase-specific strategy adjustment."""
        adjustments = {
            'early_game_weak': {
                'type': 'conservative_exploration',
                'priority': 'high',
                'reason': 'early_game_establishment'
            },
            'mid_game_struggling': {
                'type': 'aggressive_exploration',
                'priority': 'high',
                'reason': 'mid_game_recovery'
            },
            'endgame_recovery': {
                'type': 'high_risk_exploration',
                'priority': 'high',
                'reason': 'endgame_recovery_needed'
            }
        }
        return adjustments.get(game_phase)
        
    def _calculate_phase_certainty(self, game_phase: str,
                                 progression: Dict[str, float]) -> float:
        """Calculate certainty of game phase detection."""
        # Base certainty on progression metrics stability
        certainty = 0.7  # Start with moderate certainty
        
        # Adjust based on metric clarity
        if progression['completion_estimate'] > 0.9:
            certainty = 0.9  # Very certain of late game
        elif progression['completion_estimate'] < 0.1:
            certainty = 0.9  # Very certain of early game
        else:
            # Mid-game certainty based on metric stability
            metric_stability = abs(progression['score_velocity'])
            certainty = 0.7 + (0.2 * metric_stability)
            
        return min(1.0, certainty)
        
    def _estimate_endgame_proximity(self, current_step: int,
                                  score: float,
                                  historical_data: Dict[str, Any]) -> float:
        """Estimate how close we are to the end game phase."""
        if not historical_data:
            # Use heuristics without historical data
            return min(current_step / 100, 0.99)
            
        try:
            avg_completion_steps = historical_data.get('avg_completion_steps', 100)
            avg_final_score = historical_data.get('avg_final_score', 100)
            
            # Step-based estimation
            step_progress = current_step / avg_completion_steps
            
            # Score-based estimation
            score_progress = score / avg_final_score if avg_final_score > 0 else 0.5
            
            # Combine estimates with weights
            weights = {'step': 0.7, 'score': 0.3}
            combined_estimate = (
                step_progress * weights['step'] +
                score_progress * weights['score']
            )
            
            return min(max(combined_estimate, 0.0), 0.99)
            
        except Exception as e:
            logger.warning(f"Error estimating endgame proximity: {e}")
            return min(current_step / 100, 0.99)

    def _get_historical_data(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Get historical game data from database."""
        if not self.db_interface:
            return None
            
        try:
            query = """
                SELECT 
                    AVG(total_steps) as avg_completion_steps,
                    AVG(final_score) as avg_final_score,
                    AVG(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_rate,
                    GROUP_CONCAT(score_progression) as score_progressions
                FROM game_completion_stats 
                WHERE game_type = ?
                GROUP BY game_type
            """
            results = self.db_interface.execute_query(query, (game_id,))
            
            if results and len(results) > 0:
                result = results[0]
                return {
                    'avg_completion_steps': result['avg_completion_steps'],
                    'avg_final_score': result['avg_final_score'],
                    'success_rate': result['success_rate'],
                    'score_progression': self._parse_score_progression(result['score_progressions'])
                }
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            
        return None
        
    def _parse_score_progression(self, progression_str: str) -> Dict[str, float]:
        """Parse concatenated score progression string into a dict."""
        try:
            if not progression_str:
                return {}
                
            progressions = []
            for prog in progression_str.split(','):
                try:
                    step_scores = json.loads(prog)
                    progressions.append(step_scores)
                except:
                    continue
                    
            # Average scores at each step
            avg_progression = defaultdict(list)
            for prog in progressions:
                for step, score in prog.items():
                    avg_progression[step].append(score)
                    
            return {
                step: sum(scores) / len(scores)
                for step, scores in avg_progression.items()
            }
            
        except Exception as e:
            logger.error(f"Error parsing score progression: {e}")
            return {}
            
    def _update_metrics(self, game_id: str, current_step: int,
                       score: float, recommendations: Dict[str, Any]) -> None:
        """Update internal metrics and history."""
        try:
            # Update stage metrics
            self.stage_metrics[game_id][current_step] = {
                'score': score,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update history
            self.history[game_id].append({
                'step': current_step,
                'score': score,
                'stagnation_risk': recommendations.get('stagnation_risk', 0),
                'diversity_need': recommendations.get('diversity_target', 0),
                'timestamp': datetime.now().isoformat()
            })
            
            # Trim history if too long
            if len(self.history[game_id]) > 1000:
                self.history[game_id] = self.history[game_id][-1000:]
                
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            
    def _get_safe_fallback_analysis(self) -> Dict[str, Any]:
        """Get safe fallback analysis when errors occur."""
        return {
            'game_phase': 'mid_game',
            'stagnation_risk': {'total_risk': 0.5, 'risk_factors': {}},
            'diversity_metrics': {'total_need': 0.5},
            'recommendations': {
                'force_exploration': False,
                'exploration_type': 'balanced',
                'diversity_target': 0.5,
                'strategy_adjustments': [],
                'confidence': 0.3
            },
            'progression_metrics': {
                'completion_estimate': 0.5,
                'action_efficiency': 0.5,
                'score_velocity': 0.0,
                'progress_rate': 0.5,
                'historical_performance': 1.0
            }
        }
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics and performance metrics."""
        stats = {
            'games_analyzed': len(self.stage_metrics),
            'total_analyses': sum(len(stages) for stages in self.stage_metrics.values()),
            'average_stagnation_risk': 0.0,
            'average_diversity_need': 0.0,
            'phase_distributions': defaultdict(int)
        }
        
        total_risk = 0.0
        total_diversity = 0.0
        analyses_count = 0
        
        for game_id, stages in self.stage_metrics.items():
            for step_data in stages.values():
                recommendations = step_data.get('recommendations', {})
                total_risk += recommendations.get('stagnation_risk', {}).get('total_risk', 0.0)
                total_diversity += recommendations.get('diversity_target', 0.0)
                analyses_count += 1
                
                if 'game_phase' in recommendations:
                    stats['phase_distributions'][recommendations['game_phase']] += 1
                    
        if analyses_count > 0:
            stats['average_stagnation_risk'] = total_risk / analyses_count
            stats['average_diversity_need'] = total_diversity / analyses_count
            
        return stats


def create_progression_analyzer(db_interface=None) -> ProgressionAnalyzer:
    """Create a new ProgressionAnalyzer instance.
    
    Args:
        db_interface: Optional database interface for historical data
        
    Returns:
        Configured ProgressionAnalyzer instance
    """
    return ProgressionAnalyzer(db_interface=db_interface)