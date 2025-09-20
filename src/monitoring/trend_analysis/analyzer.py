"""
Trend Analyzer

Analyzes trends and patterns in performance data.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
import time

logger = logging.getLogger(__name__)

class TrendAnalyzer:
    """Analyzes trends and patterns in performance data."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.trend_cache = {}
    
    def analyze_trends(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in performance data."""
        try:
            if not performance_data:
                return {'error': 'No performance data provided'}
            
            # Sort data by timestamp
            sorted_data = sorted(performance_data, key=lambda x: x.get('timestamp', 0))
            
            # Analyze different types of trends
            success_trends = self._analyze_success_trends(sorted_data)
            score_trends = self._analyze_score_trends(sorted_data)
            time_trends = self._analyze_time_trends(sorted_data)
            action_trends = self._analyze_action_trends(sorted_data)
            
            # Generate trend summary
            trend_summary = self._generate_trend_summary(success_trends, score_trends, time_trends, action_trends)
            
            return {
                'success_trends': success_trends,
                'score_trends': score_trends,
                'time_trends': time_trends,
                'action_trends': action_trends,
                'trend_summary': trend_summary,
                'analysis_timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return {'error': str(e)}
    
    def _analyze_success_trends(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze success rate trends."""
        try:
            if not data:
                return {'trend': 'no_data', 'direction': 'unknown'}
            
            # Extract success data
            success_data = [1 if d.get('success', False) else 0 for d in data]
            
            # Calculate rolling success rate
            rolling_success_rates = []
            for i in range(self.window_size, len(success_data) + 1):
                window_data = success_data[i - self.window_size:i]
                rolling_success_rates.append(np.mean(window_data))
            
            if not rolling_success_rates:
                return {'trend': 'insufficient_data', 'direction': 'unknown'}
            
            # Calculate trend direction
            trend_direction = self._calculate_trend_direction(rolling_success_rates)
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(rolling_success_rates)
            
            # Calculate overall success rate
            overall_success_rate = np.mean(success_data)
            
            return {
                'trend': 'success_rate',
                'direction': trend_direction,
                'strength': trend_strength,
                'overall_rate': overall_success_rate,
                'recent_rate': rolling_success_rates[-1] if rolling_success_rates else 0.0,
                'rolling_rates': rolling_success_rates
            }
            
        except Exception as e:
            logger.error(f"Error analyzing success trends: {e}")
            return {'trend': 'error', 'direction': 'unknown'}
    
    def _analyze_score_trends(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze score trends."""
        try:
            if not data:
                return {'trend': 'no_data', 'direction': 'unknown'}
            
            # Extract score data
            scores = [d.get('score', 0) for d in data if d.get('score', 0) > 0]
            
            if not scores:
                return {'trend': 'no_scores', 'direction': 'unknown'}
            
            # Calculate rolling average scores
            rolling_scores = []
            for i in range(self.window_size, len(scores) + 1):
                window_data = scores[i - self.window_size:i]
                rolling_scores.append(np.mean(window_data))
            
            if not rolling_scores:
                return {'trend': 'insufficient_data', 'direction': 'unknown'}
            
            # Calculate trend direction
            trend_direction = self._calculate_trend_direction(rolling_scores)
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(rolling_scores)
            
            # Calculate score statistics
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            min_score = np.min(scores)
            
            return {
                'trend': 'scores',
                'direction': trend_direction,
                'strength': trend_strength,
                'average_score': avg_score,
                'max_score': max_score,
                'min_score': min_score,
                'recent_average': rolling_scores[-1] if rolling_scores else 0.0,
                'rolling_scores': rolling_scores
            }
            
        except Exception as e:
            logger.error(f"Error analyzing score trends: {e}")
            return {'trend': 'error', 'direction': 'unknown'}
    
    def _analyze_time_trends(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze completion time trends."""
        try:
            if not data:
                return {'trend': 'no_data', 'direction': 'unknown'}
            
            # Extract completion time data
            times = [d.get('completion_time', 0) for d in data if d.get('completion_time', 0) > 0]
            
            if not times:
                return {'trend': 'no_times', 'direction': 'unknown'}
            
            # Calculate rolling average times
            rolling_times = []
            for i in range(self.window_size, len(times) + 1):
                window_data = times[i - self.window_size:i]
                rolling_times.append(np.mean(window_data))
            
            if not rolling_times:
                return {'trend': 'insufficient_data', 'direction': 'unknown'}
            
            # Calculate trend direction (inverted - lower times are better)
            trend_direction = self._calculate_trend_direction(rolling_times, inverted=True)
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(rolling_times)
            
            # Calculate time statistics
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            return {
                'trend': 'completion_time',
                'direction': trend_direction,
                'strength': trend_strength,
                'average_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'recent_average': rolling_times[-1] if rolling_times else 0.0,
                'rolling_times': rolling_times
            }
            
        except Exception as e:
            logger.error(f"Error analyzing time trends: {e}")
            return {'trend': 'error', 'direction': 'unknown'}
    
    def _analyze_action_trends(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze action count trends."""
        try:
            if not data:
                return {'trend': 'no_data', 'direction': 'unknown'}
            
            # Extract action count data
            actions = [d.get('actions_taken', 0) for d in data if d.get('actions_taken', 0) > 0]
            
            if not actions:
                return {'trend': 'no_actions', 'direction': 'unknown'}
            
            # Calculate rolling average actions
            rolling_actions = []
            for i in range(self.window_size, len(actions) + 1):
                window_data = actions[i - self.window_size:i]
                rolling_actions.append(np.mean(window_data))
            
            if not rolling_actions:
                return {'trend': 'insufficient_data', 'direction': 'unknown'}
            
            # Calculate trend direction (inverted - fewer actions are better)
            trend_direction = self._calculate_trend_direction(rolling_actions, inverted=True)
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(rolling_actions)
            
            # Calculate action statistics
            avg_actions = np.mean(actions)
            min_actions = np.min(actions)
            max_actions = np.max(actions)
            
            return {
                'trend': 'action_count',
                'direction': trend_direction,
                'strength': trend_strength,
                'average_actions': avg_actions,
                'min_actions': min_actions,
                'max_actions': max_actions,
                'recent_average': rolling_actions[-1] if rolling_actions else 0.0,
                'rolling_actions': rolling_actions
            }
            
        except Exception as e:
            logger.error(f"Error analyzing action trends: {e}")
            return {'trend': 'error', 'direction': 'unknown'}
    
    def _calculate_trend_direction(self, values: List[float], inverted: bool = False) -> str:
        """Calculate trend direction from a list of values."""
        try:
            if len(values) < 2:
                return 'insufficient_data'
            
            # Simple linear regression slope
            x = np.arange(len(values))
            y = np.array(values)
            
            # Calculate slope
            n = len(values)
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_x2 = np.sum(x * x)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # Apply inversion if needed
            if inverted:
                slope = -slope
            
            if slope > 0.01:
                return 'improving'
            elif slope < -0.01:
                return 'declining'
            else:
                return 'stable'
                
        except Exception as e:
            logger.error(f"Error calculating trend direction: {e}")
            return 'unknown'
    
    def _calculate_trend_strength(self, values: List[float]) -> float:
        """Calculate trend strength from a list of values."""
        try:
            if len(values) < 2:
                return 0.0
            
            # Calculate correlation with time
            x = np.arange(len(values))
            y = np.array(values)
            
            correlation = np.corrcoef(x, y)[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.0
    
    def _generate_trend_summary(self, success_trends: Dict[str, Any], score_trends: Dict[str, Any], 
                               time_trends: Dict[str, Any], action_trends: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall trend summary."""
        try:
            # Count improving trends
            improving_trends = 0
            declining_trends = 0
            stable_trends = 0
            
            for trends in [success_trends, score_trends, time_trends, action_trends]:
                direction = trends.get('direction', 'unknown')
                if direction == 'improving':
                    improving_trends += 1
                elif direction == 'declining':
                    declining_trends += 1
                elif direction == 'stable':
                    stable_trends += 1
            
            # Determine overall trend
            if improving_trends > declining_trends:
                overall_trend = 'improving'
            elif declining_trends > improving_trends:
                overall_trend = 'declining'
            else:
                overall_trend = 'stable'
            
            # Calculate overall trend strength
            trend_strengths = []
            for trends in [success_trends, score_trends, time_trends, action_trends]:
                strength = trends.get('strength', 0.0)
                if strength > 0:
                    trend_strengths.append(strength)
            
            overall_strength = np.mean(trend_strengths) if trend_strengths else 0.0
            
            return {
                'overall_trend': overall_trend,
                'overall_strength': overall_strength,
                'improving_trends': improving_trends,
                'declining_trends': declining_trends,
                'stable_trends': stable_trends,
                'trend_confidence': min(1.0, overall_strength * 2)  # Scale to 0-1
            }
            
        except Exception as e:
            logger.error(f"Error generating trend summary: {e}")
            return {'overall_trend': 'unknown', 'overall_strength': 0.0}
    
    def predict_future_performance(self, performance_data: List[Dict[str, Any]], 
                                 days_ahead: int = 7) -> Dict[str, Any]:
        """Predict future performance based on trends."""
        try:
            if not performance_data or len(performance_data) < 10:
                return {'error': 'Insufficient data for prediction'}
            
            # Analyze current trends
            trends = self.analyze_trends(performance_data)
            
            # Extract recent performance
            recent_data = performance_data[-10:]  # Last 10 data points
            recent_success_rate = np.mean([1 if d.get('success', False) else 0 for d in recent_data])
            recent_avg_score = np.mean([d.get('score', 0) for d in recent_data if d.get('score', 0) > 0])
            
            # Simple linear projection
            success_trends = trends.get('success_trends', {})
            score_trends = trends.get('score_trends', {})
            
            # Project success rate
            success_direction = success_trends.get('direction', 'stable')
            success_strength = success_trends.get('strength', 0.0)
            
            if success_direction == 'improving':
                predicted_success_rate = min(1.0, recent_success_rate + (success_strength * 0.1))
            elif success_direction == 'declining':
                predicted_success_rate = max(0.0, recent_success_rate - (success_strength * 0.1))
            else:
                predicted_success_rate = recent_success_rate
            
            # Project average score
            score_direction = score_trends.get('direction', 'stable')
            score_strength = score_trends.get('strength', 0.0)
            
            if score_direction == 'improving':
                predicted_avg_score = recent_avg_score * (1 + score_strength * 0.1)
            elif score_direction == 'declining':
                predicted_avg_score = recent_avg_score * (1 - score_strength * 0.1)
            else:
                predicted_avg_score = recent_avg_score
            
            return {
                'predicted_success_rate': predicted_success_rate,
                'predicted_avg_score': predicted_avg_score,
                'prediction_confidence': min(1.0, (success_strength + score_strength) / 2),
                'days_ahead': days_ahead,
                'based_on_trends': {
                    'success_trend': success_direction,
                    'score_trend': score_direction
                }
            }
            
        except Exception as e:
            logger.error(f"Error predicting future performance: {e}")
            return {'error': str(e)}
