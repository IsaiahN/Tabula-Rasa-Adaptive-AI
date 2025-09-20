"""
Performance Tracker

Tracks performance metrics and trends in scorecard data.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Tracks performance metrics and trends in scorecard data."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.scorecard_dir = os.path.join(data_dir, "scorecards")
        self.sessions_dir = os.path.join(data_dir, "sessions")
        
        # Ensure directories exist
        os.makedirs(self.scorecard_dir, exist_ok=True)
        os.makedirs(self.sessions_dir, exist_ok=True)
        
        # Performance tracking
        self.performance_history = []
        self.game_success_patterns = {}
        self.level_completion_tracking = {}
    
    def track_performance(self, scorecard_data: Dict[str, Any]) -> Dict[str, Any]:
        """Track performance from scorecard data."""
        try:
            # Extract performance metrics
            metrics = self._extract_performance_metrics(scorecard_data)
            
            # Update performance history
            self._update_performance_history(metrics)
            
            # Update game-specific patterns
            self._update_game_patterns(scorecard_data)
            
            # Update level completion tracking
            self._update_level_completion(scorecard_data)
            
            # Generate performance summary
            summary = self._generate_performance_summary()
            
            return {
                'metrics': metrics,
                'summary': summary,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error tracking performance: {e}")
            return {'error': str(e)}
    
    def _extract_performance_metrics(self, scorecard_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance metrics from scorecard data."""
        try:
            metrics = {
                'game_id': scorecard_data.get('game_id', 'unknown'),
                'level': scorecard_data.get('level', 0),
                'score': scorecard_data.get('score', 0),
                'max_score': scorecard_data.get('max_score', 0),
                'completion_time': scorecard_data.get('completion_time', 0),
                'actions_taken': scorecard_data.get('actions_taken', 0),
                'success': scorecard_data.get('success', False),
                'timestamp': scorecard_data.get('timestamp', time.time())
            }
            
            # Calculate derived metrics
            if metrics['max_score'] > 0:
                metrics['score_ratio'] = metrics['score'] / metrics['max_score']
            else:
                metrics['score_ratio'] = 0.0
            
            if metrics['completion_time'] > 0:
                metrics['actions_per_second'] = metrics['actions_taken'] / metrics['completion_time']
            else:
                metrics['actions_per_second'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting performance metrics: {e}")
            return {}
    
    def _update_performance_history(self, metrics: Dict[str, Any]):
        """Update performance history with new metrics."""
        try:
            self.performance_history.append(metrics)
            
            # Keep only recent history (last 1000 entries)
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-500:]
                
        except Exception as e:
            logger.error(f"Error updating performance history: {e}")
    
    def _update_game_patterns(self, scorecard_data: Dict[str, Any]):
        """Update game-specific success patterns."""
        try:
            game_id = scorecard_data.get('game_id', 'unknown')
            success = scorecard_data.get('success', False)
            level = scorecard_data.get('level', 0)
            
            if game_id not in self.game_success_patterns:
                self.game_success_patterns[game_id] = {
                    'total_attempts': 0,
                    'successful_attempts': 0,
                    'levels_completed': set(),
                    'recent_performance': []
                }
            
            pattern = self.game_success_patterns[game_id]
            pattern['total_attempts'] += 1
            
            if success:
                pattern['successful_attempts'] += 1
                pattern['levels_completed'].add(level)
            
            # Update recent performance (last 10 attempts)
            pattern['recent_performance'].append(success)
            if len(pattern['recent_performance']) > 10:
                pattern['recent_performance'] = pattern['recent_performance'][-10:]
                
        except Exception as e:
            logger.error(f"Error updating game patterns: {e}")
    
    def _update_level_completion(self, scorecard_data: Dict[str, Any]):
        """Update level completion tracking."""
        try:
            level = scorecard_data.get('level', 0)
            success = scorecard_data.get('success', False)
            
            if level not in self.level_completion_tracking:
                self.level_completion_tracking[level] = {
                    'total_attempts': 0,
                    'successful_attempts': 0,
                    'average_score': 0.0,
                    'completion_times': []
                }
            
            tracking = self.level_completion_tracking[level]
            tracking['total_attempts'] += 1
            
            if success:
                tracking['successful_attempts'] += 1
                completion_time = scorecard_data.get('completion_time', 0)
                if completion_time > 0:
                    tracking['completion_times'].append(completion_time)
                    if len(tracking['completion_times']) > 50:
                        tracking['completion_times'] = tracking['completion_times'][-25:]
            
            # Update average score
            score = scorecard_data.get('score', 0)
            if score > 0:
                current_avg = tracking['average_score']
                total_attempts = tracking['total_attempts']
                tracking['average_score'] = ((current_avg * (total_attempts - 1)) + score) / total_attempts
                
        except Exception as e:
            logger.error(f"Error updating level completion: {e}")
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary from tracked data."""
        try:
            if not self.performance_history:
                return {'error': 'No performance data available'}
            
            # Calculate overall metrics
            total_attempts = len(self.performance_history)
            successful_attempts = sum(1 for m in self.performance_history if m.get('success', False))
            success_rate = successful_attempts / max(total_attempts, 1)
            
            # Calculate score metrics
            scores = [m.get('score', 0) for m in self.performance_history if m.get('score', 0) > 0]
            avg_score = np.mean(scores) if scores else 0.0
            max_score = np.max(scores) if scores else 0.0
            
            # Calculate time metrics
            completion_times = [m.get('completion_time', 0) for m in self.performance_history if m.get('completion_time', 0) > 0]
            avg_completion_time = np.mean(completion_times) if completion_times else 0.0
            
            # Calculate action metrics
            actions = [m.get('actions_taken', 0) for m in self.performance_history if m.get('actions_taken', 0) > 0]
            avg_actions = np.mean(actions) if actions else 0.0
            
            # Game-specific metrics
            game_metrics = {}
            for game_id, pattern in self.game_success_patterns.items():
                game_success_rate = pattern['successful_attempts'] / max(pattern['total_attempts'], 1)
                recent_success_rate = np.mean(pattern['recent_performance']) if pattern['recent_performance'] else 0.0
                
                game_metrics[game_id] = {
                    'total_attempts': pattern['total_attempts'],
                    'success_rate': game_success_rate,
                    'recent_success_rate': recent_success_rate,
                    'levels_completed': len(pattern['levels_completed']),
                    'max_level': max(pattern['levels_completed']) if pattern['levels_completed'] else 0
                }
            
            # Level-specific metrics
            level_metrics = {}
            for level, tracking in self.level_completion_tracking.items():
                level_success_rate = tracking['successful_attempts'] / max(tracking['total_attempts'], 1)
                avg_completion_time = np.mean(tracking['completion_times']) if tracking['completion_times'] else 0.0
                
                level_metrics[level] = {
                    'total_attempts': tracking['total_attempts'],
                    'success_rate': level_success_rate,
                    'average_score': tracking['average_score'],
                    'average_completion_time': avg_completion_time
                }
            
            return {
                'overall_metrics': {
                    'total_attempts': total_attempts,
                    'success_rate': success_rate,
                    'average_score': avg_score,
                    'max_score': max_score,
                    'average_completion_time': avg_completion_time,
                    'average_actions': avg_actions
                },
                'game_metrics': game_metrics,
                'level_metrics': level_metrics,
                'data_points': total_attempts
            }
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            return {'error': str(e)}
    
    def get_performance_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get performance trends over the specified number of days."""
        try:
            cutoff_time = time.time() - (days * 24 * 60 * 60)
            recent_metrics = [m for m in self.performance_history if m.get('timestamp', 0) > cutoff_time]
            
            if not recent_metrics:
                return {'error': f'No data available for the last {days} days'}
            
            # Calculate daily trends
            daily_trends = defaultdict(list)
            for metric in recent_metrics:
                day = int(metric.get('timestamp', 0) // (24 * 60 * 60))
                daily_trends[day].append(metric)
            
            # Calculate daily success rates
            daily_success_rates = {}
            for day, metrics in daily_trends.items():
                if metrics:
                    success_count = sum(1 for m in metrics if m.get('success', False))
                    daily_success_rates[day] = success_count / len(metrics)
            
            # Calculate trend direction
            if len(daily_success_rates) >= 2:
                days_list = sorted(daily_success_rates.keys())
                success_rates = [daily_success_rates[d] for d in days_list]
                
                # Simple linear trend
                x = np.arange(len(success_rates))
                y = np.array(success_rates)
                if len(y) > 1:
                    slope = np.polyfit(x, y, 1)[0]
                    if slope > 0.01:
                        trend_direction = 'improving'
                    elif slope < -0.01:
                        trend_direction = 'declining'
                    else:
                        trend_direction = 'stable'
                else:
                    trend_direction = 'insufficient_data'
            else:
                trend_direction = 'insufficient_data'
            
            return {
                'trend_direction': trend_direction,
                'daily_success_rates': daily_success_rates,
                'total_days': len(daily_success_rates),
                'recent_performance': {
                    'success_rate': np.mean(list(daily_success_rates.values())) if daily_success_rates else 0.0,
                    'total_attempts': len(recent_metrics)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting performance trends: {e}")
            return {'error': str(e)}
    
    def get_game_performance(self, game_id: str) -> Dict[str, Any]:
        """Get performance metrics for a specific game."""
        try:
            if game_id not in self.game_success_patterns:
                return {'error': f'No data available for game {game_id}'}
            
            pattern = self.game_success_patterns[game_id]
            game_metrics = [m for m in self.performance_history if m.get('game_id') == game_id]
            
            if not game_metrics:
                return {'error': f'No performance data available for game {game_id}'}
            
            # Calculate game-specific metrics
            success_rate = pattern['successful_attempts'] / max(pattern['total_attempts'], 1)
            recent_success_rate = np.mean(pattern['recent_performance']) if pattern['recent_performance'] else 0.0
            
            scores = [m.get('score', 0) for m in game_metrics if m.get('score', 0) > 0]
            avg_score = np.mean(scores) if scores else 0.0
            max_score = np.max(scores) if scores else 0.0
            
            completion_times = [m.get('completion_time', 0) for m in game_metrics if m.get('completion_time', 0) > 0]
            avg_completion_time = np.mean(completion_times) if completion_times else 0.0
            
            return {
                'game_id': game_id,
                'total_attempts': pattern['total_attempts'],
                'success_rate': success_rate,
                'recent_success_rate': recent_success_rate,
                'levels_completed': len(pattern['levels_completed']),
                'max_level': max(pattern['levels_completed']) if pattern['levels_completed'] else 0,
                'average_score': avg_score,
                'max_score': max_score,
                'average_completion_time': avg_completion_time,
                'performance_trend': 'improving' if recent_success_rate > success_rate else 'stable'
            }
            
        except Exception as e:
            logger.error(f"Error getting game performance: {e}")
            return {'error': str(e)}
