"""
Performance Tracker

Tracks performance metrics and trends in action traces.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
import numpy as np
import time

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Tracks performance metrics and trends in action traces."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.game_metrics = defaultdict(lambda: {'traces': 0, 'successes': 0, 'total_reward': 0.0})
        self.action_metrics = defaultdict(lambda: {'count': 0, 'successes': 0, 'total_reward': 0.0})
    
    def track_performance(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Track performance metrics from traces."""
        try:
            if not traces:
                return {'error': 'No traces provided'}
            
            # Calculate overall metrics
            overall_metrics = self._calculate_overall_metrics(traces)
            
            # Calculate game-specific metrics
            game_metrics = self._calculate_game_metrics(traces)
            
            # Calculate action-specific metrics
            action_metrics = self._calculate_action_metrics(traces)
            
            # Calculate trends
            trends = self._calculate_trends(traces)
            
            # Update history
            self._update_history(overall_metrics)
            
            return {
                'overall_metrics': overall_metrics,
                'game_metrics': game_metrics,
                'action_metrics': action_metrics,
                'trends': trends,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error tracking performance: {e}")
            return {'error': str(e)}
    
    def _calculate_overall_metrics(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall performance metrics."""
        try:
            total_traces = len(traces)
            successful_traces = sum(1 for t in traces if t.get('success', False))
            success_rate = successful_traces / max(total_traces, 1)
            
            rewards = [t.get('reward', 0.0) for t in traces if t.get('reward') is not None]
            avg_reward = np.mean(rewards) if rewards else 0.0
            total_reward = np.sum(rewards) if rewards else 0.0
            
            # Calculate action statistics
            all_actions = []
            for trace in traces:
                actions = trace.get('actions', [])
                all_actions.extend(actions)
            
            unique_actions = len(set(all_actions))
            avg_actions_per_trace = len(all_actions) / max(total_traces, 1)
            
            # Calculate trace length statistics
            trace_lengths = [len(t.get('actions', [])) for t in traces]
            avg_trace_length = np.mean(trace_lengths) if trace_lengths else 0.0
            max_trace_length = np.max(trace_lengths) if trace_lengths else 0.0
            min_trace_length = np.min(trace_lengths) if trace_lengths else 0.0
            
            return {
                'total_traces': total_traces,
                'successful_traces': successful_traces,
                'success_rate': success_rate,
                'average_reward': avg_reward,
                'total_reward': total_reward,
                'unique_actions': unique_actions,
                'average_actions_per_trace': avg_actions_per_trace,
                'average_trace_length': avg_trace_length,
                'max_trace_length': max_trace_length,
                'min_trace_length': min_trace_length
            }
            
        except Exception as e:
            logger.error(f"Error calculating overall metrics: {e}")
            return {}
    
    def _calculate_game_metrics(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate game-specific performance metrics."""
        try:
            game_stats = defaultdict(lambda: {
                'traces': 0, 'successes': 0, 'total_reward': 0.0, 
                'rewards': [], 'trace_lengths': []
            })
            
            for trace in traces:
                game_id = trace.get('game_id', 'unknown')
                success = trace.get('success', False)
                reward = trace.get('reward', 0.0)
                actions = trace.get('actions', [])
                
                game_stats[game_id]['traces'] += 1
                if success:
                    game_stats[game_id]['successes'] += 1
                game_stats[game_id]['total_reward'] += reward
                game_stats[game_id]['rewards'].append(reward)
                game_stats[game_id]['trace_lengths'].append(len(actions))
            
            # Calculate final metrics for each game
            game_metrics = {}
            for game_id, stats in game_stats.items():
                if stats['traces'] > 0:
                    success_rate = stats['successes'] / stats['traces']
                    avg_reward = np.mean(stats['rewards']) if stats['rewards'] else 0.0
                    avg_length = np.mean(stats['trace_lengths']) if stats['trace_lengths'] else 0.0
                    
                    game_metrics[game_id] = {
                        'total_traces': stats['traces'],
                        'success_rate': success_rate,
                        'average_reward': avg_reward,
                        'total_reward': stats['total_reward'],
                        'average_trace_length': avg_length,
                        'performance_score': success_rate * avg_reward
                    }
            
            return game_metrics
            
        except Exception as e:
            logger.error(f"Error calculating game metrics: {e}")
            return {}
    
    def _calculate_action_metrics(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate action-specific performance metrics."""
        try:
            action_stats = defaultdict(lambda: {
                'count': 0, 'successes': 0, 'total_reward': 0.0, 
                'rewards': [], 'contexts': []
            })
            
            for trace in traces:
                actions = trace.get('actions', [])
                success = trace.get('success', False)
                reward = trace.get('reward', 0.0)
                
                for i, action in enumerate(actions):
                    action_stats[action]['count'] += 1
                    if success:
                        action_stats[action]['successes'] += 1
                    action_stats[action]['total_reward'] += reward
                    action_stats[action]['rewards'].append(reward)
                    action_stats[action]['contexts'].append({
                        'position': i,
                        'trace_length': len(actions),
                        'is_last_action': i == len(actions) - 1
                    })
            
            # Calculate final metrics for each action
            action_metrics = {}
            for action, stats in action_stats.items():
                if stats['count'] > 0:
                    success_rate = stats['successes'] / stats['count']
                    avg_reward = np.mean(stats['rewards']) if stats['rewards'] else 0.0
                    
                    # Calculate position statistics
                    positions = [ctx['position'] for ctx in stats['contexts']]
                    avg_position = np.mean(positions) if positions else 0.0
                    last_action_rate = sum(1 for ctx in stats['contexts'] if ctx['is_last_action']) / stats['count']
                    
                    action_metrics[action] = {
                        'total_count': stats['count'],
                        'success_rate': success_rate,
                        'average_reward': avg_reward,
                        'total_reward': stats['total_reward'],
                        'average_position': avg_position,
                        'last_action_rate': last_action_rate,
                        'effectiveness_score': success_rate * avg_reward
                    }
            
            return action_metrics
            
        except Exception as e:
            logger.error(f"Error calculating action metrics: {e}")
            return {}
    
    def _calculate_trends(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance trends over time."""
        try:
            if not traces:
                return {'trends': 'no_data'}
            
            # Sort traces by timestamp if available
            sorted_traces = sorted(traces, key=lambda t: t.get('timestamp', 0))
            
            if len(sorted_traces) < 10:
                return {'trends': 'insufficient_data'}
            
            # Calculate rolling metrics
            window_size = min(20, len(sorted_traces) // 5)
            success_rates = []
            avg_rewards = []
            
            for i in range(window_size, len(sorted_traces) + 1):
                window_traces = sorted_traces[i-window_size:i]
                window_successes = sum(1 for t in window_traces if t.get('success', False))
                window_success_rate = window_successes / len(window_traces)
                success_rates.append(window_success_rate)
                
                window_rewards = [t.get('reward', 0.0) for t in window_traces if t.get('reward') is not None]
                window_avg_reward = np.mean(window_rewards) if window_rewards else 0.0
                avg_rewards.append(window_avg_reward)
            
            # Calculate trend directions
            success_trend = self._calculate_trend_direction(success_rates)
            reward_trend = self._calculate_trend_direction(avg_rewards)
            
            # Calculate trend strength
            success_trend_strength = self._calculate_trend_strength(success_rates)
            reward_trend_strength = self._calculate_trend_strength(avg_rewards)
            
            return {
                'success_rate_trend': success_trend,
                'reward_trend': reward_trend,
                'success_trend_strength': success_trend_strength,
                'reward_trend_strength': reward_trend_strength,
                'recent_success_rate': success_rates[-1] if success_rates else 0.0,
                'recent_avg_reward': avg_rewards[-1] if avg_rewards else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating trends: {e}")
            return {'trends': 'error'}
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values."""
        try:
            if len(values) < 2:
                return 'stable'
            
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
    
    def _update_history(self, metrics: Dict[str, Any]):
        """Update metrics history."""
        try:
            self.metrics_history.append({
                'timestamp': time.time(),
                'metrics': metrics
            })
        except Exception as e:
            logger.error(f"Error updating history: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from history."""
        try:
            if not self.metrics_history:
                return {'error': 'No performance data available'}
            
            # Get recent metrics
            recent_metrics = self.metrics_history[-1]['metrics']
            
            # Calculate historical trends
            success_rates = [h['metrics'].get('success_rate', 0.0) for h in self.metrics_history]
            avg_rewards = [h['metrics'].get('average_reward', 0.0) for h in self.metrics_history]
            
            return {
                'current_metrics': recent_metrics,
                'historical_success_rate': {
                    'current': success_rates[-1] if success_rates else 0.0,
                    'average': np.mean(success_rates) if success_rates else 0.0,
                    'trend': self._calculate_trend_direction(success_rates)
                },
                'historical_reward': {
                    'current': avg_rewards[-1] if avg_rewards else 0.0,
                    'average': np.mean(avg_rewards) if avg_rewards else 0.0,
                    'trend': self._calculate_trend_direction(avg_rewards)
                },
                'data_points': len(self.metrics_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {'error': str(e)}
