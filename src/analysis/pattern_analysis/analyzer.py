"""
Pattern Analyzer

Analyzes patterns in action traces to identify successful sequences.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np

logger = logging.getLogger(__name__)

class PatternAnalyzer:
    """Analyzes patterns in action traces to identify successful sequences."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.action_traces_file = os.path.join(data_dir, "action_traces.ndjson")
        self.sessions_dir = os.path.join(data_dir, "sessions")
        
        # Pattern storage
        self.successful_sequences = defaultdict(list)
        self.action_effectiveness = defaultdict(list)
        self.coordinate_patterns = defaultdict(list)
        self.game_specific_patterns = defaultdict(lambda: defaultdict(list))
    
    def analyze_patterns(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in action traces."""
        try:
            if not traces:
                logger.warning("No traces provided for pattern analysis")
                return self._create_empty_analysis()
            
            # Analyze different types of patterns
            analysis = {
                'total_traces': len(traces),
                'analysis_timestamp': time.time(),
                'successful_sequences': self._find_successful_sequences(traces),
                'action_effectiveness': self._analyze_action_effectiveness(traces),
                'coordinate_patterns': self._analyze_coordinate_patterns(traces),
                'game_specific_patterns': self._analyze_game_specific_patterns(traces)
            }
            
            logger.info(f"Pattern analysis completed for {len(traces)} traces")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            return self._create_empty_analysis()
    
    def _find_successful_sequences(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find successful action sequences from traces."""
        try:
            successful_sequences = {
                'common_sequences': [],
                'high_success_sequences': [],
                'sequence_lengths': {},
                'action_transitions': defaultdict(int)
            }
            
            # Group traces by success
            successful_traces = [t for t in traces if t.get('success', False)]
            failed_traces = [t for t in traces if not t.get('success', False)]
            
            if not successful_traces:
                return successful_sequences
            
            # Extract sequences from successful traces
            all_sequences = []
            for trace in successful_traces:
                actions = trace.get('actions', [])
                if len(actions) >= 2:
                    # Extract sequences of different lengths
                    for length in range(2, min(len(actions) + 1, 6)):
                        for i in range(len(actions) - length + 1):
                            sequence = actions[i:i + length]
                            all_sequences.append(sequence)
            
            # Count sequence frequencies
            sequence_counts = Counter(tuple(seq) for seq in all_sequences)
            
            # Find common sequences
            common_sequences = sequence_counts.most_common(10)
            successful_sequences['common_sequences'] = [
                {'sequence': list(seq), 'count': count, 'frequency': count / len(all_sequences)}
                for seq, count in common_sequences
            ]
            
            # Find high-success sequences
            high_success_sequences = []
            for seq, count in sequence_counts.items():
                if count >= 3:  # At least 3 occurrences
                    success_rate = self._calculate_sequence_success_rate(seq, traces)
                    if success_rate > 0.7:  # High success rate
                        high_success_sequences.append({
                            'sequence': list(seq),
                            'count': count,
                            'success_rate': success_rate
                        })
            
            successful_sequences['high_success_sequences'] = sorted(
                high_success_sequences, 
                key=lambda x: x['success_rate'], 
                reverse=True
            )[:10]
            
            # Analyze sequence lengths
            sequence_lengths = Counter(len(seq) for seq in all_sequences)
            successful_sequences['sequence_lengths'] = dict(sequence_lengths)
            
            # Analyze action transitions
            for trace in successful_traces:
                actions = trace.get('actions', [])
                for i in range(len(actions) - 1):
                    transition = (actions[i], actions[i + 1])
                    successful_sequences['action_transitions'][transition] += 1
            
            return successful_sequences
            
        except Exception as e:
            logger.error(f"Error finding successful sequences: {e}")
            return {'common_sequences': [], 'high_success_sequences': [], 'sequence_lengths': {}, 'action_transitions': {}}
    
    def _analyze_action_effectiveness(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze effectiveness of individual actions."""
        try:
            action_stats = defaultdict(lambda: {'total': 0, 'successful': 0, 'rewards': []})
            
            for trace in traces:
                actions = trace.get('actions', [])
                success = trace.get('success', False)
                reward = trace.get('reward', 0.0)
                
                for action in actions:
                    action_stats[action]['total'] += 1
                    if success:
                        action_stats[action]['successful'] += 1
                    action_stats[action]['rewards'].append(reward)
            
            # Calculate effectiveness metrics
            effectiveness = {}
            for action, stats in action_stats.items():
                success_rate = stats['successful'] / max(stats['total'], 1)
                avg_reward = np.mean(stats['rewards']) if stats['rewards'] else 0.0
                total_usage = stats['total']
                
                effectiveness[action] = {
                    'success_rate': success_rate,
                    'average_reward': avg_reward,
                    'total_usage': total_usage,
                    'effectiveness_score': success_rate * avg_reward
                }
            
            # Sort by effectiveness score
            sorted_actions = sorted(
                effectiveness.items(), 
                key=lambda x: x[1]['effectiveness_score'], 
                reverse=True
            )
            
            return {
                'action_effectiveness': dict(sorted_actions),
                'most_effective_action': sorted_actions[0][0] if sorted_actions else 'none',
                'least_effective_action': sorted_actions[-1][0] if sorted_actions else 'none'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing action effectiveness: {e}")
            return {'action_effectiveness': {}, 'most_effective_action': 'none', 'least_effective_action': 'none'}
    
    def _analyze_coordinate_patterns(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze coordinate patterns in action traces."""
        try:
            coordinate_data = []
            
            for trace in traces:
                actions = trace.get('actions', [])
                coordinates = trace.get('coordinates', [])
                success = trace.get('success', False)
                
                if len(actions) == len(coordinates):
                    for action, coord in zip(actions, coordinates):
                        coordinate_data.append({
                            'action': action,
                            'x': coord.get('x', 0),
                            'y': coord.get('y', 0),
                            'success': success
                        })
            
            if not coordinate_data:
                return {'coordinate_clusters': [], 'coordinate_effectiveness': {}}
            
            # Analyze coordinate clusters
            coordinate_clusters = self._find_coordinate_clusters(coordinate_data)
            
            # Analyze coordinate effectiveness
            coordinate_effectiveness = self._analyze_coordinate_effectiveness(coordinate_data)
            
            return {
                'coordinate_clusters': coordinate_clusters,
                'coordinate_effectiveness': coordinate_effectiveness,
                'total_coordinate_actions': len(coordinate_data)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing coordinate patterns: {e}")
            return {'coordinate_clusters': [], 'coordinate_effectiveness': {}, 'total_coordinate_actions': 0}
    
    def _analyze_game_specific_patterns(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze game-specific patterns."""
        try:
            game_patterns = defaultdict(lambda: {
                'total_traces': 0,
                'successful_traces': 0,
                'common_actions': Counter(),
                'successful_actions': Counter(),
                'average_reward': 0.0
            })
            
            for trace in traces:
                game_id = trace.get('game_id', 'unknown')
                success = trace.get('success', False)
                reward = trace.get('reward', 0.0)
                actions = trace.get('actions', [])
                
                game_patterns[game_id]['total_traces'] += 1
                if success:
                    game_patterns[game_id]['successful_traces'] += 1
                
                for action in actions:
                    game_patterns[game_id]['common_actions'][action] += 1
                    if success:
                        game_patterns[game_id]['successful_actions'][action] += 1
                
                game_patterns[game_id]['average_reward'] += reward
            
            # Calculate final metrics
            for game_id, patterns in game_patterns.items():
                if patterns['total_traces'] > 0:
                    patterns['success_rate'] = patterns['successful_traces'] / patterns['total_traces']
                    patterns['average_reward'] /= patterns['total_traces']
                    
                    # Find most effective actions for this game
                    action_effectiveness = {}
                    for action in patterns['common_actions']:
                        total_usage = patterns['common_actions'][action]
                        successful_usage = patterns['successful_actions'][action]
                        effectiveness = successful_usage / max(total_usage, 1)
                        action_effectiveness[action] = effectiveness
                    
                    patterns['most_effective_actions'] = sorted(
                        action_effectiveness.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:5]
            
            return dict(game_patterns)
            
        except Exception as e:
            logger.error(f"Error analyzing game-specific patterns: {e}")
            return {}
    
    def _calculate_sequence_success_rate(self, sequence: Tuple, traces: List[Dict[str, Any]]) -> float:
        """Calculate success rate for a specific sequence."""
        try:
            sequence_list = list(sequence)
            total_occurrences = 0
            successful_occurrences = 0
            
            for trace in traces:
                actions = trace.get('actions', [])
                success = trace.get('success', False)
                
                # Check if sequence appears in this trace
                for i in range(len(actions) - len(sequence_list) + 1):
                    if actions[i:i + len(sequence_list)] == sequence_list:
                        total_occurrences += 1
                        if success:
                            successful_occurrences += 1
                        break  # Only count once per trace
            
            return successful_occurrences / max(total_occurrences, 1)
            
        except Exception as e:
            logger.error(f"Error calculating sequence success rate: {e}")
            return 0.0
    
    def _find_coordinate_clusters(self, coordinate_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find coordinate clusters using simple clustering."""
        try:
            if not coordinate_data:
                return []
            
            # Simple grid-based clustering
            grid_size = 10
            clusters = defaultdict(list)
            
            for data in coordinate_data:
                x, y = data['x'], data['y']
                cluster_x = x // grid_size
                cluster_y = y // grid_size
                cluster_key = (cluster_x, cluster_y)
                clusters[cluster_key].append(data)
            
            # Convert to cluster information
            cluster_info = []
            for (cluster_x, cluster_y), points in clusters.items():
                if len(points) >= 3:  # Only include clusters with at least 3 points
                    success_rate = sum(1 for p in points if p['success']) / len(points)
                    avg_reward = np.mean([p.get('reward', 0) for p in points])
                    
                    cluster_info.append({
                        'cluster_id': f"({cluster_x}, {cluster_y})",
                        'center': (cluster_x * grid_size + grid_size // 2, cluster_y * grid_size + grid_size // 2),
                        'point_count': len(points),
                        'success_rate': success_rate,
                        'average_reward': avg_reward,
                        'actions': list(set(p['action'] for p in points))
                    })
            
            return sorted(cluster_info, key=lambda x: x['point_count'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error finding coordinate clusters: {e}")
            return []
    
    def _analyze_coordinate_effectiveness(self, coordinate_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze effectiveness of actions at different coordinates."""
        try:
            if not coordinate_data:
                return {}
            
            # Group by action type
            action_coords = defaultdict(list)
            for data in coordinate_data:
                action_coords[data['action']].append(data)
            
            effectiveness = {}
            for action, coords in action_coords.items():
                if len(coords) < 3:  # Need at least 3 data points
                    continue
                
                # Calculate success rate and average reward
                success_rate = sum(1 for c in coords if c['success']) / len(coords)
                avg_reward = np.mean([c.get('reward', 0) for c in coords])
                
                # Find most effective coordinate ranges
                x_coords = [c['x'] for c in coords]
                y_coords = [c['y'] for c in coords]
                
                effectiveness[action] = {
                    'success_rate': success_rate,
                    'average_reward': avg_reward,
                    'coordinate_range': {
                        'x_min': min(x_coords),
                        'x_max': max(x_coords),
                        'y_min': min(y_coords),
                        'y_max': max(y_coords)
                    },
                    'coordinate_center': (np.mean(x_coords), np.mean(y_coords))
                }
            
            return effectiveness
            
        except Exception as e:
            logger.error(f"Error analyzing coordinate effectiveness: {e}")
            return {}
    
    def _create_empty_analysis(self) -> Dict[str, Any]:
        """Create empty analysis result."""
        return {
            'total_traces': 0,
            'analysis_timestamp': time.time(),
            'successful_sequences': {'common_sequences': [], 'high_success_sequences': [], 'sequence_lengths': {}, 'action_transitions': {}},
            'action_effectiveness': {'action_effectiveness': {}, 'most_effective_action': 'none', 'least_effective_action': 'none'},
            'coordinate_patterns': {'coordinate_clusters': [], 'coordinate_effectiveness': {}, 'total_coordinate_actions': 0},
            'game_specific_patterns': {}
        }
