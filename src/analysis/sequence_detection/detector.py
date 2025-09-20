"""
Sequence Detector

Detects and analyzes action sequences in traces.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np

logger = logging.getLogger(__name__)

class SequenceDetector:
    """Detects and analyzes action sequences in traces."""
    
    def __init__(self, min_sequence_length: int = 2, max_sequence_length: int = 10):
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
    
    def detect_sequences(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect sequences in action traces."""
        try:
            if not traces:
                return {'sequences': [], 'sequence_stats': {}}
            
            # Extract all sequences
            all_sequences = self._extract_sequences(traces)
            
            # Analyze sequence patterns
            sequence_analysis = self._analyze_sequences(all_sequences, traces)
            
            # Find frequent sequences
            frequent_sequences = self._find_frequent_sequences(all_sequences)
            
            # Find successful sequences
            successful_sequences = self._find_successful_sequences(all_sequences, traces)
            
            return {
                'total_sequences': len(all_sequences),
                'frequent_sequences': frequent_sequences,
                'successful_sequences': successful_sequences,
                'sequence_analysis': sequence_analysis
            }
            
        except Exception as e:
            logger.error(f"Error detecting sequences: {e}")
            return {'sequences': [], 'sequence_stats': {}}
    
    def _extract_sequences(self, traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract sequences from traces."""
        sequences = []
        
        try:
            for trace in traces:
                actions = trace.get('actions', [])
                if len(actions) < self.min_sequence_length:
                    continue
                
                # Extract sequences of different lengths
                for length in range(self.min_sequence_length, min(len(actions) + 1, self.max_sequence_length + 1)):
                    for i in range(len(actions) - length + 1):
                        sequence = actions[i:i + length]
                        sequences.append({
                            'sequence': sequence,
                            'length': length,
                            'start_index': i,
                            'trace_id': trace.get('trace_id', 'unknown'),
                            'game_id': trace.get('game_id', 'unknown'),
                            'success': trace.get('success', False),
                            'reward': trace.get('reward', 0.0)
                        })
            
            return sequences
            
        except Exception as e:
            logger.error(f"Error extracting sequences: {e}")
            return []
    
    def _analyze_sequences(self, sequences: List[Dict[str, Any]], traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sequence patterns."""
        try:
            if not sequences:
                return {'length_distribution': {}, 'success_rates': {}, 'reward_distribution': {}}
            
            # Length distribution
            length_counts = Counter(seq['length'] for seq in sequences)
            length_distribution = dict(length_counts)
            
            # Success rates by length
            success_rates = {}
            for length in length_counts.keys():
                length_sequences = [seq for seq in sequences if seq['length'] == length]
                if length_sequences:
                    success_count = sum(1 for seq in length_sequences if seq['success'])
                    success_rates[length] = success_count / len(length_sequences)
            
            # Reward distribution
            rewards = [seq['reward'] for seq in sequences if seq['reward'] is not None]
            reward_distribution = {
                'mean': np.mean(rewards) if rewards else 0.0,
                'std': np.std(rewards) if rewards else 0.0,
                'min': np.min(rewards) if rewards else 0.0,
                'max': np.max(rewards) if rewards else 0.0
            }
            
            # Action transition analysis
            transitions = defaultdict(int)
            for seq in sequences:
                sequence = seq['sequence']
                for i in range(len(sequence) - 1):
                    transition = (sequence[i], sequence[i + 1])
                    transitions[transition] += 1
            
            # Most common transitions
            common_transitions = Counter(transitions).most_common(10)
            
            return {
                'length_distribution': length_distribution,
                'success_rates': success_rates,
                'reward_distribution': reward_distribution,
                'common_transitions': [{'from': t[0], 'to': t[1], 'count': count} for t, count in common_transitions]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sequences: {e}")
            return {'length_distribution': {}, 'success_rates': {}, 'reward_distribution': {}}
    
    def _find_frequent_sequences(self, sequences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find frequently occurring sequences."""
        try:
            if not sequences:
                return []
            
            # Count sequence occurrences
            sequence_counts = Counter(tuple(seq['sequence']) for seq in sequences)
            
            # Find frequent sequences (appearing at least 3 times)
            frequent_sequences = []
            for sequence_tuple, count in sequence_counts.items():
                if count >= 3:
                    sequence_list = list(sequence_tuple)
                    
                    # Calculate additional metrics
                    sequence_instances = [seq for seq in sequences if tuple(seq['sequence']) == sequence_tuple]
                    success_rate = sum(1 for seq in sequence_instances if seq['success']) / len(sequence_instances)
                    avg_reward = np.mean([seq['reward'] for seq in sequence_instances if seq['reward'] is not None])
                    
                    frequent_sequences.append({
                        'sequence': sequence_list,
                        'count': count,
                        'frequency': count / len(sequences),
                        'success_rate': success_rate,
                        'average_reward': avg_reward,
                        'length': len(sequence_list)
                    })
            
            # Sort by frequency
            frequent_sequences.sort(key=lambda x: x['count'], reverse=True)
            
            return frequent_sequences[:20]  # Return top 20
            
        except Exception as e:
            logger.error(f"Error finding frequent sequences: {e}")
            return []
    
    def _find_successful_sequences(self, sequences: List[Dict[str, Any]], traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find sequences with high success rates."""
        try:
            if not sequences:
                return []
            
            # Group sequences by their content
            sequence_groups = defaultdict(list)
            for seq in sequences:
                sequence_key = tuple(seq['sequence'])
                sequence_groups[sequence_key].append(seq)
            
            # Analyze each sequence group
            successful_sequences = []
            for sequence_tuple, group in sequence_groups.items():
                if len(group) < 2:  # Need at least 2 occurrences
                    continue
                
                sequence_list = list(sequence_tuple)
                total_count = len(group)
                success_count = sum(1 for seq in group if seq['success'])
                success_rate = success_count / total_count
                
                # Only include sequences with high success rate
                if success_rate >= 0.7:
                    rewards = [seq['reward'] for seq in group if seq['reward'] is not None]
                    avg_reward = np.mean(rewards) if rewards else 0.0
                    
                    successful_sequences.append({
                        'sequence': sequence_list,
                        'count': total_count,
                        'success_rate': success_rate,
                        'average_reward': avg_reward,
                        'length': len(sequence_list)
                    })
            
            # Sort by success rate
            successful_sequences.sort(key=lambda x: x['success_rate'], reverse=True)
            
            return successful_sequences[:15]  # Return top 15
            
        except Exception as e:
            logger.error(f"Error finding successful sequences: {e}")
            return []
    
    def find_sequence_patterns(self, sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find patterns in sequences."""
        try:
            if not sequences:
                return {'patterns': [], 'pattern_stats': {}}
            
            patterns = []
            
            # Find repeating patterns within sequences
            for seq in sequences:
                sequence = seq['sequence']
                if len(sequence) < 4:
                    continue
                
                # Look for 2-element repeating patterns
                for i in range(len(sequence) - 3):
                    pattern = sequence[i:i+2]
                    if sequence[i+2:i+4] == pattern:
                        patterns.append({
                            'pattern': pattern,
                            'sequence': sequence,
                            'start_index': i,
                            'repetitions': 2
                        })
            
            # Find common prefixes and suffixes
            prefixes = defaultdict(int)
            suffixes = defaultdict(int)
            
            for seq in sequences:
                sequence = seq['sequence']
                if len(sequence) >= 2:
                    # Common prefixes
                    for length in range(2, min(len(sequence), 5)):
                        prefix = tuple(sequence[:length])
                        prefixes[prefix] += 1
                    
                    # Common suffixes
                    for length in range(2, min(len(sequence), 5)):
                        suffix = tuple(sequence[-length:])
                        suffixes[suffix] += 1
            
            # Find most common prefixes and suffixes
            common_prefixes = Counter(prefixes).most_common(10)
            common_suffixes = Counter(suffixes).most_common(10)
            
            return {
                'repeating_patterns': patterns,
                'common_prefixes': [{'pattern': list(p), 'count': c} for p, c in common_prefixes],
                'common_suffixes': [{'pattern': list(p), 'count': c} for p, c in common_suffixes],
                'total_patterns': len(patterns)
            }
            
        except Exception as e:
            logger.error(f"Error finding sequence patterns: {e}")
            return {'patterns': [], 'pattern_stats': {}}
    
    def predict_next_action(self, current_sequence: List[str], sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict next action based on current sequence."""
        try:
            if not current_sequence or not sequences:
                return {'prediction': 'unknown', 'confidence': 0.0, 'alternatives': []}
            
            # Find sequences that start with current sequence
            matching_sequences = []
            for seq in sequences:
                sequence = seq['sequence']
                if len(sequence) > len(current_sequence):
                    if sequence[:len(current_sequence)] == current_sequence:
                        matching_sequences.append(seq)
            
            if not matching_sequences:
                return {'prediction': 'unknown', 'confidence': 0.0, 'alternatives': []}
            
            # Count next actions
            next_actions = Counter()
            for seq in matching_sequences:
                sequence = seq['sequence']
                next_action = sequence[len(current_sequence)]
                next_actions[next_action] += 1
            
            # Calculate probabilities
            total = sum(next_actions.values())
            action_probs = {action: count / total for action, count in next_actions.items()}
            
            # Get most likely action
            most_likely = max(action_probs.items(), key=lambda x: x[1])
            
            # Get alternatives
            alternatives = sorted(action_probs.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'prediction': most_likely[0],
                'confidence': most_likely[1],
                'alternatives': [{'action': action, 'probability': prob} for action, prob in alternatives]
            }
            
        except Exception as e:
            logger.error(f"Error predicting next action: {e}")
            return {'prediction': 'unknown', 'confidence': 0.0, 'alternatives': []}
