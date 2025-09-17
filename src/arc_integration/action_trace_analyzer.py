#!/usr/bin/env python3
"""
Action Trace Analyzer for ARC-AGI-3 Training System

This module analyzes action traces to identify successful sequences and patterns
that can be used to improve future action selection.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np

logger = logging.getLogger(__name__)

class ActionTraceAnalyzer:
    """
    Analyzes action traces to identify successful sequences and patterns.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.action_traces_file = os.path.join(data_dir, "action_traces.ndjson")
        self.sessions_dir = os.path.join(data_dir, "sessions")
        
        # Pattern storage
        self.successful_sequences = defaultdict(list)
        self.action_effectiveness = defaultdict(list)
        self.coordinate_patterns = defaultdict(list)
        self.game_specific_patterns = defaultdict(lambda: defaultdict(list))
        
    def analyze_action_traces(self) -> Dict[str, Any]:
        """Analyze all action traces to identify successful patterns."""
        try:
            if not os.path.exists(self.action_traces_file):
                logger.warning("No action traces file found")
                return self._create_empty_analysis()
            
            # Read action traces
            traces = self._read_action_traces()
            if not traces:
                logger.warning("No action traces found")
                return self._create_empty_analysis()
            
            # Analyze patterns
            analysis = {
                'total_traces': len(traces),
                'analysis_timestamp': time.time(),
                'successful_sequences': {},
                'action_effectiveness': {},
                'coordinate_patterns': {},
                'game_specific_insights': {},
                'recommendations': []
            }
            
            # Analyze successful sequences
            successful_sequences = self._find_successful_sequences(traces)
            analysis['successful_sequences'] = successful_sequences
            
            # Analyze action effectiveness
            action_effectiveness = self._analyze_action_effectiveness(traces)
            analysis['action_effectiveness'] = action_effectiveness
            
            # Analyze coordinate patterns
            coordinate_patterns = self._analyze_coordinate_patterns(traces)
            analysis['coordinate_patterns'] = coordinate_patterns
            
            # Analyze game-specific patterns
            game_patterns = self._analyze_game_specific_patterns(traces)
            analysis['game_specific_insights'] = game_patterns
            
            # Generate recommendations
            recommendations = self._generate_recommendations(analysis)
            analysis['recommendations'] = recommendations
            
            # Save analysis
            await self._save_analysis(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing action traces: {e}")
            return self._create_empty_analysis()
    
    def _read_action_traces(self) -> List[Dict[str, Any]]:
        """Read action traces from the ndjson file."""
        traces = []
        try:
            with open(self.action_traces_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            trace = json.loads(line)
                            traces.append(trace)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Error reading action traces: {e}")
        
        return traces
    
    def _find_successful_sequences(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find successful action sequences from traces."""
        successful_sequences = {
            'common_sequences': [],
            'high_success_sequences': [],
            'sequence_lengths': {},
            'action_transitions': defaultdict(int)
        }
        
        # Group traces by game and session
        game_sessions = defaultdict(list)
        for trace in traces:
            if 'game_id' in trace and 'session_id' in trace:
                key = f"{trace['game_id']}_{trace['session_id']}"
                game_sessions[key].append(trace)
        
        # Analyze each session for successful sequences
        for session_key, session_traces in game_sessions.items():
            # Sort by timestamp
            session_traces.sort(key=lambda x: x.get('timestamp', 0))
            
            # Extract action sequences
            action_sequence = []
            for trace in session_traces:
                if 'action' in trace:
                    action_sequence.append(trace['action'])
            
            # Look for successful patterns (sequences that led to wins or score improvements)
            if len(action_sequence) >= 3:
                # Check if this session was successful
                session_success = self._evaluate_session_success(session_traces)
                
                if session_success:
                    # Record successful sequences of different lengths
                    for length in range(3, min(len(action_sequence) + 1, 8)):
                        for i in range(len(action_sequence) - length + 1):
                            sequence = tuple(action_sequence[i:i+length])
                            successful_sequences['common_sequences'].append(sequence)
                            
                            # Track action transitions
                            for j in range(len(sequence) - 1):
                                transition = (sequence[j], sequence[j+1])
                                successful_sequences['action_transitions'][transition] += 1
        
        # Find most common successful sequences
        sequence_counts = Counter(successful_sequences['common_sequences'])
        successful_sequences['high_success_sequences'] = [
            {'sequence': seq, 'count': count, 'success_rate': count / len(game_sessions)}
            for seq, count in sequence_counts.most_common(10)
        ]
        
        # Analyze sequence lengths
        for sequence in successful_sequences['common_sequences']:
            length = len(sequence)
            if length not in successful_sequences['sequence_lengths']:
                successful_sequences['sequence_lengths'][length] = 0
            successful_sequences['sequence_lengths'][length] += 1
        
        return successful_sequences
    
    def _analyze_action_effectiveness(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze effectiveness of individual actions."""
        action_stats = defaultdict(lambda: {'total': 0, 'successful': 0, 'score_changes': []})
        
        for trace in traces:
            if 'action' in trace:
                action = trace['action']
                action_stats[action]['total'] += 1
                
                # Check if this action was effective
                if trace.get('effective', False) or trace.get('score_change', 0) > 0:
                    action_stats[action]['successful'] += 1
                
                # Record score changes
                score_change = trace.get('score_change', 0)
                if score_change != 0:
                    action_stats[action]['score_changes'].append(score_change)
        
        # Calculate effectiveness metrics
        effectiveness_analysis = {}
        for action, stats in action_stats.items():
            success_rate = stats['successful'] / stats['total'] if stats['total'] > 0 else 0
            avg_score_change = np.mean(stats['score_changes']) if stats['score_changes'] else 0
            
            effectiveness_analysis[action] = {
                'total_attempts': stats['total'],
                'successful_attempts': stats['successful'],
                'success_rate': success_rate,
                'average_score_change': avg_score_change,
                'effectiveness_score': success_rate * (1 + avg_score_change / 10)  # Combined metric
            }
        
        return effectiveness_analysis
    
    def _analyze_coordinate_patterns(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze coordinate patterns for Action 6."""
        coordinate_stats = defaultdict(lambda: {'total': 0, 'successful': 0, 'score_changes': []})
        
        for trace in traces:
            if trace.get('action') == 6 and 'coordinates' in trace:
                coords = trace['coordinates']
                coord_key = f"({coords[0]},{coords[1]})"
                
                coordinate_stats[coord_key]['total'] += 1
                
                if trace.get('effective', False) or trace.get('score_change', 0) > 0:
                    coordinate_stats[coord_key]['successful'] += 1
                
                score_change = trace.get('score_change', 0)
                if score_change != 0:
                    coordinate_stats[coord_key]['score_changes'].append(score_change)
        
        # Calculate coordinate effectiveness
        coordinate_analysis = {}
        for coord, stats in coordinate_stats.items():
            if stats['total'] >= 3:  # Only consider coordinates with multiple attempts
                success_rate = stats['successful'] / stats['total']
                avg_score_change = np.mean(stats['score_changes']) if stats['score_changes'] else 0
                
                coordinate_analysis[coord] = {
                    'total_attempts': stats['total'],
                    'successful_attempts': stats['successful'],
                    'success_rate': success_rate,
                    'average_score_change': avg_score_change,
                    'effectiveness_score': success_rate * (1 + avg_score_change / 10)
                }
        
        return coordinate_analysis
    
    def _analyze_game_specific_patterns(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns specific to individual games."""
        game_patterns = defaultdict(lambda: {
            'action_effectiveness': defaultdict(lambda: {'total': 0, 'successful': 0}),
            'coordinate_effectiveness': defaultdict(lambda: {'total': 0, 'successful': 0}),
            'successful_sequences': [],
            'total_sessions': 0,
            'successful_sessions': 0
        })
        
        # Group traces by game
        game_traces = defaultdict(list)
        for trace in traces:
            if 'game_id' in trace:
                game_traces[trace['game_id']].append(trace)
        
        for game_id, game_trace_list in game_traces.items():
            # Group by session
            sessions = defaultdict(list)
            for trace in game_trace_list:
                session_id = trace.get('session_id', 'unknown')
                sessions[session_id].append(trace)
            
            game_patterns[game_id]['total_sessions'] = len(sessions)
            
            for session_id, session_traces in sessions.items():
                session_success = self._evaluate_session_success(session_traces)
                if session_success:
                    game_patterns[game_id]['successful_sessions'] += 1
                
                # Analyze actions in this session
                for trace in session_traces:
                    if 'action' in trace:
                        action = trace['action']
                        game_patterns[game_id]['action_effectiveness'][action]['total'] += 1
                        if trace.get('effective', False) or trace.get('score_change', 0) > 0:
                            game_patterns[game_id]['action_effectiveness'][action]['successful'] += 1
                    
                    if trace.get('action') == 6 and 'coordinates' in trace:
                        coords = trace['coordinates']
                        coord_key = f"({coords[0]},{coords[1]})"
                        game_patterns[game_id]['coordinate_effectiveness'][coord_key]['total'] += 1
                        if trace.get('effective', False) or trace.get('score_change', 0) > 0:
                            game_patterns[game_id]['coordinate_effectiveness'][coord_key]['successful'] += 1
        
        # Calculate effectiveness rates
        for game_id, patterns in game_patterns.items():
            for action, stats in patterns['action_effectiveness'].items():
                if stats['total'] > 0:
                    stats['success_rate'] = stats['successful'] / stats['total']
                else:
                    stats['success_rate'] = 0
            
            for coord, stats in patterns['coordinate_effectiveness'].items():
                if stats['total'] > 0:
                    stats['success_rate'] = stats['successful'] / stats['total']
                else:
                    stats['success_rate'] = 0
            
            if patterns['total_sessions'] > 0:
                patterns['session_success_rate'] = patterns['successful_sessions'] / patterns['total_sessions']
            else:
                patterns['session_success_rate'] = 0
        
        return dict(game_patterns)
    
    def _evaluate_session_success(self, session_traces: List[Dict[str, Any]]) -> bool:
        """Evaluate if a session was successful based on traces."""
        # Check for explicit success indicators
        for trace in session_traces:
            if trace.get('success', False) or trace.get('win', False):
                return True
            
            if trace.get('score_change', 0) > 0:
                return True
        
        # Check for frame changes (indicating progress)
        frame_changes = sum(1 for trace in session_traces if trace.get('frame_changed', False))
        if frame_changes > 5:  # Multiple frame changes indicate progress
            return True
        
        return False
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Action effectiveness recommendations
        action_effectiveness = analysis.get('action_effectiveness', {})
        if action_effectiveness:
            best_actions = sorted(
                action_effectiveness.items(),
                key=lambda x: x[1].get('effectiveness_score', 0),
                reverse=True
            )[:3]
            
            if best_actions:
                recommendations.append(f"Focus on high-effectiveness actions: {[action for action, _ in best_actions]}")
        
        # Coordinate recommendations
        coordinate_patterns = analysis.get('coordinate_patterns', {})
        if coordinate_patterns:
            best_coords = sorted(
                coordinate_patterns.items(),
                key=lambda x: x[1].get('effectiveness_score', 0),
                reverse=True
            )[:3]
            
            if best_coords:
                recommendations.append(f"Prioritize high-success coordinates: {[coord for coord, _ in best_coords]}")
        
        # Sequence recommendations
        successful_sequences = analysis.get('successful_sequences', {})
        high_success_sequences = successful_sequences.get('high_success_sequences', [])
        if high_success_sequences:
            top_sequence = high_success_sequences[0]
            recommendations.append(f"Use successful action sequence: {top_sequence['sequence']} (success rate: {top_sequence['success_rate']:.2%})")
        
        # Game-specific recommendations
        game_insights = analysis.get('game_specific_insights', {})
        for game_id, patterns in game_insights.items():
            if patterns.get('session_success_rate', 0) > 0.5:
                best_actions = sorted(
                    patterns['action_effectiveness'].items(),
                    key=lambda x: x[1].get('success_rate', 0),
                    reverse=True
                )[:2]
                
                if best_actions:
                    recommendations.append(f"For {game_id}: Use actions {[action for action, _ in best_actions]}")
        
        return recommendations
    
    def _create_empty_analysis(self) -> Dict[str, Any]:
        """Create empty analysis when no data is available."""
        return {
            'total_traces': 0,
            'analysis_timestamp': time.time(),
            'successful_sequences': {},
            'action_effectiveness': {},
            'coordinate_patterns': {},
            'game_specific_insights': {},
            'recommendations': ['No action traces available for analysis']
        }
    
    async def _save_analysis(self, analysis: Dict[str, Any]):
        """Save analysis results to database."""
        try:
            integration = get_system_integration()
            
            # Log analysis to database
            await integration.log_system_event(
                level="INFO",
                component="action_trace_analyzer",
                message="Action trace analysis completed",
                data=analysis,
                session_id=f"trace_analysis_{int(time.time())}"
            )
            
            logger.info("Action trace analysis saved to database")
            
        except Exception as e:
            logger.error(f"Error saving analysis to database: {e}")
    
    def print_analysis_summary(self, analysis: Dict[str, Any]):
        """Print a human-readable summary of the analysis."""
        print("\n" + "="*80)
        print("🔍 ACTION TRACE ANALYZER - ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total Traces Analyzed: {analysis.get('total_traces', 0)}")
        
        # Action effectiveness
        action_effectiveness = analysis.get('action_effectiveness', {})
        if action_effectiveness:
            print(f"\n🎯 ACTION EFFECTIVENESS:")
            sorted_actions = sorted(
                action_effectiveness.items(),
                key=lambda x: x[1].get('effectiveness_score', 0),
                reverse=True
            )
            
            for action, stats in sorted_actions[:5]:
                print(f"   Action {action}: {stats['success_rate']:.1%} success rate "
                      f"({stats['successful_attempts']}/{stats['total_attempts']} attempts)")
        
        # Coordinate patterns
        coordinate_patterns = analysis.get('coordinate_patterns', {})
        if coordinate_patterns:
            print(f"\n📍 COORDINATE PATTERNS:")
            sorted_coords = sorted(
                coordinate_patterns.items(),
                key=lambda x: x[1].get('effectiveness_score', 0),
                reverse=True
            )
            
            for coord, stats in sorted_coords[:5]:
                print(f"   {coord}: {stats['success_rate']:.1%} success rate "
                      f"({stats['successful_attempts']}/{stats['total_attempts']} attempts)")
        
        # Successful sequences
        successful_sequences = analysis.get('successful_sequences', {})
        high_success_sequences = successful_sequences.get('high_success_sequences', [])
        if high_success_sequences:
            print(f"\n🔗 SUCCESSFUL SEQUENCES:")
            for seq_data in high_success_sequences[:3]:
                sequence = seq_data['sequence']
                count = seq_data['count']
                success_rate = seq_data['success_rate']
                print(f"   {sequence}: {count} occurrences, {success_rate:.1%} success rate")
        
        # Recommendations
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "="*80)

def main():
    """Main function for running the action trace analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Action Trace Analyzer for ARC-AGI-3')
    parser.add_argument('--data-dir', default='data', help='Data directory path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run analysis
    analyzer = ActionTraceAnalyzer(args.data_dir)
    analysis = analyzer.analyze_action_traces()
    analyzer.print_analysis_summary(analysis)
    
    return analysis

if __name__ == "__main__":
    main()
