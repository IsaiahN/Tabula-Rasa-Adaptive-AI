#!/usr/bin/env python3
"""
Action Trace Analyzer Runner

This script runs the action trace analyzer to identify successful patterns
and sequences from the training data.
"""

import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from arc_integration.action_trace_analyzer import ActionTraceAnalyzer

def main():
    """Run the action trace analyzer."""
    print("🔍 Starting Action Trace Analyzer...")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create analyzer and run analysis
    analyzer = ActionTraceAnalyzer()
    analysis = analyzer.analyze_action_traces()
    
    # Print summary
    analyzer.print_analysis_summary(analysis)
    
    # Check for insights
    action_effectiveness = analysis.get('action_effectiveness', {})
    if action_effectiveness:
        best_action = max(action_effectiveness.items(), key=lambda x: x[1].get('effectiveness_score', 0))
        print(f"\n🏆 BEST PERFORMING ACTION: {best_action[0]} "
              f"(effectiveness: {best_action[1]['effectiveness_score']:.2f})")
    
    coordinate_patterns = analysis.get('coordinate_patterns', {})
    if coordinate_patterns:
        best_coord = max(coordinate_patterns.items(), key=lambda x: x[1].get('effectiveness_score', 0))
        print(f"🎯 BEST PERFORMING COORDINATE: {best_coord[0]} "
              f"(effectiveness: {best_coord[1]['effectiveness_score']:.2f})")
    
    return analysis

if __name__ == "__main__":
    main()
