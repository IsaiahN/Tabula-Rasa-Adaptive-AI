#!/usr/bin/env python3
"""
Enhanced Scorecard Monitor Runner

This script runs the enhanced scorecard monitoring system to analyze
current performance and track level completions.
"""

import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from arc_integration.enhanced_scorecard_monitor import EnhancedScorecardMonitor

def main():
    """Run the enhanced scorecard monitor."""
    print("🎯 Starting Enhanced Scorecard Monitor...")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create monitor and run analysis
    monitor = EnhancedScorecardMonitor()
    analysis = monitor.analyze_all_scorecards()
    
    # Print summary
    monitor.print_analysis_summary(analysis)
    
    # Check for critical issues
    recent_activity = analysis.get('recent_activity', {})
    if recent_activity.get('win_rate', 0) == 0 and recent_activity.get('total_plays', 0) > 0:
        print("\n⚠️  CRITICAL: Zero win rate detected in recent activity!")
        print("   This indicates the system is not learning effectively.")
        print("   Consider checking action effectiveness and coordinate selection.")
    
    if recent_activity.get('level_completions', 0) == 0 and recent_activity.get('total_plays', 0) > 0:
        print("\n⚠️  WARNING: No level completions detected in recent activity!")
        print("   This may indicate issues with win detection or game completion logic.")
    
    return analysis

if __name__ == "__main__":
    main()