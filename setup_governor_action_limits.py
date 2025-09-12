#!/usr/bin/env python3
"""
Governor Action Limits Setup

This script helps you set up the Governor-controlled action limits system.
It allows you to set maximum boundaries that the Governor cannot exceed while
giving the Governor full control over dynamic adjustments within those boundaries.

Usage:
    python setup_governor_action_limits.py --max-scorecard 5000 --max-game 2000
    python setup_governor_action_limits.py --show-status
    python setup_governor_action_limits.py --reset-to-defaults
"""

import argparse
import sys
import json
from pathlib import Path

def setup_governor_limits(**kwargs):
    """Set up Governor action limits with user-defined maximums."""
    try:
        # Import the Governor to set limits
        from src.core.meta_cognitive_governor import MetaCognitiveGovernor
        
        # Create a temporary Governor instance to set limits
        governor = MetaCognitiveGovernor(persistence_dir="data")
        
        if governor.action_limits_manager:
            # Set the maximum boundaries
            governor.set_action_limit_maximums(**kwargs)
            
            # Get current status
            status = governor.get_action_limits_status()
            
            print("✅ Governor action limits configured successfully!")
            print("\nCurrent configuration:")
            print("=" * 50)
            
            for limit_type, config in status.get('current_limits', {}).items():
                print(f"{limit_type.upper()}:")
                print(f"  Current: {config['current']:,}")
                print(f"  Min:     {config['min']:,}")
                print(f"  Max:     {config['max']:,}")
                print(f"  Base:    {config['base']:,}")
                print()
            
            print("Global Metrics:")
            metrics = status.get('global_metrics', {})
            print(f"  Efficiency:        {metrics.get('efficiency', 0):.2f}")
            print(f"  Learning Progress: {metrics.get('learning_progress', 0):.2f}")
            print(f"  System Stress:     {metrics.get('system_stress', 0):.2f}")
            print("=" * 50)
            
            return True
        else:
            print("❌ Error: Action limits manager not available")
            return False
            
    except ImportError as e:
        print(f"❌ Error: Could not import Governor: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: Failed to setup Governor limits: {e}")
        return False

def show_status():
    """Show current Governor action limits status."""
    try:
        from src.core.meta_cognitive_governor import MetaCognitiveGovernor
        
        governor = MetaCognitiveGovernor(persistence_dir="data")
        
        if governor.action_limits_manager:
            status = governor.get_action_limits_status()
            
            print("🎯 GOVERNOR ACTION LIMITS STATUS")
            print("=" * 60)
            
            # Current limits
            print("\n📊 CURRENT LIMITS:")
            for limit_type, config in status.get('current_limits', {}).items():
                print(f"  {limit_type.upper()}:")
                print(f"    Current: {config['current']:,} actions")
                print(f"    Range:   {config['min']:,} - {config['max']:,} actions")
                print(f"    Base:    {config['base']:,} actions")
                print(f"    Scaling: {config['scaling_factor']:.2f}x")
                if config['last_adjusted'] > 0:
                    print(f"    Last Adjusted: {config['last_adjusted']:.1f}s ago ({config['adjustment_reason']})")
                print()
            
            # Global metrics
            print("📈 GLOBAL METRICS:")
            metrics = status.get('global_metrics', {})
            print(f"  Efficiency:        {metrics.get('efficiency', 0):.2f} (0.0 = poor, 1.0 = excellent)")
            print(f"  Learning Progress: {metrics.get('learning_progress', 0):.2f} (0.0 = no learning, 1.0 = rapid learning)")
            print(f"  System Stress:     {metrics.get('system_stress', 0):.2f} (0.0 = low stress, 1.0 = high stress)")
            print(f"  Last Adjustment:   {metrics.get('last_global_adjustment', 0):.1f}s ago")
            
            # Performance trend
            trend = status.get('performance_trend', 'unknown')
            trend_emoji = {
                'improving': '📈',
                'declining': '📉',
                'stable': '➡️',
                'insufficient_data': '❓'
            }.get(trend, '❓')
            print(f"  Performance Trend: {trend_emoji} {trend}")
            
            # Recent adjustments
            recent_adjustments = status.get('recent_adjustments', [])
            if recent_adjustments:
                print(f"\n🔄 RECENT ADJUSTMENTS (last {len(recent_adjustments)}):")
                for adj in recent_adjustments[-5:]:  # Show last 5
                    print(f"  {adj['limit_type']}: {adj['old_value']:,} → {adj['new_value']:,} ({adj['reason']})")
            
            print("=" * 60)
            
        else:
            print("❌ Error: Action limits manager not available")
            
    except ImportError as e:
        print(f"❌ Error: Could not import Governor: {e}")
    except Exception as e:
        print(f"❌ Error: Failed to get status: {e}")

def reset_to_defaults():
    """Reset Governor action limits to default values."""
    try:
        from src.core.meta_cognitive_governor import MetaCognitiveGovernor
        
        governor = MetaCognitiveGovernor(persistence_dir="data")
        
        if governor.action_limits_manager:
            governor.action_limits_manager.reset_to_defaults()
            print("✅ Governor action limits reset to defaults")
        else:
            print("❌ Error: Action limits manager not available")
            
    except ImportError as e:
        print(f"❌ Error: Could not import Governor: {e}")
    except Exception as e:
        print(f"❌ Error: Failed to reset limits: {e}")

def main():
    parser = argparse.ArgumentParser(description="Setup Governor-controlled action limits")
    
    # Maximum boundary options
    parser.add_argument('--max-game', type=int, help='Set maximum actions per game')
    parser.add_argument('--max-session', type=int, help='Set maximum actions per session')
    parser.add_argument('--max-scorecard', type=int, help='Set maximum actions per scorecard')
    parser.add_argument('--max-episode', type=int, help='Set maximum actions per episode')
    
    # Set all maximums to the same value
    parser.add_argument('--max-all', type=int, help='Set all maximums to the same value')
    
    # Status and management
    parser.add_argument('--show-status', action='store_true', help='Show current Governor action limits status')
    parser.add_argument('--reset', action='store_true', help='Reset to default values')
    
    args = parser.parse_args()
    
    if args.reset:
        reset_to_defaults()
        return
    
    if args.show_status:
        show_status()
        return
    
    # Check if any limits were specified
    limits_to_set = {}
    if args.max_all:
        limits_to_set = {
            'per_game': args.max_all,
            'per_session': args.max_all,
            'per_scorecard': args.max_all,
            'per_episode': args.max_all
        }
    else:
        if args.max_game:
            limits_to_set['per_game'] = args.max_game
        if args.max_session:
            limits_to_set['per_session'] = args.max_session
        if args.max_scorecard:
            limits_to_set['per_scorecard'] = args.max_scorecard
        if args.max_episode:
            limits_to_set['per_episode'] = args.max_episode
    
    if not limits_to_set:
        print("❌ Error: No action limits specified!")
        print("Use --help to see available options.")
        return
    
    # Set up the limits
    if setup_governor_limits(**limits_to_set):
        print("\n🎯 The Governor will now dynamically adjust action limits within these boundaries.")
        print("   The Governor will increase limits when performance is good and decrease them when struggling.")
        print("   Use --show-status to monitor the Governor's adjustments.")
    else:
        print("❌ Failed to setup Governor action limits!")

if __name__ == "__main__":
    main()
