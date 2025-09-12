#!/usr/bin/env python3
"""
Action Limits Changer

This script provides an easy way to change action limits without editing files directly.
It updates the action_limits_config.py file with new values.

Usage:
    python change_action_limits.py --game 2000 --session 1500 --scorecard 2000
    python change_action_limits.py --all 500
    python change_action_limits.py --show
"""

import argparse
import sys
from pathlib import Path

def update_action_limits(**kwargs):
    """Update the action limits configuration file."""
    config_file = Path("action_limits_config.py")
    
    if not config_file.exists():
        print("❌ Error: action_limits_config.py not found!")
        print("Please make sure you're running this from the project root directory.")
        return False
    
    # Read the current configuration
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Update the values
    for key, value in kwargs.items():
        if key == 'all':
            # Set all limits to the same value
            content = content.replace('MAX_ACTIONS_PER_GAME = 1000', f'MAX_ACTIONS_PER_GAME = {value}')
            content = content.replace('MAX_ACTIONS_PER_SESSION = 1000', f'MAX_ACTIONS_PER_SESSION = {value}')
            content = content.replace('MAX_ACTIONS_PER_SCORECARD = 1000', f'MAX_ACTIONS_PER_SCORECARD = {value}')
            content = content.replace('MAX_ACTIONS_PER_EPISODE = 1000', f'MAX_ACTIONS_PER_EPISODE = {value}')
        elif key == 'game':
            content = content.replace('MAX_ACTIONS_PER_GAME = 1000', f'MAX_ACTIONS_PER_GAME = {value}')
        elif key == 'session':
            content = content.replace('MAX_ACTIONS_PER_SESSION = 1000', f'MAX_ACTIONS_PER_SESSION = {value}')
        elif key == 'scorecard':
            content = content.replace('MAX_ACTIONS_PER_SCORECARD = 1000', f'MAX_ACTIONS_PER_SCORECARD = {value}')
        elif key == 'episode':
            content = content.replace('MAX_ACTIONS_PER_EPISODE = 1000', f'MAX_ACTIONS_PER_EPISODE = {value}')
    
    # Write the updated configuration
    with open(config_file, 'w') as f:
        f.write(content)
    
    return True

def show_current_limits():
    """Show the current action limits."""
    try:
        from action_limits_config import ActionLimits
        ActionLimits.print_current_limits()
    except ImportError:
        print("❌ Error: Could not import action_limits_config.py")
        print("Please make sure you're running this from the project root directory.")

def main():
    parser = argparse.ArgumentParser(description="Change action limits for the ARC training system")
    
    # Individual limit options
    parser.add_argument('--game', type=int, help='Set max actions per game')
    parser.add_argument('--session', type=int, help='Set max actions per session')
    parser.add_argument('--scorecard', type=int, help='Set max actions per scorecard')
    parser.add_argument('--episode', type=int, help='Set max actions per episode')
    
    # Set all limits to the same value
    parser.add_argument('--all', type=int, help='Set all limits to the same value')
    
    # Show current limits
    parser.add_argument('--show', action='store_true', help='Show current action limits')
    
    args = parser.parse_args()
    
    if args.show:
        show_current_limits()
        return
    
    # Check if any limits were specified
    limits_to_update = {}
    if args.all:
        limits_to_update['all'] = args.all
    else:
        if args.game:
            limits_to_update['game'] = args.game
        if args.session:
            limits_to_update['session'] = args.session
        if args.scorecard:
            limits_to_update['scorecard'] = args.scorecard
        if args.episode:
            limits_to_update['episode'] = args.episode
    
    if not limits_to_update:
        print("❌ Error: No action limits specified!")
        print("Use --help to see available options.")
        return
    
    # Update the limits
    if update_action_limits(**limits_to_update):
        print("✅ Action limits updated successfully!")
        print("\nNew configuration:")
        show_current_limits()
        print("\n🔄 Restart the training system to apply the new limits.")
    else:
        print("❌ Failed to update action limits!")

if __name__ == "__main__":
    main()
