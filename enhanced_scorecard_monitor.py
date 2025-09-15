#!/usr/bin/env python3
"""
Enhanced Scorecard Monitor
Extracts scorecard IDs from recent logs and monitors active training sessions.
"""

import json
import re
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta
import glob

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from arc_integration.scorecard_api import ScorecardAPIManager, get_api_key_from_config

def extract_scorecard_ids_from_logs():
    """Extract scorecard IDs from recent training logs."""
    
    print("🔍 CONDUCTOR: Extracting Scorecard IDs from Recent Logs")
    print("=" * 60)
    
    # Look for scorecard IDs in various log patterns
    scorecard_patterns = [
        r'scorecard[_-]?id["\s:]+([a-f0-9-]{36})',
        r'card[_-]?id["\s:]+([a-f0-9-]{36})',
        r'scorecard["\s:]+([a-f0-9-]{36})',
        r'opened[_\s]+scorecard[_\s]+([a-f0-9-]{36})',
        r'created[_\s]+scorecard[_\s]+([a-f0-9-]{36})',
        r'scorecard[_\s]+([a-f0-9-]{36})',
        r'card[_\s]+([a-f0-9-]{36})'
    ]
    
    scorecard_ids = set()
    
    # Search in log files
    log_files = [
        "data/logs/master_arc_trainer.log",
        "data/logs/master_arc_trainer_output.log",
        "data/logs/governor_decisions_phase3.log",
        "data/logs/architect_evolution.log"
    ]
    
    for log_file in log_files:
        if Path(log_file).exists():
            print(f"📋 Searching {log_file}...")
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                for pattern in scorecard_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        if len(match) == 36:  # UUID length
                            scorecard_ids.add(match)
                            print(f"   ✅ Found scorecard ID: {match}")
                
            except Exception as e:
                print(f"   ⚠️ Error reading {log_file}: {e}")
    
    # Search in session files
    session_files = glob.glob("data/sessions/*.json")
    for session_file in session_files:
        try:
            with open(session_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            for pattern in scorecard_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    if len(match) == 36:
                        scorecard_ids.add(match)
                        print(f"   ✅ Found scorecard ID in {Path(session_file).name}: {match}")
        
        except Exception as e:
            continue
    
    # Search in recent files (last 3 days)
    cutoff_time = time.time() - (3 * 24 * 60 * 60)
    recent_files = []
    
    for pattern in ["data/**/*.json", "data/**/*.log"]:
        files = glob.glob(pattern, recursive=True)
        for file in files:
            try:
                if Path(file).stat().st_mtime > cutoff_time:
                    recent_files.append(file)
            except:
                continue
    
    print(f"📁 Searching {len(recent_files)} recent files...")
    for file in recent_files:
        try:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            for pattern in scorecard_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    if len(match) == 36:
                        scorecard_ids.add(match)
                        print(f"   ✅ Found scorecard ID in {Path(file).name}: {match}")
        
        except Exception as e:
            continue
    
    print(f"\n📊 FOUND {len(scorecard_ids)} SCORECARD IDS:")
    for card_id in sorted(scorecard_ids):
        print(f"   • {card_id}")
    
    return list(scorecard_ids)

def check_active_training_sessions():
    """Check for active training sessions and their scorecards."""
    
    print(f"\n🔍 CONDUCTOR: Checking Active Training Sessions")
    print("=" * 60)
    
    # Look for active session indicators
    active_indicators = [
        "data/global_counters.json",
        "data/task_performance.json",
        "data/training/results/unified_trainer_results.json"
    ]
    
    active_sessions = []
    
    for indicator_file in active_indicators:
        if Path(indicator_file).exists():
            try:
                with open(indicator_file, 'r') as f:
                    data = json.load(f)
                
                # Look for active session data
                if isinstance(data, dict):
                    for key, value in data.items():
                        if 'active' in key.lower() or 'current' in key.lower():
                            if isinstance(value, (str, int)) and value:
                                active_sessions.append(f"{indicator_file}: {key}={value}")
                
                print(f"✅ {indicator_file}: Active session data found")
                
            except Exception as e:
                print(f"⚠️ {indicator_file}: Error reading - {e}")
    
    # Check for recent session files (last hour)
    cutoff_time = time.time() - (60 * 60)  # 1 hour
    recent_sessions = []
    
    session_files = glob.glob("data/sessions/*.json")
    for session_file in session_files:
        try:
            if Path(session_file).stat().st_mtime > cutoff_time:
                recent_sessions.append(session_file)
                print(f"🕐 Recent session: {Path(session_file).name}")
        except:
            continue
    
    if recent_sessions:
        print(f"📊 Found {len(recent_sessions)} recent sessions (last hour)")
    else:
        print("📊 No recent sessions found (last hour)")
    
    return active_sessions, recent_sessions

def monitor_scorecards_by_id(scorecard_ids, api_key):
    """Monitor specific scorecard IDs."""
    
    if not scorecard_ids:
        print("📊 No scorecard IDs to monitor")
        return {}
    
    print(f"\n📊 CONDUCTOR: Monitoring {len(scorecard_ids)} Scorecards")
    print("=" * 60)
    
    manager = ScorecardAPIManager(api_key)
    all_stats = {
        'total_scorecards': len(scorecard_ids),
        'total_wins': 0,
        'total_played': 0,
        'total_actions': 0,
        'total_score': 0,
        'total_level_completions': 0,
        'total_games_completed': 0,
        'scorecards': {}
    }
    
    for i, card_id in enumerate(scorecard_ids):
        print(f"\n📊 Monitoring Scorecard {i+1}/{len(scorecard_ids)}: {card_id}")
        
        try:
            scorecard_data = manager.get_scorecard_data(card_id)
            if scorecard_data:
                analysis = manager.analyze_level_completions(scorecard_data)
                
                # Save data
                manager.save_scorecard_data(card_id, scorecard_data, analysis)
                
                # Aggregate statistics
                all_stats['total_wins'] += analysis['total_wins']
                all_stats['total_played'] += analysis['total_played']
                all_stats['total_actions'] += analysis['total_actions']
                all_stats['total_score'] += analysis['total_score']
                all_stats['total_level_completions'] += analysis['level_completions']
                all_stats['total_games_completed'] += analysis['games_completed']
                
                all_stats['scorecards'][card_id] = {
                    'analysis': analysis,
                    'card_id': card_id,
                    'won': scorecard_data.get('won', 0),
                    'played': scorecard_data.get('played', 0),
                    'total_actions': scorecard_data.get('total_actions', 0),
                    'score': scorecard_data.get('score', 0)
                }
                
                print(f"   ✅ {analysis['level_completions']} level completions, {analysis['games_completed']} games completed")
            else:
                print(f"   ❌ Failed to retrieve data for {card_id}")
        
        except Exception as e:
            print(f"   ❌ Error monitoring {card_id}: {e}")
    
    # Calculate overall win rate
    if all_stats['total_played'] > 0:
        all_stats['overall_win_rate'] = (all_stats['total_wins'] / all_stats['total_played']) * 100
    else:
        all_stats['overall_win_rate'] = 0.0
    
    return all_stats

def main():
    """Main monitoring function."""
    
    print("🎯 CONDUCTOR: Enhanced Scorecard Monitor")
    print("=" * 60)
    print("Extracting scorecard IDs from logs and monitoring progress...")
    
    # Extract scorecard IDs from logs
    scorecard_ids = extract_scorecard_ids_from_logs()
    
    # Check for active training sessions
    active_sessions, recent_sessions = check_active_training_sessions()
    
    # Get API key
    api_key = get_api_key_from_config()
    if not api_key:
        print("\n❌ No API key found - cannot monitor scorecards")
        print("   Please ensure API key is configured")
        return
    
    # Monitor scorecards if we found any
    if scorecard_ids:
        stats = monitor_scorecards_by_id(scorecard_ids, api_key)
        
        print(f"\n📈 COMPREHENSIVE PROGRESS SUMMARY:")
        print(f"   📊 Scorecards Monitored: {stats['total_scorecards']}")
        print(f"   🎯 Total Level Completions: {stats['total_level_completions']}")
        print(f"   🏆 Total Games Completed: {stats['total_games_completed']}")
        print(f"   🎮 Total Wins: {stats['total_wins']}/{stats['total_played']}")
        print(f"   🎯 Total Actions: {stats['total_actions']}")
        print(f"   📊 Total Score: {stats['total_score']}")
        print(f"   📈 Overall Win Rate: {stats['overall_win_rate']:.1f}%")
        
        if stats['total_level_completions'] > 0 or stats['total_games_completed'] > 0:
            print(f"\n🎉 SUCCESS! Tabula Rasa IS completing levels and games!")
        else:
            print(f"\n⚠️ No level or game completions found in monitored scorecards")
    else:
        print(f"\n⚠️ No scorecard IDs found in recent logs")
        print(f"   This may indicate:")
        print(f"   • No recent training sessions with scorecard integration")
        print(f"   • Scorecard IDs stored in different format")
        print(f"   • Training using different logging system")
    
    # Show active session status
    if active_sessions or recent_sessions:
        print(f"\n🕐 ACTIVE TRAINING STATUS:")
        if active_sessions:
            for session in active_sessions:
                print(f"   • {session}")
        if recent_sessions:
            print(f"   • {len(recent_sessions)} recent session files")
    else:
        print(f"\n📊 No active training sessions detected")
    
    print(f"\n✅ ENHANCED MONITORING COMPLETE")

if __name__ == "__main__":
    main()
