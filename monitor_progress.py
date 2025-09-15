#!/usr/bin/env python3
"""
Conductor Progress Monitoring Script
Monitors the optimized training progress and tracks improvements.
"""

import json
import time
from pathlib import Path
from datetime import datetime

def monitor_action_intelligence():
    """Monitor action intelligence files for progress updates."""
    
    print("🎯 CONDUCTOR: Monitoring Action Intelligence Progress")
    print("=" * 60)
    print(f"⏰ Monitoring started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find all action intelligence files
    action_intel_files = list(Path("data").glob("action_intelligence_*.json"))
    
    if not action_intel_files:
        print("❌ No action intelligence files found!")
        return False
    
    print(f"\n📁 MONITORING {len(action_intel_files)} ACTION INTELLIGENCE FILES:")
    
    total_effective_actions = 0
    high_performers = []
    
    for file in sorted(action_intel_files):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            game_id = data.get('game_id', file.stem)
            effective_actions = data.get('effective_actions', {})
            last_updated = data.get('last_updated', 0)
            
            # Convert timestamp to readable format
            if last_updated > 0:
                update_time = datetime.fromtimestamp(last_updated).strftime('%Y-%m-%d %H:%M:%S')
            else:
                update_time = "Never"
            
            print(f"\n🎮 {game_id}:")
            print(f"   📅 Last Updated: {update_time}")
            print(f"   🎯 Effective Actions: {len(effective_actions)}")
            
            if effective_actions:
                for action, stats in effective_actions.items():
                    success_rate = stats.get('success_rate', 0)
                    attempts = stats.get('attempts', 0)
                    successes = stats.get('successes', 0)
                    
                    print(f"      ACTION{action}: {success_rate:.2f} success rate ({successes}/{attempts})")
                    
                    if success_rate > 1.0:
                        high_performers.append(f"{game_id} ACTION{action}: {success_rate:.2f}")
                        total_effective_actions += 1
            else:
                print(f"      No effective actions learned yet")
                
        except Exception as e:
            print(f"   ❌ Error reading {file}: {e}")
    
    print(f"\n📊 SUMMARY:")
    print(f"   🎯 Total Effective Actions: {total_effective_actions}")
    print(f"   🏆 High Performers (>100% success): {len(high_performers)}")
    
    if high_performers:
        print(f"\n🏆 HIGH PERFORMERS:")
        for performer in high_performers:
            print(f"   • {performer}")
    
    return True

def check_recent_logs():
    """Check recent training logs for progress indicators."""
    
    print(f"\n📋 CHECKING RECENT TRAINING LOGS:")
    
    log_file = Path("data/logs/master_arc_trainer.log")
    if not log_file.exists():
        print("   ❌ No training log found")
        return False
    
    # Read last 50 lines of log
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        recent_lines = lines[-50:] if len(lines) > 50 else lines
        
        # Look for progress indicators
        progress_indicators = []
        for line in recent_lines:
            if any(keyword in line.lower() for keyword in ['score:', 'success', 'win', 'level', 'complete']):
                progress_indicators.append(line.strip())
        
        if progress_indicators:
            print(f"   📈 Recent Progress Indicators:")
            for indicator in progress_indicators[-10:]:  # Show last 10
                print(f"      {indicator}")
        else:
            print(f"   ⚠️ No recent progress indicators found")
            
    except Exception as e:
        print(f"   ❌ Error reading log file: {e}")
    
    return True

def main():
    """Main monitoring function."""
    
    print("🎯 CONDUCTOR AUTONOMOUS EVOLUTION MODE")
    print("=" * 60)
    print("Monitoring optimized training progress...")
    
    # Monitor action intelligence
    intel_success = monitor_action_intelligence()
    
    # Check recent logs
    log_success = check_recent_logs()
    
    print(f"\n✅ MONITORING COMPLETE")
    print(f"   Action Intelligence: {'✅' if intel_success else '❌'}")
    print(f"   Recent Logs: {'✅' if log_success else '❌'}")
    
    print(f"\n🎯 NEXT MONITORING CYCLE IN 1 HOUR")
    print(f"   The system will continue autonomous evolution...")

if __name__ == "__main__":
    main()
