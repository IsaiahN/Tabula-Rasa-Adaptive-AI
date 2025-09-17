#!/usr/bin/env python3
"""
MONITOR 9-HOUR TEST
Monitor the 9-hour simple training test for errors and progress
"""

import asyncio
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database.director_commands import get_director_commands

async def monitor_test():
    """Monitor the 9-hour test progress."""
    print("🔍 MONITORING 9-HOUR SIMPLE TRAINING TEST")
    print("=" * 50)
    
    director = get_director_commands()
    start_time = time.time()
    
    while True:
        try:
            # Get system status
            status = await director.get_system_overview()
            health = await director.analyze_system_health()
            performance = await director.get_performance_summary(1)  # Last hour
            
            # Calculate elapsed time
            elapsed = time.time() - start_time
            hours = elapsed / 3600
            minutes = (elapsed % 3600) / 60
            
            print(f"\n⏰ ELAPSED TIME: {hours:.1f}h {minutes:.0f}m")
            print(f"📊 ACTIVE SESSIONS: {status['system_status']['active_sessions']}")
            print(f"🎮 TOTAL GAMES: {status['system_status']['total_games']}")
            print(f"🏆 TOTAL WINS: {status['system_status']['total_wins']}")
            print(f"📈 WIN RATE: {status['system_status']['avg_win_rate']:.1%}")
            print(f"🏥 HEALTH: {health['status']} (score: {health['health_score']:.2f})")
            
            if health['issues']:
                print(f"⚠️ ISSUES: {len(health['issues'])} - {health['issues']}")
            
            if performance['total_games'] > 0:
                print(f"🎯 RECENT GAMES: {performance['total_games']} in last hour")
                print(f"📊 RECENT WIN RATE: {performance['win_rate']:.1%}")
            
            # Check for errors in logs
            try:
                with open("data/logs/master_arc_trainer.log", "r") as f:
                    lines = f.readlines()
                    recent_lines = lines[-10:]  # Last 10 lines
                    
                error_count = sum(1 for line in recent_lines if any(word in line.lower() for word in ['error', 'exception', 'traceback', 'failed']))
                if error_count > 0:
                    print(f"❌ ERRORS DETECTED: {error_count} in recent logs")
                else:
                    print("✅ NO RECENT ERRORS")
                    
            except Exception as e:
                print(f"⚠️ Could not check logs: {e}")
            
            # Check if we should stop (9 hours elapsed)
            if hours >= 9:
                print("\n🎉 9-HOUR TEST COMPLETED!")
                break
                
            # Wait 5 minutes before next check
            print("⏳ Waiting 5 minutes for next check...")
            await asyncio.sleep(300)  # 5 minutes
            
        except Exception as e:
            print(f"❌ MONITORING ERROR: {e}")
            await asyncio.sleep(60)  # Wait 1 minute on error

if __name__ == "__main__":
    asyncio.run(monitor_test())
