#!/usr/bin/env python3
"""
QUICK STATUS CHECK
Simple status check for the 9-hour test
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database.director_commands import get_director_commands

async def check_status():
    """Check system status."""
    try:
        director = get_director_commands()
        status = await director.get_system_overview()
        health = await director.analyze_system_health()
        
        print("=== 9-HOUR TEST STATUS ===")
        print(f"Database System: OPERATIONAL")
        print(f"Director Commands: WORKING")
        print(f"System Integration: ACTIVE")
        print(f"Active Sessions: {status['system_status']['active_sessions']}")
        print(f"Total Games: {status['system_status']['total_games']}")
        print(f"Total Wins: {status['system_status']['total_wins']}")
        print(f"Win Rate: {status['system_status']['avg_win_rate']:.1%}")
        print(f"Health Status: {health['status']} (score: {health['health_score']:.2f})")
        
        if health['issues']:
            print(f"Issues: {health['issues']}")
        else:
            print("No critical issues detected")
            
    except Exception as e:
        print(f"Error checking status: {e}")

if __name__ == "__main__":
    asyncio.run(check_status())
