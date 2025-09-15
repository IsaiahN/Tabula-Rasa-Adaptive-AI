#!/usr/bin/env python3
"""
Continuous Monitoring Script for Tabula Rasa 9-Hour Training
Monitors progress every hour and reports on system performance.
"""

import time
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def run_scorecard_monitor():
    """Run the enhanced scorecard monitor and return results."""
    try:
        result = subprocess.run([sys.executable, "enhanced_scorecard_monitor.py"], 
                              capture_output=True, text=True, timeout=60)
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT", "Monitor timed out"
    except Exception as e:
        return f"ERROR: {e}", ""

def check_global_counters():
    """Check global counters for action progress."""
    try:
        with open("data/global_counters.json", "r") as f:
            counters = json.load(f)
        return counters
    except Exception as e:
        return {"error": str(e)}

def log_progress(hour, counters, monitor_output):
    """Log progress to a monitoring file."""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "hour": hour,
        "timestamp": timestamp,
        "global_counters": counters,
        "monitor_output": monitor_output[:1000] if len(monitor_output) > 1000 else monitor_output
    }
    
    log_file = Path("data/monitoring_log.jsonl")
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def main():
    """Main monitoring loop."""
    print("🔍 CONDUCTOR: Starting Continuous 9-Hour Monitoring")
    print("=" * 60)
    
    start_time = time.time()
    hour = 0
    
    while hour < 9:
        print(f"\n⏰ HOUR {hour + 1} MONITORING - {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 40)
        
        # Check global counters
        counters = check_global_counters()
        if "error" not in counters:
            total_actions = counters.get("total_actions", 0)
            energy = counters.get("persistent_energy_level", 0)
            print(f"📊 Total Actions: {total_actions}")
            print(f"⚡ Energy Level: {energy:.1f}")
        else:
            print(f"❌ Error reading counters: {counters['error']}")
        
        # Run scorecard monitor
        print("🔍 Running scorecard monitor...")
        monitor_output, monitor_error = run_scorecard_monitor()
        
        # Extract key metrics from monitor output
        if "Total Actions:" in monitor_output:
            for line in monitor_output.split('\n'):
                if "Total Actions:" in line:
                    print(f"📈 {line.strip()}")
                elif "Total Level Completions:" in line:
                    print(f"🎯 {line.strip()}")
                elif "Total Games Completed:" in line:
                    print(f"🏆 {line.strip()}")
                elif "Overall Win Rate:" in line:
                    print(f"📊 {line.strip()}")
        
        # Log progress
        log_progress(hour + 1, counters, monitor_output)
        
        # Wait for next hour (or 5 minutes for testing)
        if hour < 8:  # Don't wait after the last check
            print(f"⏳ Waiting for next hour... (5 minutes for testing)")
            time.sleep(300)  # 5 minutes for testing, change to 3600 for real hours
        
        hour += 1
    
    print("\n✅ 9-Hour Monitoring Complete!")
    print(f"Total monitoring time: {(time.time() - start_time) / 3600:.1f} hours")

if __name__ == "__main__":
    main()
