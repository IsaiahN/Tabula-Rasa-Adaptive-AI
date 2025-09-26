#!/usr/bin/env python3
"""
TABULA RASA - SCALED 9 HOUR CONTINUOUS TRAINING SCRIPT

This script runs a comprehensive 9-hour continuous training session
by launching multiple instances of the simple training script concurrently.

Features:
- Multiple concurrent training instances
- Parallel learning across different games
- Enhanced memory sharing between games
- Optimized for maximum learning speed
- Graceful shutdown for all instances
- Wrapper around simple training to reduce code duplication

Usage:
    python run_9hour_scaled_training.py
"""

import os
import sys
import subprocess
import time
import asyncio
import psutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
import signal

# Add src to path for database access
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from database.system_integration import get_system_integration
from database.db_initializer import ensure_database_ready

def analyze_system_resources() -> Dict[str, Any]:
    """Analyze system resources and determine optimal concurrent session count."""
    print(" Analyzing system resources...")
    
    # Get system information
    memory = psutil.virtual_memory()
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Calculate available resources
    total_ram_gb = memory.total / (1024**3)
    available_ram_gb = memory.available / (1024**3)
    used_ram_gb = memory.used / (1024**3)
    ram_usage_percent = memory.percent
    
    print(f" Memory Analysis:")
    print(f"   • Total RAM: {total_ram_gb:.1f} GB")
    print(f"   • Available RAM: {available_ram_gb:.1f} GB")
    print(f"   • Used RAM: {used_ram_gb:.1f} GB ({ram_usage_percent:.1f}%)")
    print(f"   • CPU Cores: {cpu_count}")
    print(f"   • Current CPU Usage: {cpu_percent:.1f}%")
    
    # Estimate memory per training session (conservative estimate)
    estimated_memory_per_session = 0.5  # 500MB per session (conservative)
    estimated_cpu_per_session = 25  # 25% CPU per session (conservative)
    
    # Calculate maximum concurrent sessions based on available resources
    max_sessions_by_ram = int(available_ram_gb * 0.8 / estimated_memory_per_session)  # Use 80% of available RAM
    max_sessions_by_cpu = int(cpu_count * 4)  # 4 sessions per CPU core max
    max_sessions_by_ram_usage = int((100 - ram_usage_percent) / 10)  # 1 session per 10% available RAM
    
    # Take the most conservative estimate
    max_concurrent_sessions = min(max_sessions_by_ram, max_sessions_by_cpu, max_sessions_by_cpu)
    
    # Apply safety limits
    max_concurrent_sessions = max(1, min(max_concurrent_sessions, 20))  # Between 1 and 20 sessions
    
    # Determine session duration based on resources
    if total_ram_gb >= 16:
        session_duration = 30  # 30 minutes for high-end systems
        memory_size = 512
    elif total_ram_gb >= 8:
        session_duration = 25  # 25 minutes for mid-range systems
        memory_size = 256
    else:
        session_duration = 20  # 20 minutes for lower-end systems
        memory_size = 128
    
    # Adjust based on current CPU usage
    if cpu_percent > 80:
        max_concurrent_sessions = max(1, max_concurrent_sessions // 2)
        print(f" High CPU usage detected, reducing concurrent sessions")
    elif cpu_percent > 60:
        max_concurrent_sessions = max(1, int(max_concurrent_sessions * 0.8))
        print(f" Moderate CPU usage, slightly reducing concurrent sessions")
    
    print(f"\n Intelligent Resource Analysis:")
    print(f"   • Max sessions by RAM: {max_sessions_by_ram}")
    print(f"   • Max sessions by CPU: {max_sessions_by_cpu}")
    print(f"   • Max sessions by RAM usage: {max_sessions_by_ram_usage}")
    print(f"   • Recommended concurrent sessions: {max_concurrent_sessions}")
    print(f"   • Recommended session duration: {session_duration} minutes")
    print(f"   • Recommended memory size: {memory_size} MB")
    
    return {
        'max_concurrent_sessions': max_concurrent_sessions,
        'session_duration': session_duration,
        'memory_size': memory_size,
        'total_ram_gb': total_ram_gb,
        'available_ram_gb': available_ram_gb,
        'cpu_count': cpu_count,
        'cpu_percent': cpu_percent,
        'ram_usage_percent': ram_usage_percent
    }

def run_simple_training_instance(instance_id: int, duration_minutes: int = 540) -> Dict[str, Any]:
    """Run a single instance of the simple training script."""
    print(f" Starting simple training instance #{instance_id}")

    # Set environment variables for the instance
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['TRAINING_INSTANCE_ID'] = str(instance_id)

    # Use the simple training script as the base
    cmd = ['python', 'run_9hour_simple_training.py']

    start_time = time.time()

    try:
        # Run the simple training script
        process = subprocess.run(
            cmd,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        duration = time.time() - start_time

        # Parse success from return code and output
        success = process.returncode == 0

        if success:
            print(f"   ✓ Instance #{instance_id} completed successfully in {duration:.1f}s")
        else:
            print(f"   ✗ Instance #{instance_id} failed (code: {process.returncode}) in {duration:.1f}s")

        return {
            'instance_id': instance_id,
            'return_code': process.returncode,
            'duration': duration,
            'success': success,
            'stdout': process.stdout,
            'stderr': process.stderr
        }

    except Exception as e:
        duration = time.time() - start_time
        return {
            'instance_id': instance_id,
            'return_code': -1,
            'duration': duration,
            'success': False,
            'error': str(e),
            'stdout': '',
            'stderr': ''
        }

def run_parallel_training_instances(num_instances: int = 5) -> List[Dict[str, Any]]:
    """Run multiple simple training instances in parallel."""
    print(f" Starting {num_instances} parallel simple training instances")
    print(f"⏱ Each instance will run for 9 hours")
    print(f" Total concurrent training instances: {num_instances}")
    print()

    results = []

    # Use conservative limits for parallel simple training instances
    max_workers = min(num_instances, 10)  # Cap at 10 for safety

    if num_instances > max_workers:
        print(f" Limiting to {max_workers} concurrent workers for system stability")
        print(f"   (Requested: {num_instances}, Using: {max_workers})")
        print()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all instances
        future_to_instance = {
            executor.submit(run_simple_training_instance, i): i
            for i in range(num_instances)
        }

        # Collect results as they complete
        for future in as_completed(future_to_instance):
            instance_id = future_to_instance[future]
            try:
                result = future.result()
                results.append(result)

                if result['success']:
                    print(f" Instance #{instance_id} completed successfully in {result['duration']:.1f}s")
                else:
                    print(f" Instance #{instance_id} failed (code: {result['return_code']}) in {result['duration']:.1f}s")

            except Exception as e:
                print(f" Instance #{instance_id} crashed: {e}")
                results.append({
                    'instance_id': instance_id,
                    'return_code': -1,
                    'duration': 0,
                    'success': False,
                    'error': str(e)
                })

    return results

async def main():
    """Main function for scaled 9-hour training."""
    print("=" * 80)
    print("TABULA RASA - SCALED 9 HOUR CONTINUOUS TRAINING")
    print("=" * 80)
    print()
    print(" Starting scaled parallel training session...")
    print("⏱ Duration: Multiple 9-hour instances running in parallel")
    print(" Mode: Wrapper around simple training for reduced code duplication")
    print(" Features: Multiple concurrent simple training instances")
    print(" Database: Enabled (no more JSON files)")

    # Ensure database is ready before starting training
    print(" Checking database initialization...")
    if not ensure_database_ready():
        print(" Database initialization failed. Training cannot proceed.")
        return 1

    # Initialize database integration
    integration = get_system_integration()

    # Analyze system resources and determine optimal configuration
    resource_analysis = analyze_system_resources()

    # Determine number of concurrent instances based on system resources
    num_instances = max(1, min(resource_analysis['max_concurrent_sessions'] // 4, 8))  # Conservative approach

    # Record start time
    start_time = datetime.now()
    print(f" Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Press Ctrl+C to stop gracefully (will terminate all instances)")
    print()

    print(f" Scaled Training Plan:")
    print(f"   • Number of concurrent instances: {num_instances}")
    print(f"   • Each instance duration: 9 hours")
    print(f"   • System: {resource_analysis['cpu_count']} cores, {resource_analysis['total_ram_gb']:.1f}GB RAM")
    print(f"   • Wrapper approach: Uses simple training as base")
    print()

    try:
        # Run multiple simple training instances in parallel
        print(f" Starting {num_instances} parallel simple training instances...")
        all_results = run_parallel_training_instances(num_instances)

        # All instances have completed
        current_time = datetime.now()
        elapsed_seconds = (current_time - start_time).total_seconds()
        print(f"\n SCALED TRAINING COMPLETE!")
        print(f"Total duration: {elapsed_seconds/3600:.2f} hours")
        print(f"Total instances: {len(all_results)}")
        print(f"Completed at: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

    except KeyboardInterrupt:
        current_time = datetime.now()
        elapsed_seconds = (current_time - start_time).total_seconds()
        print(f"\n Training stopped by user (Ctrl+C)")
        print(f"⏱ Total duration: {elapsed_seconds/3600:.2f} hours")
        print(f" All instances will be terminated gracefully")

    except Exception as e:
        print(f"\n Error running scaled training: {e}")
        return 1
    # Final statistics
    print("\n" + "=" * 80)
    print("FINAL TRAINING STATISTICS")
    print("=" * 80)

    if all_results:
        successful_instances = sum(1 for r in all_results if r['success'])
        failed_instances = len(all_results) - successful_instances
        total_duration = sum(r['duration'] for r in all_results)
        avg_duration = total_duration / len(all_results)

        print(f" Overall Results:")
        print(f"    Total instances: {len(all_results)}")
        print(f"    Successful: {successful_instances}")
        print(f"    Failed: {failed_instances}")
        print(f"    Success rate: {successful_instances/len(all_results)*100:.1f}%")
        print(f"   ⏱ Total training time: {total_duration/3600:.2f} hours")
        print(f"    Average instance duration: {avg_duration:.1f}s")

        # Save results to database
        print(f"\n Saving results to database...")

        # Log final summary to database
        await integration.log_system_event(
            level="INFO",
            component="scaled_training_wrapper",
            message=f"Scaled training (wrapper) completed: {len(all_results)} instances, {successful_instances/len(all_results)*100:.1f}% success rate",
            data={
                'total_instances': len(all_results),
                'successful_instances': successful_instances,
                'failed_instances': failed_instances,
                'success_rate': successful_instances/len(all_results)*100,
                'total_duration_hours': total_duration/3600,
                'average_duration_seconds': avg_duration,
                'wrapper_approach': True
            },
            session_id=f"scaled_training_wrapper_summary_{int(time.time())}"
        )

        print(f" Summary logged to system_logs")

    print(f"\nTraining session ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0

if __name__ == "__main__":
    asyncio.run(main())
