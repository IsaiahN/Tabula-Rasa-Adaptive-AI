#!/usr/bin/env python3
"""
Migrate simple_training_results files to database
"""
import json
import glob
import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from database.system_integration import get_system_integration

async def migrate_simple_training_results():
    """Migrate simple_training_results files to database"""
    integration = get_system_integration()
    
    files = glob.glob('simple_training_results_*.json')
    print(f'Found {len(files)} simple_training_results files to migrate')
    
    total_migrated = 0
    total_sessions = 0
    
    for file in files:
        try:
            print(f'\n📁 Processing {file}...')
            
            with open(file, 'r') as f:
                data = json.load(f)
            
            summary = data.get('summary', {})
            detailed_results = data.get('detailed_results', [])
            
            # Extract timestamp from filename (format: simple_training_results_1757989696.json)
            timestamp_str = file.replace('simple_training_results_', '').replace('.json', '')
            try:
                # Convert timestamp to datetime
                timestamp = datetime.fromtimestamp(int(timestamp_str))
            except:
                timestamp = datetime.now()
            
            print(f'   Summary: {summary.get("total_sessions", 0)} sessions, {summary.get("success_rate", 0):.2%} success rate')
            print(f'   Detailed results: {len(detailed_results)} entries')
            
            # Migrate each detailed session result
            for i, session_data in enumerate(detailed_results):
                session_id = f"simple_training_{timestamp_str}_{i+1}"
                
                # Create training session data
                session_metrics = {
                    'session_id': session_id,
                    'start_time': timestamp,
                    'mode': 'simple_training',
                    'status': 'completed' if session_data.get('success', False) else 'failed',
                    'total_actions': 0,  # Not available in simple training results
                    'total_wins': 1 if session_data.get('success', False) else 0,
                    'total_games': 1,
                    'win_rate': 1.0 if session_data.get('success', False) else 0.0,
                    'avg_score': 0.0,  # Not available in simple training results
                    'energy_level': 100.0,  # Default value
                    'memory_operations': 0,  # Not available
                    'sleep_cycles': 0,  # Not available
                    'duration_seconds': session_data.get('duration', 0),
                    'return_code': session_data.get('return_code', 0)
                }
                
                # Save to database
                success = await integration.update_session_metrics(session_id, session_metrics)
                if success:
                    total_migrated += 1
                    print(f'   ✅ Migrated session {i+1}: {session_id}')
                else:
                    print(f'   ❌ Failed to migrate session {i+1}: {session_id}')
            
            total_sessions += len(detailed_results)
            
        except Exception as e:
            print(f'❌ Error processing {file}: {e}')
    
    print(f'\n🎉 MIGRATION COMPLETE!')
    print(f'   Total sessions processed: {total_sessions}')
    print(f'   Successfully migrated: {total_migrated}')
    print(f'   Success rate: {(total_migrated/total_sessions*100) if total_sessions > 0 else 0:.1f}%')

if __name__ == '__main__':
    asyncio.run(migrate_simple_training_results())
