#!/usr/bin/env python3
"""
Analyze simple_training_results files for migration
"""
import json
import glob
import os

def analyze_simple_training_files():
    """Analyze simple_training_results files"""
    files = glob.glob('simple_training_results_*.json')
    print(f'Found {len(files)} simple_training_results files:')
    
    total_sessions = 0
    total_successful = 0
    total_duration = 0
    
    for file in files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            summary = data.get('summary', {})
            detailed = data.get('detailed_results', [])
            
            print(f'\n📁 {file}:')
            print(f'   Total sessions: {summary.get("total_sessions", 0)}')
            print(f'   Success rate: {summary.get("success_rate", 0):.2%}')
            print(f'   Duration: {summary.get("total_duration_hours", 0):.2f} hours')
            print(f'   Detailed results: {len(detailed)} entries')
            
            # Accumulate totals
            total_sessions += summary.get('total_sessions', 0)
            total_successful += summary.get('successful_sessions', 0)
            total_duration += summary.get('total_duration_hours', 0)
            
        except Exception as e:
            print(f'❌ Error reading {file}: {e}')
    
    print(f'\n📊 SUMMARY:')
    print(f'   Total sessions across all files: {total_sessions}')
    print(f'   Total successful: {total_successful}')
    print(f'   Overall success rate: {(total_successful/total_sessions*100) if total_sessions > 0 else 0:.2f}%')
    print(f'   Total duration: {total_duration:.2f} hours')

if __name__ == '__main__':
    analyze_simple_training_files()
