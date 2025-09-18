#!/usr/bin/env python3
import sqlite3
from datetime import datetime, timedelta

def analyze_errors():
    conn = sqlite3.connect('tabula_rasa.db')
    cursor = conn.cursor()
    
    print('🔍 RUNTIME ERROR ANALYSIS')
    print('=' * 50)
    
    # Get recent error logs
    cursor.execute('''
        SELECT log_level, component, message, timestamp, session_id, game_id
        FROM system_logs 
        WHERE log_level IN ('ERROR', 'WARNING') 
        AND timestamp > datetime('now', '-2 hours')
        ORDER BY timestamp DESC 
        LIMIT 20
    ''')
    
    errors = cursor.fetchall()
    print(f'📊 Recent Errors/Warnings (last 2 hours): {len(errors)}')
    print()
    
    for error in errors:
        level, component, message, timestamp, session_id, game_id = error
        print(f'🚨 {level} | {component} | {timestamp}')
        print(f'   Message: {message[:100]}...' if len(message) > 100 else f'   Message: {message}')
        if session_id:
            print(f'   Session: {session_id}')
        if game_id:
            print(f'   Game: {game_id}')
        print()
    
    # Check system health
    print('🏥 SYSTEM HEALTH CHECK')
    print('=' * 50)
    
    # Count errors by type
    cursor.execute('''
        SELECT log_level, COUNT(*) as count
        FROM system_logs 
        WHERE timestamp > datetime('now', '-2 hours')
        GROUP BY log_level
        ORDER BY count DESC
    ''')
    
    error_counts = cursor.fetchall()
    for level, count in error_counts:
        print(f'📈 {level}: {count} occurrences')
    
    print()
    
    # Check for specific error patterns
    cursor.execute('''
        SELECT message, COUNT(*) as count
        FROM system_logs 
        WHERE log_level = 'ERROR' 
        AND timestamp > datetime('now', '-2 hours')
        GROUP BY message
        ORDER BY count DESC
        LIMIT 10
    ''')
    
    error_patterns = cursor.fetchall()
    if error_patterns:
        print('🔍 TOP ERROR PATTERNS:')
        for message, count in error_patterns:
            print(f'   {count}x: {message[:80]}...' if len(message) > 80 else f'   {count}x: {message}')
    else:
        print('✅ No error patterns found in the last 2 hours')
    
    # Check recent game results
    print()
    print('🎮 RECENT GAME RESULTS')
    print('=' * 50)
    
    cursor.execute('''
        SELECT game_id, final_score, total_actions, win_detected, level_completions, status
        FROM game_results 
        WHERE start_time > datetime('now', '-2 hours')
        ORDER BY start_time DESC 
        LIMIT 10
    ''')
    
    games = cursor.fetchall()
    if games:
        print(f'📊 Recent Games (last 2 hours): {len(games)}')
        for game in games:
            game_id, score, actions, win, levels, status = game
            print(f'   {game_id}: Score={score}, Actions={actions}, Win={win}, Levels={levels}, Status={status}')
    else:
        print('📊 No recent games found in the last 2 hours')
    
    # Check error_logs table specifically
    print()
    print('🚨 ERROR LOGS TABLE')
    print('=' * 50)
    
    cursor.execute('''
        SELECT error_type, error_message, timestamp, session_id, game_id
        FROM error_logs 
        WHERE timestamp > datetime('now', '-2 hours')
        ORDER BY timestamp DESC 
        LIMIT 10
    ''')
    
    error_logs = cursor.fetchall()
    if error_logs:
        print(f'📊 Error Logs (last 2 hours): {len(error_logs)}')
        for error in error_logs:
            error_type, message, timestamp, session_id, game_id = error
            print(f'   {error_type} | {timestamp}')
            print(f'   Message: {message[:100]}...' if len(message) > 100 else f'   Message: {message}')
            if session_id:
                print(f'   Session: {session_id}')
            if game_id:
                print(f'   Game: {game_id}')
            print()
    else:
        print('📊 No error logs found in the last 2 hours')
    
    conn.close()

if __name__ == "__main__":
    analyze_errors()
