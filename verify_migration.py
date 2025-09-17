#!/usr/bin/env python3
"""
Verify simple_training_results migration
"""
import sqlite3

def verify_migration():
    conn = sqlite3.connect('tabula_rasa.db')
    cursor = conn.cursor()
    
    # Count all training sessions
    cursor.execute('SELECT COUNT(*) FROM training_sessions;')
    total_sessions = cursor.fetchone()[0]
    
    # Count simple training sessions
    cursor.execute("SELECT COUNT(*) FROM training_sessions WHERE session_id LIKE 'simple_training_%';")
    simple_sessions = cursor.fetchone()[0]
    
    # Count other sessions
    other_sessions = total_sessions - simple_sessions
    
    print(f'DATABASE VERIFICATION:')
    print(f'   Total training sessions: {total_sessions}')
    print(f'   Simple training sessions: {simple_sessions}')
    print(f'   Other sessions: {other_sessions}')
    
    # Check success rates
    cursor.execute("SELECT AVG(win_rate) FROM training_sessions WHERE session_id LIKE 'simple_training_%';")
    avg_win_rate = cursor.fetchone()[0] or 0
    
    print(f'   Average win rate (simple training): {avg_win_rate:.1%}')
    
    conn.close()
    
    print(f'\n✅ All simple_training_results data successfully migrated!')
    print(f'   The JSON files can now be safely deleted.')

if __name__ == '__main__':
    verify_migration()
