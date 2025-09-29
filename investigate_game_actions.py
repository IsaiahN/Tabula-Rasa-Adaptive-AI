#!/usr/bin/env python3
"""
Database investigation script for game As66-821a4dcad9c2
Analyzes action patterns and persistence logs to identify ACTION1 spam issue
"""

import sqlite3
import json
from collections import Counter
from datetime import datetime

# Database connection
DB_PATH = "tabula_rasa.db"

def investigate_game_actions():
    """Investigate action patterns for the stuck game."""

    print("🔍 INVESTIGATING GAME: As66-821a4dcad9c2")
    print("📋 SESSION: f9a3eeee-99e5-4566-b833-fbfdc37e6962")
    print("=" * 80)

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        cursor = conn.cursor()

        # 1. Check persistence debug logs for this game
        print("\n📊 1. PERSISTENCE DEBUG LOGS ANALYSIS")
        print("-" * 50)

        cursor.execute("""
            SELECT timestamp, function_name, operation_type, message, parameters
            FROM persistence_debug_logs
            WHERE (message LIKE '%As66-821a4dcad9c2%' OR parameters LIKE '%As66-821a4dcad9c2%')
            ORDER BY timestamp DESC
            LIMIT 50
        """)

        persistence_logs = cursor.fetchall()
        print(f"Found {len(persistence_logs)} persistence log entries")

        # Analyze action sequences
        action_sequences = []
        for log in persistence_logs:
            if log['function_name'] == 'persist_winning_sequence':
                try:
                    params = json.loads(log['parameters']) if log['parameters'] else {}
                    if 'sequence' in params:
                        action_sequences.append(params['sequence'])
                except:
                    # Try to extract from message
                    if '[' in log['message'] and ']' in log['message']:
                        sequence_str = log['message'].split('[')[1].split(']')[0]
                        try:
                            sequence = [int(x.strip()) for x in sequence_str.split(',')]
                            action_sequences.append(sequence)
                        except:
                            pass

        print(f"📈 Found {len(action_sequences)} action sequences:")
        sequence_counter = Counter(str(seq) for seq in action_sequences)
        for seq, count in sequence_counter.most_common(10):
            print(f"  {seq}: {count} times")

        # 2. Check if actions are all ACTION1
        if action_sequences:
            flat_actions = [action for seq in action_sequences for action in seq]
            action_counter = Counter(flat_actions)
            print(f"\n🎯 ACTION DISTRIBUTION:")
            for action_id, count in action_counter.most_common():
                percentage = (count / len(flat_actions)) * 100
                print(f"  ACTION{action_id}: {count} times ({percentage:.1f}%)")

            if action_counter.get(1, 0) / len(flat_actions) > 0.8:
                print("🚨 CONFIRMED: ACTION1 SPAM DETECTED!")

        # 3. Check game results table
        print(f"\n📋 2. GAME RESULTS ANALYSIS")
        print("-" * 50)

        cursor.execute("""
            SELECT game_id, session_id, status, final_score, total_actions,
                   actions_taken, win_detected, level_completions
            FROM game_results
            WHERE game_id LIKE '%As66-821a4dcad9c2%' OR session_id LIKE '%f9a3eeee-99e5-4566-b833-fbfdc37e6962%'
        """)

        game_results = cursor.fetchall()
        print(f"Found {len(game_results)} game result entries")

        for result in game_results:
            print(f"  Game: {result['game_id']}")
            print(f"  Session: {result['session_id']}")
            print(f"  Status: {result['status']}")
            print(f"  Score: {result['final_score']}")
            print(f"  Actions: {result['total_actions']}")
            print(f"  Win: {result['win_detected']}")
            if result['actions_taken']:
                try:
                    actions_list = json.loads(result['actions_taken'])
                    print(f"  Action sequence: {actions_list}")
                except:
                    print(f"  Actions raw: {result['actions_taken']}")
            print()

        # 4. Check action effectiveness table
        print(f"\n🎯 3. ACTION EFFECTIVENESS ANALYSIS")
        print("-" * 50)

        cursor.execute("""
            SELECT action_number, attempts, successes, success_rate, avg_score_impact, last_used
            FROM action_effectiveness
            WHERE game_id LIKE '%As66-821a4dcad9c2%'
            ORDER BY action_number
        """)

        effectiveness_data = cursor.fetchall()
        if effectiveness_data:
            print(f"Found {len(effectiveness_data)} action effectiveness entries")
            for eff in effectiveness_data:
                print(f"  ACTION{eff['action_number']}: {eff['attempts']} attempts, {eff['successes']} successes ({eff['success_rate']:.2f})")
        else:
            print("❌ No action effectiveness data found")

        # 5. Check training sessions
        print(f"\n📊 4. TRAINING SESSION ANALYSIS")
        print("-" * 50)

        cursor.execute("""
            SELECT session_id, game_id, start_time, end_time, status, total_actions,
                   total_wins, total_games, win_rate, avg_score
            FROM training_sessions
            WHERE session_id LIKE '%f9a3eeee-99e5-4566-b833-fbfdc37e6962%' OR game_id LIKE '%As66-821a4dcad9c2%'
        """)

        session_data = cursor.fetchall()
        print(f"Found {len(session_data)} session entries")

        for session in session_data:
            print(f"  Session: {session['session_id']}")
            print(f"  Game: {session['game_id']}")
            print(f"  Status: {session['status']}")
            print(f"  Total Actions: {session['total_actions']}")
            print(f"  Win Rate: {session['win_rate']}")
            print(f"  Start: {session['start_time']}")
            print(f"  End: {session['end_time']}")
            print()

        conn.close()

    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"💥 Unexpected error: {e}")

if __name__ == "__main__":
    investigate_game_actions()