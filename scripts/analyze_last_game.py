#!/usr/bin/env python3
"""
Analyze the last game played to verify comprehensive data saving.
"""

import sqlite3
import json
from datetime import datetime

def analyze_last_game():
    """Analyze the most recent game and its data coverage."""
    print("=== LAST GAME DATA ANALYSIS ===")

    try:
        db = sqlite3.connect('tabula_rasa.db')
        cursor = db.cursor()

        # Get the most recent game
        cursor.execute("""
            SELECT game_id, session_id, start_time, end_time, status,
                   final_score, total_actions, win_detected
            FROM game_results
            ORDER BY start_time DESC
            LIMIT 1
        """)

        recent_game = cursor.fetchone()
        if not recent_game:
            print("No games found in database")
            return False

        game_id, session_id, start_time, end_time, status, final_score, total_actions, win_detected = recent_game

        print(f"\nMost Recent Game:")
        print(f"  Game ID: {game_id}")
        print(f"  Session ID: {session_id}")
        print(f"  Start Time: {start_time}")
        print(f"  End Time: {end_time}")
        print(f"  Status: {status}")
        print(f"  Final Score: {final_score}")
        print(f"  Total Actions: {total_actions}")
        print(f"  Win Detected: {win_detected}")

        # Check for data completeness issues
        issues = []

        if end_time and start_time == end_time:
            issues.append("Game ended at same time it started (no gameplay time)")

        if status == "NOT_FINISHED":
            issues.append("Game status is NOT_FINISHED")

        if total_actions == 0:
            issues.append("No actions recorded")

        if final_score == 0.0 and not win_detected:
            issues.append("Zero score and no win")

        # Check related data tables
        related_tables = [
            ('action_effectiveness', 'game_id', 'Action effectiveness tracking'),
            ('coordinate_intelligence', 'game_id', 'Coordinate intelligence'),
            ('action_traces', 'game_id', 'Action traces'),
            ('system_logs', 'game_id', 'System logs for game'),
            ('system_logs', 'session_id', 'System logs for session')
        ]

        print(f"\nRelated Data Coverage:")
        data_coverage = {}

        for table, id_column, description in related_tables:
            try:
                if id_column == 'game_id':
                    cursor.execute(f'SELECT COUNT(*) FROM {table} WHERE {id_column} = ?', (game_id,))
                else:
                    cursor.execute(f'SELECT COUNT(*) FROM {table} WHERE {id_column} = ?', (session_id,))

                count = cursor.fetchone()[0]
                data_coverage[table] = count
                print(f"  {description}: {count} entries")

                if count == 0:
                    issues.append(f"No {description.lower()} recorded")

            except Exception as e:
                print(f"  {description}: Error checking - {e}")
                issues.append(f"Error checking {description}")

        # Check session data
        print(f"\nSession Analysis:")
        cursor.execute("""
            SELECT session_id, start_time, end_time, status, total_games,
                   total_wins, win_rate, avg_score
            FROM training_sessions
            WHERE session_id = ?
        """, (session_id,))

        session_data = cursor.fetchone()
        if session_data:
            sess_id, sess_start, sess_end, sess_status, total_games, total_wins, win_rate, avg_score = session_data
            print(f"  Session ID: {sess_id}")
            print(f"  Start Time: {sess_start}")
            print(f"  End Time: {sess_end}")
            print(f"  Status: {sess_status}")
            print(f"  Total Games: {total_games}")
            print(f"  Total Wins: {total_wins}")
            print(f"  Win Rate: {win_rate}")
            print(f"  Avg Score: {avg_score}")

            if sess_end is None and sess_status == "running":
                issues.append("Session is still marked as running")

            if total_games == 0:
                issues.append("Session shows 0 total games despite having game records")

        else:
            issues.append("No session data found for this game")

        # Check for scorecard data
        print(f"\nScorecard Analysis:")
        cursor.execute("""
            SELECT COUNT(*) FROM system_logs
            WHERE component = 'experiment' AND message LIKE '%scorecard%'
        """)
        scorecard_logs = cursor.fetchone()[0]
        print(f"  Scorecard-related logs: {scorecard_logs}")

        if scorecard_logs == 0:
            issues.append("No scorecard data found")

        # Summary
        print(f"\n=== ANALYSIS SUMMARY ===")

        total_data_points = sum(data_coverage.values())
        print(f"Total related data points: {total_data_points}")

        if issues:
            print(f"\nIssues Found ({len(issues)}):")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("\nNo issues found - data appears to be saved correctly!")

        # Recommendations
        print(f"\n=== RECOMMENDATIONS ===")

        critical_issues = [issue for issue in issues if any(word in issue.lower()
                          for word in ['no actions', 'no session', 'error checking', 'same time'])]

        if critical_issues:
            print("Critical issues requiring immediate attention:")
            for issue in critical_issues:
                print(f"  - {issue}")

            print("\nSuggested fixes:")
            if any('same time' in issue for issue in issues):
                print("  - Fix game timing: Ensure games run for actual time duration")
            if any('no actions' in issue for issue in issues):
                print("  - Fix action recording: Ensure actions are properly saved")
            if any('no session' in issue for issue in issues):
                print("  - Fix session linking: Ensure games are linked to sessions")

        else:
            print("Data saving appears to be working correctly.")
            print("Minor improvements could be made in:")
            for issue in issues:
                print(f"  - {issue}")

        db.close()
        return len(critical_issues) == 0

    except Exception as e:
        print(f"Error analyzing last game: {e}")
        return False

if __name__ == "__main__":
    success = analyze_last_game()
    exit(0 if success else 1)