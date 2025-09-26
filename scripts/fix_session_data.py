#!/usr/bin/env python3
"""
Script to fix session data and ensure proper game-session linking.
"""

import sqlite3
import sys
import os
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def fix_session_data():
    """Fix session data and create missing sessions."""
    print("=== FIXING SESSION DATA ===")

    try:
        db = sqlite3.connect('tabula_rasa.db')
        cursor = db.cursor()

        # 1. Find all unique session_ids in game_results that don't exist in training_sessions
        cursor.execute("""
            SELECT DISTINCT g.session_id, MIN(g.start_time) as first_game, MAX(g.start_time) as last_game,
                   COUNT(*) as game_count, AVG(g.final_score) as avg_score,
                   SUM(CASE WHEN g.win_detected = 1 THEN 1 ELSE 0 END) as wins
            FROM game_results g
            LEFT JOIN training_sessions t ON g.session_id = t.session_id
            WHERE t.session_id IS NULL
            GROUP BY g.session_id
            ORDER BY first_game DESC
        """)

        orphaned_sessions = cursor.fetchall()

        if orphaned_sessions:
            print(f"Found {len(orphaned_sessions)} orphaned session IDs:")

            for session_id, first_game, last_game, game_count, avg_score, wins in orphaned_sessions:
                print(f"  {session_id}: {game_count} games, {wins} wins, avg score {avg_score:.2f}")

                # Create the missing training session
                win_rate = wins / game_count if game_count > 0 else 0.0

                cursor.execute("""
                    INSERT INTO training_sessions
                    (session_id, start_time, end_time, mode, status, total_games,
                     total_wins, win_rate, avg_score, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    first_game,
                    last_game,
                    'recovered',  # Mark as recovered session
                    'completed',  # Assume completed since games exist
                    game_count,
                    wins,
                    win_rate,
                    avg_score or 0.0,
                    first_game,
                    last_game
                ))

                print(f"    Created training session for {session_id}")

            db.commit()
            print(f"Fixed {len(orphaned_sessions)} orphaned sessions")
        else:
            print("No orphaned sessions found")

        # 2. Fix games with immediate end times (same as start time)
        cursor.execute("""
            SELECT COUNT(*) FROM game_results
            WHERE start_time = end_time
        """)
        immediate_games = cursor.fetchone()[0]

        if immediate_games > 0:
            print(f"\nFound {immediate_games} games with immediate end times")

            # Update games to have NULL end_time if they ended immediately
            # This indicates they weren't properly finished
            cursor.execute("""
                UPDATE game_results
                SET end_time = NULL,
                    status = 'incomplete'
                WHERE start_time = end_time AND status = 'NOT_FINISHED'
            """)

            affected = cursor.rowcount
            print(f"Updated {affected} games to have NULL end_time (indicating incomplete)")
            db.commit()

        # 3. Update session statistics based on actual game data
        print("\nUpdating session statistics...")

        cursor.execute("""
            SELECT session_id FROM training_sessions
        """)
        all_sessions = cursor.fetchall()

        for (session_id,) in all_sessions:
            # Calculate actual statistics from games
            cursor.execute("""
                SELECT COUNT(*) as total_games,
                       SUM(CASE WHEN win_detected = 1 THEN 1 ELSE 0 END) as total_wins,
                       AVG(final_score) as avg_score,
                       SUM(total_actions) as total_actions
                FROM game_results
                WHERE session_id = ?
            """, (session_id,))

            stats = cursor.fetchone()
            if stats:
                total_games, total_wins, avg_score, total_actions = stats
                total_games = total_games or 0
                total_wins = total_wins or 0
                avg_score = avg_score or 0.0
                total_actions = total_actions or 0
                win_rate = total_wins / total_games if total_games > 0 else 0.0

                cursor.execute("""
                    UPDATE training_sessions
                    SET total_games = ?,
                        total_wins = ?,
                        win_rate = ?,
                        avg_score = ?,
                        total_actions = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                """, (total_games, total_wins, win_rate, avg_score, total_actions, session_id))

        db.commit()
        print("Updated all session statistics")

        # 4. Close any sessions that are still marked as "running"
        cursor.execute("""
            UPDATE training_sessions
            SET status = 'completed',
                end_time = updated_at
            WHERE status = 'running' AND session_id != (
                SELECT session_id FROM training_sessions
                WHERE status = 'running'
                ORDER BY start_time DESC
                LIMIT 1
            )
        """)

        closed_sessions = cursor.rowcount
        if closed_sessions > 0:
            print(f"Closed {closed_sessions} old running sessions")
            db.commit()

        print("\n=== SESSION DATA FIX COMPLETE ===")

        # Show summary
        cursor.execute("SELECT COUNT(*) FROM training_sessions")
        total_sessions = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM game_results")
        total_games = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM game_results g
            JOIN training_sessions t ON g.session_id = t.session_id
        """)
        linked_games = cursor.fetchone()[0]

        print(f"\nFinal Summary:")
        print(f"  Total sessions: {total_sessions}")
        print(f"  Total games: {total_games}")
        print(f"  Properly linked games: {linked_games}")
        print(f"  Link success rate: {linked_games/total_games*100:.1f}%" if total_games > 0 else "  No games")

        db.close()
        return True

    except Exception as e:
        print(f"Error fixing session data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = fix_session_data()
    sys.exit(0 if success else 1)