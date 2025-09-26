#!/usr/bin/env python3
"""
Clean Up Duplicate Action Traces

This script removes duplicate action_traces records and fixes the action count
synchronization properly.
"""

import sqlite3
import sys

def cleanup_duplicate_action_traces(db_path: str = "tabula_rasa.db"):
    """Remove duplicate action traces and fix synchronization."""

    print("=== DUPLICATE ACTION TRACES CLEANUP ===")

    conn = sqlite3.connect(db_path)

    try:
        # Check initial state
        print("1. Checking initial state...")
        cursor = conn.execute('SELECT COUNT(*) FROM action_traces')
        initial_count = cursor.fetchone()[0]
        print(f"   Initial action traces: {initial_count:,}")

        # Find games with massive duplicates
        cursor = conn.execute('''
            SELECT
                game_id,
                action_number,
                COUNT(*) as duplicates
            FROM action_traces
            GROUP BY game_id, action_number
            HAVING COUNT(*) > 1
            ORDER BY duplicates DESC
            LIMIT 10
        ''')

        duplicates = cursor.fetchall()
        print(f"   Found {len(duplicates)} sets of duplicates:")
        total_duplicates = 0
        for game_id, action_num, count in duplicates:
            excess = count - 1  # Keep one, remove the rest
            total_duplicates += excess
            print(f"     Game {game_id}, action {action_num}: {count:,} copies ({excess:,} to remove)")

        print(f"   Total duplicate records to remove: {total_duplicates:,}")

        # Remove duplicates by keeping only the oldest record for each game+action combination
        print("\n2. Removing duplicates...")

        cursor = conn.execute('''
            DELETE FROM action_traces
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM action_traces
                GROUP BY game_id, action_number
            )
        ''')

        removed_count = cursor.rowcount
        conn.commit()

        print(f"   Removed {removed_count:,} duplicate records")

        # Check final state
        cursor = conn.execute('SELECT COUNT(*) FROM action_traces')
        final_count = cursor.fetchone()[0]
        print(f"   Final action traces: {final_count:,}")
        print(f"   Net reduction: {initial_count - final_count:,}")

        # Update game_results.total_actions to match actual unique actions
        print("\n3. Updating game_results.total_actions...")

        cursor = conn.execute('''
            UPDATE game_results
            SET total_actions = (
                SELECT COUNT(DISTINCT action_number)
                FROM action_traces
                WHERE action_traces.game_id = game_results.game_id
            )
        ''')

        updated_games = cursor.rowcount
        conn.commit()

        print(f"   Updated {updated_games} game records")

        # Verification
        print("\n4. Verification:")

        # Check for remaining duplicates
        cursor = conn.execute('''
            SELECT COUNT(*) FROM (
                SELECT game_id, action_number, COUNT(*)
                FROM action_traces
                GROUP BY game_id, action_number
                HAVING COUNT(*) > 1
            )
        ''')

        remaining_dups = cursor.fetchone()[0]
        print(f"   Remaining duplicate sets: {remaining_dups}")

        # Check synchronization
        cursor = conn.execute('SELECT SUM(total_actions) FROM game_results')
        total_recorded = cursor.fetchone()[0] or 0

        cursor = conn.execute('SELECT COUNT(*) FROM action_traces')
        total_traced = cursor.fetchone()[0] or 0

        print(f"   Total recorded actions: {total_recorded:,}")
        print(f"   Total traced actions: {total_traced:,}")
        print(f"   Difference: {abs(total_traced - total_recorded):,}")

        if remaining_dups == 0 and abs(total_traced - total_recorded) < 100:
            print("SUCCESS: Duplicate cleanup completed successfully!")
        else:
            print("WARNING: Some issues may remain")

    except Exception as e:
        print(f"ERROR during cleanup: {e}")
        conn.rollback()
        return False

    finally:
        conn.close()

    return True

if __name__ == "__main__":
    success = cleanup_duplicate_action_traces()
    sys.exit(0 if success else 1)