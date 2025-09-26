#!/usr/bin/env python3
"""
Fix Action Count Synchronization

This script fixes the mismatch between game_results.total_actions and the actual
number of action_traces for each game.
"""

import sqlite3
import sys
from pathlib import Path

def fix_action_count_sync(db_path: str = "tabula_rasa.db"):
    """Fix action count synchronization between game_results and action_traces."""

    print("=== ACTION COUNT SYNCHRONIZATION FIX ===")

    conn = sqlite3.connect(db_path)

    try:
        # Get all games with action count mismatches
        print("1. Identifying games with action count mismatches...")

        cursor = conn.execute('''
            SELECT
                g.game_id,
                g.total_actions as current_recorded,
                COUNT(at.id) as actual_traces
            FROM game_results g
            LEFT JOIN action_traces at ON g.game_id = at.game_id
            GROUP BY g.game_id
            HAVING g.total_actions != COUNT(at.id)
            ORDER BY COUNT(at.id) DESC
        ''')

        mismatches = cursor.fetchall()
        print(f"   Found {len(mismatches)} games with action count mismatches")

        # Fix each game
        fixed_count = 0
        total_actions_fixed = 0

        for game_id, current_recorded, actual_traces in mismatches:
            print(f"   Fixing game {game_id}: {current_recorded} -> {actual_traces}")

            # Update the total_actions in game_results to match actual traces
            conn.execute('''
                UPDATE game_results
                SET total_actions = ?
                WHERE game_id = ?
            ''', (actual_traces, game_id))

            fixed_count += 1
            total_actions_fixed += (actual_traces - current_recorded)

            # Commit every 100 updates to avoid large transactions
            if fixed_count % 100 == 0:
                conn.commit()
                print(f"   Committed {fixed_count} fixes...")

        # Final commit
        conn.commit()

        print(f"\n2. Synchronization completed:")
        print(f"   Fixed {fixed_count} games")
        print(f"   Added {total_actions_fixed} action count corrections")

        # Verify the fix
        print("\n3. Verification:")
        cursor = conn.execute('''
            SELECT COUNT(*) FROM game_results g
            LEFT JOIN action_traces at ON g.game_id = at.game_id
            WHERE g.total_actions != (
                SELECT COUNT(*) FROM action_traces at2
                WHERE at2.game_id = g.game_id
            )
        ''')

        remaining_mismatches = cursor.fetchone()[0]
        print(f"   Remaining mismatches: {remaining_mismatches}")

        # Show overall stats after fix
        cursor = conn.execute('SELECT SUM(total_actions) FROM game_results')
        total_recorded = cursor.fetchone()[0] or 0

        cursor = conn.execute('SELECT COUNT(*) FROM action_traces')
        total_traced = cursor.fetchone()[0] or 0

        print(f"   Total recorded actions now: {total_recorded}")
        print(f"   Total traced actions: {total_traced}")
        print(f"   Difference: {total_traced - total_recorded}")

        if remaining_mismatches == 0:
            print("✅ Action count synchronization completed successfully!")
        else:
            print("⚠️ Some mismatches remain - manual review may be needed")

    except Exception as e:
        print(f"❌ Error during synchronization: {e}")
        conn.rollback()
        return False

    finally:
        conn.close()

    return True

if __name__ == "__main__":
    success = fix_action_count_sync()
    sys.exit(0 if success else 1)