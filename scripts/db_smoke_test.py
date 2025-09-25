import asyncio
from pathlib import Path
import sys

# Ensure repo root on path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


async def _main():
    try:
        from src.database.performance_data_manager import get_performance_manager
        from src.database.api import get_database
        from src.database.system_integration import get_system_integration
        from src.database.persistence_helpers import persist_winning_sequence, persist_button_priorities, persist_governor_decision
    except Exception as e:
        print('Import failed:', e)
        return

    pm = get_performance_manager()
    integration = get_system_integration()
    db = get_database()

    # Insert a performance row via the manager
    perf_ok = await pm.store_performance_data('smoke_session', {
        'game_id': 'smoke_game',
        'score': 100,
        'win_rate': 1.0,
        'learning_efficiency': 0.9,
        'metadata': {'source': 'smoke_test'}
    })
    print('store_performance_data:', perf_ok)

    # Use persistence helpers to persist winning sequence and button priorities
    try:
        ok_ws = await persist_winning_sequence('smoke_game', [1, 2, 3])
        print('persist_winning_sequence:', ok_ws)
    except Exception as e:
        print('persist_winning_sequence failed:', e)

    try:
        ok_bp = await persist_button_priorities('smoke_type', 1, 1, 'score_button', 0.9)
        print('persist_button_priorities:', ok_bp)
    except Exception as e:
        print('persist_button_priorities failed:', e)

    # Use SystemIntegration APIs where signature-known
    try:
        ok_store_strategy = await integration.store_winning_strategy(
            strategy_id='smoke_strategy',
            game_type='smoke_type',
            game_id='smoke_game',
            action_sequence=[1,2,3],
            score_progression=[0,10,100],
            total_score_increase=100,
            efficiency=0.9
        )
        print('store_winning_strategy:', ok_store_strategy)
    except Exception as e:
        print('store_winning_strategy failed:', e)

    # Persist a governor decision via helper
    try:
        ok_gd = await persist_governor_decision('smoke_session', 'test_decision', {'k': 'v'}, 0.5, {'action': 'continue'})
        print('persist_governor_decision:', ok_gd)
    except Exception as e:
        print('persist_governor_decision failed:', e)

    # Verify counts by querying DB via async connection
    try:
        async with db.get_connection() as conn:
            for table in ['winning_sequences', 'button_priorities', 'strategy_refinements', 'strategy_replications', 'action_effectiveness_detailed']:
                try:
                    cur = conn.execute(f'SELECT COUNT(*) as count FROM {table}')
                    row = cur.fetchone()
                    print(table, row['count'] if row else 0)
                except Exception as e:
                    print(f'Query failed for {table}:', e)
    except Exception as e:
        print('Count query failed:', e)

    # Insert minimal rows for remaining empty tables if they are still empty
    try:
        async with db.get_connection() as conn:
            # strategy_refinements (schema: strategy_id, refinement_attempt, original_efficiency, new_efficiency, improvement, action_sequence)
            try:
                conn.execute("""
                    INSERT INTO strategy_refinements (strategy_id, refinement_attempt, original_efficiency, new_efficiency, improvement, action_sequence)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, ('smoke_strategy', 1, 0.5, 0.7, 0.2, '[1,2,3]'))
            except Exception as e:
                print('strategy_refinements insert failed:', e)

            # strategy_replications (schema: strategy_id, game_id, replication_attempt, expected_efficiency, actual_efficiency, success)
            try:
                conn.execute("""
                    INSERT INTO strategy_replications (strategy_id, game_id, replication_attempt, expected_efficiency, actual_efficiency, success)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, ('smoke_strategy', 'smoke_game', 1, 0.7, 0.65, 0))
            except Exception as e:
                print('strategy_replications insert failed:', e)

            # action_effectiveness_detailed (schema: game_id, action_number, coordinates_x, coordinates_y, frame_changes, movement_detected, score_changes, action_unlocks, stagnation_breaks, success_rate, efficiency_score)
            try:
                conn.execute("""
                    INSERT INTO action_effectiveness_detailed (game_id, action_number, coordinates_x, coordinates_y, frame_changes, movement_detected, score_changes, action_unlocks, stagnation_breaks, success_rate, efficiency_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, ('smoke_game', 1, 0, 0, 1, 1, 2, 0, 0, 0.5, 0.4))
            except Exception as e:
                print('action_effectiveness_detailed insert failed:', e)

            conn.commit()
        print('Additional INSERTs executed')
    except Exception as e:
        print('Additional INSERTs failed:', e)


if __name__ == '__main__':
    asyncio.run(_main())
