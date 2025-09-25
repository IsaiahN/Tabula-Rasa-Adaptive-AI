# Simple test script to exercise LearnedPatternsManager.add_pattern
import sys
from pathlib import Path
# Ensure project root is on sys.path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.database.learned_patterns_manager import LearnedPatternsManager

if __name__ == '__main__':
    # Use a temporary file-backed database so multiple connections share schema
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(prefix='tr_test_db_', suffix='.db', delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        mgr = LearnedPatternsManager(db_path=tmp_path)
        pattern = {
        'type': 'unit_test',
        'actions': [1, 2, 3],
        'context': {'level': 'test', 'meta': {'a': 1}},
        'confidence': 0.42
        }
        print('Inserting pattern...')
        try:
            res = mgr.add_pattern(pattern, pattern_type='unit_test', success_rate=0.42)
            print('Insert result:')
            print(res)
        except Exception as e:
            print('Exception during insert:', e)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
