import importlib
import sys
from pathlib import Path

# Ensure the repository root is on sys.path so `src` can be imported when
# running this script directly from the tools/ folder.
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

try:
    m = importlib.import_module('src.arc_integration.continuous_learning_loop')
    print('OK', getattr(m, 'ContinuousLearningLoop', None) is not None)
except Exception as e:
    print('IMPORT ERROR:', type(e).__name__, e)
    raise
