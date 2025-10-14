import subprocess
import sys
import os
import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'run_minimal.py')

@pytest.mark.skipif(not os.path.exists(SCRIPT), reason="run_minimal.py not present")
def test_smoke_run_minimal():
    # Run minimal script with the explicit stub flag
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    p = subprocess.run([sys.executable, SCRIPT, '--use-stub-api-for-ci', '--max-actions', '2'], capture_output=True, text=True, timeout=20, env=env)
    # script should exit with code 0 under normal circumstances
    assert p.returncode == 0, f"run_minimal failed: stdout={p.stdout}\nstderr={p.stderr}"
