import json, sys, time
from pathlib import Path
sys.path.insert(0, 'src')
try:
    from arc_integration.action_trace_logger import log_action_trace
except Exception as e:
    print('import error', e)
    raise
rec = {
    'ts': int(time.time()),
    'event':'action_executed',
    'game_id':'test-game-001',
    'guid':'testguid1234',
    'action_number':1,
    'action_type':'ACTION6',
    'x':42,
    'y':7,
    'response_score':5,
    'success':True
}
log_action_trace(rec)
p = Path('data/action_traces.ndjson')
if p.exists():
    print(p.read_text().strip().splitlines()[-1])
else:
    print('missing')
