import time
from pathlib import Path

LOG = Path('master_trainer.log')
OUT = Path('monitor_capture.log')
DURATION = 1200  # 20 minutes
INTERVAL = 120   # 2 minutes
SNAP_TAIL_LINES = 500

start = time.time()
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(f"=== Monitor start: {time.ctime()} ===\n")

while time.time() - start < DURATION:
    time.sleep(INTERVAL)
    OUT.write_text(f"--- Snapshot: {time.ctime()} ---\n", append=False)
    if LOG.exists():
        try:
            with LOG.open('r', encoding='utf-8', errors='replace') as f:
                lines = f.read().splitlines()
                tail = '\n'.join(lines[-SNAP_TAIL_LINES:])
        except Exception as e:
            tail = f"ERROR reading log: {e}"
    else:
        tail = "LOG MISSING"
    with OUT.open('a', encoding='utf-8', errors='replace') as out:
        out.write('--- Snapshot: ' + time.ctime() + '\n')
        out.write(tail + '\n')

OUT.write_text(f"=== Monitor end: {time.ctime()} ===\n")
print('Monitor completed')
