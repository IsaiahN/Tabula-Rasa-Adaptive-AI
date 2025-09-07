import time
from src.core.meta_cognitive_dashboard import MetaCognitiveDashboard, DashboardMode

print('Instantiating dashboard...')
d = MetaCognitiveDashboard(mode=DashboardMode.CONSOLE, update_interval=0.5)
d.start('smoke_test_session')
# send a few updates
for i in range(3):
    d.log_performance_update({'score': i*10, 'win_score': 50+i*5}, source='game')
    d.log_governor_decision({'type':'adjust_params'}, confidence=0.6 + i*0.1, context={'iter':i})
    time.sleep(0.6)

print('Stopping...')
d.stop()
print('Exporting session...')
print(d.export_session_data('smoke_session.json'))
