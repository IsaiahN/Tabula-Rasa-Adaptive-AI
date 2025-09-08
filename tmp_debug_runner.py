import logging
import runpy

# Configure root logger to DEBUG and ensure output to console
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

# Run the trainer script as __main__ so CLI args are used
runpy.run_path('master_arc_trainer.py', run_name='__main__')
