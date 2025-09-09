import importlib, inspect
m = importlib.import_module('src.arc_integration.continuous_learning_loop')
cls = getattr(m, 'ContinuousLearningLoop')
lines, start = inspect.getsourcelines(cls)
print('CLASS_START_LINE', start)
print('CLASS_SOURCE_LINES', len(lines))
print('CLASS_LAST_LINE_PREVIEW:', lines[-1][:120])
print('HAS_method_start', hasattr(cls, 'start_training_session'))
# print all attribute names
print('DIR_LEN', len(dir(cls)))
for n in dir(cls):
    if 'start' in n or 'training' in n or n.startswith('_load'):
        print('ATTR', n)
