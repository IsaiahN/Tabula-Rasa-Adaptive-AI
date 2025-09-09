import importlib, inspect
m=importlib.import_module('src.arc_integration.continuous_learning_loop')
print('module_has_start', hasattr(m, 'start_training_session'))
cls = getattr(m,'ContinuousLearningLoop')
print('class_has_start', hasattr(cls, 'start_training_session'))
print('callable_module_start', callable(getattr(m, 'start_training_session', None)))
if hasattr(m,'start_training_session'):
    print('module_start_source_firstline:')
    print(inspect.getsource(m.start_training_session).splitlines()[0])
else:
    print('no module-level start_training_session')
