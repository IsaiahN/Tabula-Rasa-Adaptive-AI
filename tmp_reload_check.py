import importlib, inspect
m = importlib.import_module('src.arc_integration.continuous_learning_loop')
importlib.reload(m)
print('MODULE_FILE', getattr(m,'__file__',None))
print('HAS_CLASS', hasattr(m,'ContinuousLearningLoop'))
cls = getattr(m,'ContinuousLearningLoop')
print('CLASS_HAS_start', hasattr(cls,'start_training_session'))
print('MODULE_HAS_start', hasattr(m,'start_training_session'))
if hasattr(cls,'start_training_session'):
    print('OK method attached')
    print('callable?', callable(getattr(cls,'start_training_session')))
    try:
        print('source first line for method attached:', inspect.getsource(getattr(cls,'start_training_session')).splitlines()[0])
    except Exception as e:
        print('src err', e)
