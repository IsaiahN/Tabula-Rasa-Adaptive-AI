import importlib
m = importlib.import_module('src.arc_integration.continuous_learning_loop')
print('MODULE_FILE:', getattr(m, '__file__', None))
print('HAS_CLASS:', hasattr(m, 'ContinuousLearningLoop'))
# Check for method on class
cls = getattr(m, 'ContinuousLearningLoop', None)
if cls is None:
    print('CLASS_NONE')
else:
    print('HAS_METHOD_ON_CLASS:', hasattr(cls, 'start_training_session'))
    # Also check module-level attributes that match
print('MODULE_DIRLIST_STARTS_WITH_START_TRAINING:', any('start_training_session' == n for n in dir(m)))
