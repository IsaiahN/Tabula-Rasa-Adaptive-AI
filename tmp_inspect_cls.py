import importlib, inspect
m = importlib.import_module('src.arc_integration.continuous_learning_loop')
cls = getattr(m, 'ContinuousLearningLoop')
print('CLASS_NAME', cls.__name__)
print('CLASS_MODULE', cls.__module__)
print('HAS_METHOD_ATTR', hasattr(cls, 'start_training_session'))
print('DIR_SNIPPET', [n for n in dir(cls) if 'start' in n or 'training' in n][:50])
try:
    src = inspect.getsource(cls)
    print('SOURCE_SNIPPET:\n', '\n'.join(src.splitlines()[:200]))
except Exception as e:
    print('GETSOURCE_ERR', e)
