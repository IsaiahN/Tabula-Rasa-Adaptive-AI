"""
Compatibility Shim

Provides backward compatibility for functions that were accidentally
defined at module level but expect 'self' as the first argument.
"""

import inspect
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class CompatibilityShim:
    """Handles compatibility for module-level functions that expect 'self'."""
    
    @staticmethod
    def attach_module_functions_to_class(module_globals: Dict[str, Any], target_class: type):
        """
        Attach module-level functions that expect 'self' to a class.
        
        This preserves backward compatibility for functions that were
        accidentally defined at module level but expect 'self' as the first argument.
        """
        try:
            for name, obj in list(module_globals.items()):
                if inspect.isfunction(obj) or inspect.iscoroutinefunction(obj):
                    try:
                        sig = inspect.signature(obj)
                        params = list(sig.parameters.keys())
                        if params and params[0] == 'self':
                            # Only attach if the class doesn't already provide it
                            if not hasattr(target_class, name):
                                setattr(target_class, name, obj)
                                logger.debug(f"Attached function {name} to {target_class.__name__}")
                    except Exception as e:
                        # Ignore any functions we cannot introspect
                        logger.debug(f"Could not introspect function {name}: {e}")
                        continue
        except Exception as e:
            # Best-effort shim - failure here is non-fatal
            logger.warning(f"Error in compatibility shim: {e}")
            # Missing methods will be handled at runtime
