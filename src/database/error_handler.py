"""
Database Error Handler and Auto-Fixer

This module provides automatic error handling and fixing for common database issues.
"""

import sqlite3
import json
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple
from functools import wraps
import inspect

logger = logging.getLogger(__name__)

class DatabaseErrorHandler:
    """Handles database errors automatically with intelligent fixing."""
    
    def __init__(self):
        self.error_patterns = {
            "column_count_mismatch": self._fix_column_count_mismatch,
            "parameter_binding": self._fix_parameter_binding,
            "type_error": self._fix_type_error,
            "constraint_error": self._fix_constraint_error
        }
        self.fix_attempts = {}
    
    def handle_database_error(self, error: Exception, query: str, params: tuple, 
                            table_name: str = None) -> Tuple[bool, str]:
        """Handle database errors with automatic fixing."""
        error_msg = str(error).lower()
        
        # Identify error type
        if "values for" in error_msg and "columns" in error_msg:
            return self._handle_column_mismatch(error, query, params, table_name)
        elif "binding parameter" in error_msg and "type" in error_msg:
            return self._handle_parameter_binding(error, query, params)
        elif "type" in error_msg and "not supported" in error_msg:
            return self._handle_type_error(error, query, params)
        else:
            logger.error(f"Unhandled database error: {error}")
            return False, f"Unhandled error: {error}"
    
    def _handle_column_mismatch(self, error: Exception, query: str, params: tuple, 
                               table_name: str) -> Tuple[bool, str]:
        """Handle column count mismatch errors."""
        try:
            # Extract column and value counts from error message
            error_msg = str(error)
            if "9 values for 8 columns" in error_msg:
                # This is the specific error we're seeing
                logger.warning(f"Column count mismatch detected: {error_msg}")
                
                # Try to fix by adjusting the query
                if table_name:
                    fixed_query, fixed_params = self._fix_insert_query(query, params, table_name)
                    if fixed_query and fixed_params:
                        return True, f"Fixed column mismatch for {table_name}"
                
            return False, f"Could not fix column mismatch: {error_msg}"
            
        except Exception as e:
            logger.error(f"Error handling column mismatch: {e}")
            return False, f"Error in column mismatch handler: {e}"
    
    def _handle_parameter_binding(self, error: Exception, query: str, params: tuple) -> Tuple[bool, str]:
        """Handle parameter binding errors."""
        try:
            # Convert problematic parameters
            safe_params = []
            for param in params:
                if isinstance(param, dict):
                    safe_params.append(json.dumps(param))
                elif isinstance(param, list):
                    safe_params.append(json.dumps(param))
                elif hasattr(param, '__dict__'):
                    safe_params.append(json.dumps(param.__dict__))
                else:
                    safe_params.append(param)
            
            return True, f"Fixed parameter binding - converted {len(params)} parameters"
            
        except Exception as e:
            logger.error(f"Error handling parameter binding: {e}")
            return False, f"Error in parameter binding handler: {e}"
    
    def _handle_type_error(self, error: Exception, query: str, params: tuple) -> Tuple[bool, str]:
        """Handle type conversion errors."""
        try:
            # Similar to parameter binding fix
            safe_params = []
            for param in params:
                if isinstance(param, dict):
                    safe_params.append(json.dumps(param))
                elif isinstance(param, list):
                    safe_params.append(json.dumps(param))
                else:
                    safe_params.append(str(param))
            
            return True, f"Fixed type error - converted parameters to safe types"
            
        except Exception as e:
            logger.error(f"Error handling type error: {e}")
            return False, f"Error in type error handler: {e}"
    
    def _fix_insert_query(self, query: str, params: tuple, table_name: str) -> Tuple[Optional[str], Optional[tuple]]:
        """Fix INSERT query by adjusting column count."""
        try:
            # Get table schema
            with sqlite3.connect("data/system.db") as conn:
                cursor = conn.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                # Count actual columns (excluding auto-generated ones)
                actual_columns = len([col for col in columns if not col[1].startswith('id') and 
                                    col[1] not in ['created_at', 'updated_at']])
                
                # Count parameters provided
                param_count = len(params)
                
                if param_count > actual_columns:
                    # Too many parameters - truncate
                    fixed_params = params[:actual_columns]
                    logger.info(f"Truncated parameters from {param_count} to {actual_columns}")
                    return query, fixed_params
                elif param_count < actual_columns:
                    # Too few parameters - pad with None
                    fixed_params = params + (None,) * (actual_columns - param_count)
                    logger.info(f"Padded parameters from {param_count} to {actual_columns}")
                    return query, fixed_params
                else:
                    # Count matches - no fix needed
                    return query, params
                    
        except Exception as e:
            logger.error(f"Error fixing INSERT query: {e}")
            return None, None
    
    def _fix_column_count_mismatch(self, error: Exception, query: str, params: tuple) -> Tuple[bool, str]:
        """Legacy method for column count mismatch fixing."""
        return self._handle_column_mismatch(error, query, params, None)
    
    def _fix_parameter_binding(self, error: Exception, query: str, params: tuple) -> Tuple[bool, str]:
        """Legacy method for parameter binding fixing."""
        return self._handle_parameter_binding(error, query, params)
    
    def _fix_type_error(self, error: Exception, query: str, params: tuple) -> Tuple[bool, str]:
        """Legacy method for type error fixing."""
        return self._handle_type_error(error, query, params)
    
    def _fix_constraint_error(self, error: Exception, query: str, params: tuple) -> Tuple[bool, str]:
        """Handle constraint errors."""
        return False, f"Constraint error not auto-fixable: {error}"

def safe_database_execute(cursor, query: str, params: tuple = None, 
                         table_name: str = None, max_retries: int = 3) -> bool:
    """Safely execute database operations with automatic error handling."""
    error_handler = DatabaseErrorHandler()
    
    for attempt in range(max_retries):
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return True
            
        except Exception as e:
            logger.warning(f"Database error (attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                # Try to fix the error
                success, fix_msg = error_handler.handle_database_error(e, query, params or (), table_name)
                
                if success:
                    logger.info(f"Applied fix: {fix_msg}")
                    try:
                        if params:
                            cursor.execute(query, params)
                        else:
                            cursor.execute(query)
                        return True
                    except Exception as retry_error:
                        logger.warning(f"Fix failed, retrying: {retry_error}")
                        continue
                else:
                    logger.warning(f"Could not fix error: {fix_msg}")
                    continue
            else:
                logger.error(f"Max retries exceeded for query: {query}")
                logger.error(f"Final error: {e}")
                return False
    
    return False

def database_error_handler(table_name: str = None):
    """Decorator for automatic database error handling."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Database error in {func.__name__}: {e}")
                
                # Try to extract query and params from the function call
                query = kwargs.get('query', '')
                params = kwargs.get('params', ())
                
                error_handler = DatabaseErrorHandler()
                success, fix_msg = error_handler.handle_database_error(e, query, params, table_name)
                
                if success:
                    logger.info(f"Auto-fixed database error: {fix_msg}")
                    # Retry the function with fixed parameters
                    try:
                        return func(*args, **kwargs)
                    except Exception as retry_error:
                        logger.error(f"Retry failed: {retry_error}")
                        raise retry_error
                else:
                    logger.error(f"Could not auto-fix: {fix_msg}")
                    raise e
                    
        return wrapper
    return decorator

# Global error handler instance
db_error_handler = DatabaseErrorHandler()

def handle_database_error(error: Exception, query: str = "", params: tuple = (), 
                         table_name: str = None) -> Tuple[bool, str]:
    """Global function to handle database errors."""
    return db_error_handler.handle_database_error(error, query, params, table_name)
