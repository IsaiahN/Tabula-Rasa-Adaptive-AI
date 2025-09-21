#!/usr/bin/env python3
"""
Data Validation System - Comprehensive input validation and sanitization.

This module provides robust data validation to prevent:
- Malformed input data
- Injection attacks
- Type confusion attacks
- Buffer overflow attempts
- Data corruption propagation
"""

import re
import json
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import hashlib
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception raised when data validation fails."""
    pass


class ValidationSeverity(Enum):
    """Severity levels for validation failures."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_data: Any
    validation_time: float
    severity: ValidationSeverity
    metadata: Dict[str, Any]


class DataValidator:
    """
    Comprehensive data validation system.
    
    Provides multi-layer validation including:
    - Type checking and conversion
    - Range and format validation
    - Injection attack prevention
    - Data integrity verification
    - Sanitization and normalization
    """
    
    def __init__(self, strict_mode: bool = True, max_depth: int = 10):
        self.strict_mode = strict_mode
        self.max_depth = max_depth
        self.validation_rules = {}
        self.custom_validators = {}
        self.validation_history = []
        self.logger = logging.getLogger(__name__)
        
        # Initialize default validation rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default validation rules."""
        self.validation_rules = {
            'string': {
                'max_length': 10000,
                'min_length': 0,
                'allowed_chars': r'[\w\s\-_.,!?@#$%^&*()+=\[\]{}|\\:";\'<>?/~`]',
                'forbidden_patterns': [
                    r'<script.*?>.*?</script>',  # XSS prevention
                    r'javascript:',  # JavaScript injection
                    r'data:text/html',  # Data URI injection
                    r'vbscript:',  # VBScript injection
                    r'on\w+\s*=',  # Event handler injection
                ]
            },
            'number': {
                'min_value': -1e10,
                'max_value': 1e10,
                'allow_nan': False,
                'allow_inf': False
            },
            'array': {
                'max_length': 1000,
                'min_length': 0,
                'max_depth': 5
            },
            'object': {
                'max_keys': 100,
                'max_depth': 10,
                'forbidden_keys': ['__proto__', 'constructor', 'prototype']
            }
        }
    
    def validate(self, data: Any, data_type: str = "auto", 
                context: str = "general") -> ValidationResult:
        """
        Validate data with comprehensive checks.
        
        Args:
            data: Data to validate
            data_type: Expected data type or "auto" for detection
            context: Validation context for specialized rules
            
        Returns:
            ValidationResult with validation details
        """
        start_time = time.time()
        errors = []
        warnings = []
        sanitized_data = data
        
        try:
            # Auto-detect type if not specified
            if data_type == "auto":
                data_type = self._detect_type(data)
            
            # Basic type validation
            if not self._validate_type(data, data_type):
                errors.append(f"Expected {data_type}, got {type(data).__name__}")
                if self.strict_mode:
                    raise ValidationError(f"Type mismatch: expected {data_type}")
            
            # Context-specific validation
            if context in self.custom_validators:
                context_result = self.custom_validators[context](data)
                if not context_result.is_valid:
                    errors.extend(context_result.errors)
                    warnings.extend(context_result.warnings)
            
            # Type-specific validation
            if data_type == "string":
                sanitized_data, str_errors, str_warnings = self._validate_string(data)
                errors.extend(str_errors)
                warnings.extend(str_warnings)
            elif data_type == "number":
                sanitized_data, num_errors, num_warnings = self._validate_number(data)
                errors.extend(num_errors)
                warnings.extend(num_warnings)
            elif data_type == "array":
                sanitized_data, arr_errors, arr_warnings = self._validate_array(data)
                errors.extend(arr_errors)
                warnings.extend(arr_warnings)
            elif data_type == "object":
                sanitized_data, obj_errors, obj_warnings = self._validate_object(data)
                errors.extend(obj_errors)
                warnings.extend(obj_warnings)
            
            # Security checks
            security_errors, security_warnings = self._security_checks(data)
            errors.extend(security_errors)
            warnings.extend(security_warnings)
            
            # Data integrity verification
            integrity_errors = self._verify_integrity(data)
            errors.extend(integrity_errors)
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            if self.strict_mode:
                raise ValidationError(f"Validation failed: {str(e)}")
        
        # Determine severity
        severity = self._determine_severity(errors, warnings)
        
        # Create result
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=sanitized_data,
            validation_time=time.time() - start_time,
            severity=severity,
            metadata={
                'data_type': data_type,
                'context': context,
                'data_size': self._calculate_size(data),
                'validation_timestamp': datetime.now().isoformat()
            }
        )
        
        # Log validation result
        self._log_validation_result(result)
        
        return result
    
    def _detect_type(self, data: Any) -> str:
        """Detect the data type."""
        if isinstance(data, str):
            return "string"
        elif isinstance(data, (int, float, np.number)):
            return "number"
        elif isinstance(data, (list, tuple, np.ndarray)):
            return "array"
        elif isinstance(data, dict):
            return "object"
        elif data is None:
            return "null"
        elif isinstance(data, bool):
            return "boolean"
        else:
            return "unknown"
    
    def _validate_type(self, data: Any, expected_type: str) -> bool:
        """Validate that data matches expected type."""
        detected_type = self._detect_type(data)
        return detected_type == expected_type or expected_type == "any"
    
    def _validate_string(self, data: str) -> Tuple[str, List[str], List[str]]:
        """Validate string data."""
        errors = []
        warnings = []
        sanitized = data
        
        rules = self.validation_rules['string']
        
        # Length validation
        if len(data) > rules['max_length']:
            errors.append(f"String too long: {len(data)} > {rules['max_length']}")
            sanitized = data[:rules['max_length']]
        elif len(data) < rules['min_length']:
            errors.append(f"String too short: {len(data)} < {rules['min_length']}")
        
        # Character validation
        if not re.match(f"^{rules['allowed_chars']}+$", data):
            warnings.append("String contains potentially unsafe characters")
            # Sanitize by removing unsafe characters
            sanitized = re.sub(f"[^{rules['allowed_chars']}]", "", data)
        
        # Pattern-based security checks
        for pattern in rules['forbidden_patterns']:
            if re.search(pattern, data, re.IGNORECASE):
                errors.append(f"String contains forbidden pattern: {pattern}")
                # Remove the pattern
                sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        
        return sanitized, errors, warnings
    
    def _validate_number(self, data: Union[int, float]) -> Tuple[Union[int, float], List[str], List[str]]:
        """Validate numeric data."""
        errors = []
        warnings = []
        sanitized = data
        
        rules = self.validation_rules['number']
        
        # Range validation
        if data < rules['min_value']:
            errors.append(f"Number too small: {data} < {rules['min_value']}")
            sanitized = rules['min_value']
        elif data > rules['max_value']:
            errors.append(f"Number too large: {data} > {rules['max_value']}")
            sanitized = rules['max_value']
        
        # Special value checks
        if isinstance(data, float):
            if np.isnan(data) and not rules['allow_nan']:
                errors.append("NaN values not allowed")
                sanitized = 0.0
            elif np.isinf(data) and not rules['allow_inf']:
                errors.append("Infinite values not allowed")
                sanitized = rules['max_value'] if data > 0 else rules['min_value']
        
        return sanitized, errors, warnings
    
    def _validate_array(self, data: Union[List, Tuple, np.ndarray]) -> Tuple[List, List[str], List[str]]:
        """Validate array data."""
        errors = []
        warnings = []
        sanitized = list(data)
        
        rules = self.validation_rules['array']
        
        # Length validation
        if len(data) > rules['max_length']:
            errors.append(f"Array too long: {len(data)} > {rules['max_length']}")
            sanitized = sanitized[:rules['max_length']]
        elif len(data) < rules['min_length']:
            errors.append(f"Array too short: {len(data)} < {rules['min_length']}")
        
        # Depth validation
        depth = self._calculate_depth(data)
        if depth > rules['max_depth']:
            errors.append(f"Array too deep: {depth} > {rules['max_depth']}")
        
        # Element validation
        for i, element in enumerate(sanitized):
            try:
                element_result = self.validate(element, context="array_element")
                if not element_result.is_valid:
                    errors.extend([f"Element {i}: {error}" for error in element_result.errors])
                    warnings.extend([f"Element {i}: {warning}" for warning in element_result.warnings])
                    sanitized[i] = element_result.sanitized_data
            except Exception as e:
                errors.append(f"Element {i} validation failed: {str(e)}")
        
        return sanitized, errors, warnings
    
    def _validate_object(self, data: Dict) -> Tuple[Dict, List[str], List[str]]:
        """Validate object data."""
        errors = []
        warnings = []
        sanitized = data.copy()
        
        rules = self.validation_rules['object']
        
        # Key count validation
        if len(data) > rules['max_keys']:
            errors.append(f"Object has too many keys: {len(data)} > {rules['max_keys']}")
            # Keep only the first max_keys
            keys_to_keep = list(data.keys())[:rules['max_keys']]
            sanitized = {k: v for k, v in data.items() if k in keys_to_keep}
        
        # Forbidden keys check
        for forbidden_key in rules['forbidden_keys']:
            if forbidden_key in data:
                errors.append(f"Object contains forbidden key: {forbidden_key}")
                sanitized.pop(forbidden_key, None)
        
        # Depth validation
        depth = self._calculate_depth(data)
        if depth > rules['max_depth']:
            errors.append(f"Object too deep: {depth} > {rules['max_depth']}")
        
        # Key and value validation
        for key, value in sanitized.items():
            # Key validation
            if not isinstance(key, str):
                errors.append(f"Object key must be string: {key}")
                continue
            
            # Value validation
            try:
                value_result = self.validate(value, context="object_value")
                if not value_result.is_valid:
                    errors.extend([f"Key '{key}': {error}" for error in value_result.errors])
                    warnings.extend([f"Key '{key}': {warning}" for warning in value_result.warnings])
                    sanitized[key] = value_result.sanitized_data
            except Exception as e:
                errors.append(f"Key '{key}' validation failed: {str(e)}")
        
        return sanitized, errors, warnings
    
    def _security_checks(self, data: Any) -> Tuple[List[str], List[str]]:
        """Perform security-specific checks."""
        errors = []
        warnings = []
        
        # Check for potential injection attacks
        data_str = str(data)
        
        # SQL injection patterns
        sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(--|\#|\/\*|\*\/)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, data_str, re.IGNORECASE):
                errors.append(f"Potential SQL injection detected: {pattern}")
        
        # XSS patterns
        xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, data_str, re.IGNORECASE):
                errors.append(f"Potential XSS detected: {pattern}")
        
        # Command injection patterns
        cmd_patterns = [
            r"[;&|`$]",
            r"(\b(cat|ls|pwd|whoami|id|uname)\b)",
            r"(\$\{.*\})",
        ]
        
        for pattern in cmd_patterns:
            if re.search(pattern, data_str, re.IGNORECASE):
                warnings.append(f"Potential command injection: {pattern}")
        
        return errors, warnings
    
    def _verify_integrity(self, data: Any) -> List[str]:
        """Verify data integrity."""
        errors = []
        
        # Check for data corruption indicators
        if isinstance(data, (list, tuple)):
            # Check for unexpected None values in arrays
            if None in data:
                errors.append("Array contains unexpected None values")
            
            # Check for mixed types in arrays
            if len(data) > 1:
                types = [type(item) for item in data]
                if len(set(types)) > 1:
                    errors.append("Array contains mixed types")
        
        # Check for circular references
        if self._has_circular_reference(data):
            errors.append("Data contains circular references")
        
        return errors
    
    def _has_circular_reference(self, data: Any, visited: Optional[set] = None) -> bool:
        """Check for circular references in data."""
        if visited is None:
            visited = set()
        
        if id(data) in visited:
            return True
        
        if isinstance(data, (list, tuple)):
            visited.add(id(data))
            for item in data:
                if self._has_circular_reference(item, visited):
                    return True
            visited.remove(id(data))
        elif isinstance(data, dict):
            visited.add(id(data))
            for value in data.values():
                if self._has_circular_reference(value, visited):
                    return True
            visited.remove(id(data))
        
        return False
    
    def _calculate_depth(self, data: Any, current_depth: int = 0) -> int:
        """Calculate the depth of nested data structures."""
        if current_depth > self.max_depth:
            return current_depth
        
        if isinstance(data, (list, tuple)):
            if not data:
                return current_depth
            return max(self._calculate_depth(item, current_depth + 1) for item in data)
        elif isinstance(data, dict):
            if not data:
                return current_depth
            return max(self._calculate_depth(value, current_depth + 1) for value in data.values())
        else:
            return current_depth
    
    def _calculate_size(self, data: Any) -> int:
        """Calculate the approximate size of data in bytes."""
        try:
            return len(json.dumps(data, default=str).encode('utf-8'))
        except:
            return 0
    
    def _determine_severity(self, errors: List[str], warnings: List[str]) -> ValidationSeverity:
        """Determine the severity of validation issues."""
        if not errors and not warnings:
            return ValidationSeverity.LOW
        
        if any("injection" in error.lower() for error in errors):
            return ValidationSeverity.CRITICAL
        
        if any("forbidden" in error.lower() for error in errors):
            return ValidationSeverity.HIGH
        
        if len(errors) > 5:
            return ValidationSeverity.HIGH
        
        if errors:
            return ValidationSeverity.MEDIUM
        
        return ValidationSeverity.LOW
    
    def _log_validation_result(self, result: ValidationResult):
        """Log validation result for monitoring."""
        self.validation_history.append(result)
        
        # Keep only recent history
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-500:]
        
        # Log based on severity
        if result.severity == ValidationSeverity.CRITICAL:
            self.logger.error(f"Critical validation failure: {result.errors}")
        elif result.severity == ValidationSeverity.HIGH:
            self.logger.warning(f"High severity validation issues: {result.errors}")
        elif result.errors:
            self.logger.info(f"Validation issues: {result.errors}")
    
    def add_custom_validator(self, context: str, validator: Callable[[Any], ValidationResult]):
        """Add a custom validator for specific contexts."""
        self.custom_validators[context] = validator
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        if not self.validation_history:
            return {"total_validations": 0}
        
        total = len(self.validation_history)
        valid = sum(1 for r in self.validation_history if r.is_valid)
        invalid = total - valid
        
        severity_counts = {}
        for severity in ValidationSeverity:
            severity_counts[severity.value] = sum(
                1 for r in self.validation_history if r.severity == severity
            )
        
        avg_time = sum(r.validation_time for r in self.validation_history) / total
        
        return {
            "total_validations": total,
            "valid_count": valid,
            "invalid_count": invalid,
            "success_rate": valid / total,
            "severity_distribution": severity_counts,
            "average_validation_time": avg_time
        }
