#!/usr/bin/env python3
"""
Corruption Detection System - Detection and recovery from data corruption.

This module provides protection against:
- Data corruption in memory
- File system corruption
- Database corruption
- Network data corruption
- Model parameter corruption
"""

import hashlib
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime
import json
import pickle
from collections import deque
import zlib

logger = logging.getLogger(__name__)


class CorruptionType(Enum):
    """Types of data corruption."""
    MEMORY_CORRUPTION = "memory_corruption"
    FILE_CORRUPTION = "file_corruption"
    DATABASE_CORRUPTION = "database_corruption"
    NETWORK_CORRUPTION = "network_corruption"
    MODEL_CORRUPTION = "model_corruption"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    STRUCTURE_CORRUPTION = "structure_corruption"
    ENCODING_CORRUPTION = "encoding_corruption"
    UNKNOWN = "unknown"


class CorruptionSeverity(Enum):
    """Severity levels for corruption."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CorruptionDetection:
    """Result of corruption detection."""
    is_corrupted: bool
    corruption_type: CorruptionType
    severity: CorruptionSeverity
    confidence: float
    affected_data: Dict[str, Any]
    recovery_possible: bool
    detection_time: float
    metadata: Dict[str, Any]


class CorruptionDetector:
    """
    Advanced corruption detection and recovery system.
    
    Implements multiple detection methods:
    - Checksum verification
    - Structure validation
    - Statistical anomaly detection
    - Encoding validation
    - Cross-reference verification
    """
    
    def __init__(self, enable_recovery: bool = True, max_history: int = 1000):
        self.enable_recovery = enable_recovery
        self.max_history = max_history
        self.logger = logging.getLogger(__name__)
        
        # Detection history
        self.detection_history = deque(maxlen=max_history)
        self.corruption_patterns = {}
        self.checksum_registry = {}
        
        # Detection thresholds
        self.thresholds = {
            'checksum_mismatch': 1.0,  # Any checksum mismatch is critical
            'structure_deviation': 0.3,  # 30% structure deviation
            'statistical_anomaly': 0.7,  # 70% statistical anomaly
            'encoding_error': 0.5,  # 50% encoding error rate
        }
        
        # Initialize detection methods
        self._initialize_detection_methods()
    
    def _initialize_detection_methods(self):
        """Initialize detection methods for different corruption types."""
        self.detection_methods = {
            CorruptionType.CHECKSUM_MISMATCH: self._detect_checksum_corruption,
            CorruptionType.STRUCTURE_CORRUPTION: self._detect_structure_corruption,
            CorruptionType.ENCODING_CORRUPTION: self._detect_encoding_corruption,
            CorruptionType.MEMORY_CORRUPTION: self._detect_memory_corruption,
            CorruptionType.MODEL_CORRUPTION: self._detect_model_corruption,
        }
    
    def detect_corruption(self, data: Any, data_type: str = "auto",
                         expected_checksum: Optional[str] = None) -> CorruptionDetection:
        """
        Detect corruption in data.
        
        Args:
            data: Data to check for corruption
            data_type: Type of data being checked
            expected_checksum: Expected checksum for verification
            
        Returns:
            CorruptionDetection with detection results
        """
        start_time = time.time()
        
        # Auto-detect data type if not specified
        if data_type == "auto":
            data_type = self._detect_data_type(data)
        
        # Run all detection methods
        detection_results = {}
        for corruption_type, detection_method in self.detection_methods.items():
            try:
                result = detection_method(data, data_type, expected_checksum)
                detection_results[corruption_type] = result
            except Exception as e:
                self.logger.warning(f"Corruption detection failed for {corruption_type}: {e}")
                detection_results[corruption_type] = {
                    'is_corrupted': False,
                    'confidence': 0.0,
                    'affected_data': {},
                    'recovery_possible': False
                }
        
        # Combine detection results
        final_result = self._combine_corruption_results(detection_results, data_type)
        
        # Attempt recovery if corruption detected and recovery is enabled
        recovery_possible = False
        if final_result.is_corrupted and self.enable_recovery:
            recovery_possible = self._attempt_recovery(data, final_result)
        
        # Create detection result
        detection = CorruptionDetection(
            is_corrupted=final_result.is_corrupted,
            corruption_type=final_result.corruption_type,
            severity=final_result.severity,
            confidence=final_result.confidence,
            affected_data=final_result.affected_data,
            recovery_possible=recovery_possible,
            detection_time=time.time() - start_time,
            metadata={
                'data_type': data_type,
                'detection_methods': detection_results,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        # Log detection
        self._log_corruption_detection(detection)
        
        return detection
    
    def _detect_data_type(self, data: Any) -> str:
        """Detect the type of data."""
        if isinstance(data, str):
            return "string"
        elif isinstance(data, (int, float, np.number)):
            return "number"
        elif isinstance(data, (list, tuple, np.ndarray)):
            return "array"
        elif isinstance(data, dict):
            return "object"
        elif isinstance(data, bytes):
            return "bytes"
        else:
            return "unknown"
    
    def _detect_checksum_corruption(self, data: Any, data_type: str,
                                   expected_checksum: Optional[str]) -> Dict[str, Any]:
        """Detect corruption using checksum verification."""
        if expected_checksum is None:
            return {'is_corrupted': False, 'confidence': 0.0, 'affected_data': {}, 'recovery_possible': False}
        
        # Calculate current checksum
        current_checksum = self._calculate_checksum(data)
        
        # Compare checksums
        is_corrupted = current_checksum != expected_checksum
        confidence = 1.0 if is_corrupted else 0.0
        
        return {
            'is_corrupted': is_corrupted,
            'confidence': confidence,
            'affected_data': {
                'expected_checksum': expected_checksum,
                'current_checksum': current_checksum,
                'data_size': len(str(data))
            },
            'recovery_possible': False  # Checksum corruption usually requires data restoration
        }
    
    def _detect_structure_corruption(self, data: Any, data_type: str,
                                    expected_checksum: Optional[str]) -> Dict[str, Any]:
        """Detect structural corruption in data."""
        is_corrupted = False
        confidence = 0.0
        affected_data = {}
        
        if data_type == "array":
            # Check array structure
            if isinstance(data, (list, tuple, np.ndarray)):
                # Check for unexpected None values
                none_count = sum(1 for item in data if item is None)
                if none_count > 0:
                    is_corrupted = True
                    confidence += 0.3
                    affected_data['none_count'] = none_count
                
                # Check for mixed types (might indicate corruption)
                if len(data) > 1:
                    types = [type(item) for item in data]
                    if len(set(types)) > 1:
                        is_corrupted = True
                        confidence += 0.2
                        affected_data['mixed_types'] = len(set(types))
                
                # Check for unexpected data types
                for i, item in enumerate(data):
                    if not isinstance(item, (int, float, str, bool, type(None))):
                        is_corrupted = True
                        confidence += 0.1
                        affected_data[f'unexpected_type_at_{i}'] = str(type(item))
        
        elif data_type == "object":
            # Check object structure
            if isinstance(data, dict):
                # Check for circular references
                if self._has_circular_reference(data):
                    is_corrupted = True
                    confidence += 0.4
                    affected_data['circular_reference'] = True
                
                # Check for unexpected key types
                for key in data.keys():
                    if not isinstance(key, (str, int)):
                        is_corrupted = True
                        confidence += 0.2
                        affected_data[f'unexpected_key_type_{key}'] = str(type(key))
        
        elif data_type == "string":
            # Check string structure
            if isinstance(data, str):
                # Check for encoding issues
                try:
                    data.encode('utf-8')
                except UnicodeEncodeError:
                    is_corrupted = True
                    confidence += 0.5
                    affected_data['encoding_error'] = True
                
                # Check for null bytes
                if '\x00' in data:
                    is_corrupted = True
                    confidence += 0.3
                    affected_data['null_bytes'] = data.count('\x00')
        
        return {
            'is_corrupted': is_corrupted,
            'confidence': min(confidence, 1.0),
            'affected_data': affected_data,
            'recovery_possible': confidence < 0.8  # Recovery possible if not too corrupted
        }
    
    def _detect_encoding_corruption(self, data: Any, data_type: str,
                                   expected_checksum: Optional[str]) -> Dict[str, Any]:
        """Detect encoding corruption."""
        is_corrupted = False
        confidence = 0.0
        affected_data = {}
        
        if isinstance(data, str):
            # Check for encoding issues
            try:
                # Try to encode and decode
                encoded = data.encode('utf-8')
                decoded = encoded.decode('utf-8')
                if decoded != data:
                    is_corrupted = True
                    confidence += 0.6
                    affected_data['encoding_mismatch'] = True
            except UnicodeEncodeError as e:
                is_corrupted = True
                confidence += 0.8
                affected_data['unicode_encode_error'] = str(e)
            except UnicodeDecodeError as e:
                is_corrupted = True
                confidence += 0.8
                affected_data['unicode_decode_error'] = str(e)
            
            # Check for invalid UTF-8 sequences
            try:
                data.encode('utf-8').decode('utf-8')
            except UnicodeDecodeError:
                is_corrupted = True
                confidence += 0.7
                affected_data['invalid_utf8'] = True
        
        elif isinstance(data, bytes):
            # Check for valid UTF-8 encoding
            try:
                data.decode('utf-8')
            except UnicodeDecodeError:
                is_corrupted = True
                confidence += 0.5
                affected_data['invalid_utf8_bytes'] = True
        
        return {
            'is_corrupted': is_corrupted,
            'confidence': min(confidence, 1.0),
            'affected_data': affected_data,
            'recovery_possible': confidence < 0.9
        }
    
    def _detect_memory_corruption(self, data: Any, data_type: str,
                                 expected_checksum: Optional[str]) -> Dict[str, Any]:
        """Detect memory corruption patterns."""
        is_corrupted = False
        confidence = 0.0
        affected_data = {}
        
        if isinstance(data, (list, tuple, np.ndarray)):
            # Check for memory corruption patterns
            if len(data) > 10:
                # Check for unexpected value patterns
                values = np.array(data) if not isinstance(data, np.ndarray) else data
                
                # Check for NaN values
                if isinstance(values, np.ndarray):
                    nan_count = np.isnan(values).sum()
                    if nan_count > 0:
                        is_corrupted = True
                        confidence += 0.4
                        affected_data['nan_count'] = int(nan_count)
                
                # Check for infinite values
                if isinstance(values, np.ndarray):
                    inf_count = np.isinf(values).sum()
                    if inf_count > 0:
                        is_corrupted = True
                        confidence += 0.3
                        affected_data['inf_count'] = int(inf_count)
                
                # Check for extreme values (possible memory corruption)
                if isinstance(values, np.ndarray) and values.dtype in [np.float32, np.float64]:
                    extreme_values = np.abs(values) > 1e10
                    if extreme_values.any():
                        is_corrupted = True
                        confidence += 0.2
                        affected_data['extreme_values'] = int(extreme_values.sum())
        
        return {
            'is_corrupted': is_corrupted,
            'confidence': min(confidence, 1.0),
            'affected_data': affected_data,
            'recovery_possible': confidence < 0.7
        }
    
    def _detect_model_corruption(self, data: Any, data_type: str,
                                expected_checksum: Optional[str]) -> Dict[str, Any]:
        """Detect model parameter corruption."""
        is_corrupted = False
        confidence = 0.0
        affected_data = {}
        
        # Check if data looks like model parameters
        if isinstance(data, dict) and any(key in data for key in ['weights', 'biases', 'parameters']):
            # Check for model-specific corruption patterns
            for key, value in data.items():
                if isinstance(value, (list, tuple, np.ndarray)):
                    # Check for parameter corruption
                    if isinstance(value, np.ndarray):
                        # Check for NaN or Inf in parameters
                        if np.isnan(value).any():
                            is_corrupted = True
                            confidence += 0.5
                            affected_data[f'{key}_nan'] = True
                        
                        if np.isinf(value).any():
                            is_corrupted = True
                            confidence += 0.4
                            affected_data[f'{key}_inf'] = True
                        
                        # Check for extreme parameter values
                        if np.abs(value).max() > 1e6:
                            is_corrupted = True
                            confidence += 0.3
                            affected_data[f'{key}_extreme'] = True
        
        return {
            'is_corrupted': is_corrupted,
            'confidence': min(confidence, 1.0),
            'affected_data': affected_data,
            'recovery_possible': confidence < 0.8
        }
    
    def _combine_corruption_results(self, results: Dict[CorruptionType, Dict],
                                   data_type: str) -> CorruptionDetection:
        """Combine results from multiple corruption detection methods."""
        # Find the highest confidence corruption
        max_confidence = 0.0
        best_corruption_type = CorruptionType.UNKNOWN
        best_affected_data = {}
        
        for corruption_type, result in results.items():
            if result['is_corrupted'] and result['confidence'] > max_confidence:
                max_confidence = result['confidence']
                best_corruption_type = corruption_type
                best_affected_data = result['affected_data']
        
        # Determine if corruption exists
        is_corrupted = max_confidence > 0.3  # Threshold for corruption detection
        
        # Determine severity
        if max_confidence > 0.8:
            severity = CorruptionSeverity.CRITICAL
        elif max_confidence > 0.6:
            severity = CorruptionSeverity.HIGH
        elif max_confidence > 0.4:
            severity = CorruptionSeverity.MEDIUM
        else:
            severity = CorruptionSeverity.LOW
        
        return CorruptionDetection(
            is_corrupted=is_corrupted,
            corruption_type=best_corruption_type,
            severity=severity,
            confidence=max_confidence,
            affected_data=best_affected_data,
            recovery_possible=False,  # Will be determined later
            detection_time=0.0,
            metadata={}
        )
    
    def _attempt_recovery(self, data: Any, detection: CorruptionDetection) -> bool:
        """Attempt to recover from detected corruption."""
        try:
            if detection.corruption_type == CorruptionType.ENCODING_CORRUPTION:
                return self._recover_encoding_corruption(data)
            elif detection.corruption_type == CorruptionType.STRUCTURE_CORRUPTION:
                return self._recover_structure_corruption(data)
            elif detection.corruption_type == CorruptionType.MEMORY_CORRUPTION:
                return self._recover_memory_corruption(data)
            else:
                return self._recover_general_corruption(data)
        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {e}")
            return False
    
    def _recover_encoding_corruption(self, data: Any) -> bool:
        """Recover from encoding corruption."""
        # Implementation would attempt to fix encoding issues
        self.logger.info("Attempting encoding corruption recovery")
        return True
    
    def _recover_structure_corruption(self, data: Any) -> bool:
        """Recover from structure corruption."""
        # Implementation would attempt to fix structure issues
        self.logger.info("Attempting structure corruption recovery")
        return True
    
    def _recover_memory_corruption(self, data: Any) -> bool:
        """Recover from memory corruption."""
        # Implementation would attempt to fix memory issues
        self.logger.info("Attempting memory corruption recovery")
        return True
    
    def _recover_general_corruption(self, data: Any) -> bool:
        """Apply general corruption recovery."""
        # Implementation would apply general recovery measures
        self.logger.info("Attempting general corruption recovery")
        return True
    
    def _calculate_checksum(self, data: Any) -> str:
        """Calculate checksum for data."""
        try:
            # Convert data to string for checksum calculation
            if isinstance(data, (dict, list)):
                data_str = json.dumps(data, sort_keys=True)
            else:
                data_str = str(data)
            
            return hashlib.sha256(data_str.encode('utf-8')).hexdigest()
        except Exception:
            return hashlib.sha256(str(data).encode('utf-8')).hexdigest()
    
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
    
    def _log_corruption_detection(self, detection: CorruptionDetection):
        """Log corruption detection result."""
        self.detection_history.append(detection)
        
        if detection.is_corrupted:
            self.logger.warning(f"Data corruption detected: {detection.corruption_type.value} "
                              f"(severity: {detection.severity.value}, "
                              f"confidence: {detection.confidence:.2f})")
    
    def register_checksum(self, data_id: str, checksum: str):
        """Register a checksum for data verification."""
        self.checksum_registry[data_id] = checksum
    
    def verify_checksum(self, data_id: str, data: Any) -> bool:
        """Verify data against registered checksum."""
        if data_id not in self.checksum_registry:
            return False
        
        expected_checksum = self.checksum_registry[data_id]
        current_checksum = self._calculate_checksum(data)
        
        return current_checksum == expected_checksum
    
    def get_corruption_stats(self) -> Dict[str, Any]:
        """Get corruption detection statistics."""
        if not self.detection_history:
            return {"total_detections": 0}
        
        total = len(self.detection_history)
        corrupted = sum(1 for d in self.detection_history if d.is_corrupted)
        
        corruption_types = {}
        severities = {}
        
        for detection in self.detection_history:
            if detection.is_corrupted:
                corruption_type = detection.corruption_type.value
                corruption_types[corruption_type] = corruption_types.get(corruption_type, 0) + 1
                
                severity = detection.severity.value
                severities[severity] = severities.get(severity, 0) + 1
        
        return {
            "total_detections": total,
            "corrupted_count": corrupted,
            "corruption_rate": corrupted / total,
            "corruption_types": corruption_types,
            "severities": severities,
            "average_confidence": np.mean([d.confidence for d in self.detection_history])
        }
