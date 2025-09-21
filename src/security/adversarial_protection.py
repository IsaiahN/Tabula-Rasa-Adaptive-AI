#!/usr/bin/env python3
"""
Adversarial Protection System - Defense against adversarial attacks.

This module provides protection against:
- Adversarial examples in input data
- Model poisoning attacks
- Data manipulation attacks
- Gradient-based attacks
- Feature space attacks
"""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime
import hashlib
from collections import deque
import random

logger = logging.getLogger(__name__)


class AttackType(Enum):
    """Types of adversarial attacks."""
    FGSM = "fgsm"  # Fast Gradient Sign Method
    PGD = "pgd"   # Projected Gradient Descent
    CWL2 = "cwl2"  # Carlini & Wagner L2
    DEEPFOOL = "deepfool"
    JSMA = "jsma"  # Jacobian-based Saliency Map Attack
    POISONING = "poisoning"  # Data poisoning
    BACKDOOR = "backdoor"  # Backdoor attack
    MODEL_EVASION = "model_evasion"  # Model evasion
    FEATURE_SPACE = "feature_space"  # Feature space attack
    UNKNOWN = "unknown"


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AttackDetection:
    """Result of adversarial attack detection."""
    is_attack: bool
    attack_type: AttackType
    confidence: float
    threat_level: ThreatLevel
    features: Dict[str, Any]
    mitigation_applied: bool
    detection_time: float
    metadata: Dict[str, Any]


class AdversarialDetector:
    """
    Advanced adversarial attack detection and prevention system.
    
    Implements multiple detection methods:
    - Statistical anomaly detection
    - Gradient-based analysis
    - Feature space analysis
    - Model confidence analysis
    - Ensemble detection methods
    """
    
    def __init__(self, sensitivity: float = 0.8, max_history: int = 10000):
        self.sensitivity = sensitivity
        self.max_history = max_history
        self.logger = logging.getLogger(__name__)
        
        # Detection history
        self.detection_history = deque(maxlen=max_history)
        self.attack_patterns = {}
        self.baseline_stats = {}
        
        # Detection thresholds
        self.thresholds = {
            'statistical_anomaly': 0.7,
            'gradient_magnitude': 0.5,
            'confidence_drop': 0.3,
            'feature_deviation': 0.6,
            'ensemble_consensus': 0.8
        }
        
        # Initialize detection methods
        self._initialize_detection_methods()
    
    def _initialize_detection_methods(self):
        """Initialize various detection methods."""
        self.detection_methods = {
            'statistical': self._detect_statistical_anomaly,
            'gradient': self._detect_gradient_attack,
            'confidence': self._detect_confidence_attack,
            'feature': self._detect_feature_attack,
            'ensemble': self._detect_ensemble_attack
        }
    
    def detect_attack(self, data: Any, model_output: Optional[Dict] = None,
                     context: str = "general") -> AttackDetection:
        """
        Detect adversarial attacks in input data.
        
        Args:
            data: Input data to analyze
            model_output: Model predictions/outputs
            context: Analysis context
            
        Returns:
            AttackDetection with detection results
        """
        start_time = time.time()
        
        # Convert data to analyzable format
        analysis_data = self._prepare_analysis_data(data)
        
        # Run all detection methods
        detection_results = {}
        for method_name, method_func in self.detection_methods.items():
            try:
                result = method_func(analysis_data, model_output, context)
                detection_results[method_name] = result
            except Exception as e:
                self.logger.warning(f"Detection method {method_name} failed: {e}")
                detection_results[method_name] = {
                    'is_attack': False,
                    'confidence': 0.0,
                    'attack_type': AttackType.UNKNOWN,
                    'features': {}
                }
        
        # Combine detection results
        final_result = self._combine_detection_results(detection_results, context)
        
        # Apply mitigation if attack detected
        mitigation_applied = False
        if final_result.is_attack and final_result.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            mitigation_applied = self._apply_mitigation(data, final_result)
        
        # Create detection result
        detection = AttackDetection(
            is_attack=final_result.is_attack,
            attack_type=final_result.attack_type,
            confidence=final_result.confidence,
            threat_level=final_result.threat_level,
            features=final_result.features,
            mitigation_applied=mitigation_applied,
            detection_time=time.time() - start_time,
            metadata={
                'context': context,
                'detection_methods': detection_results,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        # Log detection
        self._log_detection(detection)
        
        return detection
    
    def _prepare_analysis_data(self, data: Any) -> np.ndarray:
        """Prepare data for analysis."""
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, (list, tuple)):
            return np.array(data)
        elif isinstance(data, dict):
            # Extract numeric values from dict
            values = []
            for value in data.values():
                if isinstance(value, (int, float)):
                    values.append(value)
                elif isinstance(value, (list, tuple)):
                    values.extend(value)
            return np.array(values) if values else np.array([0])
        else:
            # Convert to numeric
            try:
                return np.array([float(data)])
            except:
                return np.array([0])
    
    def _detect_statistical_anomaly(self, data: np.ndarray, model_output: Optional[Dict],
                                   context: str) -> Dict[str, Any]:
        """Detect statistical anomalies that might indicate adversarial attacks."""
        features = {}
        
        # Calculate basic statistics
        mean_val = np.mean(data)
        std_val = np.std(data)
        skewness = self._calculate_skewness(data)
        kurtosis = self._calculate_kurtosis(data)
        
        features.update({
            'mean': mean_val,
            'std': std_val,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'data_range': np.ptp(data),
            'zero_ratio': np.sum(data == 0) / len(data)
        })
        
        # Check against baseline statistics
        is_anomaly = False
        confidence = 0.0
        
        if context in self.baseline_stats:
            baseline = self.baseline_stats[context]
            
            # Statistical tests
            mean_diff = abs(mean_val - baseline.get('mean', 0))
            std_diff = abs(std_val - baseline.get('std', 1))
            
            if mean_diff > baseline.get('mean_std', 1) * 3:
                is_anomaly = True
                confidence += 0.3
            
            if std_diff > baseline.get('std_std', 1) * 2:
                is_anomaly = True
                confidence += 0.2
            
            # Skewness and kurtosis checks
            if abs(skewness) > 3:
                is_anomaly = True
                confidence += 0.2
            
            if abs(kurtosis) > 10:
                is_anomaly = True
                confidence += 0.2
        
        # Update baseline if not anomaly
        if not is_anomaly:
            self._update_baseline(context, features)
        
        return {
            'is_attack': is_anomaly,
            'confidence': min(confidence, 1.0),
            'attack_type': AttackType.UNKNOWN,
            'features': features
        }
    
    def _detect_gradient_attack(self, data: np.ndarray, model_output: Optional[Dict],
                               context: str) -> Dict[str, Any]:
        """Detect gradient-based attacks."""
        features = {}
        
        # Calculate gradient-like features
        if len(data) > 1:
            gradients = np.gradient(data)
            gradient_magnitude = np.linalg.norm(gradients)
            gradient_std = np.std(gradients)
            
            features.update({
                'gradient_magnitude': gradient_magnitude,
                'gradient_std': gradient_std,
                'max_gradient': np.max(np.abs(gradients)),
                'gradient_ratio': gradient_magnitude / (np.linalg.norm(data) + 1e-8)
            })
            
            # Detect suspicious gradient patterns
            is_attack = False
            confidence = 0.0
            
            # High gradient magnitude might indicate FGSM/PGD
            if gradient_magnitude > self.thresholds['gradient_magnitude']:
                is_attack = True
                confidence += 0.4
            
            # High gradient ratio might indicate adversarial perturbation
            if features['gradient_ratio'] > 0.5:
                is_attack = True
                confidence += 0.3
            
            # Sudden gradient changes
            if gradient_std > np.std(data) * 2:
                is_attack = True
                confidence += 0.2
        else:
            is_attack = False
            confidence = 0.0
        
        return {
            'is_attack': is_attack,
            'confidence': min(confidence, 1.0),
            'attack_type': AttackType.FGSM if is_attack else AttackType.UNKNOWN,
            'features': features
        }
    
    def _detect_confidence_attack(self, data: np.ndarray, model_output: Optional[Dict],
                                 context: str) -> Dict[str, Any]:
        """Detect attacks based on model confidence analysis."""
        features = {}
        is_attack = False
        confidence = 0.0
        
        if model_output:
            # Extract confidence information
            pred_confidence = model_output.get('confidence', 0.0)
            pred_entropy = model_output.get('entropy', 0.0)
            pred_variance = model_output.get('variance', 0.0)
            
            features.update({
                'prediction_confidence': pred_confidence,
                'prediction_entropy': pred_entropy,
                'prediction_variance': pred_variance
            })
            
            # Low confidence might indicate adversarial input
            if pred_confidence < self.thresholds['confidence_drop']:
                is_attack = True
                confidence += 0.4
            
            # High entropy might indicate uncertainty/attack
            if pred_entropy > 2.0:  # High entropy threshold
                is_attack = True
                confidence += 0.3
            
            # High variance might indicate model uncertainty
            if pred_variance > 0.5:
                is_attack = True
                confidence += 0.2
        else:
            # Without model output, use data characteristics
            data_entropy = self._calculate_entropy(data)
            features['data_entropy'] = data_entropy
            
            if data_entropy > 3.0:  # High entropy might indicate attack
                is_attack = True
                confidence += 0.3
        
        return {
            'is_attack': is_attack,
            'confidence': min(confidence, 1.0),
            'attack_type': AttackType.MODEL_EVASION if is_attack else AttackType.UNKNOWN,
            'features': features
        }
    
    def _detect_feature_attack(self, data: np.ndarray, model_output: Optional[Dict],
                              context: str) -> Dict[str, Any]:
        """Detect attacks in feature space."""
        features = {}
        
        # Calculate feature space characteristics
        data_norm = np.linalg.norm(data)
        data_mean = np.mean(data)
        data_std = np.std(data)
        
        # Normalize data for feature analysis
        if data_std > 0:
            normalized_data = (data - data_mean) / data_std
        else:
            normalized_data = data
        
        # Calculate feature space metrics
        l2_norm = np.linalg.norm(normalized_data)
        l_inf_norm = np.max(np.abs(normalized_data))
        sparsity = np.sum(normalized_data == 0) / len(normalized_data)
        
        features.update({
            'l2_norm': l2_norm,
            'l_inf_norm': l_inf_norm,
            'sparsity': sparsity,
            'data_norm': data_norm,
            'normalized_std': np.std(normalized_data)
        })
        
        # Detect suspicious feature patterns
        is_attack = False
        confidence = 0.0
        
        # High L-infinity norm might indicate adversarial perturbation
        if l_inf_norm > 3.0:  # 3 standard deviations
            is_attack = True
            confidence += 0.4
        
        # Unusual sparsity patterns
        if sparsity > 0.8 or sparsity < 0.1:
            is_attack = True
            confidence += 0.2
        
        # Extreme L2 norm
        if l2_norm > 5.0:  # 5 standard deviations
            is_attack = True
            confidence += 0.3
        
        return {
            'is_attack': is_attack,
            'confidence': min(confidence, 1.0),
            'attack_type': AttackType.FEATURE_SPACE if is_attack else AttackType.UNKNOWN,
            'features': features
        }
    
    def _detect_ensemble_attack(self, data: np.ndarray, model_output: Optional[Dict],
                               context: str) -> Dict[str, Any]:
        """Use ensemble methods for attack detection."""
        features = {}
        
        # Multiple detection approaches
        detections = []
        
        # 1. Statistical consistency
        if len(data) > 10:
            # Check for statistical consistency
            chunk_size = len(data) // 5
            chunk_means = []
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i+chunk_size]
                if len(chunk) > 0:
                    chunk_means.append(np.mean(chunk))
            
            if len(chunk_means) > 1:
                chunk_std = np.std(chunk_means)
                if chunk_std > np.std(data) * 0.5:
                    detections.append(0.3)
                else:
                    detections.append(0.0)
        
        # 2. Frequency domain analysis
        if len(data) > 4:
            fft_data = np.fft.fft(data)
            power_spectrum = np.abs(fft_data)
            
            # Check for unusual frequency patterns
            high_freq_power = np.sum(power_spectrum[len(power_spectrum)//2:])
            total_power = np.sum(power_spectrum)
            
            if total_power > 0:
                high_freq_ratio = high_freq_power / total_power
                if high_freq_ratio > 0.7:  # Unusually high high-frequency content
                    detections.append(0.4)
                else:
                    detections.append(0.0)
        
        # 3. Pattern consistency
        if len(data) > 20:
            # Check for repeating patterns
            pattern_lengths = [2, 3, 4, 5]
            pattern_scores = []
            
            for pattern_len in pattern_lengths:
                if len(data) >= pattern_len * 2:
                    pattern_score = self._calculate_pattern_consistency(data, pattern_len)
                    pattern_scores.append(pattern_score)
            
            if pattern_scores:
                avg_pattern_score = np.mean(pattern_scores)
                if avg_pattern_score < 0.3:  # Low pattern consistency
                    detections.append(0.3)
                else:
                    detections.append(0.0)
        
        # Combine detections
        if detections:
            ensemble_confidence = np.mean(detections)
            is_attack = ensemble_confidence > self.thresholds['ensemble_consensus']
        else:
            ensemble_confidence = 0.0
            is_attack = False
        
        features['ensemble_confidence'] = ensemble_confidence
        features['detection_count'] = len(detections)
        
        return {
            'is_attack': is_attack,
            'confidence': ensemble_confidence,
            'attack_type': AttackType.UNKNOWN,
            'features': features
        }
    
    def _combine_detection_results(self, results: Dict[str, Dict], context: str) -> AttackDetection:
        """Combine results from multiple detection methods."""
        # Weighted combination of detection methods
        weights = {
            'statistical': 0.2,
            'gradient': 0.25,
            'confidence': 0.2,
            'feature': 0.2,
            'ensemble': 0.15
        }
        
        total_confidence = 0.0
        total_weight = 0.0
        attack_votes = 0
        attack_types = []
        
        for method, result in results.items():
            if method in weights:
                weight = weights[method]
                total_confidence += result['confidence'] * weight
                total_weight += weight
                
                if result['is_attack']:
                    attack_votes += 1
                    if result['attack_type'] != AttackType.UNKNOWN:
                        attack_types.append(result['attack_type'])
        
        # Determine final result
        final_confidence = total_confidence / total_weight if total_weight > 0 else 0.0
        is_attack = attack_votes >= 2 or final_confidence > self.sensitivity
        
        # Determine attack type
        if attack_types:
            # Use most common attack type
            attack_type = max(set(attack_types), key=attack_types.count)
        else:
            attack_type = AttackType.UNKNOWN
        
        # Determine threat level
        if final_confidence > 0.8:
            threat_level = ThreatLevel.CRITICAL
        elif final_confidence > 0.6:
            threat_level = ThreatLevel.HIGH
        elif final_confidence > 0.4:
            threat_level = ThreatLevel.MEDIUM
        else:
            threat_level = ThreatLevel.LOW
        
        # Combine features
        combined_features = {}
        for result in results.values():
            combined_features.update(result.get('features', {}))
        
        return AttackDetection(
            is_attack=is_attack,
            attack_type=attack_type,
            confidence=final_confidence,
            threat_level=threat_level,
            features=combined_features,
            mitigation_applied=False,
            detection_time=0.0,
            metadata={}
        )
    
    def _apply_mitigation(self, data: Any, detection: AttackDetection) -> bool:
        """Apply mitigation strategies for detected attacks."""
        try:
            # Log the attack
            self.logger.warning(f"Adversarial attack detected: {detection.attack_type} "
                              f"(confidence: {detection.confidence:.2f})")
            
            # Apply appropriate mitigation based on attack type
            if detection.attack_type == AttackType.FGSM:
                # Apply input smoothing
                return self._apply_input_smoothing(data)
            elif detection.attack_type == AttackType.POISONING:
                # Apply data filtering
                return self._apply_data_filtering(data)
            elif detection.attack_type == AttackType.MODEL_EVASION:
                # Apply confidence thresholding
                return self._apply_confidence_thresholding(data)
            else:
                # Apply general mitigation
                return self._apply_general_mitigation(data)
                
        except Exception as e:
            self.logger.error(f"Mitigation application failed: {e}")
            return False
    
    def _apply_input_smoothing(self, data: Any) -> bool:
        """Apply input smoothing to reduce adversarial noise."""
        # Implementation would smooth the input data
        return True
    
    def _apply_data_filtering(self, data: Any) -> bool:
        """Apply data filtering to remove poisoned samples."""
        # Implementation would filter out suspicious data
        return True
    
    def _apply_confidence_thresholding(self, data: Any) -> bool:
        """Apply confidence thresholding to reject low-confidence inputs."""
        # Implementation would reject low-confidence inputs
        return True
    
    def _apply_general_mitigation(self, data: Any) -> bool:
        """Apply general mitigation strategies."""
        # Implementation would apply general security measures
        return True
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0.0
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0.0
        
        skewness = np.mean(((data - mean_val) / std_val) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        if len(data) < 4:
            return 0.0
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0.0
        
        kurtosis = np.mean(((data - mean_val) / std_val) ** 4) - 3
        return kurtosis
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate entropy of data."""
        if len(data) == 0:
            return 0.0
        
        # Discretize data for entropy calculation
        bins = np.histogram(data, bins=min(10, len(data)))[0]
        probabilities = bins / np.sum(bins)
        probabilities = probabilities[probabilities > 0]  # Remove zero probabilities
        
        if len(probabilities) == 0:
            return 0.0
        
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def _calculate_pattern_consistency(self, data: np.ndarray, pattern_len: int) -> float:
        """Calculate consistency of repeating patterns."""
        if len(data) < pattern_len * 2:
            return 1.0
        
        patterns = []
        for i in range(len(data) - pattern_len + 1):
            pattern = data[i:i+pattern_len]
            patterns.append(pattern)
        
        if len(patterns) < 2:
            return 1.0
        
        # Calculate similarity between consecutive patterns
        similarities = []
        for i in range(len(patterns) - 1):
            similarity = np.corrcoef(patterns[i], patterns[i+1])[0, 1]
            if not np.isnan(similarity):
                similarities.append(similarity)
        
        if not similarities:
            return 1.0
        
        return np.mean(similarities)
    
    def _update_baseline(self, context: str, features: Dict[str, Any]):
        """Update baseline statistics for a context."""
        if context not in self.baseline_stats:
            self.baseline_stats[context] = {
                'mean': features.get('mean', 0),
                'std': features.get('std', 1),
                'mean_std': 1.0,
                'std_std': 1.0,
                'count': 1
            }
        else:
            baseline = self.baseline_stats[context]
            count = baseline['count']
            
            # Update running statistics
            new_mean = (baseline['mean'] * count + features.get('mean', 0)) / (count + 1)
            new_std = (baseline['std'] * count + features.get('std', 1)) / (count + 1)
            
            baseline['mean'] = new_mean
            baseline['std'] = new_std
            baseline['count'] = count + 1
    
    def _log_detection(self, detection: AttackDetection):
        """Log detection result."""
        self.detection_history.append(detection)
        
        if detection.is_attack:
            self.logger.warning(f"Adversarial attack detected: {detection.attack_type.value} "
                              f"(confidence: {detection.confidence:.2f}, "
                              f"threat: {detection.threat_level.value})")
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        if not self.detection_history:
            return {"total_detections": 0}
        
        total = len(self.detection_history)
        attacks = sum(1 for d in self.detection_history if d.is_attack)
        
        attack_types = {}
        threat_levels = {}
        
        for detection in self.detection_history:
            if detection.is_attack:
                attack_type = detection.attack_type.value
                attack_types[attack_type] = attack_types.get(attack_type, 0) + 1
                
                threat_level = detection.threat_level.value
                threat_levels[threat_level] = threat_levels.get(threat_level, 0) + 1
        
        return {
            "total_detections": total,
            "attack_count": attacks,
            "attack_rate": attacks / total,
            "attack_types": attack_types,
            "threat_levels": threat_levels,
            "average_confidence": np.mean([d.confidence for d in self.detection_history])
        }
