#!/usr/bin/env python3
"""
Threat Intelligence System - Advanced threat detection and analysis.

This module provides:
- Threat pattern recognition
- Risk assessment
- Threat intelligence gathering
- Predictive threat analysis
- Threat response recommendations
"""

import logging
import time
import json
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import hashlib

logger = logging.getLogger(__name__)


class ThreatCategory(Enum):
    """Categories of threats."""
    MALWARE = "malware"
    INTRUSION = "intrusion"
    DATA_BREACH = "data_breach"
    DOS_ATTACK = "dos_attack"
    PHISHING = "phishing"
    SOCIAL_ENGINEERING = "social_engineering"
    INSIDER_THREAT = "insider_threat"
    SUPPLY_CHAIN = "supply_chain"
    ZERO_DAY = "zero_day"
    APT = "apt"  # Advanced Persistent Threat
    UNKNOWN = "unknown"


class RiskLevel(Enum):
    """Risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ThreatPattern:
    """Threat pattern definition."""
    pattern_id: str
    name: str
    category: ThreatCategory
    description: str
    indicators: List[Dict[str, Any]]
    severity: RiskLevel
    confidence: float
    last_seen: Optional[datetime] = None
    frequency: int = 0
    mitigation_strategies: List[str] = None


@dataclass
class RiskAssessment:
    """Risk assessment result."""
    threat_id: str
    risk_level: RiskLevel
    confidence: float
    factors: List[str]
    recommendations: List[str]
    mitigation_priority: int
    assessment_time: datetime
    metadata: Dict[str, Any]


class ThreatIntelligence:
    """
    Advanced threat intelligence system.
    
    Provides:
    - Threat pattern recognition
    - Risk assessment
    - Threat intelligence gathering
    - Predictive analysis
    - Response recommendations
    """
    
    def __init__(self, max_patterns: int = 1000, learning_rate: float = 0.1):
        self.max_patterns = max_patterns
        self.learning_rate = learning_rate
        self.logger = logging.getLogger(__name__)
        
        # Threat patterns database
        self.threat_patterns: Dict[str, ThreatPattern] = {}
        self.pattern_frequency: Dict[str, int] = defaultdict(int)
        self.pattern_indicators: Dict[str, Set[str]] = defaultdict(set)
        
        # Risk assessment history
        self.risk_assessments: deque = deque(maxlen=1000)
        self.risk_trends: Dict[str, List[float]] = defaultdict(list)
        
        # Threat intelligence sources
        self.intelligence_sources: List[str] = []
        self.threat_feeds: Dict[str, List[Dict]] = {}
        
        # Learning and adaptation
        self.learning_enabled = True
        self.adaptation_threshold = 0.7
        
        # Initialize threat patterns
        self._initialize_threat_patterns()
    
    def _initialize_threat_patterns(self):
        """Initialize known threat patterns."""
        # Malware patterns
        malware_patterns = [
            {
                "pattern_id": "malware_executable",
                "name": "Malicious Executable",
                "category": ThreatCategory.MALWARE,
                "description": "Suspicious executable file patterns",
                "indicators": [
                    {"type": "file_extension", "value": ".exe", "weight": 0.3},
                    {"type": "file_size", "value": "suspicious", "weight": 0.2},
                    {"type": "entropy", "value": "high", "weight": 0.4},
                    {"type": "digital_signature", "value": "missing", "weight": 0.1}
                ],
                "severity": RiskLevel.HIGH,
                "confidence": 0.8
            },
            {
                "pattern_id": "suspicious_network",
                "name": "Suspicious Network Activity",
                "category": ThreatCategory.INTRUSION,
                "description": "Unusual network communication patterns",
                "indicators": [
                    {"type": "port_scan", "value": "detected", "weight": 0.4},
                    {"type": "unusual_protocol", "value": "detected", "weight": 0.3},
                    {"type": "data_exfiltration", "value": "suspected", "weight": 0.3}
                ],
                "severity": RiskLevel.MEDIUM,
                "confidence": 0.7
            },
            {
                "pattern_id": "data_breach_indicators",
                "name": "Data Breach Indicators",
                "category": ThreatCategory.DATA_BREACH,
                "description": "Signs of potential data breach",
                "indicators": [
                    {"type": "unusual_data_access", "value": "detected", "weight": 0.3},
                    {"type": "large_data_transfer", "value": "detected", "weight": 0.4},
                    {"type": "privilege_escalation", "value": "detected", "weight": 0.3}
                ],
                "severity": RiskLevel.CRITICAL,
                "confidence": 0.9
            }
        ]
        
        for pattern_data in malware_patterns:
            pattern = ThreatPattern(
                pattern_id=pattern_data["pattern_id"],
                name=pattern_data["name"],
                category=pattern_data["category"],
                description=pattern_data["description"],
                indicators=pattern_data["indicators"],
                severity=pattern_data["severity"],
                confidence=pattern_data["confidence"],
                mitigation_strategies=self._get_default_mitigation_strategies(pattern_data["category"])
            )
            self.threat_patterns[pattern.pattern_id] = pattern
    
    def _get_default_mitigation_strategies(self, category: ThreatCategory) -> List[str]:
        """Get default mitigation strategies for threat category."""
        strategies = {
            ThreatCategory.MALWARE: [
                "Quarantine suspicious files",
                "Update antivirus signatures",
                "Scan system for malware",
                "Isolate affected systems"
            ],
            ThreatCategory.INTRUSION: [
                "Block suspicious IP addresses",
                "Increase network monitoring",
                "Review access logs",
                "Implement additional authentication"
            ],
            ThreatCategory.DATA_BREACH: [
                "Immediately isolate affected systems",
                "Change all passwords",
                "Notify stakeholders",
                "Conduct forensic analysis"
            ],
            ThreatCategory.DOS_ATTACK: [
                "Implement rate limiting",
                "Use DDoS protection services",
                "Scale resources",
                "Block attack sources"
            ],
            ThreatCategory.PHISHING: [
                "Educate users about phishing",
                "Implement email filtering",
                "Monitor for credential theft",
                "Enable multi-factor authentication"
            ]
        }
        
        return strategies.get(category, ["Investigate threat", "Implement security measures"])
    
    def analyze_threat(self, threat_data: Dict[str, Any]) -> RiskAssessment:
        """
        Analyze a potential threat and provide risk assessment.
        
        Args:
            threat_data: Data about the potential threat
            
        Returns:
            RiskAssessment with analysis results
        """
        threat_id = self._generate_threat_id(threat_data)
        
        # Match against known patterns
        matched_patterns = self._match_threat_patterns(threat_data)
        
        # Calculate risk level
        risk_level, confidence, factors = self._calculate_risk_level(threat_data, matched_patterns)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(matched_patterns, risk_level)
        
        # Determine mitigation priority
        mitigation_priority = self._calculate_mitigation_priority(risk_level, confidence)
        
        # Create risk assessment
        assessment = RiskAssessment(
            threat_id=threat_id,
            risk_level=risk_level,
            confidence=confidence,
            factors=factors,
            recommendations=recommendations,
            mitigation_priority=mitigation_priority,
            assessment_time=datetime.now(),
            metadata={
                "matched_patterns": [p.pattern_id for p in matched_patterns],
                "threat_data": threat_data
            }
        )
        
        # Store assessment
        self.risk_assessments.append(assessment)
        
        # Update learning if enabled
        if self.learning_enabled:
            self._update_learning(threat_data, matched_patterns, assessment)
        
        return assessment
    
    def _generate_threat_id(self, threat_data: Dict[str, Any]) -> str:
        """Generate unique threat ID."""
        threat_string = json.dumps(threat_data, sort_keys=True)
        return hashlib.sha256(threat_string.encode()).hexdigest()[:16]
    
    def _match_threat_patterns(self, threat_data: Dict[str, Any]) -> List[ThreatPattern]:
        """Match threat data against known patterns."""
        matched_patterns = []
        
        for pattern in self.threat_patterns.values():
            match_score = self._calculate_pattern_match_score(threat_data, pattern)
            
            if match_score > pattern.confidence:
                matched_patterns.append(pattern)
                # Update pattern frequency
                self.pattern_frequency[pattern.pattern_id] += 1
                pattern.frequency += 1
                pattern.last_seen = datetime.now()
        
        return matched_patterns
    
    def _calculate_pattern_match_score(self, threat_data: Dict[str, Any], 
                                     pattern: ThreatPattern) -> float:
        """Calculate how well threat data matches a pattern."""
        total_score = 0.0
        total_weight = 0.0
        
        for indicator in pattern.indicators:
            indicator_type = indicator["type"]
            indicator_value = indicator["value"]
            weight = indicator["weight"]
            
            # Check if threat data contains this indicator
            if self._check_indicator_match(threat_data, indicator_type, indicator_value):
                total_score += weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _check_indicator_match(self, threat_data: Dict[str, Any], 
                              indicator_type: str, indicator_value: str) -> bool:
        """Check if threat data matches a specific indicator."""
        # This is a simplified implementation
        # In practice, this would be much more sophisticated
        
        if indicator_type == "file_extension":
            return threat_data.get("file_extension") == indicator_value
        elif indicator_type == "file_size":
            return threat_data.get("file_size") == indicator_value
        elif indicator_type == "entropy":
            return threat_data.get("entropy") == indicator_value
        elif indicator_type == "digital_signature":
            return threat_data.get("digital_signature") == indicator_value
        elif indicator_type == "port_scan":
            return threat_data.get("port_scan") == indicator_value
        elif indicator_type == "unusual_protocol":
            return threat_data.get("unusual_protocol") == indicator_value
        elif indicator_type == "data_exfiltration":
            return threat_data.get("data_exfiltration") == indicator_value
        elif indicator_type == "unusual_data_access":
            return threat_data.get("unusual_data_access") == indicator_value
        elif indicator_type == "large_data_transfer":
            return threat_data.get("large_data_transfer") == indicator_value
        elif indicator_type == "privilege_escalation":
            return threat_data.get("privilege_escalation") == indicator_value
        else:
            return False
    
    def _calculate_risk_level(self, threat_data: Dict[str, Any], 
                            matched_patterns: List[ThreatPattern]) -> Tuple[RiskLevel, float, List[str]]:
        """Calculate risk level based on threat data and matched patterns."""
        if not matched_patterns:
            return RiskLevel.LOW, 0.1, ["No known threat patterns matched"]
        
        # Calculate weighted risk level
        total_risk_score = 0.0
        total_confidence = 0.0
        factors = []
        
        for pattern in matched_patterns:
            # Map severity to numeric score
            severity_scores = {
                RiskLevel.LOW: 1.0,
                RiskLevel.MEDIUM: 2.0,
                RiskLevel.HIGH: 3.0,
                RiskLevel.CRITICAL: 4.0
            }
            
            risk_score = severity_scores[pattern.severity]
            confidence = pattern.confidence
            
            total_risk_score += risk_score * confidence
            total_confidence += confidence
            
            factors.append(f"Matched pattern: {pattern.name} (severity: {pattern.severity.value})")
        
        # Calculate average risk score
        avg_risk_score = total_risk_score / total_confidence if total_confidence > 0 else 0.0
        avg_confidence = total_confidence / len(matched_patterns) if matched_patterns else 0.0
        
        # Determine risk level
        if avg_risk_score >= 3.5:
            risk_level = RiskLevel.CRITICAL
        elif avg_risk_score >= 2.5:
            risk_level = RiskLevel.HIGH
        elif avg_risk_score >= 1.5:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # Add additional factors
        if threat_data.get("immediate_threat", False):
            risk_level = RiskLevel.CRITICAL
            factors.append("Immediate threat detected")
        
        if threat_data.get("data_sensitivity", "low") == "high":
            if risk_level == RiskLevel.LOW:
                risk_level = RiskLevel.MEDIUM
            elif risk_level == RiskLevel.MEDIUM:
                risk_level = RiskLevel.HIGH
            factors.append("High sensitivity data involved")
        
        return risk_level, avg_confidence, factors
    
    def _generate_recommendations(self, matched_patterns: List[ThreatPattern], 
                                risk_level: RiskLevel) -> List[str]:
        """Generate security recommendations based on matched patterns and risk level."""
        recommendations = []
        
        # Add pattern-specific recommendations
        for pattern in matched_patterns:
            recommendations.extend(pattern.mitigation_strategies)
        
        # Add risk-level specific recommendations
        if risk_level == RiskLevel.CRITICAL:
            recommendations.extend([
                "Immediately isolate affected systems",
                "Activate incident response team",
                "Notify senior management",
                "Consider system shutdown if necessary"
            ])
        elif risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "Increase monitoring frequency",
                "Implement additional security controls",
                "Review and update security policies",
                "Conduct security assessment"
            ])
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.extend([
                "Monitor for additional indicators",
                "Review security logs",
                "Update threat intelligence",
                "Consider preventive measures"
            ])
        else:
            recommendations.extend([
                "Continue normal monitoring",
                "Document incident for future reference",
                "Review security procedures"
            ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _calculate_mitigation_priority(self, risk_level: RiskLevel, confidence: float) -> int:
        """Calculate mitigation priority (1 = highest, 5 = lowest)."""
        priority_scores = {
            RiskLevel.CRITICAL: 1,
            RiskLevel.HIGH: 2,
            RiskLevel.MEDIUM: 3,
            RiskLevel.LOW: 4
        }
        
        base_priority = priority_scores[risk_level]
        
        # Adjust based on confidence
        if confidence > 0.8:
            return max(1, base_priority - 1)
        elif confidence < 0.5:
            return min(5, base_priority + 1)
        else:
            return base_priority
    
    def _update_learning(self, threat_data: Dict[str, Any], matched_patterns: List[ThreatPattern],
                        assessment: RiskAssessment):
        """Update learning based on threat analysis results."""
        try:
            # Update pattern confidence based on assessment accuracy
            for pattern in matched_patterns:
                if assessment.risk_level == pattern.severity:
                    # Correct assessment - increase confidence
                    pattern.confidence = min(1.0, pattern.confidence + self.learning_rate * 0.1)
                else:
                    # Incorrect assessment - decrease confidence
                    pattern.confidence = max(0.1, pattern.confidence - self.learning_rate * 0.05)
            
            # Update risk trends
            risk_scores = {
                RiskLevel.LOW: 1.0,
                RiskLevel.MEDIUM: 2.0,
                RiskLevel.HIGH: 3.0,
                RiskLevel.CRITICAL: 4.0
            }
            
            current_time = datetime.now()
            for pattern in matched_patterns:
                pattern_id = pattern.pattern_id
                self.risk_trends[pattern_id].append(risk_scores[assessment.risk_level])
                
                # Keep only recent trends (last 100 assessments)
                if len(self.risk_trends[pattern_id]) > 100:
                    self.risk_trends[pattern_id] = self.risk_trends[pattern_id][-100:]
            
        except Exception as e:
            self.logger.error(f"Error updating learning: {e}")
    
    def add_threat_pattern(self, pattern: ThreatPattern):
        """Add a new threat pattern."""
        self.threat_patterns[pattern.pattern_id] = pattern
        self.logger.info(f"Added threat pattern: {pattern.name}")
    
    def remove_threat_pattern(self, pattern_id: str):
        """Remove a threat pattern."""
        if pattern_id in self.threat_patterns:
            del self.threat_patterns[pattern_id]
            self.logger.info(f"Removed threat pattern: {pattern_id}")
    
    def get_threat_patterns(self, category: Optional[ThreatCategory] = None) -> List[ThreatPattern]:
        """Get threat patterns, optionally filtered by category."""
        patterns = list(self.threat_patterns.values())
        
        if category:
            patterns = [p for p in patterns if p.category == category]
        
        return patterns
    
    def get_risk_trends(self, pattern_id: Optional[str] = None) -> Dict[str, List[float]]:
        """Get risk trends for patterns."""
        if pattern_id:
            return {pattern_id: self.risk_trends.get(pattern_id, [])}
        else:
            return dict(self.risk_trends)
    
    def get_threat_intelligence_summary(self) -> Dict[str, Any]:
        """Get threat intelligence summary."""
        total_patterns = len(self.threat_patterns)
        total_assessments = len(self.risk_assessments)
        
        # Calculate pattern distribution by category
        category_distribution = defaultdict(int)
        for pattern in self.threat_patterns.values():
            category_distribution[pattern.category.value] += 1
        
        # Calculate risk level distribution
        risk_distribution = defaultdict(int)
        for assessment in self.risk_assessments:
            risk_distribution[assessment.risk_level.value] += 1
        
        # Calculate average confidence
        avg_confidence = 0.0
        if self.risk_assessments:
            avg_confidence = sum(a.confidence for a in self.risk_assessments) / len(self.risk_assessments)
        
        return {
            "total_patterns": total_patterns,
            "total_assessments": total_assessments,
            "category_distribution": dict(category_distribution),
            "risk_distribution": dict(risk_distribution),
            "average_confidence": avg_confidence,
            "learning_enabled": self.learning_enabled,
            "recent_assessments": len([a for a in self.risk_assessments 
                                     if a.assessment_time > datetime.now() - timedelta(hours=24)])
        }
    
    def predict_threat_likelihood(self, threat_data: Dict[str, Any]) -> float:
        """Predict the likelihood of a threat based on historical data."""
        if not self.risk_assessments:
            return 0.5  # Default probability
        
        # Find similar historical threats
        similar_threats = []
        for assessment in self.risk_assessments:
            similarity = self._calculate_threat_similarity(threat_data, assessment.metadata.get("threat_data", {}))
            if similarity > 0.7:  # Similarity threshold
                similar_threats.append(assessment)
        
        if not similar_threats:
            return 0.5  # No similar threats found
        
        # Calculate weighted likelihood based on similar threats
        total_weight = 0.0
        weighted_risk = 0.0
        
        for assessment in similar_threats:
            weight = assessment.confidence
            risk_score = {
                RiskLevel.LOW: 0.2,
                RiskLevel.MEDIUM: 0.5,
                RiskLevel.HIGH: 0.8,
                RiskLevel.CRITICAL: 1.0
            }[assessment.risk_level]
            
            weighted_risk += risk_score * weight
            total_weight += weight
        
        return weighted_risk / total_weight if total_weight > 0 else 0.5
    
    def _calculate_threat_similarity(self, threat1: Dict[str, Any], threat2: Dict[str, Any]) -> float:
        """Calculate similarity between two threat data sets."""
        # Simple similarity calculation based on common keys
        keys1 = set(threat1.keys())
        keys2 = set(threat2.keys())
        
        if not keys1 or not keys2:
            return 0.0
        
        common_keys = keys1.intersection(keys2)
        total_keys = keys1.union(keys2)
        
        if not common_keys:
            return 0.0
        
        # Calculate value similarity for common keys
        value_similarity = 0.0
        for key in common_keys:
            if threat1[key] == threat2[key]:
                value_similarity += 1.0
            elif isinstance(threat1[key], (int, float)) and isinstance(threat2[key], (int, float)):
                # Numeric similarity
                max_val = max(abs(threat1[key]), abs(threat2[key]))
                if max_val > 0:
                    similarity = 1.0 - abs(threat1[key] - threat2[key]) / max_val
                    value_similarity += max(0, similarity)
        
        # Combine key similarity and value similarity
        key_similarity = len(common_keys) / len(total_keys)
        value_similarity = value_similarity / len(common_keys) if common_keys else 0.0
        
        return (key_similarity + value_similarity) / 2.0
