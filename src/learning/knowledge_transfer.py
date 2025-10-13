"""
Knowledge Transfer System for ARC Games

This module implements a sophisticated knowledge transfer system that enables
learning from previous games and applying that knowledge to new situations.
It includes pattern abstraction, similarity scoring, adaptation rules,
and validation checks.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import sqlite3
import logging
from datetime import datetime
import json

# Disable pycache
import sys
sys.dont_write_bytecode = True

logger = logging.getLogger(__name__)

@dataclass
class TransferablePattern:
    """Represents a pattern that can be transferred between games."""
    pattern_id: str
    pattern_type: str
    features: Dict[str, Any]
    abstraction_level: int
    confidence: float
    success_rate: float
    usage_count: int
    last_used: datetime
    game_ids: Set[str]

@dataclass
class PatternSimilarity:
    """Represents similarity between two patterns."""
    score: float
    matching_features: Dict[str, Any]
    differences: Dict[str, Any]
    confidence: float

@dataclass
class AdaptationRule:
    """Represents a rule for adapting patterns to new contexts."""
    rule_id: str
    condition: Dict[str, Any]
    transformation: Dict[str, Any]
    success_rate: float
    usage_count: int

class KnowledgeTransferSystem:
    """Core system for transferring knowledge between games."""

    def __init__(self, db_connection: Optional[sqlite3.Connection] = None):
        self.db_connection = db_connection
        self.patterns: Dict[str, TransferablePattern] = {}
        self.adaptation_rules: Dict[str, AdaptationRule] = {}
        self.similarity_threshold = 0.7
        self.confidence_threshold = 0.6

    def store_pattern(self,
                     game_id: str,
                     pattern_type: str,
                     pattern_data: Dict[str, Any],
                     success: bool) -> None:
        """Store a new pattern from gameplay."""
        
        # Create abstracted version of pattern
        abstracted = self._create_pattern_abstraction(pattern_data)
        
        # Generate pattern ID
        pattern_id = self._generate_pattern_id(pattern_type, abstracted)
        
        if pattern_id in self.patterns:
            # Update existing pattern
            pattern = self.patterns[pattern_id]
            pattern.usage_count += 1
            pattern.last_used = datetime.now()
            pattern.game_ids.add(game_id)
            
            # Update success rate
            total = pattern.usage_count
            current_success = pattern.success_rate * (total - 1)
            pattern.success_rate = (current_success + (1 if success else 0)) / total
            
        else:
            # Create new pattern
            pattern = TransferablePattern(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                features=abstracted,
                abstraction_level=self._calculate_abstraction_level(abstracted),
                confidence=self.confidence_threshold,
                success_rate=1.0 if success else 0.0,
                usage_count=1,
                last_used=datetime.now(),
                game_ids={game_id}
            )
            self.patterns[pattern_id] = pattern
            
        # Store in database
        self._store_pattern(pattern)

    def find_similar_patterns(self,
                            pattern_data: Dict[str, Any],
                            min_similarity: float = 0.7) -> List[Tuple[TransferablePattern, PatternSimilarity]]:
        """Find patterns similar to the given pattern data."""
        
        similar_patterns = []
        
        # Create abstracted version of input pattern
        abstracted = self._create_pattern_abstraction(pattern_data)
        
        for pattern in self.patterns.values():
            # Calculate similarity
            similarity = self._calculate_pattern_similarity(abstracted, pattern.features)
            
            if similarity.score >= min_similarity:
                similar_patterns.append((pattern, similarity))
                
        # Sort by similarity score
        similar_patterns.sort(key=lambda x: x[1].score, reverse=True)
        
        return similar_patterns

    def adapt_pattern(self,
                     pattern: TransferablePattern,
                     target_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Adapt a pattern to a new context."""
        
        try:
            # Find applicable adaptation rules
            applicable_rules = self._find_applicable_rules(pattern, target_context)
            
            if not applicable_rules:
                return pattern.features.copy()
                
            # Apply rules in order of success rate
            adapted_features = pattern.features.copy()
            for rule in applicable_rules:
                adapted_features = self._apply_adaptation_rule(adapted_features, rule)
                
            return adapted_features
            
        except Exception as e:
            logger.error(f"Error adapting pattern: {e}")
            return None

    def validate_transfer(self,
                         original_pattern: TransferablePattern,
                         adapted_pattern: Dict[str, Any],
                         target_context: Dict[str, Any]) -> bool:
        """Validate if pattern transfer is likely to succeed."""
        
        # Check confidence thresholds
        if original_pattern.confidence < self.confidence_threshold:
            return False
            
        # Perform validation checks
        checks = [
            self._validate_feature_compatibility(adapted_pattern, target_context),
            self._validate_complexity_match(adapted_pattern, target_context),
            self._validate_constraints(adapted_pattern, target_context)
        ]
        
        return all(checks)

    def _create_pattern_abstraction(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create an abstracted version of a pattern."""
        
        abstracted = {}
        
        # Abstract numerical values into ranges
        for key, value in pattern_data.items():
            if isinstance(value, (int, float)):
                abstracted[key] = self._abstract_numerical_value(value)
            elif isinstance(value, (list, tuple)):
                abstracted[key] = self._abstract_sequence(value)
            elif isinstance(value, dict):
                abstracted[key] = self._create_pattern_abstraction(value)
            else:
                abstracted[key] = value
                
        return abstracted

    def _abstract_numerical_value(self, value: float) -> str:
        """Convert numerical value to abstract range."""
        if value == 0:
            return "zero"
        elif 0 < value <= 0.33:
            return "low"
        elif 0.33 < value <= 0.66:
            return "medium"
        else:
            return "high"

    def _abstract_sequence(self, sequence: Any) -> List[Any]:
        """Abstract a sequence of values."""
        if not sequence:
            return []
            
        # Convert to list if tuple
        sequence_list = list(sequence)
            
        # Detect patterns in sequence
        if all(isinstance(x, (int, float)) for x in sequence_list):
            return self._abstract_numerical_sequence(sequence_list)
        
        return [self._create_pattern_abstraction(x) if isinstance(x, dict) else x
                for x in sequence_list]

    def _abstract_numerical_sequence(self, sequence: List[float]) -> List[str]:
        """Abstract a sequence of numerical values."""
        differences = [sequence[i+1] - sequence[i] for i in range(len(sequence)-1)]
        
        if all(abs(d - differences[0]) < 0.0001 for d in differences):
            return ["arithmetic_sequence", str(differences[0])]
        
        ratios = [sequence[i+1]/sequence[i] for i in range(len(sequence)-1)
                 if sequence[i] != 0]
        if ratios and all(abs(r - ratios[0]) < 0.0001 for r in ratios):
            return ["geometric_sequence", str(ratios[0])]
            
        return [self._abstract_numerical_value(x) for x in sequence]

    def _calculate_abstraction_level(self, abstracted: Dict[str, Any]) -> int:
        """Calculate the level of abstraction of a pattern."""
        level = 0
        
        for value in abstracted.values():
            if isinstance(value, (list, tuple)):
                level += sum(isinstance(x, str) and x in 
                           ["low", "medium", "high", "arithmetic_sequence", "geometric_sequence"]
                           for x in value)
            elif isinstance(value, str) and value in ["low", "medium", "high"]:
                level += 1
            elif isinstance(value, dict):
                level += self._calculate_abstraction_level(value)
                
        return level

    def _calculate_pattern_similarity(self,
                                    pattern1: Dict[str, Any],
                                    pattern2: Dict[str, Any]) -> PatternSimilarity:
        """Calculate similarity between two patterns."""
        
        matching = {}
        differences = {}
        total_features = 0
        matched_features = 0
        
        for key in set(pattern1.keys()) | set(pattern2.keys()):
            total_features += 1
            
            if key in pattern1 and key in pattern2:
                if isinstance(pattern1[key], dict) and isinstance(pattern2[key], dict):
                    subsimilarity = self._calculate_pattern_similarity(pattern1[key], pattern2[key])
                    matched_features += subsimilarity.score
                    matching[key] = subsimilarity.matching_features
                    if subsimilarity.differences:
                        differences[key] = subsimilarity.differences
                elif pattern1[key] == pattern2[key]:
                    matched_features += 1
                    matching[key] = pattern1[key]
                else:
                    differences[key] = (pattern1[key], pattern2[key])
            else:
                differences[key] = (pattern1.get(key, None), pattern2.get(key, None))
                
        similarity_score = matched_features / total_features if total_features > 0 else 0.0
        
        return PatternSimilarity(
            score=similarity_score,
            matching_features=matching,
            differences=differences,
            confidence=self._calculate_similarity_confidence(matched_features, total_features)
        )

    def _calculate_similarity_confidence(self, matched: float, total: int) -> float:
        """Calculate confidence in similarity assessment."""
        if total == 0:
            return 0.0
            
        # Consider both match ratio and sample size
        match_ratio = matched / total
        size_factor = min(1.0, total / 10.0)  # Saturates at 10 features
        
        return match_ratio * size_factor

    def _find_applicable_rules(self,
                             pattern: TransferablePattern,
                             target_context: Dict[str, Any]) -> List[AdaptationRule]:
        """Find adaptation rules applicable to the pattern and context."""
        
        applicable_rules = []
        
        for rule in self.adaptation_rules.values():
            if self._check_rule_applicability(rule, pattern, target_context):
                applicable_rules.append(rule)
                
        # Sort by success rate
        applicable_rules.sort(key=lambda r: r.success_rate, reverse=True)
        
        return applicable_rules

    def _check_rule_applicability(self,
                                rule: AdaptationRule,
                                pattern: TransferablePattern,
                                target_context: Dict[str, Any]) -> bool:
        """Check if an adaptation rule is applicable."""
        
        try:
            for key, condition in rule.condition.items():
                if key.startswith('pattern.'):
                    pattern_key = key[8:]  # Remove 'pattern.' prefix
                    if pattern_key not in pattern.features:
                        return False
                    if not self._match_condition(pattern.features[pattern_key], condition):
                        return False
                elif key.startswith('context.'):
                    context_key = key[8:]  # Remove 'context.' prefix
                    if context_key not in target_context:
                        return False
                    if not self._match_condition(target_context[context_key], condition):
                        return False
                        
            return True
            
        except Exception as e:
            logger.error(f"Error checking rule applicability: {e}")
            return False

    def _match_condition(self, value: Any, condition: Any) -> bool:
        """Check if a value matches a condition."""
        
        if isinstance(condition, dict):
            if 'type' in condition:
                if condition['type'] == 'range':
                    return (condition.get('min', float('-inf')) <= value <=
                            condition.get('max', float('inf')))
                elif condition['type'] == 'set':
                    return value in condition.get('values', [])
                elif condition['type'] == 'regex':
                    import re
                    return bool(re.match(condition['pattern'], str(value)))
        else:
            return value == condition
            
        return False

    def _apply_adaptation_rule(self,
                             features: Dict[str, Any],
                             rule: AdaptationRule) -> Dict[str, Any]:
        """Apply an adaptation rule to pattern features."""
        
        adapted = features.copy()
        
        for key, transform in rule.transformation.items():
            if key in adapted:
                if isinstance(transform, dict):
                    if transform.get('type') == 'scale':
                        adapted[key] = adapted[key] * transform['factor']
                    elif transform.get('type') == 'offset':
                        adapted[key] = adapted[key] + transform['value']
                    elif transform.get('type') == 'map':
                        if adapted[key] in transform.get('mapping', {}):
                            adapted[key] = transform['mapping'][adapted[key]]
                else:
                    adapted[key] = transform
                    
        return adapted

    def _validate_feature_compatibility(self,
                                      adapted_pattern: Dict[str, Any],
                                      target_context: Dict[str, Any]) -> bool:
        """Validate feature compatibility with target context."""
        
        # Check for required features
        required_features = target_context.get('required_features', set())
        if not required_features.issubset(adapted_pattern.keys()):
            return False
            
        # Check value ranges
        valid_ranges = target_context.get('valid_ranges', {})
        for feature, range_info in valid_ranges.items():
            if feature in adapted_pattern:
                value = adapted_pattern[feature]
                if not (range_info['min'] <= value <= range_info['max']):
                    return False
                    
        return True

    def _validate_complexity_match(self,
                                 adapted_pattern: Dict[str, Any],
                                 target_context: Dict[str, Any]) -> bool:
        """Validate complexity compatibility."""
        
        pattern_complexity = self._calculate_pattern_complexity(adapted_pattern)
        target_complexity = target_context.get('complexity', 0)
        
        # Allow some flexibility in complexity matching
        return abs(pattern_complexity - target_complexity) <= 1

    def _calculate_pattern_complexity(self, pattern: Dict[str, Any]) -> float:
        """Calculate the complexity of a pattern based on its features."""
        complexity = 0.0
        
        for value in pattern.values():
            if isinstance(value, dict):
                complexity += 1 + self._calculate_pattern_complexity(value)
            elif isinstance(value, (list, tuple)):
                complexity += 0.5 * len(value)
            elif isinstance(value, (int, float)):
                complexity += 0.1
                
        return complexity

    def _validate_constraints(self,
                            adapted_pattern: Dict[str, Any],
                            target_context: Dict[str, Any]) -> bool:
        """Validate pattern against target constraints."""
        
        constraints = target_context.get('constraints', [])
        
        for constraint in constraints:
            if not self._check_constraint(adapted_pattern, constraint):
                return False
                
        return True

    def _check_constraint(self, pattern: Dict[str, Any], constraint: Dict[str, Any]) -> bool:
        """Check if pattern satisfies a constraint."""
        
        constraint_type = constraint.get('type')
        
        if constraint_type == 'dependency':
            return self._check_dependency_constraint(pattern, constraint)
        elif constraint_type == 'exclusion':
            return self._check_exclusion_constraint(pattern, constraint)
        elif constraint_type == 'composition':
            return self._check_composition_constraint(pattern, constraint)
            
        return True

    def _check_dependency_constraint(self,
                                   pattern: Dict[str, Any],
                                   constraint: Dict[str, Any]) -> bool:
        """Check dependency constraint."""
        required = constraint.get('required', [])
        if any(req not in pattern for req in required):
            return False
        return True

    def _check_exclusion_constraint(self,
                                  pattern: Dict[str, Any],
                                  constraint: Dict[str, Any]) -> bool:
        """Check exclusion constraint."""
        excluded = constraint.get('excluded', [])
        if any(excl in pattern for excl in excluded):
            return False
        return True

    def _check_composition_constraint(self,
                                    pattern: Dict[str, Any],
                                    constraint: Dict[str, Any]) -> bool:
        """Check composition constraint."""
        required_composition = constraint.get('composition', {})
        for key, required_value in required_composition.items():
            if pattern.get(key) != required_value:
                return False
        return True

    def _store_pattern(self, pattern: TransferablePattern) -> None:
        """Store pattern in database."""
        
        if not self.db_connection:
            return
            
        try:
            cursor = self.db_connection.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO transferable_patterns
                (pattern_id, pattern_type, features, abstraction_level,
                 confidence, success_rate, usage_count, last_used, game_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern.pattern_id,
                pattern.pattern_type,
                json.dumps(pattern.features),
                pattern.abstraction_level,
                pattern.confidence,
                pattern.success_rate,
                pattern.usage_count,
                pattern.last_used.isoformat(),
                json.dumps(list(pattern.game_ids))
            ))
            
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing pattern: {e}")

    def _generate_pattern_id(self, pattern_type: str, features: Dict[str, Any]) -> str:
        """Generate unique ID for a pattern."""
        import hashlib
        feature_str = json.dumps(features, sort_keys=True)
        hash_obj = hashlib.md5(feature_str.encode())
        return f"{pattern_type}_{hash_obj.hexdigest()[:8]}"

# Create singleton instance
_knowledge_transfer_system = None

def get_knowledge_transfer_system(db_connection: Optional[sqlite3.Connection] = None) -> KnowledgeTransferSystem:
    """Get or create the knowledge transfer system singleton."""
    global _knowledge_transfer_system
    if _knowledge_transfer_system is None:
        _knowledge_transfer_system = KnowledgeTransferSystem(db_connection)
    return _knowledge_transfer_system