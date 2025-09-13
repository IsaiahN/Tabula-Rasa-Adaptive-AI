#!/usr/bin/env python3
"""
Cross-Session Learning Persistence System

This module provides comprehensive persistence capabilities for meta-cognitive systems,
enabling them to maintain learned patterns, successful strategies, and accumulated
knowledge across different training sessions and system restarts.
"""

import json
import pickle
import time
import logging
import hashlib
import os
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
from enum import Enum
from collections import defaultdict, deque
import statistics
import threading
from datetime import datetime, timedelta

class PersistenceLevel(Enum):
    """Different levels of persistence for different data types."""
    EPHEMERAL = "ephemeral"      # Session only, not persisted
    SESSION = "session"          # Persisted for current session
    PERMANENT = "permanent"      # Persisted across all sessions
    CRITICAL = "critical"        # Critical knowledge, backed up

class KnowledgeType(Enum):
    """Types of knowledge that can be persisted."""
    STRATEGY_PATTERN = "strategy_pattern"
    PARAMETER_OPTIMIZATION = "parameter_optimization"  
    MUTATION_SUCCESS = "mutation_success"
    CONFIGURATION_PROFILE = "configuration_profile"
    PERFORMANCE_BASELINE = "performance_baseline"
    FAILURE_PATTERN = "failure_pattern"
    CONTEXTUAL_INSIGHT = "contextual_insight"
    ACTION_PATTERN = "action_pattern"
    SPATIAL_PATTERN = "spatial_pattern"

@dataclass
class LearnedPattern:
    """A pattern learned from experience that should be preserved."""
    pattern_id: str
    knowledge_type: KnowledgeType
    pattern_data: Dict[str, Any]
    
    # Context information
    puzzle_types: List[str] = field(default_factory=list)
    performance_conditions: Dict[str, Any] = field(default_factory=dict)
    environmental_factors: Dict[str, Any] = field(default_factory=dict)
    
    # Success metrics
    success_rate: float = 0.0
    confidence: float = 0.0
    sample_size: int = 0
    total_applications: int = 0
    successful_applications: int = 0
    
    # Temporal information
    first_learned: float = field(default_factory=time.time)
    last_applied: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    
    # Persistence metadata
    persistence_level: PersistenceLevel = PersistenceLevel.SESSION
    importance_score: float = 0.0
    decay_factor: float = 0.95  # How quickly importance decays over time
    
    def update_success_metrics(self, was_successful: bool):
        """Update success metrics based on application result."""
        self.total_applications += 1
        if was_successful:
            self.successful_applications += 1
        
        self.success_rate = self.successful_applications / self.total_applications
        self.last_applied = time.time()
        self.last_updated = time.time()
        
        # Update confidence based on sample size and consistency
        self.confidence = min(0.95, 
                             (self.total_applications / 20.0) * 0.6 +
                             self.success_rate * 0.4)
    
    def calculate_importance(self) -> float:
        """Calculate current importance score for prioritization."""
        # Base importance from success metrics
        base_importance = self.success_rate * self.confidence
        
        # Boost for high-impact patterns
        if self.success_rate > 0.8 and self.total_applications >= 5:
            base_importance *= 1.5
        
        # Time decay factor
        days_since_update = (time.time() - self.last_updated) / 86400
        time_factor = self.decay_factor ** days_since_update
        
        # Frequency bonus
        frequency_factor = min(2.0, 1.0 + (self.total_applications / 50.0))
        
        self.importance_score = base_importance * time_factor * frequency_factor
        return self.importance_score
    
    def is_applicable(self, context: Dict[str, Any]) -> bool:
        """Check if this pattern is applicable to the given context."""
        # Check puzzle type compatibility
        if self.puzzle_types and context.get('puzzle_type'):
            if context['puzzle_type'] not in self.puzzle_types:
                # Allow similar puzzle types (more flexible matching)
                context_type = context['puzzle_type']
                has_similar = any(
                    context_type in puzzle_type or puzzle_type in context_type 
                    for puzzle_type in self.puzzle_types
                )
                if not has_similar:
                    return False
        
        # Check performance conditions (more lenient)
        current_perf = context.get('current_performance', {})
        for condition, threshold in self.performance_conditions.items():
            if condition in current_perf:
                if isinstance(threshold, dict):
                    min_val = threshold.get('min', float('-inf'))
                    max_val = threshold.get('max', float('inf'))
                    # Allow some tolerance in the range
                    tolerance = 0.1
                    if not ((min_val - tolerance) <= current_perf[condition] <= (max_val + tolerance)):
                        return False
                else:
                    # Simple threshold check with tolerance
                    if current_perf[condition] < (threshold - 0.1):
                        return False
        
        return True

@dataclass
class SessionMetaLearningState:
    """Comprehensive state of meta-learning across sessions."""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    
    # Learned patterns from this session
    session_patterns: List[LearnedPattern] = field(default_factory=list)
    
    # Performance summary
    total_decisions: int = 0
    successful_decisions: int = 0
    average_performance_improvement: float = 0.0
    
    # Key insights discovered
    key_insights: List[str] = field(default_factory=list)
    
    # Configuration evolution
    initial_config: Dict[str, Any] = field(default_factory=dict)
    final_config: Dict[str, Any] = field(default_factory=dict)
    config_changes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Environmental context
    training_context: Dict[str, Any] = field(default_factory=dict)

class CrossSessionLearningManager:
    """Manages persistence and recovery of learned patterns across sessions."""
    
    def __init__(self, persistence_dir: Path, logger: Optional[logging.Logger] = None):
        self.persistence_dir = Path(persistence_dir)
        self.persistence_dir.mkdir(exist_ok=True)
        self.logger = logger or logging.getLogger(f"{__name__}.CrossSessionLearning")
        
        # Current session state
        self.current_session = None
        self.session_start_time = time.time()
        
        # Learned patterns storage
        self.learned_patterns = {}  # pattern_id -> LearnedPattern
        self.patterns_by_type = defaultdict(list)  # KnowledgeType -> List[pattern_id]
        self.patterns_by_context = defaultdict(list)  # context_hash -> List[pattern_id]
        
        # Performance tracking
        self.session_history = deque(maxlen=100)  # Recent sessions
        self.global_performance_trends = defaultdict(lambda: deque(maxlen=1000))
        
        # Auto-save settings
        self.auto_save_interval = 300  # 5 minutes
        self.last_save_time = time.time()
        self.auto_save_thread = None
        self.shutdown_flag = threading.Event()
        
        # Load existing state
        self._load_persistent_state()
        self._start_auto_save()
        
        self.logger.info(f"Cross-session learning manager initialized with "
                        f"{len(self.learned_patterns)} existing patterns")
    
    def start_session(self, session_context: Dict[str, Any] = None) -> str:
        """Start a new learning session."""
        session_id = f"session_{int(time.time())}_{hash(str(session_context)) % 10000}"
        
        self.current_session = SessionMetaLearningState(
            session_id=session_id,
            start_time=time.time(),
            training_context=session_context or {}
        )
        
        self.logger.info(f"Started meta-learning session: {session_id}")
        return session_id
    
    def end_session(self, performance_summary: Dict[str, Any] = None):
        """End the current learning session and persist state."""
        if not self.current_session:
            return
        
        self.current_session.end_time = time.time()
        
        if performance_summary:
            self.current_session.total_decisions = performance_summary.get('total_decisions', 0)
            self.current_session.successful_decisions = performance_summary.get('successful_decisions', 0)
            self.current_session.average_performance_improvement = performance_summary.get('avg_improvement', 0.0)
        
        # Generate session insights
        self._generate_session_insights()
        
        # Add to session history
        self.session_history.append(self.current_session)
        
        # Persist state
        self._save_persistent_state()
        
        session_duration = self.current_session.end_time - self.current_session.start_time
        self.logger.info(f"Ended session {self.current_session.session_id} "
                        f"({session_duration:.1f}s, {len(self.current_session.session_patterns)} new patterns)")
        
        self.current_session = None
    
    def learn_pattern(self, knowledge_type: KnowledgeType, pattern_data: Dict[str, Any],
                     context: Dict[str, Any], success_rate: float = 0.0,
                     persistence_level: PersistenceLevel = PersistenceLevel.SESSION) -> str:
        """Learn a new pattern from experience."""
        
        # Generate unique pattern ID
        pattern_content = json.dumps(pattern_data, sort_keys=True)
        context_hash = self._hash_context(context)
        pattern_id = f"{knowledge_type.value}_{hashlib.md5(pattern_content.encode()).hexdigest()[:8]}"
        
        # Check if pattern already exists
        if pattern_id in self.learned_patterns:
            existing_pattern = self.learned_patterns[pattern_id]
            existing_pattern.update_success_metrics(success_rate > 0.5)
            existing_pattern.last_updated = time.time()
            return pattern_id
        
        # Create new pattern
        pattern = LearnedPattern(
            pattern_id=pattern_id,
            knowledge_type=knowledge_type,
            pattern_data=pattern_data,
            puzzle_types=[context.get('puzzle_type', 'unknown')],
            performance_conditions=self._extract_performance_conditions(context),
            environmental_factors=self._extract_environmental_factors(context),
            success_rate=success_rate,
            confidence=min(0.8, max(0.3, success_rate * 0.8 + 0.2)),  # Initialize with reasonable confidence
            sample_size=1,
            total_applications=1,
            successful_applications=1 if success_rate > 0.5 else 0,
            persistence_level=persistence_level
        )
        
        # Store pattern
        self.learned_patterns[pattern_id] = pattern
        self.patterns_by_type[knowledge_type].append(pattern_id)
        self.patterns_by_context[context_hash].append(pattern_id)
        
        # Add to current session
        if self.current_session:
            self.current_session.session_patterns.append(pattern)
        
        # CRITICAL FIX: Save patterns immediately for real-time learning
        try:
            self._save_persistent_state()
            self.logger.debug(f"Pattern {pattern_id} saved immediately to disk")
        except Exception as e:
            self.logger.warning(f"Failed to save pattern {pattern_id} immediately: {e}")
        
        self.logger.debug(f"Learned new pattern: {pattern_id} ({knowledge_type.value})")
        return pattern_id
    
    def retrieve_applicable_patterns(self, knowledge_type: KnowledgeType,
                                   context: Dict[str, Any],
                                   min_confidence: float = 0.3,
                                   max_results: int = 10) -> List[LearnedPattern]:
        """Retrieve patterns applicable to the current context."""
        
        # Get candidate patterns of the requested type
        candidate_ids = self.patterns_by_type.get(knowledge_type, [])
        
        applicable_patterns = []
        for pattern_id in candidate_ids:
            pattern = self.learned_patterns[pattern_id]
            
            # Check if pattern is applicable and meets confidence threshold
            if (pattern.confidence >= min_confidence and 
                pattern.is_applicable(context)):
                
                # Update importance score
                pattern.calculate_importance()
                applicable_patterns.append(pattern)
        
        # Sort by importance and return top results
        applicable_patterns.sort(key=lambda p: p.importance_score, reverse=True)
        return applicable_patterns[:max_results]
    
    def get_best_configuration_for_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get the best known configuration for a given context."""
        
        config_patterns = self.retrieve_applicable_patterns(
            KnowledgeType.CONFIGURATION_PROFILE, 
            context,
            min_confidence=0.4
        )
        
        if not config_patterns:
            return {}
        
        # Return configuration from the most successful pattern
        best_pattern = config_patterns[0]
        return best_pattern.pattern_data.get('configuration', {})
    
    def record_strategy_success(self, strategy_name: str, context: Dict[str, Any],
                              success_metrics: Dict[str, float]):
        """Record the success of a strategy for future reference."""
        
        strategy_data = {
            'strategy_name': strategy_name,
            'success_metrics': success_metrics,
            'context_snapshot': context
        }
        
        # Calculate overall success score
        success_score = success_metrics.get('win_rate_improvement', 0) * 0.4 + \
                       success_metrics.get('score_improvement', 0) / 20.0 * 0.3 + \
                       success_metrics.get('efficiency_improvement', 0) * 0.3
        
        self.learn_pattern(
            KnowledgeType.STRATEGY_PATTERN,
            strategy_data,
            context,
            success_score,
            PersistenceLevel.PERMANENT if success_score > 0.7 else PersistenceLevel.SESSION
        )
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Generate insights from accumulated learning data."""
        insights = {
            'total_patterns': len(self.learned_patterns),
            'patterns_by_type': {},
            'most_successful_patterns': [],
            'learning_trends': {},
            'session_summary': {}
        }
        
        # Patterns by type
        for knowledge_type, pattern_ids in self.patterns_by_type.items():
            insights['patterns_by_type'][knowledge_type.value] = len(pattern_ids)
        
        # Most successful patterns
        all_patterns = list(self.learned_patterns.values())
        all_patterns.sort(key=lambda p: p.importance_score, reverse=True)
        
        for pattern in all_patterns[:5]:
            insights['most_successful_patterns'].append({
                'type': pattern.knowledge_type.value,
                'success_rate': pattern.success_rate,
                'confidence': pattern.confidence,
                'applications': pattern.total_applications,
                'importance': pattern.importance_score
            })
        
        # Learning trends
        if self.session_history:
            recent_sessions = list(self.session_history)[-10:]
            insights['learning_trends'] = {
                'average_patterns_per_session': statistics.mean([len(s.session_patterns) for s in recent_sessions]),
                'success_rate_trend': [s.successful_decisions / max(s.total_decisions, 1) for s in recent_sessions],
                'performance_improvement_trend': [s.average_performance_improvement for s in recent_sessions]
            }
        
        # Current session summary
        if self.current_session:
            insights['session_summary'] = {
                'session_id': self.current_session.session_id,
                'duration': time.time() - self.current_session.start_time,
                'patterns_learned': len(self.current_session.session_patterns),
                'decisions_made': self.current_session.total_decisions
            }
        
        return insights
    
    def _extract_performance_conditions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance conditions that define when a pattern is applicable."""
        current_perf = context.get('current_performance', {})
        conditions = {}
        
        # Define threshold conditions for key performance metrics
        if 'win_rate' in current_perf:
            win_rate = current_perf['win_rate']
            if win_rate < 0.3:
                conditions['win_rate'] = {'max': 0.4}  # Low win rate condition
            elif win_rate > 0.7:
                conditions['win_rate'] = {'min': 0.6}  # High win rate condition
        
        if 'learning_efficiency' in current_perf:
            eff = current_perf['learning_efficiency']
            if eff < 0.5:
                conditions['learning_efficiency'] = {'max': 0.6}
        
        return conditions
    
    def _extract_environmental_factors(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract environmental factors from context."""
        return {
            'puzzle_type': context.get('puzzle_type', 'unknown'),
            'system_load': context.get('system_load', 'normal'),
            'training_phase': context.get('training_phase', 'standard')
        }
    
    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Generate a hash for context-based pattern indexing."""
        context_str = json.dumps({
            'puzzle_type': context.get('puzzle_type', ''),
            'performance_level': self._categorize_performance(context.get('current_performance', {}))
        }, sort_keys=True)
        return hashlib.md5(context_str.encode()).hexdigest()[:8]
    
    def _categorize_performance(self, performance: Dict[str, Any]) -> str:
        """Categorize performance level for pattern matching."""
        win_rate = performance.get('win_rate', 0)
        if win_rate < 0.3:
            return 'low'
        elif win_rate < 0.7:
            return 'medium'
        else:
            return 'high'
    
    def _generate_session_insights(self):
        """Generate insights from the current session."""
        if not self.current_session:
            return
        
        insights = []
        
        # Analyze patterns learned in this session
        if self.current_session.session_patterns:
            success_rates = [p.success_rate for p in self.current_session.session_patterns]
            avg_success = statistics.mean(success_rates)
            
            if avg_success > 0.7:
                insights.append(f"Highly successful session: {avg_success:.1%} average success rate")
            elif avg_success < 0.3:
                insights.append(f"Challenging session: {avg_success:.1%} average success rate")
        
        # Analyze decision effectiveness
        if self.current_session.total_decisions > 0:
            decision_success = self.current_session.successful_decisions / self.current_session.total_decisions
            if decision_success > 0.8:
                insights.append("Excellent decision-making performance")
            elif decision_success < 0.4:
                insights.append("Need to improve decision-making strategies")
        
        self.current_session.key_insights = insights
    
    def _load_persistent_state(self):
        """Load persistent state from disk."""
        patterns_file = self.persistence_dir / "learned_patterns.pkl"
        # Ensure the directory exists
        patterns_file.parent.mkdir(parents=True, exist_ok=True)
        sessions_file = self.persistence_dir / "session_history.jsonl"
        
        # Load learned patterns
        if patterns_file.exists():
            try:
                with open(patterns_file, 'rb') as f:
                    saved_patterns = pickle.load(f)
                    
                for pattern_id, pattern_data in saved_patterns.items():
                    # Convert string enums back to enum objects
                    if isinstance(pattern_data.get('knowledge_type'), str):
                        pattern_data['knowledge_type'] = KnowledgeType(pattern_data['knowledge_type'])
                    if isinstance(pattern_data.get('persistence_level'), str):
                        pattern_data['persistence_level'] = PersistenceLevel(pattern_data['persistence_level'])
                    
                    pattern = LearnedPattern(**pattern_data)
                    self.learned_patterns[pattern_id] = pattern
                    self.patterns_by_type[pattern.knowledge_type].append(pattern_id)
                    
                self.logger.info(f"Loaded {len(saved_patterns)} learned patterns from disk")
            except Exception as e:
                self.logger.error(f"Failed to load learned patterns: {e}")
        
        # Load session history
        if sessions_file.exists():
            try:
                with open(sessions_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            session_data = json.loads(line.strip())
                            # Convert session data back to SessionMetaLearningState
                            # (simplified for this example)
                            pass
            except Exception as e:
                self.logger.error(f"Failed to load session history: {e}")
    
    def _save_persistent_state(self):
        """Save persistent state to disk."""
        # Save learned patterns
        patterns_file = self.persistence_dir / "learned_patterns.pkl"
        # Ensure the directory exists
        patterns_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Debug: Log the actual path being used
        self.logger.debug(f"Attempting to save patterns to: {patterns_file}")
        self.logger.debug(f"Persistence dir type: {type(self.persistence_dir)}, value: {self.persistence_dir}")
        self.logger.debug(f"Patterns file type: {type(patterns_file)}, value: {patterns_file}")
        self.logger.debug(f"Patterns file parent: {patterns_file.parent}")
        self.logger.debug(f"Parent exists: {patterns_file.parent.exists()}")
        self.logger.debug(f"Parent is dir: {patterns_file.parent.is_dir()}")
        
        try:
            # Save ALL patterns, not just permanent ones (for testing and gradual learning)
            # Create a copy to avoid "dictionary changed size during iteration" error
            patterns_copy = dict(self.learned_patterns)
            persistent_patterns = {}
            
            # Safely convert patterns to dictionaries, handling any serialization issues
            for pattern_id, pattern in patterns_copy.items():
                try:
                    pattern_dict = asdict(pattern)
                    # Remove any problematic fields that might not be serializable
                    if 'metadata' in pattern_dict and isinstance(pattern_dict['metadata'], dict):
                        # Ensure metadata is serializable
                        cleaned_metadata = {}
                        for k, v in pattern_dict['metadata'].items():
                            try:
                                # Test if the value is picklable
                                pickle.dumps(v)
                                cleaned_metadata[k] = v
                            except (TypeError, pickle.PicklingError):
                                # Convert non-picklable values to strings
                                cleaned_metadata[k] = str(v)
                        pattern_dict['metadata'] = cleaned_metadata
                    persistent_patterns[pattern_id] = pattern_dict
                except Exception as pattern_error:
                    self.logger.warning(f"Failed to serialize pattern {pattern_id}: {pattern_error}")
                    continue
            
            # Use absolute path to avoid Windows path issues
            absolute_path = patterns_file.absolute()
            with open(absolute_path, 'wb') as f:
                pickle.dump(persistent_patterns, f)
                
            self.logger.debug(f"Saved {len(persistent_patterns)} patterns to disk at {absolute_path}")
        except Exception as e:
            self.logger.error(f"Failed to save learned patterns: {e}")
            self.logger.error(f"Patterns file path: {patterns_file}")
            self.logger.error(f"Patterns file path (str): {str(patterns_file)}")
            self.logger.error(f"Persistence dir: {self.persistence_dir}")
            self.logger.error(f"Persistence dir (str): {str(self.persistence_dir)}")
            self.logger.error(f"File exists: {patterns_file.exists()}")
            self.logger.error(f"Parent dir exists: {patterns_file.parent.exists()}")
            self.logger.error(f"Parent dir is dir: {patterns_file.parent.is_dir()}")
            self.logger.error(f"Error type: {type(e)}")
            self.logger.error(f"Error args: {e.args}")
            
            # Fallback: Try saving to a temporary file first, then move it
            try:
                import tempfile
                import shutil
                
                # Create a temporary file in the same directory
                temp_file = patterns_file.parent / f"learned_patterns_temp_{os.getpid()}.pkl"
                
                # Try to save to temp file first
                with open(temp_file, 'wb') as f:
                    pickle.dump(persistent_patterns, f)
                
                # If successful, move temp file to final location
                shutil.move(str(temp_file), str(patterns_file))
                self.logger.info(f"Successfully saved patterns using fallback method")
                
            except Exception as fallback_error:
                self.logger.error(f"Fallback save also failed: {fallback_error}")
                # Clean up temp file if it exists
                try:
                    if 'temp_file' in locals() and temp_file.exists():
                        temp_file.unlink()
                except:
                    pass
            
            # Try to create the directory again with more explicit error handling
            try:
                patterns_file.parent.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Successfully created directory: {patterns_file.parent}")
            except Exception as mkdir_error:
                self.logger.error(f"Failed to create directory: {mkdir_error}")
                self.logger.error(f"Directory path: {patterns_file.parent}")
                self.logger.error(f"Directory path (str): {str(patterns_file.parent)}")
        
        # Save current session to history
        if self.current_session:
            sessions_file = self.persistence_dir / "session_history.jsonl"
            try:
                with open(sessions_file, 'a') as f:
                    f.write(json.dumps(asdict(self.current_session), default=str) + '\n')
            except Exception as e:
                self.logger.error(f"Failed to save session history: {e}")
        
        self.last_save_time = time.time()
    
    def _start_auto_save(self):
        """Start auto-save thread."""
        def auto_save_worker():
            while not self.shutdown_flag.wait(self.auto_save_interval):
                try:
                    self._save_persistent_state()
                except Exception as e:
                    self.logger.error(f"Auto-save failed: {e}")
        
        self.auto_save_thread = threading.Thread(target=auto_save_worker, daemon=True)
        self.auto_save_thread.start()
    
    def shutdown(self):
        """Gracefully shutdown the persistence manager."""
        self.shutdown_flag.set()
        
        if self.current_session:
            self.end_session()
        
        self._save_persistent_state()
        
        if self.auto_save_thread:
            self.auto_save_thread.join(timeout=5.0)
        
        self.logger.info("Cross-session learning manager shut down")
