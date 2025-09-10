import os
from git import Repo
import glob
import json
#!/usr/bin/env python3
"""
MetaCognitiveGovernor - The "Third Brain"

A runtime supervisor that dynamically manages the AI's "cognitive economy."
Makes high-level decisions to switch training modes, allocate "attention" 
(compute to algorithms), and trigger consolidation based on real-time 
efficacy analysis of its own software processes.

This module is completely hardware-agnostic and reasons in terms of 
abstract computational cycles and algorithmic efficiency.
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
from collections import defaultdict, deque

# Import existing system components for integration
try:
    from src.core.salience_system import SalienceMode
except ImportError:
    # Fallback for direct execution
    class SalienceMode(Enum):
        LOSSLESS = "lossless"
        DECAY_COMPRESSION = "decay_compression"

class GovernorRecommendationType(Enum):
    """Types of recommendations the Governor can make."""
    MODE_SWITCH = "mode_switch"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    CONSOLIDATION_TRIGGER = "consolidation_trigger"
    RESOURCE_REALLOCATION = "resource_reallocation"
    ARCHITECT_REQUEST = "architect_request"

@dataclass
class CognitiveCost:
    """Abstract cost model for cognitive operations."""
    compute_units: float  # Abstract compute cost
    memory_operations: int  # Memory read/write operations
    decision_complexity: float  # Complexity of decisions made
    coordination_overhead: float  # Inter-system coordination cost
    
    def total_cost(self) -> float:
        """Calculate total abstract cost."""
        return (self.compute_units + 
                self.memory_operations * 0.1 + 
                self.decision_complexity + 
                self.coordination_overhead)

@dataclass
class CognitiveBenefit:
    """Benefit measurement for cognitive operations."""
    win_rate_improvement: float  # Improvement in win rate
    score_improvement: float  # Improvement in average score
    learning_efficiency: float  # Learning speed improvement
    knowledge_transfer: float  # Cross-domain learning benefit
    
    def total_benefit(self) -> float:
        """Calculate total benefit score."""
        return (self.win_rate_improvement * 100 + 
                self.score_improvement + 
                self.learning_efficiency * 10 + 
                self.knowledge_transfer * 5)

@dataclass
class GovernorRecommendation:
    """Recommendation from the Governor."""
    type: GovernorRecommendationType
    configuration_changes: Dict[str, Any]
    confidence: float  # 0.0 to 1.0
    expected_benefit: CognitiveBenefit
    rationale: str
    urgency: float  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/transmission."""
        return {
            'type': self.type.value,
            'configuration_changes': self.configuration_changes,
            'confidence': self.confidence,
            'expected_benefit': asdict(self.expected_benefit),
            'rationale': self.rationale,
            'urgency': self.urgency,
            'timestamp': time.time()
        }

@dataclass
class ArchitectRequest:
    """Request to the Architect for architectural changes."""
    issue_type: str
    persistent_problem: str
    failed_solutions: List[Dict[str, Any]]
    performance_data: Dict[str, Any]
    suggested_research_directions: List[str]
    priority: float  # 0.0 to 1.0
    frame_data: Optional[Dict[str, Any]] = None  # Enhanced with frame analysis
    memory_context: Optional[Dict[str, Any]] = None  # Enhanced with memory context
    object_analysis: Optional[Dict[str, Any]] = None  # Enhanced with object analysis
    learning_progress: Optional[float] = None  # Current learning progress
    energy_state: Optional[Dict[str, Any]] = None  # Current energy state

class CognitiveSystemMonitor:
    """Monitors individual cognitive systems for cost-benefit analysis."""
    
    def __init__(self, system_name: str):
        self.system_name = system_name
        self.cost_history = deque(maxlen=100)
        self.benefit_history = deque(maxlen=100)
        self.activation_count = 0
        self.total_runtime = 0.0
        self.last_activation = None
        self.performance_impact = {}
        
    def record_activation(self, cost: CognitiveCost, benefit: CognitiveBenefit):
        """Record an activation of this cognitive system."""
        self.cost_history.append(cost)
        self.benefit_history.append(benefit)
        self.activation_count += 1
        self.last_activation = time.time()
        
    def get_efficiency_ratio(self) -> float:
        """Get benefit/cost ratio for this system."""
        if not self.cost_history or not self.benefit_history:
            return 1.0  # Default neutral efficiency
            
        avg_cost = sum(c.total_cost() for c in self.cost_history) / len(self.cost_history)
        avg_benefit = sum(b.total_benefit() for b in self.benefit_history) / len(self.benefit_history)
        
        if avg_cost == 0:
            return float('inf') if avg_benefit > 0 else 1.0
        return avg_benefit / avg_cost
    
    def get_recent_trend(self, window_size: int = 10) -> str:
        """Get trend of recent efficiency changes."""
        if len(self.benefit_history) < window_size:
            return "insufficient_data"
            
        recent_benefits = list(self.benefit_history)[-window_size:]
        earlier_benefits = list(self.benefit_history)[-window_size*2:-window_size]
        
        if not earlier_benefits:
            return "insufficient_data"
            
        recent_avg = sum(b.total_benefit() for b in recent_benefits) / len(recent_benefits)
        earlier_avg = sum(b.total_benefit() for b in earlier_benefits) / len(earlier_benefits)
        
        if recent_avg > earlier_avg * 1.1:
            return "improving"
        elif recent_avg < earlier_avg * 0.9:
            return "declining"
        else:
            return "stable"

class MetaCognitiveGovernor:

    def archive_and_cleanup_logs(self, log_dir, results_dir, keep_per_game=1, keep_recent=10):
        """
        Archive and clean up training logs and results using GitPython.
        Prioritize by effectiveness, recency, and game coverage.
        """
        repo = Repo(os.path.abspath(os.path.join(log_dir, '..')))
        logs = self._parse_logs(log_dir)
        results = self._parse_results(results_dir)
        to_keep = set()
        # 1. Keep best per game (logs)
        for game_id, group in self._group_by(logs, 'game_id').items():
            best = max(group, key=lambda x: x['score'])
            to_keep.add(best['filename'])
        # 2. Keep most recent overall (logs)
        recent = sorted(logs, key=lambda x: x['timestamp'], reverse=True)[:keep_recent]
        to_keep.update(log['filename'] for log in recent)
        # 3. Repeat for results
        for game_id, group in self._group_by(results, 'game_id').items():
            best = max(group, key=lambda x: x['score'])
            to_keep.add(best['filename'])
        recent_results = sorted(results, key=lambda x: x['timestamp'], reverse=True)[:keep_recent]
        to_keep.update(r['filename'] for r in recent_results)
        # 4. Archive and delete the rest
        for file in logs + results:
            if file['filename'] not in to_keep:
                abs_path = os.path.join(log_dir if file in logs else results_dir, file['filename'])
                repo.index.add([abs_path])
                repo.index.commit(f"Archive log: {file['filename']} (score={file['score']}, game={file['game_id']})")
                os.remove(abs_path)

    def _parse_logs(self, log_dir):
        files = glob.glob(os.path.join(log_dir, '*.log'))
        parsed = []
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
                    content = fh.read()
                # Extract game_id, score, timestamp (customize as needed)
                game_id = self._extract_game_id(content)
                score = self._extract_score(content)
                timestamp = os.path.getmtime(f)
                parsed.append({'filename': os.path.basename(f), 'game_id': game_id, 'score': score, 'timestamp': timestamp})
            except Exception:
                continue
        return parsed

    def _parse_results(self, results_dir):
        files = glob.glob(os.path.join(results_dir, '*.json'))
        parsed = []
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
                    data = json.load(fh)
                game_id = data.get('game_id', 'unknown')
                score = data.get('score', 0)
                timestamp = os.path.getmtime(f)
                parsed.append({'filename': os.path.basename(f), 'game_id': game_id, 'score': score, 'timestamp': timestamp})
            except Exception:
                continue
        return parsed

    def _group_by(self, items, key):
        grouped = {}
        for item in items:
            grouped.setdefault(item[key], []).append(item)
        return grouped

    def _extract_game_id(self, content):
        # Implement logic to extract game_id from log content
        # Placeholder: look for 'Game: <id>'
        import re
        m = re.search(r'Game[s]?: ([\w\-, ]+)', content)
        if m:
            return m.group(1).split(',')[0].strip()
        return 'unknown'

    def _extract_score(self, content):
        # Implement logic to extract score from log content
        # Placeholder: look for 'Score: <number>'
        import re
        m = re.search(r'Score: (\d+)', content)
        if m:
            return int(m.group(1))
        return 0
class MetaCognitiveGovernor:
    """
    The "Third Brain" - Meta-Cognitive Resource Allocator
    
    Acts as an internal superintendent of cognitive processes, making dynamic
    high-level decisions about resource allocation between software components.
    """
    
    def __init__(self, memory_capacity: int = 1000, decision_threshold: float = 0.7, adaptation_rate: float = 0.1,
                 log_file: Optional[str] = None, outcome_tracking_dir: Optional[str] = None,
                 persistence_dir: Optional[str] = None):
        self.logger = logging.getLogger(f"{__name__}.Governor")
        self.log_file = log_file
        
        # Outcome tracking integration
        self.outcome_tracker = None
        if outcome_tracking_dir:
            try:
                from src.core.outcome_tracker import OutcomeTracker, PerformanceMetrics
                self.outcome_tracker = OutcomeTracker(Path(outcome_tracking_dir), self.logger)
                self.logger.info("Outcome tracking enabled")
            except ImportError:
                self.logger.warning("Outcome tracker not available")
        
        # Cross-session learning integration
        self.learning_manager = None
        if persistence_dir:
            try:
                from src.core.cross_session_learning import CrossSessionLearningManager, KnowledgeType
                self.learning_manager = CrossSessionLearningManager(Path(persistence_dir), self.logger)
                self.logger.info("Cross-session learning enabled")
            except ImportError:
                self.logger.warning("Cross-session learning not available")
        
        # Meta-cognitive memory management integration
        self.memory_manager = None
        try:
            from src.core.meta_cognitive_memory_manager import MetaCognitiveMemoryManager
            base_path = Path(persistence_dir) if persistence_dir else Path(".")
            self.memory_manager = MetaCognitiveMemoryManager(base_path, self.logger)
            self.logger.info("Meta-cognitive memory management enabled")
        except ImportError:
            self.logger.warning("Meta-cognitive memory manager not available")
        
        # Memory pattern optimization (Phase 1 enhancement)
        self.pattern_optimizer = None
        try:
            from src.core.memory_pattern_optimizer import MemoryPatternOptimizer
            self.pattern_optimizer = MemoryPatternOptimizer()
            self.logger.info("Memory pattern optimization enabled - Phase 1 immediate wins")
        except ImportError:
            self.logger.warning("Memory pattern optimizer not available")
        
        # Log file management system
        self.log_cleanup_threshold = 10000  # Lines for regular logs
        self.log_cleanup_remove_lines = 5000  # Lines to remove when threshold exceeded
        self.log_cleanup_patterns = [
            "master_arc_trainer*.log",
            "governor_decisions*.log",
            "arc_trainer*.log"
        ]
        
        # Archive cleanup configuration
        self.archive_cleanup_config = {
            'max_archive_days': 30,  # Keep archives for 30 days
            'keep_recent': 5,        # Always keep 5 most recent archives
            'cleanup_frequency': 10, # Clean every 10 governor decisions
            'max_cleanup_interval': 3600  # Max 1 hour between cleanups
        }
        
        # Training data cleanup configuration
        self.training_data_cleanup_config = {
            'min_score_threshold': 0.1,    # Minimum score to consider valuable
            'min_episodes_threshold': 5,   # Minimum episodes to consider meaningful
            'max_age_days': 70,             # Maximum age of data to keep
            'cleanup_frequency': 7,        # Clean every 5 governor decisions
            'max_cleanup_interval': 1800   # Max 30 minutes between cleanups
        }
        
        # Special handling for master_arc_trainer logs - keep under 100k
        self.master_trainer_threshold = 100000  # 100k lines for master trainer logs
        self.master_trainer_remove_lines = 50000  # Remove 50k lines when threshold exceeded
        
        # Hierarchical memory clustering (Phase 2 enhancement)
        self.memory_clusterer = None
        try:
            from src.core.hierarchical_memory_clusterer import HierarchicalMemoryClusterer
            self.memory_clusterer = HierarchicalMemoryClusterer()
            self.logger.info("Hierarchical memory clustering enabled - Phase 2 intelligent clusters")
        except ImportError:
            self.logger.warning("Hierarchical memory clusterer not available")
        
        # Architect Evolution Engine (Phase 3 enhancement)
        self.architect_engine = None
        try:
            from src.core.architect_evolution_engine import ArchitectEvolutionEngine
            self.architect_engine = ArchitectEvolutionEngine(
                persistence_dir=persistence_dir or ".",
                evolution_data_dir="architecture/evolution",  # Relative to persistence_dir
                enable_autonomous_evolution=True
            )
            self.logger.info("Architect Evolution Engine enabled - Phase 3 autonomous evolution")
        except ImportError:
            self.logger.warning("Architect Evolution Engine not available")
        
        # Performance Optimization Engine (Phase 4 enhancement)
        self.performance_engine = None
        try:
            from src.core.performance_optimization_engine import PerformanceOptimizationEngine
            self.performance_engine = PerformanceOptimizationEngine(
                persistence_dir=persistence_dir or ".",
                performance_data_dir="experiments/performance",  # Relative to persistence_dir
                enable_real_time_optimization=True
            )
            self.logger.info("Performance Optimization Engine enabled - Phase 4 performance maximization")
        except ImportError:
            self.logger.warning("Performance Optimization Engine not available")
        
        # Cognitive system monitors
        self.system_monitors = {}
        self.initialize_system_monitors()
        
        # Decision history
        self.decision_history = deque(maxlen=1000)
        self.performance_baseline = {}
        self.current_config = None
        
        # Governor state
        self.total_decisions_made = 0
        self.successful_recommendations = 0
        self.start_time = time.time()
        
        # Tracking for outcome measurement
        self.pending_outcome_measurements = {}  # decision_id -> outcome_id
        
        # Architect communication
        self.pending_architect_requests = []
        self.architect_response_history = []
        
        self.logger.info("🧠 MetaCognitiveGovernor initialized - Third Brain online")
    
    def make_decision(self, available_actions: List[int], context: Dict[str, Any], 
                     performance_history: List[Dict[str, Any]], current_energy: float) -> Dict[str, Any]:
        """
        Make a high-level decision about action selection based on meta-cognitive analysis.
        Enhanced to utilize frame data, object analysis, and memory context.
        
        Args:
            available_actions: List of available actions
            context: Current game context including frame analysis, object data, memory context
            performance_history: Historical performance data
            current_energy: Current energy level
            
        Returns:
            Dictionary containing decision, reasoning, and confidence
        """
        try:
            # Analyze current situation with enhanced data utilization
            game_id = context.get('game_id', 'unknown')
            frame_analysis = context.get('frame_analysis', {})
            object_analysis = context.get('object_analysis', {})
            memory_context = context.get('memory_context', {})
            
            # Enhanced analysis using all available data
            visual_insights = self._analyze_visual_context(frame_analysis)
            object_insights = self._analyze_object_context(object_analysis)
            memory_insights = self._analyze_memory_context(memory_context)
            energy_insights = self._analyze_energy_context(current_energy)
            
            # Calculate confidence based on multiple factors with enhanced data
            confidence = self._calculate_enhanced_decision_confidence(
                available_actions, context, performance_history, current_energy,
                visual_insights, object_insights, memory_insights, energy_insights
            )
            
            # Make action recommendation based on enhanced meta-cognitive analysis
            recommended_action = self._select_enhanced_meta_cognitive_action(
                available_actions, context, performance_history, current_energy
            )
            
            # Generate reasoning
            reasoning = self._generate_decision_reasoning(
                recommended_action, available_actions, context, confidence
            )
            
            # Track decision
            decision_record = {
                'timestamp': time.time(),
                'game_id': game_id,
                'available_actions': available_actions,
                'recommended_action': recommended_action,
                'confidence': confidence,
                'reasoning': reasoning,
                'energy_level': current_energy
            }
            
            self.decision_history.append(decision_record)
            self.total_decisions_made += 1
            
            return {
                'recommended_action': recommended_action,
                'confidence': confidence,
                'reasoning': reasoning,
                'meta_analysis': {
                    'energy_factor': current_energy / 100.0,
                    'performance_trend': self._analyze_performance_trend(performance_history),
                    'decision_count': self.total_decisions_made
                }
            }
            
        except Exception as e:
            self.logger.error(f"Governor decision failed: {e}")
            # Fallback to first available action with low confidence
            return {
                'recommended_action': available_actions[0] if available_actions else 1,
                'confidence': 0.1,
                'reasoning': f"Emergency fallback due to error: {e}",
                'meta_analysis': {'error': str(e)}
            }
    
    def _calculate_decision_confidence(self, available_actions: List[int], context: Dict[str, Any],
                                     performance_history: List[Dict[str, Any]], current_energy: float) -> float:
        """Calculate confidence in decision making based on various factors."""
        base_confidence = 0.5
        
        # Energy factor (higher energy = higher confidence)
        energy_factor = current_energy / 100.0
        
        # Performance history factor
        if performance_history:
            recent_performance = performance_history[-10:]  # Last 10 sessions
            success_rate = sum(1 for p in recent_performance if p.get('success', False)) / len(recent_performance)
            performance_factor = success_rate
        else:
            performance_factor = 0.5
        
        # Action diversity factor (more actions = more confidence)
        diversity_factor = min(1.0, len(available_actions) / 7.0)
        
        # Frame analysis factor
        frame_analysis = context.get('frame_analysis', {})
        analysis_factor = 0.8 if frame_analysis else 0.5
        
        # Combine factors
        confidence = (base_confidence + energy_factor + performance_factor + diversity_factor + analysis_factor) / 5.0
        
        return max(0.1, min(1.0, confidence))
    
    def _select_meta_cognitive_action(self, available_actions: List[int], context: Dict[str, Any],
                                    performance_history: List[Dict[str, Any]], current_energy: float) -> int:
        """Select action based on meta-cognitive analysis."""
        if not available_actions:
            return 1
        
        # Analyze recent performance to guide action selection
        if performance_history:
            # Find most successful actions from recent history
            recent_sessions = performance_history[-5:]
            action_success = {}
            
            for session in recent_sessions:
                actions = session.get('actions_taken', [])
                success = session.get('success', False)
                score = session.get('score', 0)
                
                for action in actions:
                    if action not in action_success:
                        action_success[action] = {'count': 0, 'success': 0, 'total_score': 0}
                    action_success[action]['count'] += 1
                    if success:
                        action_success[action]['success'] += 1
                    action_success[action]['total_score'] += score
            
            # Find best performing available action
            best_action = None
            best_score = -1
            
            for action in available_actions:
                if action in action_success:
                    stats = action_success[action]
                    if stats['count'] > 0:
                        success_rate = stats['success'] / stats['count']
                        avg_score = stats['total_score'] / stats['count']
                        combined_score = success_rate * 0.7 + (avg_score / 100.0) * 0.3
                        
                        if combined_score > best_score:
                            best_score = combined_score
                            best_action = action
            
            if best_action is not None:
                return best_action
        
        # Fallback: prefer ACTION6 if available (visual interaction)
        if 6 in available_actions:
            return 6
        
        # Otherwise return first available action, but ensure we have actions
        if available_actions:
            return available_actions[0]
        else:
            # This should never happen, but provide a safe fallback
            return 1
    
    def _generate_decision_reasoning(self, recommended_action: int, available_actions: List[int], 
                                   context: Dict[str, Any], confidence: float) -> str:
        """Generate human-readable reasoning for the decision."""
        game_id = context.get('game_id', 'unknown')
        frame_analysis = context.get('frame_analysis', {})
        
        reasoning_parts = [f"Meta-cognitive analysis for {game_id}"]
        
        if recommended_action == 6:
            reasoning_parts.append("Selected ACTION6 for visual-interactive exploration")
            if frame_analysis:
                reasoning_parts.append("Frame analysis available for targeting")
        else:
            reasoning_parts.append(f"Selected ACTION{recommended_action} based on performance history")
        
        reasoning_parts.append(f"Confidence: {confidence:.2f}")
        reasoning_parts.append(f"Available options: {available_actions}")
        
        return " | ".join(reasoning_parts)
    
    def _analyze_performance_trend(self, performance_history: List[Dict[str, Any]]) -> str:
        """Analyze performance trend from history."""
        if len(performance_history) < 3:
            return "insufficient_data"
        
        recent_scores = [p.get('score', 0) for p in performance_history[-5:]]
        older_scores = [p.get('score', 0) for p in performance_history[-10:-5]] if len(performance_history) >= 10 else []
        
        if not older_scores:
            return "improving" if recent_scores[-1] > recent_scores[0] else "stable"
        
        recent_avg = sum(recent_scores) / len(recent_scores)
        older_avg = sum(older_scores) / len(older_scores)
        
        if recent_avg > older_avg * 1.1:
            return "improving"
        elif recent_avg < older_avg * 0.9:
            return "declining"
        else:
            return "stable"
    
    def initialize_system_monitors(self):
        """Initialize monitors for all cognitive systems."""
        # Core cognitive systems from the existing architecture
        cognitive_systems = [
            "swarm_intelligence",
            "dnc_memory", 
            "meta_learning_system",
            "energy_management",
            "sleep_cycles",
            "coordinate_intelligence",
            "frame_analysis",
            "boundary_detection",
            "memory_consolidation",
            "action_intelligence",
            "goal_invention",
            "learning_progress_drive",
            "death_manager",
            "exploration_strategies",
            "pattern_recognition",
            "knowledge_transfer",
            "boredom_detection",
            "mid_game_sleep",
            "action_experimentation",
            "reset_decisions",
            "curriculum_learning",
            "multi_modal_input",
            "temporal_memory",
            "hebbian_bonuses",
            "memory_regularization",
            "gradient_flow_monitoring",
            "usage_tracking",
            "salient_memory_retrieval",
            "anti_bias_weighting",
            "stagnation_detection",
            "emergency_movement",
            "cluster_formation",
            "danger_zone_avoidance",
            "predictive_coordinates",
            "rate_limiting_management",
            "contrarian_strategy",
            "salience_system"
        ]
        
        for system in cognitive_systems:
            self.system_monitors[system] = CognitiveSystemMonitor(system)
            
        self.logger.info(f"🔍 Initialized {len(cognitive_systems)} cognitive system monitors")
    
    def get_recommended_configuration(self, 
                                    puzzle_type: str,
                                    current_performance: Dict[str, Any],
                                    current_config: Dict[str, Any]) -> Optional[GovernorRecommendation]:
        """
        Main API: Get recommended configuration changes based on current state.
        
        Args:
            puzzle_type: Type of puzzle being solved
            current_performance: Current performance metrics
            current_config: Current system configuration
            
        Returns:
            GovernorRecommendation or None if no changes recommended
        """
        self.current_config = current_config
        start_time = time.time()
        
        try:
            # Manage log files first (low priority maintenance)
            log_cleanup_results = self.manage_log_files()
            if log_cleanup_results['files_cleaned'] > 0:
                self.logger.info(f"🧹 Log maintenance: cleaned {log_cleanup_results['files_cleaned']} files")
            
            # Clean up old archives periodically (every 10th call or when archives are old)
            if not hasattr(self, '_archive_cleanup_counter'):
                self._archive_cleanup_counter = 0
            self._archive_cleanup_counter += 1
            
            # Clean archives based on configuration
            should_clean_archives = (
                self._archive_cleanup_counter % self.archive_cleanup_config['cleanup_frequency'] == 0 or 
                not hasattr(self, '_last_archive_cleanup') or 
                (time.time() - getattr(self, '_last_archive_cleanup', 0)) > self.archive_cleanup_config['max_cleanup_interval']
            )
            
            if should_clean_archives:
                archive_cleanup_results = self.cleanup_archive_non_improving(
                    max_archive_days=self.archive_cleanup_config['max_archive_days'],
                    keep_recent=self.archive_cleanup_config['keep_recent']
                )
                if archive_cleanup_results['archives_deleted'] > 0:
                    self.logger.info(f"🗑️ Archive cleanup: deleted {archive_cleanup_results['archives_deleted']} old archives")
                self._last_archive_cleanup = time.time()
            
            # Clean up low-value training data periodically
            if not hasattr(self, '_training_data_cleanup_counter'):
                self._training_data_cleanup_counter = 0
            self._training_data_cleanup_counter += 1
            
            # Clean training data every 5 governor decisions or if we haven't cleaned in a while
            should_clean_training_data = (
                self._training_data_cleanup_counter % self.training_data_cleanup_config['cleanup_frequency'] == 0 or 
                not hasattr(self, '_last_training_data_cleanup') or 
                (time.time() - getattr(self, '_last_training_data_cleanup', 0)) > self.training_data_cleanup_config['max_cleanup_interval']
            )
            
            if should_clean_training_data:
                training_cleanup_results = self.cleanup_low_value_training_data(
                    min_score_threshold=self.training_data_cleanup_config['min_score_threshold'],
                    min_episodes_threshold=self.training_data_cleanup_config['min_episodes_threshold'],
                    max_age_days=self.training_data_cleanup_config['max_age_days']
                )
                if training_cleanup_results['files_deleted'] > 0:
                    self.logger.info(f"🧹 Training data cleanup: deleted {training_cleanup_results['files_deleted']} low-value files")
                self._last_training_data_cleanup = time.time()
            
            # Analyze current system state
            system_analysis = self._analyze_cognitive_systems()
            performance_analysis = self._analyze_performance_trends(current_performance)
            resource_analysis = self._analyze_resource_utilization()
            
            # Generate recommendations based on analysis
            recommendations = []
            
            # First, check for learned patterns that might apply
            learned_recs = self.get_learned_recommendations(puzzle_type, current_performance)
            for learned_rec in learned_recs:
                # Convert learned recommendation to GovernorRecommendation if confidence is high enough
                if learned_rec['confidence'] >= 0.6 and learned_rec['success_rate'] >= 0.5:
                    rec_type_str = learned_rec['type']
                    if rec_type_str in [t.value for t in GovernorRecommendationType]:
                        rec_type = GovernorRecommendationType(rec_type_str)
                        
                        learned_recommendation = GovernorRecommendation(
                            type=rec_type,
                            configuration_changes=learned_rec['configuration_changes'],
                            confidence=min(0.95, learned_rec['confidence'] + 0.1),  # Boost confidence slightly
                            expected_benefit=CognitiveBenefit(
                                win_rate_improvement=0.1 * learned_rec['success_rate'],
                                score_improvement=5.0 * learned_rec['success_rate'],
                                learning_efficiency=0.05 * learned_rec['success_rate'],
                                knowledge_transfer=0.05 * learned_rec['success_rate']
                            ),
                            rationale=learned_rec['rationale'],
                            urgency=0.5
                        )
                        recommendations.append(learned_recommendation)
            
            # Then generate standard recommendations
            # Check for mode switching opportunities
            mode_rec = self._evaluate_mode_switching(puzzle_type, performance_analysis)
            if mode_rec:
                recommendations.append(mode_rec)
            
            # Check for parameter adjustments
            param_rec = self._evaluate_parameter_adjustments(system_analysis)
            if param_rec:
                recommendations.append(param_rec)
            
            # Check for consolidation needs
            consolidation_rec = self._evaluate_consolidation_trigger(resource_analysis)
            if consolidation_rec:
                recommendations.append(consolidation_rec)
            
            # Check if Architect intervention is needed
            architect_rec = self._evaluate_architect_request(system_analysis, performance_analysis)
            if architect_rec:
                recommendations.append(architect_rec)
            
            # Select best recommendation
            best_recommendation = self._select_best_recommendation(recommendations)
            
            # Record decision
            decision_time = time.time() - start_time
            decision_id = self._record_decision(best_recommendation, decision_time, system_analysis, current_performance)
            
            self.total_decisions_made += 1
            
            if best_recommendation:
                # Start outcome tracking for this decision
                self.start_outcome_measurement(decision_id, best_recommendation, current_performance)
                
                self.logger.info(f"🎯 Governor recommendation: {best_recommendation.type.value} "
                               f"(confidence: {best_recommendation.confidence:.2f})")
                return best_recommendation
            else:
                self.logger.debug("📊 No configuration changes recommended")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error in Governor decision-making: {e}")
            return None
    
    def record_system_activation(self, system_name: str, 
                               cost: CognitiveCost, 
                               benefit: CognitiveBenefit):
        """Record activation of a cognitive system for monitoring."""
        if system_name in self.system_monitors:
            self.system_monitors[system_name].record_activation(cost, benefit)
            self.logger.debug(f"📈 Recorded {system_name} activation: "
                            f"cost={cost.total_cost():.2f}, benefit={benefit.total_benefit():.2f}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all monitored systems."""
        status = {
            'total_decisions': self.total_decisions_made,
            'successful_recommendations': self.successful_recommendations,
            'success_rate': (self.successful_recommendations / max(1, self.total_decisions_made)),
            'uptime_hours': (time.time() - self.start_time) / 3600,
            'system_efficiencies': {},
            'pending_architect_requests': len(self.pending_architect_requests),
            'top_performers': [],
            'underperformers': []
        }
        
        # Calculate system efficiencies
        efficiencies = {}
        for name, monitor in self.system_monitors.items():
            efficiencies[name] = {
                'efficiency_ratio': monitor.get_efficiency_ratio(),
                'activation_count': monitor.activation_count,
                'trend': monitor.get_recent_trend()
            }
        
        status['system_efficiencies'] = efficiencies
        
        # Identify top performers and underperformers
        sorted_systems = sorted(efficiencies.items(), 
                              key=lambda x: x[1]['efficiency_ratio'], 
                              reverse=True)
        
        status['top_performers'] = [name for name, data in sorted_systems[:5] 
                                  if data['activation_count'] > 0]
        status['underperformers'] = [name for name, data in sorted_systems[-5:] 
                                   if data['activation_count'] > 0 and data['efficiency_ratio'] < 0.5]
        
        return status
    
    def cleanup_archive_non_improving(self, max_archive_days: int = 30, keep_recent: int = 5) -> Dict[str, Any]:
        """
        Clean up old archives in data/archive_non_improving using git.
        
        Args:
            max_archive_days: Maximum age of archives to keep (in days)
            keep_recent: Number of most recent archives to always keep
            
        Returns:
            Dictionary with cleanup results
        """
        cleanup_results = {
            'archives_checked': 0,
            'archives_deleted': 0,
            'bytes_freed': 0,
            'errors': []
        }
        
        try:
            archive_dir = Path("data/archive_non_improving")
            if not archive_dir.exists():
                return cleanup_results
            
            # Get all archive directories (timestamp-based)
            archive_dirs = [d for d in archive_dir.iterdir() if d.is_dir() and d.name.startswith('20')]
            cleanup_results['archives_checked'] = len(archive_dirs)
            
            if len(archive_dirs) <= keep_recent:
                return cleanup_results
            
            # Sort by modification time (newest first)
            archive_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Keep the most recent ones
            archives_to_keep = archive_dirs[:keep_recent]
            archives_to_delete = archive_dirs[keep_recent:]
            
            # Filter by age
            current_time = time.time()
            max_age_seconds = max_archive_days * 24 * 60 * 60
            
            archives_to_delete = [
                archive for archive in archives_to_delete
                if (current_time - archive.stat().st_mtime) > max_age_seconds
            ]
            
            if not archives_to_delete:
                return cleanup_results
            
            # Initialize git repo if not already done
            try:
                repo = Repo(os.path.abspath("."))
            except Exception as e:
                cleanup_results['errors'].append(f"Git repository not available: {e}")
                return cleanup_results
            
            # Delete old archives using git
            for archive_dir_path in archives_to_delete:
                try:
                    # Calculate size before deletion
                    archive_size = sum(f.stat().st_size for f in archive_dir_path.rglob('*') if f.is_file())
                    
                    # Add to git staging
                    repo.index.add([str(archive_dir_path)])
                    
                    # Commit the deletion
                    commit_message = f"Governor cleanup: Remove old archive {archive_dir_path.name} (age: {int((current_time - archive_dir_path.stat().st_mtime) / 86400)} days)"
                    repo.index.commit(commit_message)
                    
                    # Remove the directory
                    import shutil
                    shutil.rmtree(archive_dir_path)
                    
                    cleanup_results['archives_deleted'] += 1
                    cleanup_results['bytes_freed'] += archive_size
                    
                    self.logger.info(f"🗑️ Deleted old archive: {archive_dir_path.name} ({archive_size / 1024 / 1024:.1f} MB)")
                    
                except Exception as e:
                    error_msg = f"Error deleting archive {archive_dir_path.name}: {e}"
                    cleanup_results['errors'].append(error_msg)
                    self.logger.error(error_msg)
            
            if cleanup_results['archives_deleted'] > 0:
                self.logger.info(f"🧹 Archive cleanup completed: {cleanup_results['archives_deleted']} archives deleted, {cleanup_results['bytes_freed'] / 1024 / 1024:.1f} MB freed")
                
                # Log Governor decision
                self.log_governor_decision({
                    "decision_type": "archive_cleanup",
                    "archives_deleted": cleanup_results['archives_deleted'],
                    "bytes_freed": cleanup_results['bytes_freed'],
                    "max_archive_days": max_archive_days,
                    "keep_recent": keep_recent
                })
            
        except Exception as e:
            error_msg = f"Archive cleanup error: {e}"
            cleanup_results['errors'].append(error_msg)
            self.logger.error(error_msg)
        
        return cleanup_results

    def force_archive_cleanup(self, max_archive_days: int = None, keep_recent: int = None) -> Dict[str, Any]:
        """
        Force immediate cleanup of archives, bypassing normal timing.
        
        Args:
            max_archive_days: Override max archive days (uses config default if None)
            keep_recent: Override keep recent count (uses config default if None)
            
        Returns:
            Dictionary with cleanup results
        """
        max_days = max_archive_days or self.archive_cleanup_config['max_archive_days']
        keep_count = keep_recent or self.archive_cleanup_config['keep_recent']
        
        self.logger.info(f"🔧 Forcing archive cleanup: max_days={max_days}, keep_recent={keep_count}")
        
        results = self.cleanup_archive_non_improving(max_archive_days=max_days, keep_recent=keep_count)
        
        # Reset the counter to avoid immediate re-cleanup
        self._last_archive_cleanup = time.time()
        
        return results

    def cleanup_low_value_training_data(self, min_score_threshold: float = 0.1, 
                                      min_episodes_threshold: int = 5,
                                      max_age_days: int = 7) -> Dict[str, Any]:
        """
        Clean up low-value training data that wasn't producing score benefits.
        
        Args:
            min_score_threshold: Minimum score to consider data valuable
            min_episodes_threshold: Minimum episodes to consider data meaningful
            max_age_days: Maximum age of data to keep (in days)
            
        Returns:
            Dictionary with cleanup results
        """
        cleanup_results = {
            'files_checked': 0,
            'files_deleted': 0,
            'bytes_freed': 0,
            'sessions_cleaned': 0,
            'meta_learning_cleaned': 0,
            'performance_entries_cleaned': 0,
            'errors': []
        }
        
        try:
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 60 * 60
            
            # 1. Clean up sessions with no score benefit
            sessions_dir = Path("data/sessions")
            if sessions_dir.exists():
                session_files = list(sessions_dir.glob("*.json"))
                cleanup_results['files_checked'] += len(session_files)
                
                for session_file in session_files:
                    try:
                        with open(session_file, 'r') as f:
                            session_data = json.load(f)
                        
                        # Check if session is valuable
                        should_delete = False
                        reason = ""
                        
                        # Check score
                        final_score = session_data.get('session_result', {}).get('final_score', 0)
                        if final_score < min_score_threshold:
                            should_delete = True
                            reason = f"low_score_{final_score}"
                        
                        # Check if too old
                        file_age = current_time - session_file.stat().st_mtime
                        if file_age > max_age_seconds:
                            should_delete = True
                            reason = f"old_data_{int(file_age/86400)}_days"
                        
                        # Check if no meaningful actions
                        total_actions = session_data.get('session_result', {}).get('total_actions', 0)
                        if total_actions < min_episodes_threshold:
                            should_delete = True
                            reason = f"insufficient_actions_{total_actions}"
                        
                        if should_delete:
                            file_size = session_file.stat().st_size
                            session_file.unlink()
                            cleanup_results['files_deleted'] += 1
                            cleanup_results['bytes_freed'] += file_size
                            cleanup_results['sessions_cleaned'] += 1
                            self.logger.info(f"🗑️ Deleted low-value session: {session_file.name} (reason: {reason})")
                            
                    except Exception as e:
                        error_msg = f"Error processing session {session_file.name}: {e}"
                        cleanup_results['errors'].append(error_msg)
                        self.logger.error(error_msg)
            
            # 2. Clean up meta-learning data with no insights
            meta_learning_dir = Path("data/meta_learning_data")
            if meta_learning_dir.exists():
                meta_files = list(meta_learning_dir.glob("*.json"))
                cleanup_results['files_checked'] += len(meta_files)
                
                for meta_file in meta_files:
                    try:
                        with open(meta_file, 'r') as f:
                            meta_data = json.load(f)
                        
                        # Check if meta-learning data is valuable
                        should_delete = False
                        reason = ""
                        
                        # Check for meaningful patterns/insights
                        patterns = meta_data.get('patterns', [])
                        insights = meta_data.get('insights', [])
                        stats = meta_data.get('stats', {})
                        
                        if len(patterns) == 0 and len(insights) == 0:
                            should_delete = True
                            reason = "no_patterns_or_insights"
                        
                        # Check if too old
                        file_age = current_time - meta_file.stat().st_mtime
                        if file_age > max_age_seconds:
                            should_delete = True
                            reason = f"old_data_{int(file_age/86400)}_days"
                        
                        # Check if no successful transfers
                        successful_transfers = stats.get('successful_transfers', 0)
                        if successful_transfers == 0 and len(patterns) == 0:
                            should_delete = True
                            reason = "no_successful_transfers"
                        
                        if should_delete:
                            file_size = meta_file.stat().st_size
                            meta_file.unlink()
                            cleanup_results['files_deleted'] += 1
                            cleanup_results['bytes_freed'] += file_size
                            cleanup_results['meta_learning_cleaned'] += 1
                            self.logger.info(f"🗑️ Deleted low-value meta-learning: {meta_file.name} (reason: {reason})")
                            
                    except Exception as e:
                        error_msg = f"Error processing meta-learning {meta_file.name}: {e}"
                        cleanup_results['errors'].append(error_msg)
                        self.logger.error(error_msg)
            
            # 3. Clean up task performance entries with no wins
            task_performance_file = Path("data/task_performance.json")
            if task_performance_file.exists():
                try:
                    with open(task_performance_file, 'r') as f:
                        task_performance = json.load(f)
                    
                    original_count = len(task_performance)
                    cleaned_performance = {}
                    
                    for game_id, performance_data in task_performance.items():
                        win_rate = performance_data.get('win_rate', 0.0)
                        avg_score = performance_data.get('avg_score', 0.0)
                        episodes = performance_data.get('episodes', 0)
                        last_updated = performance_data.get('last_updated', 0)
                        
                        # Keep if has meaningful performance or recent
                        should_keep = (
                            win_rate > min_score_threshold or 
                            avg_score > min_score_threshold or
                            episodes >= min_episodes_threshold or
                            (current_time - last_updated) < max_age_seconds
                        )
                        
                        if should_keep:
                            cleaned_performance[game_id] = performance_data
                        else:
                            cleanup_results['performance_entries_cleaned'] += 1
                            self.logger.info(f"🗑️ Removed low-value performance entry: {game_id} (win_rate: {win_rate}, score: {avg_score})")
                    
                    # Write cleaned performance data back
                    if len(cleaned_performance) < original_count:
                        with open(task_performance_file, 'w') as f:
                            json.dump(cleaned_performance, f, indent=2)
                        
                        cleanup_results['files_checked'] += 1
                        self.logger.info(f"📊 Cleaned task performance: removed {original_count - len(cleaned_performance)} entries")
                        
                except Exception as e:
                    error_msg = f"Error processing task performance: {e}"
                    cleanup_results['errors'].append(error_msg)
                    self.logger.error(error_msg)
            
            # 4. Clean up empty or low-value action intelligence files
            action_intel_files = list(Path("data").glob("action_intelligence_*.json"))
            for intel_file in action_intel_files:
                try:
                    with open(intel_file, 'r') as f:
                        intel_data = json.load(f)
                    
                    # Check if action intelligence data is valuable
                    should_delete = False
                    reason = ""
                    
                    # Check for meaningful data
                    if isinstance(intel_data, dict):
                        if len(intel_data) == 0:
                            should_delete = True
                            reason = "empty_data"
                        elif all(v == 0 or v == [] for v in intel_data.values()):
                            should_delete = True
                            reason = "all_zero_values"
                    
                    # Check if too old
                    file_age = current_time - intel_file.stat().st_mtime
                    if file_age > max_age_seconds:
                        should_delete = True
                        reason = f"old_data_{int(file_age/86400)}_days"
                    
                    if should_delete:
                        file_size = intel_file.stat().st_size
                        intel_file.unlink()
                        cleanup_results['files_deleted'] += 1
                        cleanup_results['bytes_freed'] += file_size
                        self.logger.info(f"🗑️ Deleted low-value action intelligence: {intel_file.name} (reason: {reason})")
                        
                except Exception as e:
                    error_msg = f"Error processing action intelligence {intel_file.name}: {e}"
                    cleanup_results['errors'].append(error_msg)
                    self.logger.error(error_msg)
            
            if cleanup_results['files_deleted'] > 0:
                self.logger.info(f"🧹 Training data cleanup completed: {cleanup_results['files_deleted']} files deleted, {cleanup_results['bytes_freed'] / 1024 / 1024:.1f} MB freed")
                
                # Log Governor decision
                self.log_governor_decision({
                    "decision_type": "training_data_cleanup",
                    "files_deleted": cleanup_results['files_deleted'],
                    "bytes_freed": cleanup_results['bytes_freed'],
                    "sessions_cleaned": cleanup_results['sessions_cleaned'],
                    "meta_learning_cleaned": cleanup_results['meta_learning_cleaned'],
                    "performance_entries_cleaned": cleanup_results['performance_entries_cleaned'],
                    "min_score_threshold": min_score_threshold,
                    "min_episodes_threshold": min_episodes_threshold,
                    "max_age_days": max_age_days
                })
            
        except Exception as e:
            error_msg = f"Training data cleanup error: {e}"
            cleanup_results['errors'].append(error_msg)
            self.logger.error(error_msg)
        
        return cleanup_results

    def force_training_data_cleanup(self, min_score_threshold: float = None, 
                                  min_episodes_threshold: int = None,
                                  max_age_days: int = None) -> Dict[str, Any]:
        """
        Force immediate cleanup of low-value training data.
        
        Args:
            min_score_threshold: Override min score threshold (uses config default if None)
            min_episodes_threshold: Override min episodes threshold (uses config default if None)
            max_age_days: Override max age days (uses config default if None)
            
        Returns:
            Dictionary with cleanup results
        """
        min_score = min_score_threshold or self.training_data_cleanup_config['min_score_threshold']
        min_episodes = min_episodes_threshold or self.training_data_cleanup_config['min_episodes_threshold']
        max_age = max_age_days or self.training_data_cleanup_config['max_age_days']
        
        self.logger.info(f"🔧 Forcing training data cleanup: min_score={min_score}, min_episodes={min_episodes}, max_age={max_age}")
        
        results = self.cleanup_low_value_training_data(
            min_score_threshold=min_score,
            min_episodes_threshold=min_episodes,
            max_age_days=max_age
        )
        
        # Reset the counter to avoid immediate re-cleanup
        self._last_training_data_cleanup = time.time()
        
        return results

    def manage_log_files(self) -> Dict[str, Any]:
        """Manage log file sizes by implementing rolling cleanup when files exceed threshold."""
        cleanup_results = {
            'files_checked': 0,
            'files_cleaned': 0,
            'lines_removed': 0,
            'errors': []
        }
        
        try:
            # Find all log files matching our patterns
            log_files = []
            for pattern in self.log_cleanup_patterns:
                # Search in data/logs directory
                log_dir = Path("data/logs")
                if log_dir.exists():
                    log_files.extend(log_dir.glob(pattern))
                
                # Also search in current directory
                log_files.extend(Path(".").glob(pattern))
            
            cleanup_results['files_checked'] = len(log_files)
            
            for log_file in log_files:
                try:
                    result = self._cleanup_log_file(log_file)
                    if result['cleaned']:
                        cleanup_results['files_cleaned'] += 1
                        cleanup_results['lines_removed'] += result['lines_removed']
                        self.logger.info(f"🧹 Cleaned log file {log_file.name}: removed {result['lines_removed']} lines")
                except Exception as e:
                    error_msg = f"Error cleaning {log_file}: {e}"
                    cleanup_results['errors'].append(error_msg)
                    self.logger.error(error_msg)
            
            if cleanup_results['files_cleaned'] > 0:
                self.logger.info(f"📊 Log cleanup completed: {cleanup_results['files_cleaned']} files cleaned, {cleanup_results['lines_removed']} lines removed")
            
        except Exception as e:
            error_msg = f"Log management error: {e}"
            cleanup_results['errors'].append(error_msg)
            self.logger.error(error_msg)
        
        return cleanup_results
    
    def _cleanup_log_file(self, log_file: Path) -> Dict[str, Any]:
        """Clean up a single log file by removing old lines if it exceeds threshold."""
        result = {
            'cleaned': False,
            'lines_removed': 0,
            'original_lines': 0,
            'final_lines': 0
        }
        
        try:
            # Count lines in file
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            result['original_lines'] = len(lines)
            
            # Special handling for master_arc_trainer logs - keep under 100k
            if 'master_arc_trainer' in log_file.name.lower():
                threshold = self.master_trainer_threshold
                remove_lines = self.master_trainer_remove_lines
                min_keep = 10000  # Keep at least 10k lines for master trainer logs
            else:
                threshold = self.log_cleanup_threshold
                remove_lines = self.log_cleanup_remove_lines
                min_keep = 1000  # Keep at least 1k lines for other logs
            
            # Check if file exceeds threshold
            if len(lines) <= threshold:
                return result
            
            # Remove first N lines (oldest entries)
            lines_to_remove = min(remove_lines, len(lines) - min_keep)
            cleaned_lines = lines[lines_to_remove:]
            
            # Write cleaned content back to file
            with open(log_file, 'w', encoding='utf-8') as f:
                f.writelines(cleaned_lines)
            
            result['cleaned'] = True
            result['lines_removed'] = lines_to_remove
            result['final_lines'] = len(cleaned_lines)
            
        except Exception as e:
            self.logger.error(f"Error cleaning log file {log_file}: {e}")
            result['error'] = str(e)
        
        return result
    
    def create_architect_request(self, issue_type: str, 
                               problem_description: str,
                               performance_data: Dict[str, Any],
                               frame_data: Optional[Dict[str, Any]] = None,
                               memory_context: Optional[Dict[str, Any]] = None,
                               object_analysis: Optional[Dict[str, Any]] = None,
                               learning_progress: Optional[float] = None,
                               energy_state: Optional[Dict[str, Any]] = None) -> ArchitectRequest:
        """Create a request for the Architect to address systemic issues with enhanced context."""
        request = ArchitectRequest(
            issue_type=issue_type,
            persistent_problem=problem_description,
            failed_solutions=self._get_failed_solutions_history(issue_type),
            performance_data=performance_data,
            suggested_research_directions=self._suggest_research_directions(issue_type),
            priority=self._calculate_issue_priority(issue_type, performance_data),
            frame_data=frame_data,
            memory_context=memory_context,
            object_analysis=object_analysis,
            learning_progress=learning_progress,
            energy_state=energy_state
        )
        
        self.pending_architect_requests.append(request)
        self.logger.warning(f"🔬 Enhanced Architect request created: {issue_type} with context data")
        
        return request
    
    def _analyze_visual_context(self, frame_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze visual context from frame analysis data."""
        insights = {
            'has_interactive_targets': False,
            'target_quality': 0.0,
            'environment_complexity': 0.0,
            'visual_clarity': 0.0
        }
        
        if not frame_analysis:
            return insights
        
        # Analyze interactive targets
        targets = frame_analysis.get('interactive_targets', [])
        insights['has_interactive_targets'] = len(targets) > 0
        insights['target_quality'] = len(targets) / 10.0  # Normalize to 0-1
        
        # Analyze environment complexity
        total_objects = frame_analysis.get('total_objects', 0)
        insights['environment_complexity'] = min(1.0, total_objects / 20.0)
        
        # Analyze visual clarity
        clarity_score = frame_analysis.get('clarity_score', 0.5)
        insights['visual_clarity'] = clarity_score
        
        return insights
    
    def _analyze_object_context(self, object_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze object context for decision making."""
        insights = {
            'object_diversity': 0.0,
            'interaction_potential': 0.0,
            'spatial_relationships': 0.0,
            'novel_objects': 0.0
        }
        
        if not object_analysis:
            return insights
        
        # Analyze object diversity
        object_types = object_analysis.get('object_types', [])
        insights['object_diversity'] = min(1.0, len(set(object_types)) / 10.0)
        
        # Analyze interaction potential
        interactive_objects = object_analysis.get('interactive_objects', [])
        insights['interaction_potential'] = min(1.0, len(interactive_objects) / 5.0)
        
        # Analyze spatial relationships
        spatial_data = object_analysis.get('spatial_relationships', {})
        insights['spatial_relationships'] = min(1.0, len(spatial_data) / 10.0)
        
        # Analyze novel objects
        novel_objects = object_analysis.get('novel_objects', [])
        insights['novel_objects'] = min(1.0, len(novel_objects) / 3.0)
        
        return insights
    
    def _analyze_memory_context(self, memory_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze memory context for decision making."""
        insights = {
            'recent_success_rate': 0.0,
            'learning_progress': 0.0,
            'pattern_recognition': 0.0,
            'memory_consolidation': 0.0
        }
        
        if not memory_context:
            return insights
        
        # Analyze recent success rate
        recent_actions = memory_context.get('recent_actions', [])
        if recent_actions:
            successful_actions = sum(1 for action in recent_actions if action.get('success', False))
            insights['recent_success_rate'] = successful_actions / len(recent_actions)
        
        # Analyze learning progress
        learning_data = memory_context.get('learning_progress', {})
        insights['learning_progress'] = learning_data.get('current_progress', 0.0)
        
        # Analyze pattern recognition
        patterns = memory_context.get('recognized_patterns', [])
        insights['pattern_recognition'] = min(1.0, len(patterns) / 5.0)
        
        # Analyze memory consolidation
        consolidation_strength = memory_context.get('consolidation_strength', 0.5)
        insights['memory_consolidation'] = consolidation_strength
        
        return insights
    
    def _analyze_energy_context(self, current_energy: float) -> Dict[str, Any]:
        """Analyze energy context for decision making."""
        insights = {
            'energy_level': current_energy / 100.0,  # Normalize to 0-1
            'needs_conservation': current_energy < 30.0,
            'can_explore': current_energy > 50.0,
            'energy_trend': 0.0  # Would need historical data for trend
        }
        
        return insights
    
    def _calculate_enhanced_decision_confidence(self, available_actions: List[int], 
                                              context: Dict[str, Any], 
                                              performance_history: List[Dict[str, Any]], 
                                              current_energy: float,
                                              visual_insights: Dict[str, Any],
                                              object_insights: Dict[str, Any],
                                              memory_insights: Dict[str, Any],
                                              energy_insights: Dict[str, Any]) -> float:
        """Calculate decision confidence using enhanced data analysis."""
        base_confidence = 0.5
        
        # Visual confidence factors
        if visual_insights['has_interactive_targets']:
            base_confidence += 0.2
        if visual_insights['visual_clarity'] > 0.7:
            base_confidence += 0.1
        
        # Object confidence factors
        if object_insights['interaction_potential'] > 0.5:
            base_confidence += 0.15
        if object_insights['object_diversity'] > 0.6:
            base_confidence += 0.1
        
        # Memory confidence factors
        if memory_insights['recent_success_rate'] > 0.6:
            base_confidence += 0.15
        if memory_insights['learning_progress'] > 0.3:
            base_confidence += 0.1
        
        # Energy confidence factors
        if energy_insights['can_explore']:
            base_confidence += 0.1
        if energy_insights['needs_conservation']:
            base_confidence -= 0.1
        
        return min(1.0, max(0.0, base_confidence))
    
    def _select_enhanced_meta_cognitive_action(self, available_actions: List[int], 
                                             context: Dict[str, Any], 
                                             performance_history: List[Dict[str, Any]], 
                                             current_energy: float) -> int:
        """Select action using enhanced meta-cognitive analysis."""
        # Enhanced action selection based on comprehensive data analysis
        if not available_actions:
            return 0
        
        # Get enhanced insights
        frame_analysis = context.get('frame_analysis', {})
        object_analysis = context.get('object_analysis', {})
        memory_context = context.get('memory_context', {})
        
        visual_insights = self._analyze_visual_context(frame_analysis)
        object_insights = self._analyze_object_context(object_analysis)
        memory_insights = self._analyze_memory_context(memory_context)
        energy_insights = self._analyze_energy_context(current_energy)
        
        # Enhanced decision logic
        if energy_insights['needs_conservation']:
            # Conservative actions when energy is low
            return available_actions[0]  # Simple action
        
        if visual_insights['has_interactive_targets'] and object_insights['interaction_potential'] > 0.5:
            # High interaction potential - explore
            return available_actions[-1] if len(available_actions) > 1 else available_actions[0]
        
        if memory_insights['learning_progress'] > 0.5 and not memory_insights['recent_success_rate'] > 0.7:
            # High learning progress but low success - try different approach
            return available_actions[len(available_actions)//2] if len(available_actions) > 2 else available_actions[0]
        
        # Default to balanced exploration
        return available_actions[len(available_actions)//2] if len(available_actions) > 1 else available_actions[0]
    
    def has_persistent_issues(self) -> bool:
        """Check if there are persistent issues that require architectural changes."""
        return len(self.pending_architect_requests) > 0
    
    # Private methods for internal analysis
    
    def _analyze_cognitive_systems(self) -> Dict[str, Any]:
        """Analyze the current state of all cognitive systems."""
        analysis = {
            'total_systems': len(self.system_monitors),
            'active_systems': 0,
            'high_efficiency_systems': [],
            'low_efficiency_systems': [],
            'trending_up': [],
            'trending_down': [],
            'average_efficiency': 0.0
        }
        
        total_efficiency = 0.0
        active_count = 0
        
        for name, monitor in self.system_monitors.items():
            if monitor.activation_count > 0:
                active_count += 1
                efficiency = monitor.get_efficiency_ratio()
                total_efficiency += efficiency
                trend = monitor.get_recent_trend()
                
                if efficiency > 2.0:
                    analysis['high_efficiency_systems'].append(name)
                elif efficiency < 0.5:
                    analysis['low_efficiency_systems'].append(name)
                
                if trend == "improving":
                    analysis['trending_up'].append(name)
                elif trend == "declining":
                    analysis['trending_down'].append(name)
        
        analysis['active_systems'] = active_count
        analysis['average_efficiency'] = total_efficiency / max(1, active_count)
        
        return analysis
    
    def _analyze_performance_trends(self, current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        # This would integrate with existing performance tracking
        return {
            'current_win_rate': current_performance.get('win_rate', 0.0),
            'current_score': current_performance.get('average_score', 0.0),
            'trend': 'stable',  # Would calculate from history
            'stagnation_detected': False,
            'breakthrough_potential': 0.5
        }
    
    def _analyze_resource_utilization(self) -> Dict[str, Any]:
        """Analyze abstract resource utilization across systems."""
        return {
            'memory_pressure': 0.3,  # Would calculate from actual memory usage
            'compute_utilization': 0.7,
            'coordination_overhead': 0.2,
            'consolidation_needed': False
        }
    
    def _evaluate_mode_switching(self, puzzle_type: str, performance: Dict[str, Any]) -> Optional[GovernorRecommendation]:
        """Evaluate if mode switching would be beneficial."""
        current_win_rate = performance.get('current_win_rate', 0)
        
        # Get history of recent mode switches to avoid oscillation
        recent_mode_switches = [
            decision for decision in list(self.decision_history)[-5:]
            if decision['recommendation_type'] == GovernorRecommendationType.MODE_SWITCH.value
        ]
        
        # Avoid too frequent mode switching
        if len(recent_mode_switches) >= 2:
            return None
            
        # Calculate adaptive confidence based on past mode switch success
        success_rate = self._calculate_mode_switch_success_rate(recent_mode_switches)
        base_confidence = 0.8
        adaptive_confidence = min(0.95, max(0.4, base_confidence + (success_rate - 0.5) * 0.3))
        
        # Different thresholds and strategies based on puzzle type and performance
        if current_win_rate < 0.3:
            strategy_config, rationale = self._select_mode_switch_strategy(
                puzzle_type, current_win_rate, performance
            )
            
            return GovernorRecommendation(
                type=GovernorRecommendationType.MODE_SWITCH,
                configuration_changes=strategy_config,
                confidence=adaptive_confidence,
                expected_benefit=CognitiveBenefit(
                    win_rate_improvement=0.2 * adaptive_confidence,
                    score_improvement=10.0 * adaptive_confidence,
                    learning_efficiency=0.1,
                    knowledge_transfer=0.05
                ),
                rationale=rationale,
                urgency=0.7 if current_win_rate < 0.2 else 0.5
            )
        return None
    
    def _select_mode_switch_strategy(self, puzzle_type: str, win_rate: float, performance: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """Select appropriate mode switch strategy based on context."""
        if win_rate < 0.1:
            # Emergency mode - try contrarian approaches
            return (
                {'enable_contrarian_strategy': True, 'increase_exploration': True},
                f"Critical win rate {win_rate:.2f} requires contrarian exploration"
            )
        elif win_rate < 0.2:
            # Try different cognitive approach
            return (
                {'switch_to_pattern_mode': True, 'reduce_noise_threshold': 0.1},
                f"Very low win rate {win_rate:.2f} suggests pattern recognition focus needed"
            )
        elif win_rate < 0.3:
            # Moderate adjustment
            avg_score = performance.get('average_score', 0)
            if avg_score < 5:
                return (
                    {'enable_systematic_search': True, 'increase_depth_limit': 50},
                    f"Low scores with {win_rate:.2f} win rate suggest systematic search needed"
                )
            else:
                return (
                    {'enable_intuitive_mode': True, 'reduce_analysis_time': 0.8},
                    f"Decent scores but {win_rate:.2f} win rate suggest faster intuitive decisions"
                )
        
        # Fallback
        return (
            {'enable_balanced_mode': True},
            f"Balanced approach for {win_rate:.2f} win rate"
        )
    
    def _calculate_mode_switch_success_rate(self, recent_switches: List[Dict]) -> float:
        """Calculate success rate of recent mode switches."""
        if not recent_switches:
            return 0.5  # Neutral baseline
        
        successful = 0
        for switch in recent_switches:
            outcome = switch.get('outcome_metrics', {})
            if outcome.get('win_rate_change', 0) > 0.05:  # Meaningful improvement
                successful += 1
        
        return successful / len(recent_switches) if recent_switches else 0.5
    
    def _evaluate_parameter_adjustments(self, system_analysis: Dict[str, Any]) -> Optional[GovernorRecommendation]:
        """Evaluate if parameter adjustments would be beneficial."""
        if system_analysis['average_efficiency'] < 1.0:
            # Get history of recent parameter adjustments
            recent_param_adjustments = [
                decision for decision in list(self.decision_history)[-10:]
                if decision['recommendation_type'] == GovernorRecommendationType.PARAMETER_ADJUSTMENT.value
            ]
            
            # Determine which parameter to adjust based on history and system state
            config_changes, confidence, rationale = self._select_parameter_adjustment(
                system_analysis, recent_param_adjustments
            )
            
            if config_changes:
                return GovernorRecommendation(
                    type=GovernorRecommendationType.PARAMETER_ADJUSTMENT,
                    configuration_changes=config_changes,
                    confidence=confidence,
                    expected_benefit=CognitiveBenefit(
                        win_rate_improvement=0.05 * confidence,
                        score_improvement=2.0 * confidence,
                        learning_efficiency=0.1 * confidence,
                        knowledge_transfer=0.0
                    ),
                    rationale=rationale,
                    urgency=0.4
                )
        return None
    
    def _select_parameter_adjustment(self, system_analysis: Dict[str, Any], recent_adjustments: List[Dict]) -> Tuple[Dict[str, Any], float, str]:
        """Select intelligent parameter adjustment based on system state and history."""
        efficiency = system_analysis['average_efficiency']
        win_rate = system_analysis.get('win_rate', 0.0)
        
        # Calculate adaptive confidence based on recent adjustment success
        base_confidence = 0.6
        success_rate = self._calculate_adjustment_success_rate(recent_adjustments)
        adaptive_confidence = min(0.95, max(0.3, base_confidence + (success_rate - 0.5) * 0.4))
        
        # Check what parameters were recently adjusted to avoid repetition
        recent_param_types = set()
        for adj in recent_adjustments[-3:]:  # Last 3 adjustments
            config = adj.get('configuration_changes', {})
            recent_param_types.update(config.keys())
        
        # Parameter adjustment strategies based on system state
        if efficiency < 0.7 and 'max_actions_per_game' not in recent_param_types:
            return (
                {'max_actions_per_game': 750},
                adaptive_confidence,
                "Low efficiency suggests need for more exploration actions"
            )
        elif efficiency < 0.8 and win_rate < 0.4 and 'learning_rate' not in recent_param_types:
            return (
                {'learning_rate': 0.001},
                adaptive_confidence * 0.9,
                "Poor win rate indicates learning rate adjustment needed"
            )
        elif efficiency > 0.6 and efficiency < 0.9 and 'batch_size' not in recent_param_types:
            return (
                {'batch_size': 64},
                adaptive_confidence * 0.8,
                "Moderate efficiency suggests batch size optimization"
            )
        elif 'temperature' not in recent_param_types:
            # Exploration vs exploitation balance
            temp_value = 0.8 if win_rate < 0.5 else 0.3
            return (
                {'temperature': temp_value},
                adaptive_confidence * 0.7,
                f"Adjusting exploration temperature based on win rate {win_rate:.2f}"
            )
        else:
            # All main parameters recently adjusted, suggest multi-parameter fine-tuning
            return (
                {
                    'max_actions_per_game': min(1000, int(750 * (1 + (1 - efficiency)))),
                    'exploration_bonus': 0.1 if efficiency < 0.8 else 0.05
                },
                adaptive_confidence * 0.6,
                "Multi-parameter fine-tuning after recent individual adjustments"
            )
    
    def _calculate_adjustment_success_rate(self, recent_adjustments: List[Dict]) -> float:
        """Calculate success rate of recent parameter adjustments."""
        if not recent_adjustments:
            return 0.5  # Neutral baseline
        
        successful = 0
        for adj in recent_adjustments:
            # Consider adjustment successful if it led to improvement
            # This is a simplified heuristic - in practice would track actual outcomes
            outcome = adj.get('outcome_metrics', {})
            if outcome.get('win_rate_change', 0) > 0 or outcome.get('efficiency_change', 0) > 0:
                successful += 1
        
        return successful / len(recent_adjustments) if recent_adjustments else 0.5
    
    def _evaluate_consolidation_trigger(self, resource_analysis: Dict[str, Any]) -> Optional[GovernorRecommendation]:
        """Evaluate if consolidation should be triggered."""
        if resource_analysis.get('memory_pressure', 0) > 0.8:
            return GovernorRecommendation(
                type=GovernorRecommendationType.CONSOLIDATION_TRIGGER,
                configuration_changes={'trigger_consolidation': True},
                confidence=0.9,
                expected_benefit=CognitiveBenefit(
                    win_rate_improvement=0.0,
                    score_improvement=0.0,
                    learning_efficiency=0.2,
                    knowledge_transfer=0.1
                ),
                rationale="High memory pressure requires consolidation",
                urgency=0.8
            )
        return None
    
    def _evaluate_architect_request(self, system_analysis: Dict[str, Any], 
                                  performance_analysis: Dict[str, Any]) -> Optional[GovernorRecommendation]:
        """Evaluate if Architect intervention is needed."""
        # Check for persistent low performance across multiple systems
        if (len(system_analysis['low_efficiency_systems']) > 5 and 
            performance_analysis.get('stagnation_detected', False)):
            
            return GovernorRecommendation(
                type=GovernorRecommendationType.ARCHITECT_REQUEST,
                configuration_changes={'request_architectural_review': True},
                confidence=0.7,
                expected_benefit=CognitiveBenefit(
                    win_rate_improvement=0.1,
                    score_improvement=5.0,
                    learning_efficiency=0.3,
                    knowledge_transfer=0.2
                ),
                rationale="Multiple low-efficiency systems suggest architectural issues",
                urgency=0.6
            )
        return None
    
    def _select_best_recommendation(self, recommendations: List[GovernorRecommendation]) -> Optional[GovernorRecommendation]:
        """Select the best recommendation from available options."""
        if not recommendations:
            return None
        
        # Sort by urgency * confidence * expected benefit
        def score_recommendation(rec):
            return rec.urgency * rec.confidence * rec.expected_benefit.total_benefit()
        
        return max(recommendations, key=score_recommendation)
    
    def _record_decision(self, recommendation: Optional[GovernorRecommendation], 
                        decision_time: float, system_analysis: Dict[str, Any],
                        current_performance: Dict[str, Any] = None) -> str:
        """Record a decision for future analysis."""
        decision_id = f"decision_{self.total_decisions_made}_{int(time.time())}"
        
        decision_record = {
            'timestamp': time.time(),
            'recommendation': recommendation.to_dict() if recommendation else None,
            'recommendation_type': recommendation.type.value if recommendation else 'no_action',
            'decision_time_ms': decision_time * 1000,
            'system_state': system_analysis,
            'decision_id': decision_id,
            'current_performance': current_performance or {}
        }
        
        self.decision_history.append(decision_record)
        
        # Log to file if specified
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(decision_record) + '\n')
            except Exception as e:
                self.logger.error(f"Failed to log decision: {e}")
        
        return decision_id
    
    def start_outcome_measurement(self, decision_id: str, recommendation: GovernorRecommendation,
                                 current_performance: Dict[str, Any]) -> Optional[str]:
        """Start tracking outcomes for a recommendation."""
        if not self.outcome_tracker:
            return None
        
        try:
            from src.core.outcome_tracker import PerformanceMetrics
            
            # Convert current performance to PerformanceMetrics
            baseline_metrics = PerformanceMetrics(
                win_rate=current_performance.get('win_rate', 0.0),
                average_score=current_performance.get('average_score', 0.0),
                learning_efficiency=current_performance.get('learning_efficiency', 0.0),
                knowledge_transfer=current_performance.get('knowledge_transfer', 0.0),
                computational_efficiency=current_performance.get('computational_efficiency', 1.0),
                memory_usage=current_performance.get('memory_usage', 0.5),
                inference_speed=current_performance.get('inference_speed', 1.0)
            )
            
            outcome_id = self.outcome_tracker.start_outcome_measurement(
                decision_id=decision_id,
                intervention_type=f"governor_{recommendation.type.value}",
                intervention_details=recommendation.configuration_changes,
                baseline_metrics=baseline_metrics
            )
            
            self.pending_outcome_measurements[decision_id] = outcome_id
            return outcome_id
            
        except Exception as e:
            self.logger.error(f"Failed to start outcome measurement: {e}")
            return None
    
    def complete_outcome_measurement(self, decision_id: str, 
                                   post_performance: Dict[str, Any],
                                   sample_size: int = 1,
                                   notes: str = ""):
        """Complete outcome measurement for a decision."""
        if not self.outcome_tracker or decision_id not in self.pending_outcome_measurements:
            return
        
        try:
            from src.core.outcome_tracker import PerformanceMetrics
            
            # Convert post performance to PerformanceMetrics
            post_metrics = PerformanceMetrics(
                win_rate=post_performance.get('win_rate', 0.0),
                average_score=post_performance.get('average_score', 0.0),
                learning_efficiency=post_performance.get('learning_efficiency', 0.0),
                knowledge_transfer=post_performance.get('knowledge_transfer', 0.0),
                computational_efficiency=post_performance.get('computational_efficiency', 1.0),
                memory_usage=post_performance.get('memory_usage', 0.5),
                inference_speed=post_performance.get('inference_speed', 1.0)
            )
            
            outcome_id = self.pending_outcome_measurements[decision_id]
            outcome_record = self.outcome_tracker.complete_outcome_measurement(
                outcome_id=outcome_id,
                post_metrics=post_metrics,
                sample_size=sample_size,
                notes=notes
            )
            
            # Update decision history with outcome
            for decision in reversed(self.decision_history):
                if decision.get('decision_id') == decision_id:
                    decision['outcome_metrics'] = {
                        'success_score': outcome_record.success_score,
                        'status': outcome_record.status.value,
                        'win_rate_change': outcome_record.performance_deltas.get('win_rate_delta', 0),
                        'score_change': outcome_record.performance_deltas.get('score_delta', 0),
                        'efficiency_change': outcome_record.performance_deltas.get('learning_efficiency_delta', 0)
                    }
                    break
            
            # Update success counter
            if outcome_record.success_score >= 0.4:
                self.successful_recommendations += 1
            
            # Clean up pending measurements
            del self.pending_outcome_measurements[decision_id]
            
        except Exception as e:
            self.logger.error(f"Failed to complete outcome measurement: {e}")
    
    def get_effectiveness_insights(self) -> Dict[str, Any]:
        """Get insights about Governor effectiveness from outcome tracking."""
        if not self.outcome_tracker:
            return {'insights_available': False}
        
        insights = self.outcome_tracker.get_learning_insights()
        
        # Add Governor-specific insights
        governor_insights = {
            'total_decisions': self.total_decisions_made,
            'tracked_outcomes': len(self.outcome_tracker.outcome_history),
            'pending_measurements': len(self.pending_outcome_measurements),
            'estimated_success_rate': self.successful_recommendations / max(self.total_decisions_made, 1)
        }
        
        # Get effectiveness by recommendation type
        recommendation_types = [
            'governor_mode_switch',
            'governor_parameter_adjustment', 
            'governor_consolidation_trigger'
        ]
        
        for rec_type in recommendation_types:
            stats = self.outcome_tracker.get_intervention_effectiveness(rec_type)
            governor_insights[f'{rec_type}_effectiveness'] = stats
        
        insights['governor_specific'] = governor_insights
        insights['insights_available'] = True
        
        return insights
    
    def start_learning_session(self, session_context: Dict[str, Any] = None) -> Optional[str]:
        """Start a cross-session learning session."""
        if not self.learning_manager:
            return None
        
        session_id = self.learning_manager.start_session(session_context)
        self.logger.info(f"Started cross-session learning: {session_id}")
        return session_id
    
    def end_learning_session(self, performance_summary: Dict[str, Any] = None):
        """End the current learning session."""
        if not self.learning_manager:
            return
        
        if not performance_summary:
            performance_summary = {
                'total_decisions': self.total_decisions_made,
                'successful_decisions': self.successful_recommendations,
                'avg_improvement': 0.1 if self.successful_recommendations > 0 else 0.0
            }
        
        self.learning_manager.end_session(performance_summary)
    
    def learn_from_recommendation_outcome(self, recommendation: GovernorRecommendation,
                                        context: Dict[str, Any], success_metrics: Dict[str, float]):
        """Learn from the outcome of a recommendation."""
        if not self.learning_manager:
            return
        
        from src.core.cross_session_learning import KnowledgeType, PersistenceLevel
        
        # Learn strategy pattern
        strategy_data = {
            'recommendation_type': recommendation.type.value,
            'configuration_changes': recommendation.configuration_changes,
            'original_confidence': recommendation.confidence,
            'expected_benefit': asdict(recommendation.expected_benefit)
        }
        
        # Determine success score
        success_score = (
            success_metrics.get('win_rate_improvement', 0) * 0.4 +
            success_metrics.get('score_improvement', 0) / 20.0 * 0.3 +
            success_metrics.get('efficiency_improvement', 0) * 0.3
        )
        
        # Determine persistence level based on success
        persistence_level = PersistenceLevel.PERMANENT if success_score > 0.6 else PersistenceLevel.SESSION
        
        pattern_id = self.learning_manager.learn_pattern(
            KnowledgeType.STRATEGY_PATTERN,
            strategy_data,
            context,
            success_score,
            persistence_level
        )
        
        self.logger.debug(f"Learned strategy pattern {pattern_id} with success score {success_score:.3f}")
        
        # Also learn parameter optimization patterns
        if recommendation.type.value == 'parameter_adjustment':
            param_data = {
                'parameter_changes': recommendation.configuration_changes,
                'system_state': context.get('system_state', {}),
                'performance_improvement': success_score
            }
            
            self.learning_manager.learn_pattern(
                KnowledgeType.PARAMETER_OPTIMIZATION,
                param_data,
                context,
                success_score,
                PersistenceLevel.PERMANENT if success_score > 0.7 else PersistenceLevel.SESSION
            )
    
    def get_learned_recommendations(self, puzzle_type: str, current_performance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get recommendations based on learned patterns."""
        if not self.learning_manager:
            return []
        
        from src.core.cross_session_learning import KnowledgeType
        
        context = {
            'puzzle_type': puzzle_type,
            'current_performance': current_performance
        }
        
        # Get applicable strategy patterns
        strategy_patterns = self.learning_manager.retrieve_applicable_patterns(
            KnowledgeType.STRATEGY_PATTERN,
            context,
            min_confidence=0.4,
            max_results=3
        )
        
        recommendations = []
        for pattern in strategy_patterns:
            pattern_data = pattern.pattern_data
            
            rec_data = {
                'type': pattern_data.get('recommendation_type', 'unknown'),
                'configuration_changes': pattern_data.get('configuration_changes', {}),
                'confidence': pattern.confidence,
                'success_rate': pattern.success_rate,
                'applications': pattern.total_applications,
                'rationale': f"Learned strategy (success rate: {pattern.success_rate:.1%})"
            }
            
            recommendations.append(rec_data)
        
        return recommendations
    
    def get_best_configuration_for_context(self, puzzle_type: str, current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Get the best known configuration for the current context."""
        if not self.learning_manager:
            return {}
        
        context = {
            'puzzle_type': puzzle_type,
            'current_performance': current_performance
        }
        
        return self.learning_manager.get_best_configuration_for_context(context)
    
    def get_cross_session_insights(self) -> Dict[str, Any]:
        """Get insights from cross-session learning."""
        if not self.learning_manager:
            return {'learning_available': False}
        
        insights = self.learning_manager.get_performance_insights()
        insights['learning_available'] = True
        
        return insights
    
    def _get_failed_solutions_history(self, issue_type: str) -> List[Dict[str, Any]]:
        """Get history of failed solutions for this issue type."""
        # Would implement based on decision history
        return []
    
    def _suggest_research_directions(self, issue_type: str) -> List[str]:
        """Suggest research directions for architectural improvements."""
        suggestions = {
            'low_efficiency': [
                "Investigate memory allocation optimization",
                "Research cognitive load balancing",
                "Explore dynamic system activation patterns"
            ],
            'stagnation': [
                "Design new exploration strategies", 
                "Research meta-learning enhancements",
                "Investigate curriculum learning improvements"
            ]
        }
        return suggestions.get(issue_type, ["General architectural review"])
    
    def _calculate_issue_priority(self, issue_type: str, performance_data: Dict[str, Any]) -> float:
        """Calculate priority level for an issue."""
        base_priority = {
            'low_efficiency': 0.6,
            'stagnation': 0.8,
            'resource_exhaustion': 0.9,
            'system_failure': 1.0
        }.get(issue_type, 0.5)
        
        # Adjust based on performance impact
        performance_impact = 1.0 - performance_data.get('win_rate', 0.5)
        
        return min(1.0, base_priority * (1 + performance_impact))

    def perform_memory_management(self, emergency_cleanup: bool = False, 
                                 target_size_mb: Optional[float] = None) -> Dict[str, Any]:
        """
        Perform intelligent memory management with meta-cognitive awareness.
        
        Args:
            emergency_cleanup: If True, perform aggressive cleanup
            target_size_mb: Target size for emergency cleanup
        
        Returns:
            Dictionary with memory management results
        """
        if not self.memory_manager:
            self.logger.warning("Memory manager not available")
            return {"status": "unavailable"}
        
        try:
            if emergency_cleanup and target_size_mb:
                self.logger.info(f"Performing emergency memory cleanup (target: {target_size_mb} MB)")
                results = self.memory_manager.emergency_cleanup(target_size_mb)
                
                # Log Governor decision
                self.log_governor_decision({
                    "decision_type": "emergency_memory_cleanup",
                    "target_size_mb": target_size_mb,
                    "files_deleted": results["files_deleted"],
                    "bytes_freed": results["bytes_freed"],
                    "critical_files_protected": results["critical_files_protected"]
                })
                
            else:
                self.logger.info("Performing routine memory management")
                results = self.memory_manager.perform_garbage_collection(dry_run=False)
                
                # Log Governor decision
                self.log_governor_decision({
                    "decision_type": "routine_memory_management",
                    "files_processed": results["files_processed"],
                    "files_deleted": results["files_deleted"],
                    "bytes_freed": results["bytes_freed"],
                    "critical_files_protected": results["critical_files_protected"]
                })
            
            # Update system health metrics
            memory_status = self.memory_manager.get_memory_status()
            self.system_monitors["memory_system"] = {
                "total_files": memory_status["total_files"],
                "total_size_mb": memory_status["total_size_mb"],
                "status": "healthy" if memory_status["total_size_mb"] < 1000 else "attention_needed",
                "last_cleanup": time.time()
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Memory management failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status with Governor analysis and pattern optimization."""
        if not self.memory_manager:
            return {"status": "unavailable"}
        
        try:
            base_status = self.memory_manager.get_memory_status()
            
            # Add Governor analysis
            analysis = {
                "health_status": "healthy",
                "recommendations": [],
                "critical_files_count": base_status["classifications"].get("critical_lossless", {}).get("file_count", 0),
                "cleanup_needed": False
            }
            
            # Phase 1 Enhancement: Add pattern-based analysis
            pattern_recommendations = []
            if self.pattern_optimizer:
                try:
                    # Get pattern-based recommendations for immediate optimization
                    governor_recommendations = self.pattern_optimizer.get_governor_recommendations()
                    analysis["pattern_analysis"] = governor_recommendations
                    
                    # Extract immediate actions for Governor recommendations
                    for action in governor_recommendations.get('immediate_actions', []):
                        pattern_recommendations.append(f"Pattern Analysis: {action['reason']}")
                    
                    # Add efficiency status
                    efficiency_status = governor_recommendations.get('efficiency_status', {})
                    if efficiency_status.get('trend') == 'declining':
                        analysis["health_status"] = "attention_needed"
                        pattern_recommendations.append("Memory access efficiency declining - optimization needed")
                    elif efficiency_status.get('success_rate', 1.0) < 0.8:
                        pattern_recommendations.append(f"Memory success rate low: {efficiency_status['success_rate']:.1%}")
                    
                except Exception as e:
                    self.logger.warning(f"Pattern analysis failed: {e}")
            
            # Phase 2 Enhancement: Add cluster-based analysis
            cluster_recommendations = []
            if self.memory_clusterer:
                try:
                    # Get cluster-based optimization recommendations
                    cluster_opts = self.memory_clusterer.get_cluster_optimization_recommendations()
                    analysis["cluster_analysis"] = {
                        "total_recommendations": len(cluster_opts),
                        "high_priority_count": len([r for r in cluster_opts if r.get('priority') == 'high']),
                        "cluster_summary": self.memory_clusterer.get_clustering_summary()
                    }
                    
                    # Extract high-priority cluster recommendations
                    for rec in cluster_opts[:3]:  # Top 3 recommendations
                        if rec.get('priority') in ['high', 'medium']:
                            cluster_recommendations.append(f"Cluster Analysis: {rec['reason']}")
                    
                except Exception as e:
                    self.logger.warning(f"Cluster analysis failed: {e}")
            
            # Traditional memory health analysis
            total_size = base_status["total_size_mb"]
            if total_size > 2000:  # Over 2GB
                analysis["health_status"] = "critical"
                analysis["cleanup_needed"] = True
                analysis["recommendations"].append("Immediate cleanup recommended")
            elif total_size > 1000:  # Over 1GB
                analysis["health_status"] = "attention_needed"
                analysis["recommendations"].append("Consider cleanup soon")
            
            # Check for too many temporary files
            temp_files = base_status["classifications"].get("temporary_purge", {}).get("file_count", 0)
            if temp_files > 100:
                analysis["recommendations"].append("High number of temporary files detected")
            
            # Combine traditional, pattern-based, and cluster-based recommendations
            analysis["recommendations"].extend(pattern_recommendations)
            analysis["recommendations"].extend(cluster_recommendations)
            
            base_status["governor_analysis"] = analysis
            return base_status
            
        except Exception as e:
            self.logger.error(f"Memory status check failed: {e}")
            return {"status": "failed", "error": str(e)}

    def record_memory_access(self, access_info: Dict[str, Any]) -> None:
        """
        Record memory access for pattern analysis (Phase 1 enhancement).
        
        This enables the Governor to learn from memory access patterns
        and provide immediate optimization recommendations.
        """
        if self.pattern_optimizer:
            try:
                # Add timing information if not present
                if 'timestamp' not in access_info:
                    access_info['timestamp'] = time.time()
                
                self.pattern_optimizer.record_memory_access(access_info)
                
                # Log significant pattern changes
                pattern_summary = self.pattern_optimizer.get_pattern_summary()
                if pattern_summary['total_patterns'] > 0:
                    self.logger.debug(f"Memory patterns detected: {pattern_summary['total_patterns']}")
                
            except Exception as e:
                self.logger.warning(f"Failed to record memory access pattern: {e}")

    def optimize_memory_patterns(self) -> Dict[str, Any]:
        """
        Phase 1 Enhanced Memory Pattern Optimization
        
        Provides immediate Governor recommendations based on detected
        memory access patterns. This is the first phase enhancement
        that delivers immediate value.
        """
        if not self.pattern_optimizer:
            return {"status": "optimizer_unavailable", "optimizations": []}
        
        try:
            # Get comprehensive pattern analysis and recommendations
            recommendations = self.pattern_optimizer.get_governor_recommendations()
            
            # Apply high-priority optimizations immediately
            applied_optimizations = []
            for action in recommendations.get('immediate_actions', []):
                if action.get('urgency') == 'high':
                    # In Phase 1, we log and recommend rather than auto-apply
                    optimization_result = {
                        'action': action['action'],
                        'reason': action['reason'],
                        'status': 'recommended',
                        'priority': action['urgency']
                    }
                    applied_optimizations.append(optimization_result)
                    
                    self.logger.info(f"🧠 Pattern Optimization Recommended: {action['action']} - {action['reason']}")
            
            # Get pattern summary for decision logging
            pattern_summary = self.pattern_optimizer.get_pattern_summary()
            
            # Log Governor decision with pattern analysis
            governor_decision = {
                "decision_type": "memory_pattern_optimization",
                "patterns_detected": pattern_summary['total_patterns'],
                "optimization_potential": pattern_summary['top_optimization_potential'],
                "efficiency_status": pattern_summary['efficiency_status'],
                "immediate_recommendations": len(applied_optimizations),
                "applied_optimizations": applied_optimizations,
                "timestamp": time.time()
            }
            
            self.log_governor_decision(governor_decision)
            
            # Return comprehensive optimization status
            optimization_status = {
                "status": "analysis_complete",
                "patterns_analyzed": pattern_summary,
                "recommendations": recommendations,
                "applied_optimizations": applied_optimizations,
                "next_analysis_recommended": time.time() + 3600,  # 1 hour
                "governor_decision": governor_decision
            }
            
            self.logger.info(f"🎯 Governor Memory Pattern Analysis Complete: "
                           f"{pattern_summary['total_patterns']} patterns, "
                           f"{len(applied_optimizations)} recommendations")
            
            return optimization_status
            
        except Exception as e:
            error_msg = f"Memory pattern optimization failed: {e}"
            self.logger.error(error_msg)
            
            # Log failed attempt
            self.log_governor_decision({
                "decision_type": "memory_pattern_optimization_failed",
                "error": str(e),
                "timestamp": time.time()
            })
            
            return {"status": "failed", "error": error_msg}

    def create_intelligent_memory_clusters(self) -> Dict[str, Any]:
        """
        Phase 2 Enhanced Memory Clustering
        
        Creates intelligent, dynamic memory clusters that replace static 4-tier
        system with relationship-based, adaptive clustering that improves
        Governor decision-making with cluster intelligence.
        """
        if not self.memory_clusterer:
            return {"status": "clusterer_unavailable", "clusters": {}}
        
        try:
            # Get current memory data for clustering
            memory_status = self.get_memory_status()
            if memory_status.get("status") == "failed":
                return {"status": "memory_data_unavailable", "error": "Could not get memory status"}
            
            # Prepare memory data for clustering
            memories = []
            for classification, class_data in memory_status.get("classifications", {}).items():
                # Convert classification data to memory records
                # This is a simplified version - full implementation would have detailed file data
                for i in range(class_data.get("file_count", 0)):
                    memories.append({
                        "file_path": f"{classification}_file_{i}",
                        "memory_type": classification.upper(),
                        "classification": classification,
                        "size_mb": class_data.get("total_size_mb", 0) / max(class_data.get("file_count", 1), 1)
                    })
            
            # Get access patterns from pattern optimizer
            access_patterns = []
            if self.pattern_optimizer and hasattr(self.pattern_optimizer, 'access_history'):
                access_patterns = list(self.pattern_optimizer.access_history)
            
            # Create intelligent clusters
            clusters = self.memory_clusterer.create_intelligent_clusters(memories, access_patterns)
            cluster_summary = self.memory_clusterer.get_clustering_summary()
            cluster_recommendations = self.memory_clusterer.get_cluster_optimization_recommendations()
            
            # Log Governor decision with cluster analysis
            governor_decision = {
                "decision_type": "intelligent_memory_clustering",
                "clusters_created": len(clusters),
                "cluster_types": cluster_summary.get("cluster_types", {}),
                "total_clustered_memories": cluster_summary.get("total_clustered_memories", 0),
                "avg_cluster_health": cluster_summary.get("cluster_health", {}).get("avg_health_score", 0),
                "optimization_recommendations": len(cluster_recommendations),
                "timestamp": time.time()
            }
            
            self.log_governor_decision(governor_decision)
            
            # Return comprehensive clustering status
            clustering_status = {
                "status": "clustering_complete",
                "clusters_created": clusters,
                "cluster_summary": cluster_summary,
                "optimization_recommendations": cluster_recommendations,
                "governor_decision": governor_decision,
                "next_clustering_recommended": time.time() + 7200,  # 2 hours
                "enhancement_level": "Phase 2 - Intelligent Hierarchical Clustering"
            }
            
            self.logger.info(f"🗂️ Governor Intelligent Clustering Complete: "
                           f"{len(clusters)} clusters created, "
                           f"{len(cluster_recommendations)} optimizations identified")
            
            return clustering_status
            
        except Exception as e:
            error_msg = f"Intelligent memory clustering failed: {e}"
            self.logger.error(error_msg)
            
            # Log failed attempt
            self.log_governor_decision({
                "decision_type": "intelligent_clustering_failed",
                "error": str(e),
                "timestamp": time.time()
            })
            
            return {"status": "failed", "error": error_msg}

    def get_cluster_based_retention_policy(self, memory_id: str) -> Dict[str, Any]:
        """
        Get cluster-based retention policy for a memory (Phase 2 enhancement)
        
        This replaces static 4-tier thresholds with dynamic, cluster-aware
        retention decisions based on relationships and cluster health.
        """
        if not self.memory_clusterer:
            return {
                "policy": "fallback_static",
                "retention_priority": 0.5,
                "reason": "No cluster information available"
            }
        
        try:
            # Get cluster information for this memory
            cluster_info = self.memory_clusterer.get_memory_cluster_info(memory_id)
            
            if cluster_info.get("status") == "unclustered":
                # Unclustered memories get default policy
                return {
                    "policy": "unclustered_default",
                    "retention_priority": 0.4,
                    "reason": "Memory not part of any cluster"
                }
            
            # Use cluster-based retention priority
            max_retention = cluster_info.get("max_retention_priority", 0.5)
            avg_relationship = cluster_info.get("avg_relationship_strength", 0.5)
            cluster_count = cluster_info.get("cluster_count", 0)
            
            # Calculate dynamic retention priority
            dynamic_priority = max_retention
            
            # Bonus for being in multiple clusters (important connections)
            if cluster_count > 1:
                dynamic_priority = min(dynamic_priority + 0.1, 0.99)
            
            # Bonus for strong relationships
            if avg_relationship > 0.7:
                dynamic_priority = min(dynamic_priority + 0.05, 0.99)
            
            # Get cluster-specific policies
            policies = []
            for cluster_data in cluster_info.get("clusters", []):
                cluster_type = cluster_data.get("cluster_type", "unknown")
                health_score = cluster_data.get("health_score", 0.5)
                
                if cluster_type == "causal_chain":
                    policies.append("causal_chain_protection")
                elif cluster_type == "performance_cluster" and health_score > 0.7:
                    policies.append("performance_optimization_priority")
                elif cluster_type == "cross_session":
                    policies.append("cross_session_preservation")
            
            return {
                "policy": "cluster_based_dynamic",
                "retention_priority": dynamic_priority,
                "cluster_count": cluster_count,
                "relationship_strength": avg_relationship,
                "cluster_policies": policies,
                "reason": f"Cluster-based priority from {cluster_count} clusters"
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to get cluster-based retention policy: {e}")
            return {
                "policy": "error_fallback", 
                "retention_priority": 0.5,
                "reason": f"Error in cluster analysis: {e}"
            }
    
    def schedule_memory_maintenance(self, interval_hours: int = 24) -> bool:
        """Schedule regular memory maintenance with pattern optimization."""
        try:
            # Phase 1 Enhancement: Include pattern optimization in scheduled maintenance
            self.logger.info(f"Memory maintenance scheduled every {interval_hours} hours")
            
            # Trigger immediate pattern optimization analysis
            if self.pattern_optimizer:
                pattern_optimization_result = self.optimize_memory_patterns()
                self.logger.info(f"Pattern optimization integrated into maintenance schedule")
            
            # Log the enhanced scheduling decision
            self.log_governor_decision({
                "decision_type": "enhanced_memory_maintenance_schedule",
                "interval_hours": interval_hours,
                "pattern_optimization_enabled": self.pattern_optimizer is not None,
                "next_maintenance": time.time() + (interval_hours * 3600),
                "enhancement_phase": "Phase 1 - Pattern Recognition"
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to schedule enhanced memory maintenance: {e}")
            return False

    def trigger_intelligent_memory_analysis(self) -> Dict[str, Any]:
        """
        Trigger comprehensive memory analysis with pattern optimization and clustering.
        
        This method combines traditional memory analysis with Phase 1
        pattern recognition and Phase 2 hierarchical clustering for
        maximum Governor decision enhancement.
        """
        analysis_results = {
            "timestamp": time.time(),
            "analysis_type": "comprehensive_with_patterns_and_clusters",
            "results": {}
        }
        
        try:
            # 1. Traditional memory status
            memory_status = self.get_memory_status()
            analysis_results["results"]["memory_status"] = memory_status
            
            # 2. Pattern optimization analysis (Phase 1)
            if self.pattern_optimizer:
                pattern_analysis = self.optimize_memory_patterns()
                analysis_results["results"]["pattern_optimization"] = pattern_analysis
            else:
                analysis_results["results"]["pattern_optimization"] = {
                    "status": "pattern_optimizer_unavailable"
                }
            
            # 3. Hierarchical clustering analysis (Phase 2)  
            if self.memory_clusterer:
                cluster_analysis = self.create_intelligent_memory_clusters()
                analysis_results["results"]["cluster_analysis"] = cluster_analysis
            else:
                analysis_results["results"]["cluster_analysis"] = {
                    "status": "clusterer_unavailable"
                }
            
            # 4. Extract key insights from combined analysis
            analysis_results["key_insights"] = self._extract_combined_insights(analysis_results)
            
            # 5. Generate Governor recommendations based on combined analysis
            governor_recommendations = self._generate_integrated_recommendations(analysis_results)
            analysis_results["governor_recommendations"] = governor_recommendations
            
            # Log comprehensive decision
            self.log_governor_decision({
                "decision_type": "intelligent_memory_analysis_phase2",
                "analysis_results": analysis_results,
                "recommendations_generated": len(governor_recommendations),
                "enhancement_level": "Phase 2 - Pattern + Cluster Intelligence"
            })
            
            insights = analysis_results["key_insights"]
            self.logger.info(f"🧠 Enhanced Memory Analysis Complete - "
                           f"{insights.get('total_patterns_detected', 0)} patterns, "
                           f"{insights.get('total_clusters_created', 0)} clusters, "
                           f"{len(governor_recommendations)} recommendations")
            
            return analysis_results
            
        except Exception as e:
            error_msg = f"Intelligent memory analysis failed: {e}"
            self.logger.error(error_msg)
            analysis_results["status"] = "failed"
            analysis_results["error"] = error_msg
            return analysis_results

    def _extract_combined_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key insights from pattern + cluster analysis"""
        insights = {
            "total_patterns_detected": 0,
            "total_clusters_created": 0,
            "optimization_potential": 0.0,
            "immediate_actions_needed": 0,
            "efficiency_trend": "unknown",
            "cluster_health": 0.0,
            "governor_confidence": 0.5
        }
        
        try:
            # Extract pattern insights
            pattern_data = analysis_results.get("results", {}).get("pattern_optimization", {})
            if pattern_data.get("status") == "analysis_complete":
                patterns_analyzed = pattern_data.get("patterns_analyzed", {})
                insights["total_patterns_detected"] = patterns_analyzed.get("total_patterns", 0)
                insights["optimization_potential"] = patterns_analyzed.get("top_optimization_potential", 0.0)
                
                recommendations = pattern_data.get("recommendations", {})
                insights["immediate_actions_needed"] = len(recommendations.get("immediate_actions", []))
                insights["efficiency_trend"] = recommendations.get("efficiency_status", {}).get("trend", "unknown")
            
            # Extract cluster insights
            cluster_data = analysis_results.get("results", {}).get("cluster_analysis", {})
            if cluster_data.get("status") == "clustering_complete":
                cluster_summary = cluster_data.get("cluster_summary", {})
                insights["total_clusters_created"] = cluster_summary.get("total_clusters", 0)
                insights["cluster_health"] = cluster_summary.get("cluster_health", {}).get("avg_health_score", 0.0)
                
                # Add cluster-based optimization count
                cluster_recommendations = cluster_data.get("optimization_recommendations", [])
                insights["cluster_optimizations_available"] = len(cluster_recommendations)
            
            # Calculate enhanced Governor confidence
            confidence = 0.5  # Base confidence
            
            if insights["total_patterns_detected"] > 10:
                confidence += 0.2  # Pattern recognition bonus
            
            if insights["total_clusters_created"] > 3:
                confidence += 0.2  # Clustering bonus
            
            if insights["cluster_health"] > 0.7:
                confidence += 0.1  # Healthy clusters bonus
            
            if insights["optimization_potential"] > 0.5:
                confidence += 0.1  # High optimization potential bonus
            
            insights["governor_confidence"] = min(confidence, 0.95)
            
        except Exception as e:
            self.logger.warning(f"Failed to extract combined insights: {e}")
        
        return insights

    def handle_api_error(self, error_response: Dict[str, Any], game_id: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle API errors and provide Governor recommendations for recovery.
        
        Args:
            error_response: Error response from API call
            game_id: Current game ID
            context: Current game context
            
        Returns:
            Governor recommendation for error recovery or None
        """
        error_text = error_response.get('error', '')
        error_status = error_response.get('status', 0)
        
        # Handle GAME_NOT_STARTED_ERROR specifically
        if 'GAME_NOT_STARTED_ERROR' in error_text or 'has not been started' in error_text:
            self.logger.info(f"🎯 Governor detected GAME_NOT_STARTED_ERROR for {game_id} - recommending RESET")
            
            return {
                'recommended_action': 'RESET',
                'confidence': 0.95,
                'reasoning': f'Game {game_id} is available but not started - RESET required to begin playing',
                'error_type': 'GAME_NOT_STARTED_ERROR',
                'recovery_action': 'send_reset_command',
                'meta_analysis': {
                    'error_detected': True,
                    'error_type': 'GAME_NOT_STARTED_ERROR',
                    'game_id': game_id,
                    'requires_reset': True
                }
            }
        
        # Handle other 400 errors that might need RESET
        elif error_status == 400 and ('not available' in error_text.lower() or 'invalid game' in error_text.lower()):
            self.logger.warning(f"🎯 Governor detected game availability issue for {game_id} - considering RESET")
            
            return {
                'recommended_action': 'RESET',
                'confidence': 0.7,
                'reasoning': f'Game {game_id} appears unavailable or invalid - RESET may resolve the issue',
                'error_type': 'GAME_AVAILABILITY_ERROR',
                'recovery_action': 'send_reset_command',
                'meta_analysis': {
                    'error_detected': True,
                    'error_type': 'GAME_AVAILABILITY_ERROR',
                    'game_id': game_id,
                    'requires_reset': True
                }
            }
        
        # Handle authentication errors
        elif error_status == 401:
            self.logger.critical(f"🎯 Governor detected authentication error for {game_id}")
            
            return {
                'recommended_action': 'STOP',
                'confidence': 1.0,
                'reasoning': 'Authentication failed - check API key configuration',
                'error_type': 'AUTHENTICATION_ERROR',
                'recovery_action': 'check_api_key',
                'meta_analysis': {
                    'error_detected': True,
                    'error_type': 'AUTHENTICATION_ERROR',
                    'game_id': game_id,
                    'requires_manual_intervention': True
                }
            }
        
        # Handle rate limiting
        elif error_status == 429:
            self.logger.warning(f"🎯 Governor detected rate limiting for {game_id}")
            
            return {
                'recommended_action': 'WAIT',
                'confidence': 0.9,
                'reasoning': 'Rate limited - wait before retrying',
                'error_type': 'RATE_LIMIT_ERROR',
                'recovery_action': 'wait_and_retry',
                'meta_analysis': {
                    'error_detected': True,
                    'error_type': 'RATE_LIMIT_ERROR',
                    'game_id': game_id,
                    'requires_wait': True
                }
            }
        
        # For other errors, return None to let the system handle normally
        return None

    def analyze_level_completion_win(self, session_result: Dict[str, Any], game_id: str) -> Dict[str, Any]:
        """
        Analyze level completion wins and provide Governor recommendations for memory prioritization.
        
        Args:
            session_result: Session results including level progression data
            game_id: Current game ID
            
        Returns:
            Governor recommendation for level completion handling
        """
        if not session_result.get('level_progressed', False):
            return None
            
        levels_completed = session_result.get('current_level', 0)
        
        # Calculate win value based on level completion
        if levels_completed >= 2:
            win_value = 'CRITICAL'  # Multiple levels = critical win
            confidence = 0.95
            memory_priority = 'PERMANENT'
        elif levels_completed == 1:
            win_value = 'HIGH'      # Single level = high value win
            confidence = 0.85
            memory_priority = 'PROTECTED'
        else:
            win_value = 'MODERATE'  # Partial progress
            confidence = 0.7
            memory_priority = 'ENHANCED'
        
        self.logger.info(f"🎯 Governor detected level completion win: {levels_completed} levels completed for {game_id}")
        
        return {
            'recommended_action': 'PRIORITIZE_MEMORIES',
            'confidence': confidence,
            'reasoning': f'Level completion win detected: {levels_completed} levels completed',
            'priority': 'high',
            'governor_decision': True,
            'win_analysis': {
                'levels_completed': levels_completed,
                'win_value': win_value,
                'memory_priority': memory_priority,
                'should_preserve_patterns': True,
                'should_analyze_strategy': True
            },
            'meta_analysis': {
                'win_detected': True,
                'win_type': 'LEVEL_COMPLETION',
                'levels_completed': levels_completed,
                'game_id': game_id,
                'requires_memory_prioritization': True
            }
        }

    def analyze_full_game_win(self, session_result: Dict[str, Any], game_id: str) -> Dict[str, Any]:
        """
        Analyze full game wins and provide Governor recommendations for ultimate memory prioritization.
        
        Args:
            session_result: Session results including full game completion data
            game_id: Current game ID
            
        Returns:
            Governor recommendation for full game win handling
        """
        if not session_result.get('full_game_win', False):
            return None
        
        # Full game win is the ultimate achievement
        win_value = 'ULTIMATE'  # Complete game = ultimate win
        confidence = 1.0         # Maximum confidence
        memory_priority = 'IMMORTAL'  # Never delete these memories
        
        self.logger.info(f"🏆 Governor detected FULL GAME WIN: Complete game victory for {game_id}")
        
        return {
            'recommended_action': 'IMMORTALIZE_MEMORIES',
            'confidence': confidence,
            'reasoning': 'Full game win detected - complete puzzle solution achieved',
            'priority': 'critical',
            'governor_decision': True,
            'win_analysis': {
                'win_type': 'FULL_GAME_WIN',
                'win_value': win_value,
                'memory_priority': memory_priority,
                'should_preserve_patterns': True,
                'should_analyze_strategy': True,
                'should_create_master_template': True,  # Create master strategy template
                'should_share_knowledge': True          # Share with other agents
            },
            'meta_analysis': {
                'win_detected': True,
                'win_type': 'FULL_GAME_WIN',
                'game_id': game_id,
                'requires_immortal_memory_prioritization': True,
                'requires_master_template_creation': True,
                'requires_knowledge_sharing': True
            }
        }

    def _generate_integrated_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate integrated recommendations from memory, pattern, and cluster analysis"""
        recommendations = []
        
        try:
            # Extract data from analysis results
            memory_status = analysis_results.get("results", {}).get("memory_status", {})
            pattern_analysis = analysis_results.get("results", {}).get("pattern_optimization", {})
            cluster_analysis = analysis_results.get("results", {}).get("cluster_analysis", {})
            key_insights = analysis_results.get("key_insights", {})
            
            # Memory-based recommendations (traditional)
            governor_analysis = memory_status.get("governor_analysis", {})
            if governor_analysis.get("cleanup_needed"):
                recommendations.append({
                    "type": "memory_cleanup",
                    "priority": "high",
                    "action": "trigger_memory_cleanup",
                    "reason": "Memory usage exceeds threshold",
                    "source": "traditional_analysis"
                })
            
            # Pattern-based recommendations (Phase 1)
            if pattern_analysis.get("status") == "analysis_complete":
                pattern_recommendations = pattern_analysis.get("recommendations", {})
                
                for immediate_action in pattern_recommendations.get("immediate_actions", []):
                    recommendations.append({
                        "type": "pattern_optimization",
                        "priority": immediate_action.get("urgency", "medium"),
                        "action": immediate_action["action"],
                        "reason": immediate_action["reason"],
                        "source": "pattern_analysis_phase1"
                    })
                
                # High-potential pattern optimizations
                for optimization in pattern_recommendations.get("priority_optimizations", []):
                    if optimization.get("potential", 0) > 0.6:
                        recommendations.append({
                            "type": "high_potential_optimization",
                            "priority": "medium",
                            "action": optimization["action"],
                            "reason": f"High optimization potential: {optimization['expected_improvement']}",
                            "source": "pattern_analysis_phase1"
                        })
            
            # Cluster-based recommendations (Phase 2)
            if cluster_analysis.get("status") == "clustering_complete":
                cluster_recommendations = cluster_analysis.get("optimization_recommendations", [])
                
                for cluster_rec in cluster_recommendations[:5]:  # Top 5 cluster recommendations
                    recommendations.append({
                        "type": "cluster_optimization",
                        "priority": cluster_rec.get("priority", "medium"),
                        "action": cluster_rec["action"],
                        "reason": cluster_rec["reason"],
                        "source": "hierarchical_clustering_phase2",
                        "cluster_id": cluster_rec.get("cluster_id", "unknown"),
                        "expected_improvement": cluster_rec.get("expected_improvement", "Performance improvement")
                    })
            
            # Combined insights recommendations
            if key_insights.get("efficiency_trend") == "declining":
                recommendations.append({
                    "type": "efficiency_intervention",
                    "priority": "high",
                    "action": "investigate_performance_degradation",
                    "reason": (f"Memory efficiency declining - "
                             f"{key_insights.get('total_patterns_detected', 0)} patterns, "
                             f"{key_insights.get('total_clusters_created', 0)} clusters analyzed"),
                    "source": "integrated_analysis_phase2"
                })
            
            # High confidence enhancement opportunities
            if key_insights.get("governor_confidence", 0) > 0.8:
                total_optimizations = (key_insights.get("immediate_actions_needed", 0) + 
                                     key_insights.get("cluster_optimizations_available", 0))
                
                if total_optimizations >= 3:
                    recommendations.append({
                        "type": "comprehensive_optimization",
                        "priority": "medium",
                        "action": "implement_comprehensive_memory_optimization",
                        "reason": (f"High confidence ({key_insights['governor_confidence']:.2f}) "
                                 f"with {total_optimizations} optimization opportunities"),
                        "source": "integrated_analysis_phase2"
                    })
            
            # Cluster health interventions
            if key_insights.get("cluster_health", 0) < 0.5:
                recommendations.append({
                    "type": "cluster_health_intervention",
                    "priority": "medium",
                    "action": "improve_cluster_health",
                    "reason": f"Low cluster health: {key_insights['cluster_health']:.2f}",
                    "source": "hierarchical_clustering_phase2"
                })
            
            # Sort by priority
            priority_order = {"high": 0, "medium": 1, "low": 2}
            recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "medium"), 1))
            
        except Exception as e:
            self.logger.warning(f"Failed to generate integrated recommendations: {e}")
        
        return recommendations

    def log_governor_decision(self, decision_data: Dict[str, Any]) -> None:
        """
        Log a Governor decision for tracking and analysis.
        
        This method records all Governor decisions for pattern analysis,
        performance tracking, and cross-session learning.
        """
        try:
            # Add timestamp and decision ID if not present
            if 'timestamp' not in decision_data:
                decision_data['timestamp'] = time.time()
            
            if 'decision_id' not in decision_data:
                decision_data['decision_id'] = f"gov_{int(time.time() * 1000)}"
            
            # Add to decision history
            self.decision_history.append(decision_data)
            
            # Log to file if configured
            if self.log_file:
                try:
                    with open(self.log_file, 'a') as f:
                        f.write(json.dumps(decision_data) + '\n')
                except Exception as e:
                    self.logger.warning(f"Failed to write decision to log file: {e}")
            
            # Log to system logger with appropriate level
            decision_type = decision_data.get('decision_type', 'unknown')
            if decision_data.get('priority') == 'high' or 'failed' in decision_type:
                self.logger.warning(f"Governor Decision: {decision_type} - {decision_data}")
            else:
                self.logger.info(f"Governor Decision: {decision_type}")
                self.logger.debug(f"Decision details: {decision_data}")
            
        except Exception as e:
            self.logger.error(f"Failed to log Governor decision: {e}")
    
    # ==========================================
    # Phase 3: Architect Evolution Integration
    # ==========================================
    
    def trigger_architect_analysis(self) -> Dict[str, Any]:
        """
        Trigger Architect Evolution Engine analysis of Governor intelligence.
        
        Phase 3: Enable Architect to analyze Governor pattern/cluster data
        for autonomous system evolution and architectural improvements.
        """
        if not self.architect_engine:
            return {
                "status": "unavailable",
                "message": "Architect Evolution Engine not available"
            }
        
        try:
            self.logger.info("🏗️ Triggering Architect analysis of Governor intelligence")
            
            # Gather Governor intelligence data for Architect analysis
            governor_patterns = self._get_pattern_intelligence_data()
            governor_clusters = self._get_cluster_intelligence_data() 
            memory_status = self.get_memory_status()
            
            # Trigger Architect analysis
            architectural_insights = self.architect_engine.analyze_governor_intelligence(
                governor_patterns, governor_clusters, memory_status
            )
            
            # Log the Governor decision to trigger Architect analysis
            self.log_governor_decision({
                "decision_type": "architect_intelligence_analysis",
                "insights_generated": len(architectural_insights),
                "architect_status": "analysis_complete",
                "enhancement_phase": "Phase 3 - Architect Evolution"
            })
            
            result = {
                "status": "success",
                "insights_generated": len(architectural_insights),
                "architectural_insights": [
                    {
                        "insight_type": insight.insight_type,
                        "priority": insight.priority,
                        "confidence": insight.confidence,
                        "description": insight.description[:100] + "..." if len(insight.description) > 100 else insight.description
                    }
                    for insight in architectural_insights
                ],
                "message": f"Architect analysis complete: {len(architectural_insights)} insights generated"
            }
            
            self.logger.info(f"🏗️ Architect Analysis Complete: {len(architectural_insights)} insights generated")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to trigger Architect analysis: {e}")
            return {
                "status": "error",
                "error": str(e),
                "message": f"Architect analysis failed: {e}"
            }
    
    def execute_autonomous_evolution(self) -> Dict[str, Any]:
        """
        Execute autonomous evolution based on Architect's analysis.
        
        Phase 3: Allow Architect to autonomously evolve system architecture
        based on Governor intelligence analysis.
        """
        if not self.architect_engine:
            return {
                "status": "unavailable", 
                "message": "Architect Evolution Engine not available"
            }
        
        try:
            self.logger.info("🚀 Executing autonomous architecture evolution")
            
            # Execute Architect's autonomous evolution
            evolution_result = self.architect_engine.execute_autonomous_evolution()
            
            # Log the evolution attempt
            self.log_governor_decision({
                "decision_type": "autonomous_evolution_execution",
                "evolution_status": evolution_result.get("status", "unknown"),
                "evolution_success": evolution_result.get("success", False),
                "strategy_executed": evolution_result.get("strategy_id"),
                "enhancement_phase": "Phase 3 - Autonomous Evolution"
            })
            
            if evolution_result.get("success"):
                self.logger.info(f"🚀 Evolution Success: {evolution_result.get('message', 'No details')}")
            else:
                self.logger.info(f"🚀 Evolution Status: {evolution_result.get('message', 'No details')}")
            
            return evolution_result
            
        except Exception as e:
            self.logger.error(f"Failed to execute autonomous evolution: {e}")
            return {
                "status": "error",
                "error": str(e), 
                "success": False,
                "message": f"Evolution execution failed: {e}"
            }
    
    def get_architect_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of Architect Evolution Engine.
        
        Phase 3: Monitor Architect's autonomous evolution capabilities
        and architectural insight generation.
        """
        if not self.architect_engine:
            return {
                "status": "unavailable",
                "message": "Architect Evolution Engine not available"
            }
        
        try:
            # Get detailed Architect status
            architect_status = self.architect_engine.get_evolution_status()
            
            # Check if Architect should analyze Governor data
            should_analyze = self.architect_engine.should_analyze_governor_data()
            
            # Get Architect recommendations
            recommendations = self.architect_engine.get_architect_recommendations()
            
            return {
                "status": "operational",
                "architect_engine_status": architect_status,
                "analysis_needed": should_analyze,
                "recommendations_count": len(recommendations),
                "top_recommendations": recommendations[:3],  # Top 3 recommendations
                "autonomous_evolution_enabled": architect_status.get("autonomous_evolution", False),
                "recent_evolutions": architect_status.get("recent_evolution_history", [])
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get Architect status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "message": f"Architect status check failed: {e}"
            }
    
    def _get_pattern_intelligence_data(self) -> Dict[str, Any]:
        """Get Governor's pattern intelligence data for Architect analysis."""
        if not self.pattern_optimizer:
            return {}
        
        try:
            # Analyze current memory patterns
            pattern_analysis = self.pattern_optimizer.analyze_patterns()
            recommendations = self.pattern_optimizer.generate_governor_recommendations()
            
            return {
                "patterns_detected": len(pattern_analysis.get("patterns", [])),
                "optimization_potential": pattern_analysis.get("optimization_potential", 0.0),
                "confidence": pattern_analysis.get("confidence", 0.0),
                "pattern_types": pattern_analysis.get("pattern_summary", {}),
                "governor_recommendations": len(recommendations),
                "analysis_timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to get pattern intelligence data: {e}")
            return {}
    
    def _get_cluster_intelligence_data(self) -> Dict[str, Any]:
        """Get Governor's cluster intelligence data for Architect analysis."""
        if not self.memory_clusterer:
            return {}
        
        try:
            # Get current cluster state
            cluster_analysis = self.memory_clusterer.analyze_cluster_health()
            optimization_recommendations = self.memory_clusterer.generate_optimization_recommendations()
            
            return {
                "clusters_created": len(cluster_analysis.get("clusters", {})),
                "average_health": cluster_analysis.get("average_health", 0.0),
                "optimization_recommendations": optimization_recommendations,
                "cluster_types": cluster_analysis.get("cluster_type_distribution", {}),
                "total_clustered_memories": cluster_analysis.get("total_memories", 0),
                "analysis_timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to get cluster intelligence data: {e}")
            return {}
    
    def should_trigger_architect_analysis(self) -> bool:
        """
        Check if Architect analysis should be triggered based on Governor state.
        
        Phase 3: Intelligent triggering of Architect analysis when Governor
        has sufficient intelligence data for meaningful architectural insights.
        """
        if not self.architect_engine:
            return False
        
        # Check if Architect thinks it's time to analyze
        if self.architect_engine.should_analyze_governor_data():
            return True
        
        # Additional Governor-specific triggers
        pattern_data = self._get_pattern_intelligence_data()
        cluster_data = self._get_cluster_intelligence_data()
        
        # Trigger if we have rich pattern and cluster data
        patterns_sufficient = pattern_data.get("patterns_detected", 0) >= 10
        clusters_sufficient = cluster_data.get("clusters_created", 0) >= 5
        
        return patterns_sufficient and clusters_sufficient

    # ==========================================
    # Phase 4: Performance Optimization Integration
    # ==========================================
    
    def trigger_comprehensive_performance_analysis(self) -> Dict[str, Any]:
        """
        Trigger comprehensive performance analysis using all Phase 1-4 intelligence.
        
        Phase 4: Leverage pattern recognition, clustering, architectural insights,
        and performance optimization for maximum system performance.
        """
        if not self.performance_engine:
            return {
                "status": "unavailable",
                "message": "Performance Optimization Engine not available"
            }
        
        try:
            self.logger.info("⚡ Triggering comprehensive performance analysis (Phase 1-4)")
            
            # Gather all intelligence data for performance analysis
            governor_patterns = self._get_pattern_intelligence_data()
            governor_clusters = self._get_cluster_intelligence_data()
            architect_insights = self._get_architect_insights_data()
            
            # Trigger comprehensive performance analysis
            performance_analysis = self.performance_engine.analyze_performance_with_intelligence(
                governor_patterns, governor_clusters, architect_insights
            )
            
            # Log the comprehensive analysis decision
            self.log_governor_decision({
                "decision_type": "comprehensive_performance_analysis",
                "analysis_status": performance_analysis.get("status", "unknown"),
                "optimization_opportunities": performance_analysis.get("optimization_opportunities", 0),
                "optimization_strategies": performance_analysis.get("optimization_strategies", 0),
                "enhancement_phase": "Phase 4 - Performance Optimization"
            })
            
            result = {
                "status": "success",
                "performance_analysis": performance_analysis,
                "intelligence_integration": "phases_1_2_3_4_complete",
                "message": f"Comprehensive performance analysis complete using all phases"
            }
            
            self.logger.info(f"⚡ Comprehensive Performance Analysis Complete: {performance_analysis.get('optimization_opportunities', 0)} opportunities")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to trigger comprehensive performance analysis: {e}")
            return {
                "status": "error",
                "error": str(e),
                "message": f"Comprehensive performance analysis failed: {e}"
            }
    
    def execute_performance_optimization(self, optimization_id: str) -> Dict[str, Any]:
        """
        Execute a specific performance optimization strategy.
        
        Phase 4: Execute performance optimizations identified through
        comprehensive Phase 1-4 intelligence analysis.
        """
        if not self.performance_engine:
            return {
                "status": "unavailable",
                "message": "Performance Optimization Engine not available"
            }
        
        try:
            self.logger.info(f"⚡ Executing performance optimization: {optimization_id}")
            
            # Execute the optimization
            optimization_result = self.performance_engine.execute_performance_optimization(optimization_id)
            
            # Log the optimization execution
            self.log_governor_decision({
                "decision_type": "performance_optimization_execution",
                "optimization_id": optimization_id,
                "execution_success": optimization_result.get("success", False),
                "execution_time": optimization_result.get("execution_time", 0),
                "expected_improvements": optimization_result.get("expected_improvements", {}),
                "enhancement_phase": "Phase 4 - Performance Optimization"
            })
            
            if optimization_result.get("success"):
                self.logger.info(f"⚡ Performance optimization successful: {optimization_result.get('message', 'No details')}")
            else:
                self.logger.warning(f"⚡ Performance optimization failed: {optimization_result.get('message', 'No details')}")
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Failed to execute performance optimization: {e}")
            return {
                "status": "error", 
                "error": str(e),
                "success": False,
                "message": f"Performance optimization execution failed: {e}"
            }
    
    def record_system_performance_metrics(
        self,
        component: str,
        throughput: float, 
        latency: float,
        resource_utilization: float,
        **kwargs
    ) -> str:
        """
        Record performance metrics for Governor-monitored system components.
        
        Phase 4: Integrate performance monitoring with Governor decision-making
        to enable real-time performance optimization.
        """
        if not self.performance_engine:
            self.logger.warning("Performance Engine not available - metrics not recorded")
            return "performance_engine_unavailable"
        
        try:
            # Record performance metrics
            metric_id = self.performance_engine.record_performance_metrics(
                component=component,
                throughput=throughput,
                latency=latency,
                resource_utilization=resource_utilization,
                **kwargs
            )
            
            # Log performance metrics recording
            self.log_governor_decision({
                "decision_type": "performance_metrics_recorded",
                "component": component,
                "metric_id": metric_id,
                "throughput": throughput,
                "latency": latency,
                "resource_utilization": resource_utilization,
                "enhancement_phase": "Phase 4 - Performance Monitoring"
            })
            
            self.logger.debug(f"⚡ Performance metrics recorded for {component}: {metric_id}")
            return metric_id
            
        except Exception as e:
            self.logger.error(f"Failed to record performance metrics: {e}")
            return f"error_{int(time.time())}"
    
    def get_comprehensive_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status integrating all Phase 1-4 capabilities.
        
        Phase 4: Unified status view incorporating pattern recognition, clustering,
        architectural evolution, and performance optimization.
        """
        base_status = self.get_system_status()
        current_time = time.time()
        
        # Add Phase 1-4 integration status
        comprehensive_status = {
            **base_status,
            "meta_cognitive_integration": {
                "phase_1_patterns": {
                    "available": self.pattern_optimizer is not None,
                    "status": "operational" if self.pattern_optimizer else "unavailable"
                },
                "phase_2_clustering": {
                    "available": self.memory_clusterer is not None,
                    "status": "operational" if self.memory_clusterer else "unavailable"
                },
                "phase_3_architect": {
                    "available": self.architect_engine is not None,
                    "status": "operational" if self.architect_engine else "unavailable"
                },
                "phase_4_performance": {
                    "available": self.performance_engine is not None,
                    "status": "operational" if self.performance_engine else "unavailable"
                }
            }
        }
        
        # Add performance status if available
        if self.performance_engine:
            try:
                performance_status = self.performance_engine.get_performance_status()
                comprehensive_status["performance_optimization"] = performance_status
            except Exception as e:
                self.logger.warning(f"Failed to get performance status: {e}")
                comprehensive_status["performance_optimization"] = {"status": "error", "error": str(e)}
        
        # Add architect status if available
        if self.architect_engine:
            try:
                architect_status = self.get_architect_status()
                comprehensive_status["architect_evolution"] = architect_status
            except Exception as e:
                self.logger.warning(f"Failed to get architect status: {e}")
                comprehensive_status["architect_evolution"] = {"status": "error", "error": str(e)}
        
        return comprehensive_status
    
    def _get_architect_insights_data(self) -> List[Dict[str, Any]]:
        """Get Architect's insights data for performance analysis."""
        if not self.architect_engine:
            return []
        
        try:
            # Get recent architectural insights
            recommendations = self.architect_engine.get_architect_recommendations()
            
            return [
                {
                    "insight_type": rec.get("title", "unknown").replace(" ", "_").lower(),
                    "priority": rec.get("priority", 0.0),
                    "confidence": rec.get("confidence", 0.0),
                    "expected_impact": rec.get("expected_benefits", {}),
                    "description": rec.get("description", "")
                }
                for rec in recommendations
            ]
            
        except Exception as e:
            self.logger.warning(f"Failed to get architect insights data: {e}")
            return []
    
    def should_trigger_performance_optimization(self) -> bool:
        """
        Check if performance optimization should be triggered.
        
        Phase 4: Intelligent triggering of performance optimization when
        sufficient intelligence data is available from all phases.
        """
        if not self.performance_engine:
            return False
        
        # Check if we have sufficient intelligence data
        pattern_data = self._get_pattern_intelligence_data()
        cluster_data = self._get_cluster_intelligence_data()
        architect_insights = self._get_architect_insights_data()
        
        # Trigger if we have rich data from multiple phases
        patterns_available = pattern_data.get("patterns_detected", 0) >= 5
        clusters_available = cluster_data.get("clusters_created", 0) >= 3
        insights_available = len(architect_insights) >= 1
        
        # At least 2 of 3 phases should have good data
        phases_ready = sum([patterns_available, clusters_available, insights_available])
        
        return phases_ready >= 2
    
    def optimize_system_performance(self) -> Dict[str, Any]:
        """
        Execute comprehensive system performance optimization using all phases.
        
        Phase 4: Master method that coordinates all Phase 1-4 capabilities
        for maximum system performance optimization.
        """
        self.logger.info("🚀 Executing comprehensive system performance optimization")
        
        results = {
            "optimization_type": "comprehensive_phase_1_2_3_4",
            "timestamp": time.time(),
            "phases_executed": [],
            "total_optimizations": 0,
            "performance_improvements": {},
            "status": "in_progress"
        }
        
        try:
            # Phase 1: Pattern-based optimization
            if self.pattern_optimizer:
                pattern_result = self.optimize_memory_patterns()
                results["phases_executed"].append("phase_1_patterns")
                results["total_optimizations"] += pattern_result.get("optimizations_applied", 0)
                self.logger.info("✅ Phase 1 pattern optimization complete")
            
            # Phase 2: Cluster-based optimization  
            if self.memory_clusterer:
                cluster_result = self.create_intelligent_memory_clusters()
                results["phases_executed"].append("phase_2_clustering")
                results["total_optimizations"] += cluster_result.get("clusters_created", 0)
                self.logger.info("✅ Phase 2 cluster optimization complete")
            
            # Phase 3: Architectural evolution
            if self.architect_engine:
                architect_result = self.execute_autonomous_evolution()
                results["phases_executed"].append("phase_3_architect")
                if architect_result.get("success"):
                    results["total_optimizations"] += 1
                self.logger.info("✅ Phase 3 architectural evolution complete")
            
            # Phase 4: Performance optimization
            if self.performance_engine:
                performance_result = self.trigger_comprehensive_performance_analysis()
                results["phases_executed"].append("phase_4_performance")
                if performance_result.get("status") == "success":
                    perf_analysis = performance_result.get("performance_analysis", {})
                    results["total_optimizations"] += perf_analysis.get("optimization_opportunities", 0)
                self.logger.info("✅ Phase 4 performance optimization complete")
            
            # Log comprehensive optimization
            self.log_governor_decision({
                "decision_type": "comprehensive_system_optimization",
                "phases_executed": results["phases_executed"],
                "total_optimizations": results["total_optimizations"],
                "optimization_success": True,
                "enhancement_phase": "Phase 1+2+3+4 Comprehensive"
            })
            
            results["status"] = "success"
            results["message"] = f"Comprehensive optimization complete: {len(results['phases_executed'])} phases, {results['total_optimizations']} optimizations"
            
            self.logger.info(f"🚀 Comprehensive System Optimization Complete: {results['total_optimizations']} optimizations across {len(results['phases_executed'])} phases")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Comprehensive system optimization failed: {e}")
            results["status"] = "error"
            results["error"] = str(e)
            results["message"] = f"Optimization failed: {e}"
            
            return results


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create Governor instance
    governor = MetaCognitiveGovernor("governor_decisions.log")
    
    # Simulate some system activations
    test_cost = CognitiveCost(
        compute_units=10.0,
        memory_operations=5,
        decision_complexity=2.0,
        coordination_overhead=1.0
    )
    
    test_benefit = CognitiveBenefit(
        win_rate_improvement=0.1,
        score_improvement=5.0,
        learning_efficiency=0.2,
        knowledge_transfer=0.1
    )
    
    # Record some activations
    governor.record_system_activation("swarm_intelligence", test_cost, test_benefit)
    governor.record_system_activation("meta_learning_system", test_cost, test_benefit)
    
    # Test recommendation system
    current_performance = {
        'win_rate': 0.6,
        'average_score': 45.2,
        'learning_speed': 0.3
    }
    
    current_config = {
        'enable_swarm': True,
        'salience_mode': 'decay_compression',
        'max_actions_per_game': 500
    }
    
    recommendation = governor.get_recommended_configuration(
        puzzle_type="spatial_reasoning",
        current_performance=current_performance,
        current_config=current_config
    )
    
    if recommendation:
        print(f"🎯 Governor recommendation: {recommendation.rationale}")
        print(f"   Changes: {recommendation.configuration_changes}")
        print(f"   Confidence: {recommendation.confidence:.2f}")
    else:
        print("📊 No changes recommended")
    
    # Show system status
    status = governor.get_system_status()
    print(f"\n📈 Governor Status:")
    print(f"   Decisions made: {status['total_decisions']}")
    print(f"   Success rate: {status['success_rate']:.1%}")
    print(f"   Active systems: {len(status['system_efficiencies'])}")
    print(f"   Top performers: {status['top_performers']}")
