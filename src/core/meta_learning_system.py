"""
Meta-Learning System - Phase 2 Implementation

This system learns how to learn more effectively, optimizes learning processes,
and applies knowledge across different domains.
"""

import asyncio
import logging
import time
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque

from ..database.system_integration import get_system_integration

logger = logging.getLogger(__name__)

class LearningStrategy(Enum):
    """Types of learning strategies."""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    TRANSFER = "transfer"
    META = "meta"
    ADAPTIVE = "adaptive"
    CURRICULUM = "curriculum"

class LearningDomain(Enum):
    """Different learning domains."""
    GAMEPLAY = "gameplay"
    SYSTEM_OPTIMIZATION = "system_optimization"
    ERROR_RECOVERY = "error_recovery"
    CONFIGURATION = "configuration"
    PATTERN_RECOGNITION = "pattern_recognition"
    DECISION_MAKING = "decision_making"

@dataclass
class LearningExperience:
    """Represents a learning experience."""
    experience_id: str
    domain: LearningDomain
    strategy: LearningStrategy
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    success: bool
    learning_time: float
    confidence: float
    timestamp: float
    metadata: Dict[str, Any]

@dataclass
class LearningPattern:
    """Represents a learned pattern."""
    pattern_id: str
    domain: LearningDomain
    pattern_type: str
    pattern_data: Dict[str, Any]
    success_rate: float
    frequency: int
    last_used: float
    confidence: float
    transferability: float

@dataclass
class MetaLearningInsight:
    """Represents a meta-learning insight."""
    insight_id: str
    insight_type: str
    description: str
    domains: List[LearningDomain]
    effectiveness: float
    confidence: float
    timestamp: float
    applications: List[str]

class MetaLearningSystem:
    """
    Meta-Learning System that learns how to learn more effectively.
    
    Features:
    - Learning strategy optimization
    - Cross-domain knowledge transfer
    - Adaptive learning rate adjustment
    - Curriculum learning design
    - Pattern recognition and application
    - Learning effectiveness analysis
    """
    
    def __init__(self):
        self.integration = get_system_integration()
        
        # Learning state
        self.learning_active = False
        self.learning_experiences = deque(maxlen=10000)
        self.learning_patterns = {}
        self.meta_insights = []
        
        # Learning strategies
        self.strategies = {
            LearningStrategy.EXPLORATION: {
                'learning_rate': 0.1,
                'exploration_rate': 0.3,
                'success_rate': 0.0,
                'usage_count': 0
            },
            LearningStrategy.EXPLOITATION: {
                'learning_rate': 0.05,
                'exploration_rate': 0.1,
                'success_rate': 0.0,
                'usage_count': 0
            },
            LearningStrategy.TRANSFER: {
                'learning_rate': 0.15,
                'exploration_rate': 0.2,
                'success_rate': 0.0,
                'usage_count': 0
            },
            LearningStrategy.META: {
                'learning_rate': 0.2,
                'exploration_rate': 0.4,
                'success_rate': 0.0,
                'usage_count': 0
            },
            LearningStrategy.ADAPTIVE: {
                'learning_rate': 0.1,
                'exploration_rate': 0.25,
                'success_rate': 0.0,
                'usage_count': 0
            },
            LearningStrategy.CURRICULUM: {
                'learning_rate': 0.08,
                'exploration_rate': 0.15,
                'success_rate': 0.0,
                'usage_count': 0
            }
        }
        
        # Domain-specific learning
        self.domain_learning = {
            domain: {
                'experiences': deque(maxlen=1000),
                'patterns': {},
                'success_rate': 0.0,
                'learning_curve': [],
                'best_strategy': None
            }
            for domain in LearningDomain
        }
        
        # Meta-learning parameters
        self.meta_learning_rate = 0.1
        self.pattern_threshold = 0.7
        self.transfer_threshold = 0.6
        self.curriculum_difficulty = 0.5
        
        # Performance tracking
        self.metrics = {
            "total_experiences": 0,
            "successful_learnings": 0,
            "patterns_discovered": 0,
            "cross_domain_transfers": 0,
            "curriculum_advancements": 0,
            "meta_insights_generated": 0,
            "learning_acceleration": 0.0
        }
        
        # Learning cycles
        self.learning_cycle_interval = 30  # seconds
        self.last_learning_cycle = 0
        
    async def start_meta_learning(self):
        """Start the meta-learning system."""
        if self.learning_active:
            logger.warning("Meta-learning system already active")
            return
        
        self.learning_active = True
        logger.info(" Starting Meta-Learning System")
        
        # Start learning loops
        asyncio.create_task(self._learning_loop())
        asyncio.create_task(self._pattern_discovery_loop())
        asyncio.create_task(self._meta_analysis_loop())
        asyncio.create_task(self._curriculum_learning_loop())
        
    async def stop_meta_learning(self):
        """Stop the meta-learning system."""
        self.learning_active = False
        logger.info(" Stopping Meta-Learning System")
    
    async def _learning_loop(self):
        """Main learning loop."""
        while self.learning_active:
            try:
                current_time = time.time()
                
                if current_time - self.last_learning_cycle >= self.learning_cycle_interval:
                    await self._run_learning_cycle()
                    self.last_learning_cycle = current_time
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                await asyncio.sleep(10)
    
    async def _pattern_discovery_loop(self):
        """Pattern discovery loop."""
        while self.learning_active:
            try:
                # Discover patterns from experiences
                await self._discover_patterns()
                
                await asyncio.sleep(60)  # Discover patterns every minute
                
            except Exception as e:
                logger.error(f"Error in pattern discovery loop: {e}")
                await asyncio.sleep(30)
    
    async def _meta_analysis_loop(self):
        """Meta-analysis loop."""
        while self.learning_active:
            try:
                # Analyze learning effectiveness
                await self._analyze_learning_effectiveness()
                
                # Generate meta-insights
                await self._generate_meta_insights()
                
                await asyncio.sleep(120)  # Meta-analysis every 2 minutes
                
            except Exception as e:
                logger.error(f"Error in meta-analysis loop: {e}")
                await asyncio.sleep(60)
    
    async def _curriculum_learning_loop(self):
        """Curriculum learning loop."""
        while self.learning_active:
            try:
                # Design and update curriculum
                await self._design_curriculum()
                
                await asyncio.sleep(180)  # Curriculum learning every 3 minutes
                
            except Exception as e:
                logger.error(f"Error in curriculum learning loop: {e}")
                await asyncio.sleep(60)
    
    async def _run_learning_cycle(self):
        """Run a complete learning cycle."""
        try:
            logger.debug(" Running meta-learning cycle")
            
            # 1. Analyze current learning state
            learning_state = await self._analyze_learning_state()
            
            # 2. Select optimal learning strategy
            strategy = await self._select_learning_strategy(learning_state)
            
            # 3. Execute learning with selected strategy
            await self._execute_learning(strategy, learning_state)
            
            # 4. Update learning metrics
            await self._update_learning_metrics()
            
            # 5. Optimize learning parameters
            await self._optimize_learning_parameters()
            
        except Exception as e:
            logger.error(f"Error in learning cycle: {e}")
    
    async def _analyze_learning_state(self) -> Dict[str, Any]:
        """Analyze current learning state."""
        try:
            # Get recent learning experiences
            recent_experiences = list(self.learning_experiences)[-100:]  # Last 100 experiences
            
            # Calculate learning metrics
            total_experiences = len(recent_experiences)
            successful_experiences = sum(1 for exp in recent_experiences if exp.success)
            success_rate = successful_experiences / max(1, total_experiences)
            
            # Calculate learning velocity (rate of improvement)
            learning_velocity = await self._calculate_learning_velocity(recent_experiences)
            
            # Calculate domain-specific metrics
            domain_metrics = {}
            for domain in LearningDomain:
                domain_experiences = [exp for exp in recent_experiences if exp.domain == domain]
                if domain_experiences:
                    domain_success_rate = sum(1 for exp in domain_experiences if exp.success) / len(domain_experiences)
                    domain_metrics[domain.value] = {
                        'success_rate': domain_success_rate,
                        'experience_count': len(domain_experiences),
                        'learning_velocity': await self._calculate_domain_velocity(domain_experiences)
                    }
            
            return {
                'total_experiences': total_experiences,
                'success_rate': success_rate,
                'learning_velocity': learning_velocity,
                'domain_metrics': domain_metrics,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing learning state: {e}")
            return {}
    
    async def _calculate_learning_velocity(self, experiences: List[LearningExperience]) -> float:
        """Calculate learning velocity from experiences."""
        try:
            if len(experiences) < 10:
                return 0.0
            
            # Calculate success rate over time
            time_windows = 5  # 5 experience windows
            window_size = len(experiences) // time_windows
            
            success_rates = []
            for i in range(time_windows):
                start_idx = i * window_size
                end_idx = start_idx + window_size
                window_experiences = experiences[start_idx:end_idx]
                
                if window_experiences:
                    window_success_rate = sum(1 for exp in window_experiences if exp.success) / len(window_experiences)
                    success_rates.append(window_success_rate)
            
            # Calculate velocity as improvement over time
            if len(success_rates) >= 2:
                velocity = success_rates[-1] - success_rates[0]
                return velocity
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating learning velocity: {e}")
            return 0.0
    
    async def _calculate_domain_velocity(self, experiences: List[LearningExperience]) -> float:
        """Calculate learning velocity for a specific domain."""
        try:
            if len(experiences) < 5:
                return 0.0
            
            # Sort by timestamp
            sorted_experiences = sorted(experiences, key=lambda x: x.timestamp)
            
            # Calculate success rate over time
            window_size = max(1, len(sorted_experiences) // 3)
            success_rates = []
            
            for i in range(0, len(sorted_experiences), window_size):
                window = sorted_experiences[i:i + window_size]
                if window:
                    window_success_rate = sum(1 for exp in window if exp.success) / len(window)
                    success_rates.append(window_success_rate)
            
            # Calculate velocity
            if len(success_rates) >= 2:
                return success_rates[-1] - success_rates[0]
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating domain velocity: {e}")
            return 0.0
    
    async def _select_learning_strategy(self, learning_state: Dict[str, Any]) -> LearningStrategy:
        """Select optimal learning strategy based on current state."""
        try:
            success_rate = learning_state.get('success_rate', 0.5)
            learning_velocity = learning_state.get('learning_velocity', 0.0)
            
            # Strategy selection logic
            if success_rate < 0.3:
                # Low success rate - use exploration
                return LearningStrategy.EXPLORATION
            elif success_rate > 0.8 and learning_velocity > 0.1:
                # High success rate and good velocity - use exploitation
                return LearningStrategy.EXPLOITATION
            elif learning_velocity < -0.1:
                # Negative velocity - use transfer learning
                return LearningStrategy.TRANSFER
            elif success_rate > 0.6:
                # Good success rate - use meta-learning
                return LearningStrategy.META
            else:
                # Default to adaptive learning
                return LearningStrategy.ADAPTIVE
                
        except Exception as e:
            logger.error(f"Error selecting learning strategy: {e}")
            return LearningStrategy.ADAPTIVE
    
    async def _execute_learning(self, strategy: LearningStrategy, learning_state: Dict[str, Any]):
        """Execute learning with selected strategy."""
        try:
            logger.debug(f" Executing learning with strategy: {strategy.value}")
            
            if strategy == LearningStrategy.EXPLORATION:
                await self._execute_exploration_learning(learning_state)
            elif strategy == LearningStrategy.EXPLOITATION:
                await self._execute_exploitation_learning(learning_state)
            elif strategy == LearningStrategy.TRANSFER:
                await self._execute_transfer_learning(learning_state)
            elif strategy == LearningStrategy.META:
                await self._execute_meta_learning(learning_state)
            elif strategy == LearningStrategy.ADAPTIVE:
                await self._execute_adaptive_learning(learning_state)
            elif strategy == LearningStrategy.CURRICULUM:
                await self._execute_curriculum_learning(learning_state)
            
            # Update strategy usage
            self.strategies[strategy]['usage_count'] += 1
            
        except Exception as e:
            logger.error(f"Error executing learning: {e}")
    
    async def _execute_exploration_learning(self, learning_state: Dict[str, Any]):
        """Execute exploration learning strategy."""
        try:
            # High exploration rate - try new approaches
            exploration_rate = self.strategies[LearningStrategy.EXPLORATION]['exploration_rate']
            
            # Generate new learning experiences
            new_experiences = await self._generate_exploration_experiences(exploration_rate)
            
            # Execute experiences
            for experience in new_experiences:
                await self._execute_learning_experience(experience)
            
        except Exception as e:
            logger.error(f"Error in exploration learning: {e}")
    
    async def _execute_exploitation_learning(self, learning_state: Dict[str, Any]):
        """Execute exploitation learning strategy."""
        try:
            # Low exploration rate - use known successful patterns
            exploitation_rate = 1.0 - self.strategies[LearningStrategy.EXPLOITATION]['exploration_rate']
            
            # Find successful patterns
            successful_patterns = await self._find_successful_patterns()
            
            # Apply patterns with high confidence
            for pattern in successful_patterns:
                if pattern.confidence > 0.8:
                    await self._apply_learning_pattern(pattern, exploitation_rate)
            
        except Exception as e:
            logger.error(f"Error in exploitation learning: {e}")
    
    async def _execute_transfer_learning(self, learning_state: Dict[str, Any]):
        """Execute transfer learning strategy."""
        try:
            # Transfer knowledge between domains
            source_domains = await self._find_high_performing_domains()
            target_domains = await self._find_low_performing_domains()
            
            for source_domain in source_domains:
                for target_domain in target_domains:
                    if source_domain != target_domain:
                        await self._transfer_knowledge(source_domain, target_domain)
            
            self.metrics["cross_domain_transfers"] += 1
            
        except Exception as e:
            logger.error(f"Error in transfer learning: {e}")
    
    async def _execute_meta_learning(self, learning_state: Dict[str, Any]):
        """Execute meta-learning strategy."""
        try:
            # Learn about learning itself
            await self._analyze_learning_patterns()
            await self._optimize_learning_strategies()
            await self._generate_learning_insights()
            
        except Exception as e:
            logger.error(f"Error in meta-learning: {e}")
    
    async def _execute_adaptive_learning(self, learning_state: Dict[str, Any]):
        """Execute adaptive learning strategy."""
        try:
            # Adapt learning parameters based on performance
            await self._adapt_learning_parameters(learning_state)
            await self._adjust_learning_rates(learning_state)
            await self._modify_exploration_rates(learning_state)
            
        except Exception as e:
            logger.error(f"Error in adaptive learning: {e}")
    
    async def _execute_curriculum_learning(self, learning_state: Dict[str, Any]):
        """Execute curriculum learning strategy."""
        try:
            # Design curriculum based on current difficulty
            curriculum = await self._design_curriculum()
            
            # Execute curriculum tasks
            for task in curriculum:
                await self._execute_curriculum_task(task)
            
            self.metrics["curriculum_advancements"] += 1
            
        except Exception as e:
            logger.error(f"Error in curriculum learning: {e}")
    
    async def _generate_exploration_experiences(self, exploration_rate: float) -> List[LearningExperience]:
        """Generate new exploration experiences."""
        try:
            experiences = []
            
            # Generate experiences for different domains
            for domain in LearningDomain:
                if np.random.random() < exploration_rate:
                    experience = await self._create_exploration_experience(domain)
                    if experience:
                        experiences.append(experience)
            
            return experiences
            
        except Exception as e:
            logger.error(f"Error generating exploration experiences: {e}")
            return []
    
    async def _create_exploration_experience(self, domain: LearningDomain) -> Optional[LearningExperience]:
        """Create a new exploration experience for a domain."""
        try:
            experience_id = f"exp_{int(time.time() * 1000)}_{domain.value}"
            
            # Generate random input data for exploration
            input_data = await self._generate_exploration_input(domain)
            
            # Execute exploration action
            output_data, success = await self._execute_exploration_action(domain, input_data)
            
            # Calculate confidence based on success
            confidence = 0.8 if success else 0.3
            
            experience = LearningExperience(
                experience_id=experience_id,
                domain=domain,
                strategy=LearningStrategy.EXPLORATION,
                input_data=input_data,
                output_data=output_data,
                success=success,
                learning_time=time.time(),
                confidence=confidence,
                timestamp=time.time(),
                metadata={'exploration': True}
            )
            
            return experience
            
        except Exception as e:
            logger.error(f"Error creating exploration experience: {e}")
            return None
    
    async def _generate_exploration_input(self, domain: LearningDomain) -> Dict[str, Any]:
        """Generate exploration input for a domain."""
        try:
            if domain == LearningDomain.GAMEPLAY:
                return {
                    'action_type': np.random.choice(['move', 'interact', 'explore']),
                    'coordinates': (np.random.randint(0, 64), np.random.randint(0, 64)),
                    'confidence': np.random.random()
                }
            elif domain == LearningDomain.SYSTEM_OPTIMIZATION:
                return {
                    'parameter': np.random.choice(['learning_rate', 'exploration_rate', 'memory_size']),
                    'value': np.random.random(),
                    'context': 'optimization'
                }
            elif domain == LearningDomain.ERROR_RECOVERY:
                return {
                    'error_type': np.random.choice(['database', 'api', 'memory', 'performance']),
                    'severity': np.random.choice(['low', 'medium', 'high']),
                    'context': 'recovery'
                }
            else:
                return {'random_input': np.random.random()}
                
        except Exception as e:
            logger.error(f"Error generating exploration input: {e}")
            return {}
    
    async def _execute_exploration_action(self, domain: LearningDomain, input_data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Execute exploration action and return result."""
        try:
            # Simulate exploration action execution
            # In practice, this would execute actual system actions
            
            success = np.random.random() > 0.3  # 70% success rate for exploration
            
            output_data = {
                'result': 'success' if success else 'failure',
                'confidence': np.random.random(),
                'learning_gained': np.random.random() * 0.1,
                'timestamp': time.time()
            }
            
            return output_data, success
            
        except Exception as e:
            logger.error(f"Error executing exploration action: {e}")
            return {}, False
    
    async def _execute_learning_experience(self, experience: LearningExperience):
        """Execute a learning experience."""
        try:
            # Store experience
            self.learning_experiences.append(experience)
            
            # Update domain learning
            domain_data = self.domain_learning[experience.domain]
            domain_data['experiences'].append(experience)
            
            # Update success rate
            if experience.success:
                self.metrics["successful_learnings"] += 1
            
            # Update strategy success rate
            strategy_data = self.strategies[experience.strategy]
            if experience.success:
                strategy_data['success_rate'] = (strategy_data['success_rate'] * strategy_data['usage_count'] + 1) / (strategy_data['usage_count'] + 1)
            else:
                strategy_data['success_rate'] = (strategy_data['success_rate'] * strategy_data['usage_count']) / (strategy_data['usage_count'] + 1)
            
            self.metrics["total_experiences"] += 1
            
        except Exception as e:
            logger.error(f"Error executing learning experience: {e}")
    
    async def _discover_patterns(self):
        """Discover patterns from learning experiences."""
        try:
            # Analyze recent experiences for patterns
            recent_experiences = list(self.learning_experiences)[-500:]  # Last 500 experiences
            
            if len(recent_experiences) < 10:
                return
            
            # Group experiences by domain
            domain_experiences = defaultdict(list)
            for exp in recent_experiences:
                domain_experiences[exp.domain].append(exp)
            
            # Discover patterns for each domain
            for domain, experiences in domain_experiences.items():
                if len(experiences) >= 5:
                    patterns = await self._discover_domain_patterns(domain, experiences)
                    
                    for pattern in patterns:
                        self.learning_patterns[pattern.pattern_id] = pattern
                        self.metrics["patterns_discovered"] += 1
            
        except Exception as e:
            logger.error(f"Error discovering patterns: {e}")
    
    async def _discover_domain_patterns(self, domain: LearningDomain, experiences: List[LearningExperience]) -> List[LearningPattern]:
        """Discover patterns for a specific domain."""
        try:
            patterns = []
            
            # Find successful experience sequences
            successful_experiences = [exp for exp in experiences if exp.success]
            
            if len(successful_experiences) < 3:
                return patterns
            
            # Look for common patterns in successful experiences
            common_inputs = await self._find_common_inputs(successful_experiences)
            common_outputs = await self._find_common_outputs(successful_experiences)
            
            # Create patterns from common elements
            for i, (input_pattern, output_pattern) in enumerate(zip(common_inputs, common_outputs)):
                pattern_id = f"pattern_{domain.value}_{int(time.time() * 1000)}_{i}"
                
                pattern = LearningPattern(
                    pattern_id=pattern_id,
                    domain=domain,
                    pattern_type="success_sequence",
                    pattern_data={
                        'input_pattern': input_pattern,
                        'output_pattern': output_pattern,
                        'frequency': len(successful_experiences)
                    },
                    success_rate=1.0,  # All experiences in this pattern were successful
                    frequency=len(successful_experiences),
                    last_used=time.time(),
                    confidence=0.8,
                    transferability=0.6
                )
                
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error discovering domain patterns: {e}")
            return []
    
    async def _find_common_inputs(self, experiences: List[LearningExperience]) -> List[Dict[str, Any]]:
        """Find common input patterns in experiences."""
        try:
            # Simple pattern finding - in practice, this would be more sophisticated
            common_patterns = []
            
            # Group by input similarity
            input_groups = defaultdict(list)
            for exp in experiences:
                input_key = str(sorted(exp.input_data.items()))
                input_groups[input_key].append(exp)
            
            # Find groups with multiple experiences
            for input_key, group_experiences in input_groups.items():
                if len(group_experiences) >= 2:
                    # Create pattern from most common input
                    most_common_input = group_experiences[0].input_data
                    common_patterns.append(most_common_input)
            
            return common_patterns
            
        except Exception as e:
            logger.error(f"Error finding common inputs: {e}")
            return []
    
    async def _find_common_outputs(self, experiences: List[LearningExperience]) -> List[Dict[str, Any]]:
        """Find common output patterns in experiences."""
        try:
            # Similar to common inputs
            common_patterns = []
            
            # Group by output similarity
            output_groups = defaultdict(list)
            for exp in experiences:
                output_key = str(sorted(exp.output_data.items()))
                output_groups[output_key].append(exp)
            
            # Find groups with multiple experiences
            for output_key, group_experiences in output_groups.items():
                if len(group_experiences) >= 2:
                    # Create pattern from most common output
                    most_common_output = group_experiences[0].output_data
                    common_patterns.append(most_common_output)
            
            return common_patterns
            
        except Exception as e:
            logger.error(f"Error finding common outputs: {e}")
            return []
    
    async def _analyze_learning_effectiveness(self):
        """Analyze learning effectiveness across strategies and domains."""
        try:
            # Analyze strategy effectiveness
            for strategy, data in self.strategies.items():
                if data['usage_count'] > 0:
                    effectiveness = data['success_rate'] * data['usage_count']
                    logger.debug(f"Strategy {strategy.value} effectiveness: {effectiveness:.3f}")
            
            # Analyze domain effectiveness
            for domain, data in self.domain_learning.items():
                if data['experiences']:
                    domain_success_rate = sum(1 for exp in data['experiences'] if exp.success) / len(data['experiences'])
                    data['success_rate'] = domain_success_rate
                    
                    # Update learning curve
                    data['learning_curve'].append(domain_success_rate)
                    if len(data['learning_curve']) > 100:
                        data['learning_curve'] = data['learning_curve'][-100:]
            
        except Exception as e:
            logger.error(f"Error analyzing learning effectiveness: {e}")
    
    async def _generate_meta_insights(self):
        """Generate meta-learning insights."""
        try:
            # Analyze cross-domain patterns
            cross_domain_patterns = await self._find_cross_domain_patterns()
            
            # Generate insights from patterns
            for pattern in cross_domain_patterns:
                insight = await self._create_meta_insight(pattern)
                if insight:
                    self.meta_insights.append(insight)
                    self.metrics["meta_insights_generated"] += 1
            
        except Exception as e:
            logger.error(f"Error generating meta insights: {e}")
    
    async def _find_cross_domain_patterns(self) -> List[Dict[str, Any]]:
        """Find patterns that work across multiple domains."""
        try:
            cross_domain_patterns = []
            
            # Look for similar patterns across domains
            domain_patterns = defaultdict(list)
            for pattern in self.learning_patterns.values():
                domain_patterns[pattern.domain].append(pattern)
            
            # Find patterns that appear in multiple domains
            pattern_types = defaultdict(list)
            for domain, patterns in domain_patterns.items():
                for pattern in patterns:
                    pattern_key = pattern.pattern_type
                    pattern_types[pattern_key].append((domain, pattern))
            
            # Find pattern types that appear in multiple domains
            for pattern_type, domain_patterns in pattern_types.items():
                if len(set(domain for domain, _ in domain_patterns)) > 1:
                    cross_domain_patterns.append({
                        'pattern_type': pattern_type,
                        'domains': list(set(domain for domain, _ in domain_patterns)),
                        'patterns': domain_patterns
                    })
            
            return cross_domain_patterns
            
        except Exception as e:
            logger.error(f"Error finding cross-domain patterns: {e}")
            return []
    
    async def _create_meta_insight(self, pattern: Dict[str, Any]) -> Optional[MetaLearningInsight]:
        """Create a meta-learning insight from a pattern."""
        try:
            insight_id = f"insight_{int(time.time() * 1000)}"
            
            # Calculate effectiveness
            effectiveness = sum(p[1].success_rate for p in pattern['patterns']) / len(pattern['patterns'])
            
            # Calculate confidence
            confidence = min(1.0, len(pattern['domains']) / len(LearningDomain))
            
            insight = MetaLearningInsight(
                insight_id=insight_id,
                insight_type=f"cross_domain_{pattern['pattern_type']}",
                description=f"Pattern {pattern['pattern_type']} works across {len(pattern['domains'])} domains",
                domains=pattern['domains'],
                effectiveness=effectiveness,
                confidence=confidence,
                timestamp=time.time(),
                applications=[f"Apply to {domain.value}" for domain in pattern['domains']]
            )
            
            return insight
            
        except Exception as e:
            logger.error(f"Error creating meta insight: {e}")
            return None
    
    async def _design_curriculum(self) -> List[Dict[str, Any]]:
        """Design learning curriculum based on current difficulty."""
        try:
            curriculum = []
            
            # Calculate current difficulty based on success rates
            overall_success_rate = self.metrics["successful_learnings"] / max(1, self.metrics["total_experiences"])
            
            # Adjust curriculum difficulty
            if overall_success_rate > 0.8:
                self.curriculum_difficulty = min(1.0, self.curriculum_difficulty + 0.1)
            elif overall_success_rate < 0.4:
                self.curriculum_difficulty = max(0.1, self.curriculum_difficulty - 0.1)
            
            # Generate curriculum tasks
            for domain in LearningDomain:
                domain_success_rate = self.domain_learning[domain]['success_rate']
                
                if domain_success_rate < self.curriculum_difficulty:
                    # Domain needs more practice
                    task = {
                        'domain': domain,
                        'difficulty': self.curriculum_difficulty,
                        'task_type': 'practice',
                        'description': f"Practice {domain.value} with difficulty {self.curriculum_difficulty:.2f}"
                    }
                    curriculum.append(task)
            
            return curriculum
            
        except Exception as e:
            logger.error(f"Error designing curriculum: {e}")
            return []
    
    async def _execute_curriculum_task(self, task: Dict[str, Any]):
        """Execute a curriculum learning task."""
        try:
            domain = task['domain']
            difficulty = task['difficulty']
            
            # Create experience with adjusted difficulty
            experience = await self._create_curriculum_experience(domain, difficulty)
            if experience:
                await self._execute_learning_experience(experience)
            
        except Exception as e:
            logger.error(f"Error executing curriculum task: {e}")
    
    async def _create_curriculum_experience(self, domain: LearningDomain, difficulty: float) -> Optional[LearningExperience]:
        """Create a curriculum learning experience."""
        try:
            experience_id = f"curriculum_{int(time.time() * 1000)}_{domain.value}"
            
            # Generate input based on difficulty
            input_data = await self._generate_curriculum_input(domain, difficulty)
            
            # Execute with difficulty-adjusted success rate
            success_rate = 0.5 + (difficulty - 0.5) * 0.5  # Adjust success rate based on difficulty
            success = np.random.random() < success_rate
            
            output_data = {
                'result': 'success' if success else 'failure',
                'difficulty': difficulty,
                'confidence': success_rate,
                'timestamp': time.time()
            }
            
            experience = LearningExperience(
                experience_id=experience_id,
                domain=domain,
                strategy=LearningStrategy.CURRICULUM,
                input_data=input_data,
                output_data=output_data,
                success=success,
                learning_time=time.time(),
                confidence=success_rate,
                timestamp=time.time(),
                metadata={'curriculum': True, 'difficulty': difficulty}
            )
            
            return experience
            
        except Exception as e:
            logger.error(f"Error creating curriculum experience: {e}")
            return None
    
    async def _generate_curriculum_input(self, domain: LearningDomain, difficulty: float) -> Dict[str, Any]:
        """Generate curriculum input based on difficulty."""
        try:
            # Adjust input complexity based on difficulty
            complexity = int(difficulty * 10)  # 1-10 complexity scale
            
            if domain == LearningDomain.GAMEPLAY:
                return {
                    'action_type': 'complex_action',
                    'complexity': complexity,
                    'difficulty': difficulty
                }
            elif domain == LearningDomain.SYSTEM_OPTIMIZATION:
                return {
                    'parameter_count': complexity,
                    'difficulty': difficulty
                }
            else:
                return {
                    'complexity': complexity,
                    'difficulty': difficulty
                }
                
        except Exception as e:
            logger.error(f"Error generating curriculum input: {e}")
            return {}
    
    async def _update_learning_metrics(self):
        """Update learning metrics."""
        try:
            # Calculate learning acceleration
            if len(self.learning_experiences) >= 100:
                recent_experiences = list(self.learning_experiences)[-100:]
                recent_success_rate = sum(1 for exp in recent_experiences if exp.success) / len(recent_experiences)
                
                if len(self.learning_experiences) >= 200:
                    older_experiences = list(self.learning_experiences)[-200:-100]
                    older_success_rate = sum(1 for exp in older_experiences if exp.success) / len(older_experiences)
                    
                    acceleration = recent_success_rate - older_success_rate
                    self.metrics["learning_acceleration"] = acceleration
            
        except Exception as e:
            logger.error(f"Error updating learning metrics: {e}")
    
    async def _optimize_learning_parameters(self):
        """Optimize learning parameters based on performance."""
        try:
            # Adjust learning rates based on success rates
            for strategy, data in self.strategies.items():
                if data['usage_count'] > 10:
                    success_rate = data['success_rate']
                    
                    if success_rate > 0.8:
                        # High success rate - can increase learning rate
                        data['learning_rate'] = min(0.3, data['learning_rate'] * 1.1)
                    elif success_rate < 0.3:
                        # Low success rate - decrease learning rate
                        data['learning_rate'] = max(0.01, data['learning_rate'] * 0.9)
            
        except Exception as e:
            logger.error(f"Error optimizing learning parameters: {e}")
    
    def get_meta_learning_status(self) -> Dict[str, Any]:
        """Get meta-learning system status."""
        return {
            "learning_active": self.learning_active,
            "metrics": self.metrics,
            "strategies": {k.value: v for k, v in self.strategies.items()},
            "domain_learning": {
                domain.value: {
                    'experience_count': len(data['experiences']),
                    'success_rate': data['success_rate'],
                    'learning_curve_length': len(data['learning_curve'])
                }
                for domain, data in self.domain_learning.items()
            },
            "patterns_count": len(self.learning_patterns),
            "meta_insights_count": len(self.meta_insights),
            "curriculum_difficulty": self.curriculum_difficulty
        }

# Global meta-learning system instance
meta_learning_system = MetaLearningSystem()

async def start_meta_learning():
    """Start the meta-learning system."""
    await meta_learning_system.start_meta_learning()

async def stop_meta_learning():
    """Stop the meta-learning system."""
    await meta_learning_system.stop_meta_learning()

def get_meta_learning_status():
    """Get meta-learning system status."""
    return meta_learning_system.get_meta_learning_status()
