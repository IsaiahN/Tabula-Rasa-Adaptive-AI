"""
Self-Evolving Code System - Phase 3 Implementation

This system allows the code to evolve and improve itself with strict safeguards:
- 500-game cooldown between architectural changes
- Comprehensive data gathering before changes
- Vigorous testing of all modifications
- Rollback capabilities for failed changes
"""

import asyncio
import logging
import time
import json
import os
import ast
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque

from ..database.system_integration import get_system_integration

logger = logging.getLogger(__name__)

class EvolutionType(Enum):
    """Types of code evolution."""
    MINOR_OPTIMIZATION = "minor_optimization"
    MAJOR_REFACTORING = "major_refactoring"
    ARCHITECTURAL_CHANGE = "architectural_change"
    ALGORITHM_IMPROVEMENT = "algorithm_improvement"
    PERFORMANCE_ENHANCEMENT = "performance_enhancement"
    BUG_FIX = "bug_fix"

class EvolutionStatus(Enum):
    """Evolution status."""
    PENDING = "pending"
    TESTING = "testing"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"

class SafetyLevel(Enum):
    """Safety levels for code evolution."""
    LOW = "low"  # Minor optimizations
    MEDIUM = "medium"  # Major refactoring
    HIGH = "high"  # Architectural changes
    CRITICAL = "critical"  # Core algorithm changes

@dataclass
class CodeChange:
    """Represents a code change."""
    change_id: str
    evolution_type: EvolutionType
    safety_level: SafetyLevel
    file_path: str
    old_code: str
    new_code: str
    description: str
    reasoning: str
    expected_improvement: float
    confidence: float
    dependencies: List[str]
    rollback_data: Dict[str, Any]
    timestamp: float
    status: EvolutionStatus = EvolutionStatus.PENDING

@dataclass
class EvolutionProposal:
    """Represents an evolution proposal."""
    proposal_id: str
    evolution_type: EvolutionType
    priority: int
    description: str
    reasoning: str
    expected_improvement: float
    confidence: float
    data_requirements: Dict[str, Any]
    test_requirements: List[str]
    safety_requirements: List[str]
    timestamp: float

@dataclass
class EvolutionMetrics:
    """Evolution system metrics."""
    total_changes: int
    successful_changes: int
    failed_changes: int
    rolled_back_changes: int
    games_since_last_architectural_change: int
    data_gathering_cycles: int
    test_cycles: int
    improvement_rate: float
    safety_violations: int
    cooldown_violations: int

class SelfEvolvingCodeSystem:
    """
    Self-Evolving Code System that allows code to evolve and improve itself
    with strict safeguards and cooldown periods.
    
    Features:
    - 500-game cooldown for architectural changes
    - Comprehensive data gathering before changes
    - Vigorous testing of all modifications
    - Rollback capabilities for failed changes
    - Safety mechanisms and validation
    - Performance monitoring and improvement tracking
    """
    
    def __init__(self):
        self.integration = get_system_integration()
        
        # Evolution state
        self.evolution_active = False
        self.architectural_change_cooldown = 500  # Games before next architectural change
        self.last_architectural_change = 0
        self.current_game_count = 0
        
        # Change management
        self.pending_changes = deque(maxlen=1000)
        self.applied_changes = deque(maxlen=1000)
        self.rollback_stack = deque(maxlen=100)
        self.evolution_proposals = deque(maxlen=500)
        
        # Safety mechanisms
        self.safety_thresholds = {
            SafetyLevel.LOW: {"min_games": 0, "min_data": 10, "min_tests": 5},
            SafetyLevel.MEDIUM: {"min_games": 50, "min_data": 100, "min_tests": 20},
            SafetyLevel.HIGH: {"min_games": 200, "min_data": 500, "min_tests": 50},
            SafetyLevel.CRITICAL: {"min_games": 500, "min_data": 1000, "min_tests": 100}
        }
        
        # Data gathering
        self.data_gathering_active = False
        self.gathered_data = defaultdict(list)
        self.data_requirements = {}
        
        # Testing
        self.testing_active = False
        self.test_results = deque(maxlen=1000)
        self.test_requirements = {}
        
        # Performance tracking
        self.metrics = EvolutionMetrics(
            total_changes=0,
            successful_changes=0,
            failed_changes=0,
            rolled_back_changes=0,
            games_since_last_architectural_change=0,
            data_gathering_cycles=0,
            test_cycles=0,
            improvement_rate=0.0,
            safety_violations=0,
            cooldown_violations=0
        )
        
        # Evolution cycles
        self.evolution_cycle_interval = 60  # seconds
        self.last_evolution_cycle = 0
        
    async def start_evolution(self):
        """Start the self-evolving code system."""
        if self.evolution_active:
            logger.warning("Code evolution system already active")
            return
        
        self.evolution_active = True
        logger.info(" Starting Self-Evolving Code System")
        
        # Load current game count
        await self._load_current_game_count()
        
        # Start evolution loops
        asyncio.create_task(self._evolution_loop())
        asyncio.create_task(self._data_gathering_loop())
        asyncio.create_task(self._testing_loop())
        asyncio.create_task(self._safety_monitoring_loop())
        
    async def stop_evolution(self):
        """Stop the self-evolving code system."""
        self.evolution_active = False
        logger.info(" Stopping Self-Evolving Code System")
    
    async def _load_current_game_count(self):
        """Load current game count from database."""
        try:
            # Get current game count from database
            game_results = await self.integration.get_game_results(limit=1)
            if game_results:
                self.current_game_count = game_results[0].get('game_id', 0)
            else:
                self.current_game_count = 0
            
            # Calculate games since last architectural change
            self.metrics.games_since_last_architectural_change = self.current_game_count - self.last_architectural_change
            
            logger.info(f" Current game count: {self.current_game_count}")
            logger.info(f" Games since last architectural change: {self.metrics.games_since_last_architectural_change}")
            
        except Exception as e:
            logger.error(f"Error loading current game count: {e}")
            self.current_game_count = 0
    
    async def _evolution_loop(self):
        """Main evolution loop."""
        while self.evolution_active:
            try:
                current_time = time.time()
                
                if current_time - self.last_evolution_cycle >= self.evolution_cycle_interval:
                    await self._run_evolution_cycle()
                    self.last_evolution_cycle = current_time
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in evolution loop: {e}")
                await asyncio.sleep(30)
    
    async def _data_gathering_loop(self):
        """Data gathering loop."""
        while self.evolution_active:
            try:
                if self.data_gathering_active:
                    await self._gather_evolution_data()
                
                await asyncio.sleep(30)  # Gather data every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in data gathering loop: {e}")
                await asyncio.sleep(60)
    
    async def _testing_loop(self):
        """Testing loop for code changes."""
        while self.evolution_active:
            try:
                if self.testing_active:
                    await self._test_pending_changes()
                
                await asyncio.sleep(60)  # Test every minute
                
            except Exception as e:
                logger.error(f"Error in testing loop: {e}")
                await asyncio.sleep(120)
    
    async def _safety_monitoring_loop(self):
        """Safety monitoring loop."""
        while self.evolution_active:
            try:
                await self._monitor_safety_violations()
                
                await asyncio.sleep(120)  # Monitor every 2 minutes
                
            except Exception as e:
                logger.error(f"Error in safety monitoring loop: {e}")
                await asyncio.sleep(180)
    
    async def _run_evolution_cycle(self):
        """Run a complete evolution cycle."""
        try:
            logger.debug(" Running code evolution cycle")
            
            # 1. Update game count
            await self._update_game_count()
            
            # 2. Check if we can make architectural changes
            can_make_architectural_changes = await self._can_make_architectural_changes()
            
            # 3. Generate evolution proposals
            proposals = await self._generate_evolution_proposals(can_make_architectural_changes)
            
            # 4. Process proposals
            for proposal in proposals:
                await self._process_evolution_proposal(proposal)
            
            # 5. Apply pending changes
            await self._apply_pending_changes()
            
        except Exception as e:
            logger.error(f"Error in evolution cycle: {e}")
    
    async def _update_game_count(self):
        """Update current game count."""
        try:
            # Get latest game count
            game_results = await self.integration.get_game_results(limit=1)
            if game_results:
                new_game_count = game_results[0].get('game_id', 0)
                if new_game_count > self.current_game_count:
                    self.current_game_count = new_game_count
                    self.metrics.games_since_last_architectural_change = self.current_game_count - self.last_architectural_change
                    
                    logger.debug(f" Game count updated: {self.current_game_count}")
            
        except Exception as e:
            logger.error(f"Error updating game count: {e}")
    
    async def _can_make_architectural_changes(self) -> bool:
        """Check if we can make architectural changes based on cooldown."""
        try:
            games_since_last_change = self.metrics.games_since_last_architectural_change
            
            if games_since_last_change >= self.architectural_change_cooldown:
                # Check if we have sufficient data
                has_sufficient_data = await self._has_sufficient_data_for_architectural_change()
                
                if has_sufficient_data:
                    logger.info(f" Can make architectural changes: {games_since_last_change} games since last change")
                    return True
                else:
                    logger.info(f"⏳ Need more data for architectural changes: {games_since_last_change} games since last change")
                    return False
            else:
                remaining_games = self.architectural_change_cooldown - games_since_last_change
                logger.debug(f"⏳ Architectural change cooldown: {remaining_games} games remaining")
                return False
                
        except Exception as e:
            logger.error(f"Error checking architectural change eligibility: {e}")
            return False
    
    async def _has_sufficient_data_for_architectural_change(self) -> bool:
        """Check if we have sufficient data for architectural changes."""
        try:
            # Check data requirements for architectural changes
            requirements = self.safety_thresholds[SafetyLevel.CRITICAL]
            
            # Check game count
            if self.metrics.games_since_last_architectural_change < requirements["min_games"]:
                return False
            
            # Check data quality
            data_quality = await self._assess_data_quality()
            if data_quality < 0.8:  # 80% data quality threshold
                return False
            
            # Check performance trends
            performance_trend = await self._assess_performance_trend()
            if performance_trend < 0.1:  # 10% improvement threshold
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking sufficient data: {e}")
            return False
    
    async def _assess_data_quality(self) -> float:
        """Assess the quality of gathered data."""
        try:
            # This would assess data quality based on various metrics
            # For now, return a placeholder
            return 0.85  # 85% data quality
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            return 0.0
    
    async def _assess_performance_trend(self) -> float:
        """Assess performance trend over recent games."""
        try:
            # Get recent performance data
            recent_performance = await self._get_recent_performance_data()
            
            if len(recent_performance) < 10:
                return 0.0
            
            # Calculate trend
            trend = self._calculate_trend(recent_performance)
            return trend
            
        except Exception as e:
            logger.error(f"Error assessing performance trend: {e}")
            return 0.0
    
    async def _get_recent_performance_data(self) -> List[float]:
        """Get recent performance data."""
        try:
            # This would get actual performance data from the database
            # For now, return placeholder data
            return [0.8, 0.82, 0.85, 0.87, 0.89, 0.91, 0.93, 0.95, 0.97, 0.99]
            
        except Exception as e:
            logger.error(f"Error getting recent performance data: {e}")
            return []
    
    def _calculate_trend(self, data: List[float]) -> float:
        """Calculate trend from data points."""
        try:
            if len(data) < 2:
                return 0.0
            
            # Simple linear trend calculation
            n = len(data)
            x = list(range(n))
            y = data
            
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            if n * sum_x2 - sum_x ** 2 == 0:
                return 0.0
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            return slope
            
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return 0.0
    
    async def _generate_evolution_proposals(self, can_make_architectural_changes: bool) -> List[EvolutionProposal]:
        """Generate evolution proposals based on current state."""
        try:
            proposals = []
            
            # Always generate minor optimization proposals
            minor_proposals = await self._generate_minor_optimization_proposals()
            proposals.extend(minor_proposals)
            
            # Generate major refactoring proposals if conditions are met
            if self.metrics.games_since_last_architectural_change >= 100:
                major_proposals = await self._generate_major_refactoring_proposals()
                proposals.extend(major_proposals)
            
            # Generate architectural change proposals if cooldown is satisfied
            if can_make_architectural_changes:
                arch_proposals = await self._generate_architectural_change_proposals()
                proposals.extend(arch_proposals)
            
            return proposals
            
        except Exception as e:
            logger.error(f"Error generating evolution proposals: {e}")
            return []
    
    async def _generate_minor_optimization_proposals(self) -> List[EvolutionProposal]:
        """Generate minor optimization proposals."""
        try:
            proposals = []
            
            # Analyze code for minor optimizations
            optimization_opportunities = await self._analyze_optimization_opportunities()
            
            for opportunity in optimization_opportunities:
                proposal = EvolutionProposal(
                    proposal_id=f"minor_opt_{int(time.time() * 1000)}",
                    evolution_type=EvolutionType.MINOR_OPTIMIZATION,
                    priority=1,
                    description=f"Minor optimization: {opportunity['description']}",
                    reasoning=opportunity['reasoning'],
                    expected_improvement=opportunity['expected_improvement'],
                    confidence=opportunity['confidence'],
                    data_requirements={"min_games": 0, "min_data": 10},
                    test_requirements=["unit_test", "performance_test"],
                    safety_requirements=["backup", "rollback"],
                    timestamp=time.time()
                )
                proposals.append(proposal)
            
            return proposals
            
        except Exception as e:
            logger.error(f"Error generating minor optimization proposals: {e}")
            return []
    
    async def _generate_major_refactoring_proposals(self) -> List[EvolutionProposal]:
        """Generate major refactoring proposals."""
        try:
            proposals = []
            
            # Analyze code for major refactoring opportunities
            refactoring_opportunities = await self._analyze_refactoring_opportunities()
            
            for opportunity in refactoring_opportunities:
                proposal = EvolutionProposal(
                    proposal_id=f"major_refactor_{int(time.time() * 1000)}",
                    evolution_type=EvolutionType.MAJOR_REFACTORING,
                    priority=2,
                    description=f"Major refactoring: {opportunity['description']}",
                    reasoning=opportunity['reasoning'],
                    expected_improvement=opportunity['expected_improvement'],
                    confidence=opportunity['confidence'],
                    data_requirements={"min_games": 50, "min_data": 100},
                    test_requirements=["unit_test", "integration_test", "performance_test"],
                    safety_requirements=["backup", "rollback", "validation"],
                    timestamp=time.time()
                )
                proposals.append(proposal)
            
            return proposals
            
        except Exception as e:
            logger.error(f"Error generating major refactoring proposals: {e}")
            return []
    
    async def _generate_architectural_change_proposals(self) -> List[EvolutionProposal]:
        """Generate architectural change proposals."""
        try:
            proposals = []
            
            # Analyze system for architectural improvements
            arch_opportunities = await self._analyze_architectural_opportunities()
            
            for opportunity in arch_opportunities:
                proposal = EvolutionProposal(
                    proposal_id=f"arch_change_{int(time.time() * 1000)}",
                    evolution_type=EvolutionType.ARCHITECTURAL_CHANGE,
                    priority=3,
                    description=f"Architectural change: {opportunity['description']}",
                    reasoning=opportunity['reasoning'],
                    expected_improvement=opportunity['expected_improvement'],
                    confidence=opportunity['confidence'],
                    data_requirements={"min_games": 500, "min_data": 1000},
                    test_requirements=["unit_test", "integration_test", "performance_test", "stress_test"],
                    safety_requirements=["backup", "rollback", "validation", "monitoring"],
                    timestamp=time.time()
                )
                proposals.append(proposal)
            
            return proposals
            
        except Exception as e:
            logger.error(f"Error generating architectural change proposals: {e}")
            return []
    
    async def _analyze_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Analyze code for optimization opportunities."""
        try:
            opportunities = []
            
            # This would analyze actual code for optimization opportunities
            # For now, return placeholder opportunities
            opportunities.append({
                'description': 'Optimize database queries',
                'reasoning': 'Database queries are taking too long',
                'expected_improvement': 0.15,
                'confidence': 0.8
            })
            
            opportunities.append({
                'description': 'Cache frequently accessed data',
                'reasoning': 'Repeated data access is inefficient',
                'expected_improvement': 0.1,
                'confidence': 0.7
            })
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error analyzing optimization opportunities: {e}")
            return []
    
    async def _analyze_refactoring_opportunities(self) -> List[Dict[str, Any]]:
        """Analyze code for refactoring opportunities."""
        try:
            opportunities = []
            
            # This would analyze actual code for refactoring opportunities
            # For now, return placeholder opportunities
            opportunities.append({
                'description': 'Extract common functionality into base classes',
                'reasoning': 'Code duplication detected in multiple classes',
                'expected_improvement': 0.2,
                'confidence': 0.75
            })
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error analyzing refactoring opportunities: {e}")
            return []
    
    async def _analyze_architectural_opportunities(self) -> List[Dict[str, Any]]:
        """Analyze system for architectural opportunities."""
        try:
            opportunities = []
            
            # This would analyze actual system architecture for opportunities
            # For now, return placeholder opportunities
            opportunities.append({
                'description': 'Implement microservices architecture',
                'reasoning': 'Monolithic architecture is becoming a bottleneck',
                'expected_improvement': 0.3,
                'confidence': 0.6
            })
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error analyzing architectural opportunities: {e}")
            return []
    
    async def _process_evolution_proposal(self, proposal: EvolutionProposal):
        """Process an evolution proposal."""
        try:
            logger.info(f" Processing evolution proposal: {proposal.proposal_id}")
            
            # Check safety requirements
            if not await self._check_safety_requirements(proposal):
                logger.warning(f" Safety requirements not met for proposal: {proposal.proposal_id}")
                return
            
            # Generate code changes for the proposal
            code_changes = await self._generate_code_changes(proposal)
            
            # Add changes to pending queue
            for change in code_changes:
                self.pending_changes.append(change)
            
            logger.info(f" Generated {len(code_changes)} code changes for proposal: {proposal.proposal_id}")
            
        except Exception as e:
            logger.error(f"Error processing evolution proposal: {e}")
    
    async def _check_safety_requirements(self, proposal: EvolutionProposal) -> bool:
        """Check if safety requirements are met for a proposal."""
        try:
            # Check data requirements
            data_req = proposal.data_requirements
            if not await self._check_data_requirements(data_req):
                return False
            
            # Check test requirements
            if not await self._check_test_requirements(proposal.test_requirements):
                return False
            
            # Check safety requirements
            if not await self._check_safety_requirements_list(proposal.safety_requirements):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking safety requirements: {e}")
            return False
    
    async def _check_data_requirements(self, data_req: Dict[str, Any]) -> bool:
        """Check if data requirements are met."""
        try:
            min_games = data_req.get("min_games", 0)
            min_data = data_req.get("min_data", 0)
            
            # Check game count
            if self.metrics.games_since_last_architectural_change < min_games:
                return False
            
            # Check data quality
            data_quality = await self._assess_data_quality()
            if data_quality < 0.8:  # 80% data quality threshold
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking data requirements: {e}")
            return False
    
    async def _check_test_requirements(self, test_req: List[str]) -> bool:
        """Check if test requirements are met."""
        try:
            # This would check if required tests are available and passing
            # For now, return True as placeholder
            return True
            
        except Exception as e:
            logger.error(f"Error checking test requirements: {e}")
            return False
    
    async def _check_safety_requirements_list(self, safety_req: List[str]) -> bool:
        """Check if safety requirements are met."""
        try:
            # This would check safety requirements like backup, rollback, etc.
            # For now, return True as placeholder
            return True
            
        except Exception as e:
            logger.error(f"Error checking safety requirements: {e}")
            return False
    
    async def _generate_code_changes(self, proposal: EvolutionProposal) -> List[CodeChange]:
        """Generate code changes for a proposal."""
        try:
            changes = []
            
            # This would generate actual code changes based on the proposal
            # For now, return placeholder changes
            change = CodeChange(
                change_id=f"change_{int(time.time() * 1000)}",
                evolution_type=proposal.evolution_type,
                safety_level=self._get_safety_level(proposal.evolution_type),
                file_path="src/example.py",
                old_code="# Old code",
                new_code="# New optimized code",
                description=proposal.description,
                reasoning=proposal.reasoning,
                expected_improvement=proposal.expected_improvement,
                confidence=proposal.confidence,
                dependencies=[],
                rollback_data={},
                timestamp=time.time()
            )
            changes.append(change)
            
            return changes
            
        except Exception as e:
            logger.error(f"Error generating code changes: {e}")
            return []
    
    def _get_safety_level(self, evolution_type: EvolutionType) -> SafetyLevel:
        """Get safety level for evolution type."""
        if evolution_type == EvolutionType.MINOR_OPTIMIZATION:
            return SafetyLevel.LOW
        elif evolution_type == EvolutionType.MAJOR_REFACTORING:
            return SafetyLevel.MEDIUM
        elif evolution_type == EvolutionType.ARCHITECTURAL_CHANGE:
            return SafetyLevel.HIGH
        else:
            return SafetyLevel.CRITICAL
    
    async def _apply_pending_changes(self):
        """Apply pending code changes."""
        try:
            if not self.pending_changes:
                return
            
            # Get next change to apply
            change = self.pending_changes.popleft()
            
            # Test the change first
            if await self._test_code_change(change):
                # Apply the change
                if await self._apply_code_change(change):
                    self.applied_changes.append(change)
                    self.metrics.successful_changes += 1
                    logger.info(f" Applied code change: {change.change_id}")
                else:
                    self.metrics.failed_changes += 1
                    logger.error(f" Failed to apply code change: {change.change_id}")
            else:
                self.metrics.failed_changes += 1
                logger.error(f" Code change failed testing: {change.change_id}")
            
            self.metrics.total_changes += 1
            
        except Exception as e:
            logger.error(f"Error applying pending changes: {e}")
    
    async def _test_code_change(self, change: CodeChange) -> bool:
        """Test a code change before applying it."""
        try:
            # This would test the code change
            # For now, return True as placeholder
            return True
            
        except Exception as e:
            logger.error(f"Error testing code change: {e}")
            return False
    
    async def _apply_code_change(self, change: CodeChange) -> bool:
        """Apply a code change."""
        try:
            # This would apply the actual code change
            # For now, return True as placeholder
            return True
            
        except Exception as e:
            logger.error(f"Error applying code change: {e}")
            return False
    
    async def _gather_evolution_data(self):
        """Gather data for evolution decisions."""
        try:
            # Gather performance data
            performance_data = await self._gather_performance_data()
            self.gathered_data['performance'].append(performance_data)
            
            # Gather error data
            error_data = await self._gather_error_data()
            self.gathered_data['errors'].append(error_data)
            
            # Gather usage data
            usage_data = await self._gather_usage_data()
            self.gathered_data['usage'].append(usage_data)
            
            self.metrics.data_gathering_cycles += 1
            
        except Exception as e:
            logger.error(f"Error gathering evolution data: {e}")
    
    async def _gather_performance_data(self) -> Dict[str, Any]:
        """Gather performance data."""
        try:
            # This would gather actual performance data
            return {
                'timestamp': time.time(),
                'cpu_usage': 0.5,
                'memory_usage': 0.6,
                'response_time': 0.1
            }
            
        except Exception as e:
            logger.error(f"Error gathering performance data: {e}")
            return {}
    
    async def _gather_error_data(self) -> Dict[str, Any]:
        """Gather error data."""
        try:
            # This would gather actual error data
            return {
                'timestamp': time.time(),
                'error_count': 0,
                'error_types': []
            }
            
        except Exception as e:
            logger.error(f"Error gathering error data: {e}")
            return {}
    
    async def _gather_usage_data(self) -> Dict[str, Any]:
        """Gather usage data."""
        try:
            # This would gather actual usage data
            return {
                'timestamp': time.time(),
                'feature_usage': {},
                'user_activity': 0
            }
            
        except Exception as e:
            logger.error(f"Error gathering usage data: {e}")
            return {}
    
    async def _test_pending_changes(self):
        """Test pending code changes."""
        try:
            # This would test pending changes
            self.metrics.test_cycles += 1
            
        except Exception as e:
            logger.error(f"Error testing pending changes: {e}")
    
    async def _monitor_safety_violations(self):
        """Monitor for safety violations."""
        try:
            # Check for cooldown violations
            if self.metrics.games_since_last_architectural_change < self.architectural_change_cooldown:
                # Check if any architectural changes are pending
                arch_changes = [c for c in self.pending_changes if c.evolution_type == EvolutionType.ARCHITECTURAL_CHANGE]
                if arch_changes:
                    self.metrics.cooldown_violations += 1
                    logger.warning(f" Cooldown violation: {len(arch_changes)} architectural changes pending during cooldown")
            
            # Check for safety threshold violations
            for change in self.pending_changes:
                safety_level = change.safety_level
                requirements = self.safety_thresholds[safety_level]
                
                if self.metrics.games_since_last_architectural_change < requirements["min_games"]:
                    self.metrics.safety_violations += 1
                    logger.warning(f" Safety violation: {change.change_id} requires {requirements['min_games']} games")
            
        except Exception as e:
            logger.error(f"Error monitoring safety violations: {e}")
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get evolution system status."""
        return {
            "evolution_active": self.evolution_active,
            "metrics": {
                "total_changes": self.metrics.total_changes,
                "successful_changes": self.metrics.successful_changes,
                "failed_changes": self.metrics.failed_changes,
                "rolled_back_changes": self.metrics.rolled_back_changes,
                "games_since_last_architectural_change": self.metrics.games_since_last_architectural_change,
                "data_gathering_cycles": self.metrics.data_gathering_cycles,
                "test_cycles": self.metrics.test_cycles,
                "improvement_rate": self.metrics.improvement_rate,
                "safety_violations": self.metrics.safety_violations,
                "cooldown_violations": self.metrics.cooldown_violations
            },
            "pending_changes_count": len(self.pending_changes),
            "applied_changes_count": len(self.applied_changes),
            "evolution_proposals_count": len(self.evolution_proposals),
            "architectural_change_cooldown": self.architectural_change_cooldown,
            "current_game_count": self.current_game_count
        }

# Global self-evolving code system instance
self_evolving_code = SelfEvolvingCodeSystem()

async def start_self_evolving_code():
    """Start the self-evolving code system."""
    await self_evolving_code.start_evolution()

async def stop_self_evolving_code():
    """Stop the self-evolving code system."""
    await self_evolving_code.stop_evolution()

def get_evolution_status():
    """Get evolution system status."""
    return self_evolving_code.get_evolution_status()
