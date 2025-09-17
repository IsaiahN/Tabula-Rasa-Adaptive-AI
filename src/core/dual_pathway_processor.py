#!/usr/bin/env python3
"""
Dual-Pathway Processor - Conscious Architecture Enhancement

Implements Task-Positive Network (TPN) vs Default Mode Network (DMN) switching
inspired by biological consciousness research. Provides explicit cognitive mode
switching based on performance and environmental conditions.

Key Features:
- TPN Mode: Focused, goal-directed problem solving
- DMN Mode: Associative, creative exploration and pattern matching
- Performance-based mode switching
- Smooth transitions between cognitive modes
- Integration with existing meta-cognitive systems
"""

import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json

logger = logging.getLogger(__name__)

class CognitiveMode(Enum):
    """Cognitive processing modes."""
    TPN = "task_positive_network"  # Focused, goal-directed
    DMN = "default_mode_network"   # Associative, exploratory
    TRANSITION = "transition"      # Switching between modes

class ModeSwitchTrigger(Enum):
    """Triggers for mode switching."""
    PERFORMANCE_DECLINE = "performance_decline"
    HIGH_UNCERTAINTY = "high_uncertainty"
    STAGNATION = "stagnation"
    CURIOSITY_DRIVEN = "curiosity_driven"
    EXPLICIT_COMMAND = "explicit_command"

@dataclass
class CognitiveState:
    """Current cognitive state and mode information."""
    current_mode: CognitiveMode
    mode_duration: float  # Time in current mode
    switch_count: int     # Number of mode switches
    performance_metrics: Dict[str, float]
    confidence_level: float
    uncertainty_level: float
    last_switch_time: float
    mode_effectiveness: Dict[CognitiveMode, float] = field(default_factory=dict)

@dataclass
class ModeSwitchDecision:
    """Decision to switch cognitive modes."""
    should_switch: bool
    target_mode: Optional[CognitiveMode]
    trigger: Optional[ModeSwitchTrigger]
    confidence: float
    reasoning: str
    expected_benefit: float

class DualPathwayProcessor:
    """
    Dual-Pathway Processor implementing TPN/DMN mode switching.
    
    This system monitors performance and environmental conditions to determine
    when to switch between focused problem-solving (TPN) and associative
    exploration (DMN) modes.
    """
    
    def __init__(self, 
                 tpn_confidence_threshold: float = 0.7,
                 dmn_activation_threshold: float = 0.3,
                 mode_switch_cooldown: float = 5.0,
                 performance_window: int = 10):
        self.tpn_confidence_threshold = tpn_confidence_threshold
        self.dmn_activation_threshold = dmn_activation_threshold
        self.mode_switch_cooldown = mode_switch_cooldown
        self.performance_window = performance_window
        
        # Current state
        self.cognitive_state = CognitiveState(
            current_mode=CognitiveMode.TPN,
            mode_duration=0.0,
            switch_count=0,
            performance_metrics={},
            confidence_level=1.0,
            uncertainty_level=0.0,
            last_switch_time=time.time()
        )
        
        # Performance tracking
        self.performance_history = deque(maxlen=performance_window)
        self.mode_performance = {
            CognitiveMode.TPN: deque(maxlen=performance_window),
            CognitiveMode.DMN: deque(maxlen=performance_window)
        }
        
        # Mode effectiveness tracking
        self.mode_effectiveness = {
            CognitiveMode.TPN: 0.5,
            CognitiveMode.DMN: 0.5
        }
        
        # Integration components
        self.governor = None
        self.director = None
        self.memory_manager = None
        self.curiosity_system = None
        
        logger.info("Dual-Pathway Processor initialized with TPN/DMN switching")
    
    def integrate_components(self, 
                           governor=None, 
                           director=None, 
                           memory_manager=None, 
                           curiosity_system=None):
        """Integrate with existing Tabula Rasa components."""
        self.governor = governor
        self.director = director
        self.memory_manager = memory_manager
        self.curiosity_system = curiosity_system
        logger.info("Dual-Pathway Processor integrated with existing components")
    
    def update_performance(self, 
                          performance_metrics: Dict[str, Any],
                          action_taken: Optional[int] = None,
                          outcome: Optional[Dict[str, Any]] = None):
        """Update performance metrics and mode effectiveness."""
        current_time = time.time()
        
        # Update performance history
        performance_score = self._calculate_performance_score(performance_metrics)
        self.performance_history.append(performance_score)
        self.mode_performance[self.cognitive_state.current_mode].append(performance_score)
        
        # Update cognitive state
        self.cognitive_state.performance_metrics = performance_metrics
        self.cognitive_state.confidence_level = performance_metrics.get('confidence', 0.5)
        self.cognitive_state.uncertainty_level = performance_metrics.get('uncertainty', 0.5)
        self.cognitive_state.mode_duration = current_time - self.cognitive_state.last_switch_time
        
        # Update mode effectiveness
        self._update_mode_effectiveness()
        
        logger.debug(f"Performance updated: score={performance_score:.3f}, "
                    f"mode={self.cognitive_state.current_mode.value}, "
                    f"confidence={self.cognitive_state.confidence_level:.3f}")
    
    def should_switch_mode(self, 
                          current_context: Dict[str, Any],
                          available_actions: List[int]) -> ModeSwitchDecision:
        """Determine if a mode switch is needed."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.cognitive_state.last_switch_time < self.mode_switch_cooldown:
            return ModeSwitchDecision(
                should_switch=False,
                target_mode=None,
                trigger=None,
                confidence=0.0,
                reasoning="Mode switch cooldown active",
                expected_benefit=0.0
            )
        
        # Analyze current performance
        performance_analysis = self._analyze_performance()
        
        # Check for TPN to DMN switch triggers
        if self.cognitive_state.current_mode == CognitiveMode.TPN:
            dmn_decision = self._evaluate_tpn_to_dmn_switch(performance_analysis, current_context)
            if dmn_decision.should_switch:
                return dmn_decision
        
        # Check for DMN to TPN switch triggers
        elif self.cognitive_state.current_mode == CognitiveMode.DMN:
            tpn_decision = self._evaluate_dmn_to_tpn_switch(performance_analysis, current_context)
            if tpn_decision.should_switch:
                return tpn_decision
        
        return ModeSwitchDecision(
            should_switch=False,
            target_mode=None,
            trigger=None,
            confidence=0.0,
            reasoning="No mode switch needed",
            expected_benefit=0.0
        )
    
    def switch_to_mode(self, 
                      target_mode: CognitiveMode, 
                      trigger: ModeSwitchTrigger,
                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Switch to the specified cognitive mode."""
        current_time = time.time()
        previous_mode = self.cognitive_state.current_mode
        
        # Update cognitive state
        self.cognitive_state.current_mode = target_mode
        self.cognitive_state.mode_duration = 0.0
        self.cognitive_state.switch_count += 1
        self.cognitive_state.last_switch_time = current_time
        
        # Generate mode-specific context
        mode_context = self._generate_mode_context(target_mode, context)
        
        # Log mode switch
        logger.info(f"Mode switch: {previous_mode.value} -> {target_mode.value} "
                   f"(trigger: {trigger.value}, duration: {self.cognitive_state.mode_duration:.1f}s)")
        
        return {
            'previous_mode': previous_mode.value,
            'current_mode': target_mode.value,
            'trigger': trigger.value,
            'mode_context': mode_context,
            'switch_count': self.cognitive_state.switch_count,
            'mode_duration': self.cognitive_state.mode_duration
        }
    
    def get_mode_specific_actions(self, 
                                 available_actions: List[int],
                                 context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get actions appropriate for the current cognitive mode."""
        if self.cognitive_state.current_mode == CognitiveMode.TPN:
            return self._get_tpn_actions(available_actions, context)
        elif self.cognitive_state.current_mode == CognitiveMode.DMN:
            return self._get_dmn_actions(available_actions, context)
        else:
            return self._get_transition_actions(available_actions, context)
    
    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate a single performance score from metrics."""
        # Weight different performance indicators
        confidence = metrics.get('confidence', 0.5)
        success_rate = metrics.get('success_rate', 0.5)
        learning_progress = metrics.get('learning_progress', 0.0)
        energy_efficiency = metrics.get('energy_efficiency', 0.5)
        
        # Combined score (0.0 to 1.0)
        score = (0.3 * confidence + 
                0.3 * success_rate + 
                0.2 * learning_progress + 
                0.2 * energy_efficiency)
        
        return min(1.0, max(0.0, score))
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze current performance trends."""
        if len(self.performance_history) < 3:
            return {'trend': 'insufficient_data', 'confidence': 0.0}
        
        recent_scores = list(self.performance_history)[-5:]
        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        
        return {
            'trend': 'improving' if trend > 0.01 else 'declining' if trend < -0.01 else 'stable',
            'slope': trend,
            'average': np.mean(recent_scores),
            'volatility': np.std(recent_scores),
            'confidence': min(1.0, len(recent_scores) / 5.0)
        }
    
    def _evaluate_tpn_to_dmn_switch(self, 
                                   performance_analysis: Dict[str, Any],
                                   context: Dict[str, Any]) -> ModeSwitchDecision:
        """Evaluate switching from TPN to DMN mode."""
        # Check performance decline
        if (performance_analysis['trend'] == 'declining' and 
            performance_analysis['average'] < self.dmn_activation_threshold):
            return ModeSwitchDecision(
                should_switch=True,
                target_mode=CognitiveMode.DMN,
                trigger=ModeSwitchTrigger.PERFORMANCE_DECLINE,
                confidence=0.8,
                reasoning=f"Performance declining (avg: {performance_analysis['average']:.3f})",
                expected_benefit=0.3
            )
        
        # Check high uncertainty
        if self.cognitive_state.uncertainty_level > 0.7:
            return ModeSwitchDecision(
                should_switch=True,
                target_mode=CognitiveMode.DMN,
                trigger=ModeSwitchTrigger.HIGH_UNCERTAINTY,
                confidence=0.7,
                reasoning=f"High uncertainty detected ({self.cognitive_state.uncertainty_level:.3f})",
                expected_benefit=0.4
            )
        
        # Check stagnation (long time in TPN with low performance)
        if (self.cognitive_state.mode_duration > 30 and 
            performance_analysis['average'] < 0.4):
            return ModeSwitchDecision(
                should_switch=True,
                target_mode=CognitiveMode.DMN,
                trigger=ModeSwitchTrigger.STAGNATION,
                confidence=0.6,
                reasoning=f"Stagnation detected (duration: {self.cognitive_state.mode_duration:.1f}s)",
                expected_benefit=0.5
            )
        
        return ModeSwitchDecision(should_switch=False, target_mode=None, trigger=None,
                                confidence=0.0, reasoning="No TPN->DMN switch needed", expected_benefit=0.0)
    
    def _evaluate_dmn_to_tpn_switch(self, 
                                   performance_analysis: Dict[str, Any],
                                   context: Dict[str, Any]) -> ModeSwitchDecision:
        """Evaluate switching from DMN to TPN mode."""
        # Check if performance has improved
        if (performance_analysis['trend'] == 'improving' and 
            performance_analysis['average'] > self.tpn_confidence_threshold):
            return ModeSwitchDecision(
                should_switch=True,
                target_mode=CognitiveMode.TPN,
                trigger=ModeSwitchTrigger.PERFORMANCE_DECLINE,  # Reusing enum
                confidence=0.8,
                reasoning=f"Performance improved (avg: {performance_analysis['average']:.3f})",
                expected_benefit=0.4
            )
        
        # Check if we have a clear goal and high confidence
        if (context.get('goal_clarity', 0.5) > 0.7 and 
            self.cognitive_state.confidence_level > 0.6):
            return ModeSwitchDecision(
                should_switch=True,
                target_mode=CognitiveMode.TPN,
                trigger=ModeSwitchTrigger.EXPLICIT_COMMAND,
                confidence=0.7,
                reasoning="Clear goal and high confidence detected",
                expected_benefit=0.3
            )
        
        return ModeSwitchDecision(should_switch=False, target_mode=None, trigger=None,
                                confidence=0.0, reasoning="No DMN->TPN switch needed", expected_benefit=0.0)
    
    def _generate_mode_context(self, mode: CognitiveMode, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate context specific to the cognitive mode."""
        if mode == CognitiveMode.TPN:
            return {
                'focus_mode': 'goal_directed',
                'exploration_level': 'low',
                'pattern_matching_weight': 0.3,
                'logical_reasoning_weight': 0.7,
                'creativity_level': 'low',
                'directive': 'Focus on achieving the current goal efficiently'
            }
        elif mode == CognitiveMode.DMN:
            return {
                'focus_mode': 'associative',
                'exploration_level': 'high',
                'pattern_matching_weight': 0.8,
                'logical_reasoning_weight': 0.2,
                'creativity_level': 'high',
                'directive': 'Explore associatively and find novel connections'
            }
        else:
            return {
                'focus_mode': 'transitional',
                'exploration_level': 'medium',
                'pattern_matching_weight': 0.5,
                'logical_reasoning_weight': 0.5,
                'creativity_level': 'medium',
                'directive': 'Transitioning between cognitive modes'
            }
    
    def _get_tpn_actions(self, available_actions: List[int], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get actions for Task-Positive Network mode (focused, goal-directed)."""
        actions = []
        
        # Prioritize goal-directed actions
        for action in available_actions:
            if action in [1, 2, 3, 4]:  # Basic movement actions
                actions.append({
                    'action': action,
                    'priority': 0.8,
                    'reasoning': 'Direct goal pursuit',
                    'mode': 'TPN'
                })
            elif action == 6:  # ACTION6 (coordinate-based)
                actions.append({
                    'action': action,
                    'priority': 0.9,
                    'reasoning': 'Precise coordinate targeting',
                    'mode': 'TPN'
                })
        
        return sorted(actions, key=lambda x: x['priority'], reverse=True)
    
    def _get_dmn_actions(self, available_actions: List[int], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get actions for Default Mode Network (associative, exploratory)."""
        actions = []
        
        # Prioritize exploratory and creative actions
        for action in available_actions:
            if action in [5, 7]:  # Creative/experimental actions
                actions.append({
                    'action': action,
                    'priority': 0.9,
                    'reasoning': 'Creative exploration',
                    'mode': 'DMN'
                })
            elif action == 6:  # ACTION6 with pattern matching
                actions.append({
                    'action': action,
                    'priority': 0.7,
                    'reasoning': 'Pattern-based coordinate exploration',
                    'mode': 'DMN'
                })
            else:
                actions.append({
                    'action': action,
                    'priority': 0.5,
                    'reasoning': 'General exploration',
                    'mode': 'DMN'
                })
        
        return sorted(actions, key=lambda x: x['priority'], reverse=True)
    
    def _get_transition_actions(self, available_actions: List[int], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get actions during mode transition."""
        actions = []
        
        # Balanced approach during transition
        for action in available_actions:
            actions.append({
                'action': action,
                'priority': 0.6,
                'reasoning': 'Transitional exploration',
                'mode': 'TRANSITION'
            })
        
        return actions
    
    def _update_mode_effectiveness(self):
        """Update effectiveness scores for each mode."""
        for mode in CognitiveMode:
            if mode == CognitiveMode.TRANSITION:
                continue
                
            if len(self.mode_performance[mode]) > 0:
                recent_performance = list(self.mode_performance[mode])[-5:]
                self.mode_effectiveness[mode] = np.mean(recent_performance)
    
    def get_consciousness_metrics(self) -> Dict[str, Any]:
        """Get metrics related to consciousness-like behavior."""
        return {
            'current_mode': self.cognitive_state.current_mode.value,
            'mode_duration': self.cognitive_state.mode_duration,
            'switch_count': self.cognitive_state.switch_count,
            'confidence_level': self.cognitive_state.confidence_level,
            'uncertainty_level': self.cognitive_state.uncertainty_level,
            'mode_effectiveness': {mode.value: eff for mode, eff in self.mode_effectiveness.items()},
            'performance_trend': self._analyze_performance()['trend'],
            'consciousness_score': self._calculate_consciousness_score()
        }
    
    def _calculate_consciousness_score(self) -> float:
        """Calculate a consciousness-like behavior score."""
        # Factors that indicate consciousness-like behavior
        mode_switching_adaptability = min(1.0, self.cognitive_state.switch_count / 10.0)
        performance_awareness = 1.0 - abs(0.5 - self.cognitive_state.confidence_level)
        uncertainty_handling = self.cognitive_state.uncertainty_level  # Higher uncertainty = more conscious
        
        consciousness_score = (
            0.4 * mode_switching_adaptability +
            0.3 * performance_awareness +
            0.3 * uncertainty_handling
        )
        
        return min(1.0, max(0.0, consciousness_score))

# Factory function for easy integration
def create_dual_pathway_processor(**kwargs) -> DualPathwayProcessor:
    """Create a configured dual-pathway processor."""
    return DualPathwayProcessor(**kwargs)
