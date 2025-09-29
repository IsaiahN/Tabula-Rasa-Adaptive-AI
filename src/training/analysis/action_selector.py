#!/usr/bin/env python3
"""
Intelligent Action Selector for ARC-AGI-3
Uses frame analysis, available actions, and system intelligence to select optimal actions.
Integrates OpenCV analysis, pattern matching, and coordinate intelligence.
"""

import random
import logging
import numpy as np
import asyncio
import inspect
from typing import Dict, List, Optional, Any, Tuple
from .frame_analyzer import FrameAnalyzer
from ..api.api_manager import APIManager

# Import advanced cognitive systems
try:
    from ...core.tree_evaluation_simulation import TreeEvaluationSimulationEngine
    from ...core.action_sequence_optimizer import ActionSequenceOptimizer
    from ...core.enhanced_exploration_strategies import EnhancedExplorationSystem, ExplorationState
    from ...core.predictive_core import PredictiveCore, SimulationHypothesis
    TREE_EVALUATION_AVAILABLE = True
except ImportError:
    TREE_EVALUATION_AVAILABLE = False
    TreeEvaluationSimulationEngine = None
    ActionSequenceOptimizer = None
    EnhancedExplorationSystem = None
    PredictiveCore = None

# Import new advanced action systems
try:
    from src.core.visual_interactive_system import VisualInteractiveSystem
    from src.core.stagnation_intervention_system import StagnationInterventionSystem
    from src.core.strategy_discovery_system import StrategyDiscoverySystem
    from src.core.enhanced_frame_analysis import EnhancedFrameAnalysisSystem
    from src.core.systematic_exploration_system import SystematicExplorationSystem
    from src.core.safety_mechanisms import SafetyMechanisms
    from src.core.systematic_button_discovery import SystematicButtonDiscovery
    from src.core.stagnation_intervention_system import StagnationInterventionSystem

    # Import enhanced attention + communication system (TIER 1)
    from src.core.weighted_communication_system import WeightedCommunicationSystem, MessagePriority
    ADVANCED_ACTION_SYSTEMS_AVAILABLE = True
    print("[OK] ADVANCED SYSTEMS IMPORTED SUCCESSFULLY")
except ImportError as e:
    print(f"[ERROR] ADVANCED SYSTEMS IMPORT FAILED: {e}")
    ADVANCED_ACTION_SYSTEMS_AVAILABLE = False
    VisualInteractiveSystem = None
    StagnationInterventionSystem = None
    StrategyDiscoverySystem = None
    EnhancedFrameAnalysisSystem = None
    SystematicExplorationSystem = None
    EmergencyOverrideSystem = None
    SystematicButtonDiscovery = None
    StagnationInterventionSystem = None
except Exception as e:
    print(f"[ERROR] ADVANCED SYSTEMS INITIALIZATION FAILED: {e}")
    ADVANCED_ACTION_SYSTEMS_AVAILABLE = False
    VisualInteractiveSystem = None
    StagnationInterventionSystem = None
    StrategyDiscoverySystem = None
    EnhancedFrameAnalysisSystem = None
    SystematicExplorationSystem = None
    EmergencyOverrideSystem = None
    SystematicButtonDiscovery = None
    StagnationInterventionSystem = None

logger = logging.getLogger(__name__)

class ActionSelector:
    """Intelligent action selector that uses multiple sources of information and advanced systems."""
    
    def __init__(self, api_manager: APIManager, bayesian_inference_system=None, graph_traversal_system=None):
        self.api_manager = api_manager
        self.frame_analyzer = FrameAnalyzer()
        self.action_history = []
        self.success_patterns = {}
        self.coordinate_intelligence = {}
        self.performance_history = []
        self.learning_cycles = 0

        # Advanced Tier 3 systems
        self.bayesian_inference_system = bayesian_inference_system
        self.graph_traversal_system = graph_traversal_system
        self.advanced_systems_available = (bayesian_inference_system is not None and
                                         graph_traversal_system is not None)

        if self.advanced_systems_available:
            logger.debug("[OK] Tier 3 systems (Bayesian/Graph) available for action selection")
        else:
            logger.debug("[INFO] Tier 3 systems not available - using standard action selection")
        
        # Advanced analysis state
        self.last_frame_analysis = None
        self.action_effectiveness = {}
        self.coordinate_success_rates = {}
        self.pattern_matches = []
        
        # Initialize new advanced action systems
        logger.debug(f"Tier 4 systems status: {ADVANCED_ACTION_SYSTEMS_AVAILABLE}")
        if ADVANCED_ACTION_SYSTEMS_AVAILABLE:
            try:
                self.visual_interactive_system = VisualInteractiveSystem()
                self.strategy_discovery_system = StrategyDiscoverySystem()
                self.frame_analysis_system = EnhancedFrameAnalysisSystem()
                self.exploration_system = SystematicExplorationSystem()
                self.safety_mechanisms = SafetyMechanisms()
                self.button_discovery_system = SystematicButtonDiscovery()
                self.stagnation_intervention_system = StagnationInterventionSystem()

                # Communication system will be set from continuous learning loop
                self.communication_system = None
                logger.debug("[OK] Tier 4 systems (Visual/Strategy/Analysis) initialized successfully")
            except Exception as e:
                logger.error(f"[ERROR] Failed to initialize Tier 4 systems: {e}")
                # Fallback to None
                self.visual_interactive_system = None
                self.strategy_discovery_system = None
                self.frame_analysis_system = None
                self.exploration_system = None
                self.safety_mechanisms = None
                self.button_discovery_system = None
                self.stagnation_intervention_system = None
                self.communication_system = None
        else:
            logger.debug("[INFO] Tier 4 systems not available - using standard systems")
            self.visual_interactive_system = None
            self.strategy_discovery_system = None
            self.frame_analysis_system = None
            self.exploration_system = None
            self.safety_mechanisms = None
            self.button_discovery_system = None
            self.stagnation_intervention_system = None
            self.communication_system = None
        
        # Action repetition penalty system
        self.action_repetition_penalties = {}  # action_id -> penalty_score
        self.recent_actions = []  # Track recent actions for repetition detection
        self.max_recent_actions = 20  # Keep last 20 actions
        # Available buttons list (coordinates) - similar to available_actions but for Action 6
        self.available_buttons = []
        
        # OpenCV object testing system
        self.tested_objects = {}  # coordinate -> {'interactive': bool, 'test_count': int, 'frame_changes': int}
        self.interactive_objects = set()  # Set of confirmed interactive coordinates
        self.non_interactive_objects = set()  # Set of confirmed non-interactive coordinates
        self.object_test_threshold = 3  # Number of tests before marking as non-interactive
        
        # Action availability tracking system
        self.previous_available_actions = set()  # Track previous available actions
        self.action_availability_changes = {}  # action_id -> {'count': int, 'last_change': timestamp}
        self.pseudo_buttons = set()  # Actions that cause available_actions to change
        
        # Initialize advanced cognitive systems
        self.tree_evaluation_engine = None
        self.action_sequence_optimizer = None
        self.exploration_system = None
        self.predictive_core = None
        
        # Initialize Bayesian and GAN systems
        self.bayesian_scorer = None
        self.gan_system = None
        self._initialize_advanced_systems()
        
        # Initialize game type classifier
        self.game_type_classifier = None
        self._initialize_game_type_classifier()

        # Frame caching system for button candidate detection
        self.previous_frame_cache = None  # Store previous frame for comparison
        self.frame_comparison_threshold = 0.3  # 30% change threshold for detecting notable changes
        self.button_candidate_cache = {}  # Cache detected button candidates by game_id
        self.frame_change_history = []  # Track frame changes over time
        
        if TREE_EVALUATION_AVAILABLE:
            try:
                # Initialize Tree Evaluation Simulation Engine
                self.tree_evaluation_engine = TreeEvaluationSimulationEngine()
                logger.debug("Tree Evaluation Simulation Engine initialized")
                
                # Initialize Action Sequence Optimizer
                self.action_sequence_optimizer = ActionSequenceOptimizer()
                logger.debug("[TARGET] Action Sequence Optimizer initialized")
                
                # Initialize Enhanced Exploration System
                self.exploration_system = EnhancedExplorationSystem()
                logger.debug("[CHECK] Enhanced Exploration System initialized")
                
                # Initialize Predictive Core with correct parameters
                if PredictiveCore is not None:
                    self.predictive_core = PredictiveCore(
                        visual_size=(3, 64, 64),  # (channels, height, width)
                        proprioception_size=12,
                        hidden_size=256,
                        architecture="lstm"
                    )
                    logger.debug("Predictive Core initialized")
                
            except Exception as e:
                logger.warning(f"Failed to initialize Tree Evaluation systems: {e}")
                self.tree_evaluation_engine = None
                self.action_sequence_optimizer = None
                self.exploration_system = None
                self.predictive_core = None
        
        logger.info("Advanced Action Selector initialized with all available systems")
    
    def _initialize_advanced_systems(self):
        """Initialize Bayesian and GAN systems."""
        try:
            # Initialize Bayesian Success Scorer (use singleton)
            from src.core.bayesian_success_scorer import create_bayesian_success_scorer
            self.bayesian_scorer = create_bayesian_success_scorer()
            logger.debug("[OK] Bayesian Success Scorer connected")
        except Exception as e:
            logger.warning(f"Failed to initialize Bayesian Success Scorer: {e}")
        
        try:
            # Initialize GAN system
            from src.core.gan_system import PatternAwareGAN
            self.gan_system = PatternAwareGAN()
            logger.info("[OK] Pattern-Aware GAN initialized")
        except ImportError as e:
            logger.warning(f"PyTorch not available, GAN system disabled: {e}")
            self.gan_system = None
        except Exception as e:
            logger.warning(f"Failed to initialize GAN system: {e}")
            self.gan_system = None
    
    def _initialize_game_type_classifier(self):
        """Initialize game type classifier for game-specific knowledge."""
        try:
            from src.learning.game_type_classifier import get_game_type_classifier
            self.game_type_classifier = get_game_type_classifier()
            logger.info("[OK] Game Type Classifier initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Game Type Classifier: {e}")

    def set_communication_system(self, communication_system):
        """Set the communication system for coordinated action selection."""
        self.communication_system = communication_system
        if communication_system:
            logger.info("[OK] Communication system set for coordinated action selection")
        else:
            logger.warning("[WARNING] Communication system set to None")

    async def select_action(self, game_state: Dict[str, Any], available_actions: List[int]) -> Dict[str, Any]:
        """Select the best action using advanced OpenCV analysis and pattern matching."""

        try:
            # PRIORITY: Validate and prioritize available actions (removes action 0, prioritizes 1-4)
            original_actions = available_actions.copy()
            available_actions = self._validate_and_prioritize_actions(available_actions)

            if original_actions != available_actions:
                removed_actions = set(original_actions) - set(available_actions)
                if removed_actions:
                    logger.warning(f"[PRIORITY] Filtered out invalid actions: {list(removed_actions)}")
                logger.info(f"[PRIORITY] Prioritized actions: {available_actions} (original: {original_actions})")

            # Extract frame data and current state
            frame_data = game_state.get('frame', [])
            current_score = game_state.get('score', 0)
            game_state_status = game_state.get('state', 'NOT_FINISHED')
            game_id = game_state.get('game_id', 'unknown')

            logger.info(f"[CHECK] DEBUG: Starting action selection with frame_data type: {type(frame_data)}")
            if hasattr(frame_data, 'shape'):
                logger.info(f"[CHECK] DEBUG: frame_data shape: {frame_data.shape}")
            elif isinstance(frame_data, list):
                logger.info(f"[CHECK] DEBUG: frame_data length: {len(frame_data)}")
        except Exception as e:
            logger.error(f"[CHECK] DEBUG: Error in initial action selection setup: {e}")
            import traceback
            logger.error(f"[CHECK] DEBUG: Traceback: {traceback.format_exc()}")
            raise
        
        # ADVANCED SYSTEMS: Use Bayesian Inference + Graph Traversal if available
        if self.advanced_systems_available:
            try:
                logger.info("[ADVANCED] Using Bayesian Inference + Graph Traversal for action selection")

                # 1. Generate Bayesian prediction for action candidates
                action_candidates = []
                for action_id in available_actions:
                    action_candidates.append({
                        'id': action_id,
                        'x': 32 if action_id == 6 else None,  # Default coordinate for ACTION6
                        'y': 32 if action_id == 6 else None
                    })

                # Current context for Bayesian inference
                current_context = {
                    'game_id': game_id,
                    'score': current_score,
                    'state': game_state_status,
                    'frame_data': frame_data,
                    'available_actions': available_actions
                }

                # Generate Bayesian prediction
                bayesian_prediction = await self.bayesian_inference_system.generate_prediction(
                    action_candidates, current_context, game_id, f"session_{game_id}"
                )

                if bayesian_prediction and bayesian_prediction.success_probability > 0.5:
                    predicted_action = bayesian_prediction.predicted_action
                    logger.info(f"[ADVANCED] Bayesian prediction: Action {predicted_action.get('id')} "
                               f"(probability: {bayesian_prediction.success_probability:.3f}, "
                               f"confidence: {bayesian_prediction.confidence_level:.3f})")

                    return {
                        'action': f"ACTION{predicted_action.get('id')}",
                        'x': predicted_action.get('x'),
                        'y': predicted_action.get('y'),
                        'confidence': bayesian_prediction.confidence_level,
                        'reason': f'Bayesian inference prediction (probability: {bayesian_prediction.success_probability:.3f})',
                        'source': 'bayesian_inference_engine'
                    }
                else:
                    logger.info("[ADVANCED] Bayesian prediction below threshold, continuing with fallback systems")

            except Exception as e:
                logger.error(f"[ADVANCED] Error using advanced systems: {e}")
                logger.info("[ADVANCED] Falling back to traditional action selection")

        # Check if primary actions 1-4 are available
        primary_actions_available = any(action in [1, 2, 3, 4] for action in available_actions)

        if primary_actions_available:
            logger.info("[PRIORITY] Primary actions 1-4 detected - prioritizing over action 6 systems")
        else:
            logger.info("[PRIORITY] No primary actions 1-4 available - enabling action 6 fallback systems")
        
        # 0. STAGNATION INTERVENTION - Check for stagnation and trigger intervention
        if self.stagnation_intervention_system:
            try:
                logger.info(f"⚠ STAGNATION SYSTEM ACTIVE - Analyzing frame {len(self.stagnation_intervention_system.frame_history) if hasattr(self.stagnation_intervention_system, 'frame_history') else 'unknown'}")
                logger.info(f"[CHECK] DEBUG: About to call stagnation_intervention_system.analyze_frame with frame_data type: {type(frame_data)}")
                stagnation_event = await self.stagnation_intervention_system.analyze_frame(frame_data, game_state)
                
                if stagnation_event and stagnation_event.intervention_required:
                    logger.warning(f"⚠ STAGNATION DETECTED: {stagnation_event.type.value} (severity: {stagnation_event.severity:.2f})")
                    
                    # Trigger multi-system intervention
                    intervention = await self.stagnation_intervention_system.trigger_intervention(stagnation_event)
                    
                    # Use emergency actions to break stagnation
                    if intervention.get('emergency_actions'):
                        emergency_action = intervention['emergency_actions'][0]  # Use first emergency action
                        logger.warning(f"🚨 EMERGENCY INTERVENTION: {emergency_action['reason']}")
                        return emergency_action
            except Exception as e:
                logger.error(f"[CHECK] DEBUG: Error in stagnation analysis: {e}")
                import traceback
                logger.error(f"[CHECK] DEBUG: Stagnation analysis traceback: {traceback.format_exc()}")
                # Don't raise - continue with normal action selection
                logger.warning(f"⚠ STAGNATION SYSTEM ERROR - Continuing with normal action selection")
        else:
            logger.warning(f"⚠ STAGNATION SYSTEM NOT AVAILABLE - Advanced systems: {ADVANCED_ACTION_SYSTEMS_AVAILABLE}")
        
        # 0.5. SYSTEMATIC BUTTON DISCOVERY - Only when actions 1-4 are NOT available
        if (self.button_discovery_system and 6 in available_actions and 
            not primary_actions_available and 
            not self.stagnation_intervention_system.is_intervention_active()):
            # Initialize button discovery for new games
            if not hasattr(self, '_current_game_id') or self._current_game_id != game_id:
                await self.button_discovery_system.start_new_game(game_id, frame_data)
                self._current_game_id = game_id
                logger.info(f"[TARGET] BUTTON DISCOVERY INITIALIZED for game {game_id}")
            
            # Get next coordinate to test
            next_test_coord = await self.button_discovery_system.get_next_test_coordinate()
            if next_test_coord:
                logger.info(f"[TARGET] SYSTEMATIC TESTING: Testing coordinate {next_test_coord}")
                return {
                    'id': 6,
                    'x': next_test_coord[0],
                    'y': next_test_coord[1],
                    'reason': f"Systematic button discovery - testing coordinate {next_test_coord}",
                    'confidence': 0.9,
                    'source': 'button_discovery'
                }

        # BUTTON-FIRST DISCOVERY PASS - Only when actions 1-4 are NOT available
        discovered_button_suggestions = []
        try:
            if 6 in available_actions and not primary_actions_available:
                # Gather candidates from frame analyzer, persistent button_priorities, and learned coordinates
                frame_buttons = self._gather_buttons_from_frame(self.last_frame_analysis or self.frame_analyzer.analyze_frame(game_state.get('frame', [])))
                # Load persisted button priorities if available via a coordinate_intelligence_system
                persisted_buttons = []
                try:
                    from src.core.coordinate_intelligence_system import CoordinateIntelligenceSystem
                    if not hasattr(self, 'coordinate_intelligence_system'):
                        self.coordinate_intelligence_system = CoordinateIntelligenceSystem()
                    persisted_buttons = await self.coordinate_intelligence_system.get_top_buttons(game_state.get('game_id', 'unknown'))
                except Exception:
                    # If persistent system not available, ignore
                    persisted_buttons = []

                # Merge and prioritize: frame buttons first, then persisted, then learned coordinates
                seen = set()
                for b in frame_buttons + persisted_buttons + list(self.coordinate_success_rates.keys()):
                    if isinstance(b, tuple):
                        coord = b
                    elif isinstance(b, dict) and 'coordinate' in b:
                        coord = tuple(b['coordinate'])
                    else:
                        continue
                    if coord in seen:
                        continue
                    seen.add(coord)
                    discovered_button_suggestions.append({
                        'action': 'ACTION6',
                        'x': coord[0],
                        'y': coord[1],
                        'confidence': 0.9 if coord in frame_buttons else 0.75,
                        'reason': 'Discovered button candidate (fallback)',
                        'source': 'button_first_discovery'
                    })
        except Exception as e:
            logger.debug(f"Button-first discovery failed: {e}")
        
        # 1. ADVANCED FRAME ANALYSIS - OpenCV-based visual analysis
        frame_analysis = self.frame_analyzer.analyze_frame(frame_data)
        self.last_frame_analysis = frame_analysis

        # 1.1. PENALTY RESET FOR VISUAL TARGETS - Reset penalties periodically for visual objects
        current_action_count = len(self.action_history) if hasattr(self, 'action_history') else 0
        if self._should_reset_penalties(current_action_count):
            self._reset_visual_target_penalties(frame_analysis)

        # 1.2. EMERGENCY PENALTY RESET - Clear all penalties when clearly stuck
        self._emergency_penalty_reset(game_state, current_action_count)

        # 1.5. FRAME COMPARISON BUTTON CANDIDATE DETECTION - Only when actions 1-4 are NOT available
        frame_comparison_suggestions = []
        if 6 in available_actions and not primary_actions_available and self.previous_frame_cache is not None:
            frame_comparison_suggestions = self._detect_button_candidates_by_frame_changes(
                self.previous_frame_cache, frame_data
            )
            if frame_comparison_suggestions:
                logger.info(f"[FRAME] Frame comparison detected {len(frame_comparison_suggestions)} button candidates")

        # Cache current frame for next comparison
        self._cache_frame_data(frame_data, game_id)

        # 2. PATTERN MATCHING - Learn from successful patterns
        pattern_suggestions = self._analyze_patterns(frame_analysis, available_actions)

        # 3. COORDINATE INTELLIGENCE - Learn from successful coordinates
        coordinate_suggestions = await self._get_intelligent_coordinates(frame_analysis, available_actions, game_state)
        
        # 4. OPENCV OBJECT TESTING - Modified prioritization based on available actions
        opencv_suggestions = []
        if 6 in available_actions:
            # Check if this is an Action 6 centric game - only when no primary actions available
            is_action6_centric = self._is_action6_centric_game(game_state, available_actions) and not primary_actions_available
            
            if is_action6_centric:
                logger.info("[TARGET] ACTION 6 CENTRIC GAME DETECTED - Focusing on button discovery and testing")
            
            # Get confirmed interactive objects (only prioritize for Action 6 when no primary actions)
            if not primary_actions_available:
                interactive_suggestions = self._get_interactive_objects(frame_analysis)
                untested_suggestions = self._get_untested_objects(frame_analysis)
                opencv_detected_suggestions = self._detect_opencv_targets(frame_analysis, [6])
                
                # For Action 6 centric games, prioritize button testing
                if is_action6_centric:
                    # Increase priority for untested objects to discover more buttons
                    for suggestion in untested_suggestions:
                        suggestion['confidence'] = min(1.0, suggestion['confidence'] + 0.1)
                        suggestion['reason'] += " (Action 6 centric - button discovery priority)"
                
                # Combine all OpenCV-based suggestions
                opencv_suggestions = interactive_suggestions + untested_suggestions + opencv_detected_suggestions
            
            # If no OpenCV objects found or primary actions available, add fallback suggestions
            if not opencv_suggestions and len(available_actions) > 1:
                # Prioritize Action 5 as primary fallback when stuck
                if 5 in available_actions:
                    opencv_suggestions.append({
                        'action': 'ACTION5',
                        'confidence': 0.8,  # High confidence for Action 5 fallback
                        'reason': 'Action 5 fallback - no OpenCV objects found',
                        'source': 'action5_fallback'
                    })
                
                # Add other random suggestions for non-Action6 options
                non_action6_actions = [a for a in available_actions if a != 6 and a != 5]
                for action_id in non_action6_actions[:2]:  # Limit to 2 additional suggestions
                    opencv_suggestions.append({
                        'action': f'ACTION{action_id}',
                        'confidence': 0.9 if action_id in [1, 2, 3, 4] else 0.6,  # Higher confidence for primary actions
                        'reason': f'Primary action {action_id} available' if action_id in [1, 2, 3, 4] else f'Random fallback',
                        'source': 'primary_action' if action_id in [1, 2, 3, 4] else 'random_fallback'
                    })
        else:
            # Fallback to regular OpenCV detection
            opencv_suggestions = self._detect_opencv_targets(frame_analysis, available_actions)
        
        # 5. LEARNING-BASED SUGGESTIONS - Use historical success data
        learning_suggestions = self._generate_learning_suggestions(frame_analysis, available_actions)
        
        # 6. EXPLORATION STRATEGY - Balance exploration vs exploitation
        exploration_suggestions = self._generate_exploration_suggestions(available_actions)
        
        # 6.5. COORDINATE EXPLORATION - LAST RESORT for Action 6 (only when no primary actions and no OpenCV objects)
        if 6 in available_actions and not primary_actions_available and not opencv_suggestions:
            coordinate_exploration_suggestions = self._get_exploration_coordinates(frame_analysis)
            exploration_suggestions.extend(coordinate_exploration_suggestions)
            logger.info("🔍 Using coordinate exploration as LAST RESORT for Action 6 - no OpenCV objects found")
        
        # 7. ADVANCED COGNITIVE SYSTEMS - Tree evaluation and sequence optimization
        advanced_suggestions = self._generate_advanced_suggestions(
            frame_analysis, available_actions, current_score, game_state
        )
        
        # 8. BAYESIAN PATTERN DETECTION - Use Bayesian success scoring
        bayesian_suggestions = self._generate_bayesian_suggestions(
            frame_analysis, available_actions, current_score, game_state
        )
        
        # 9. GAN-GENERATED SUGGESTIONS - Use GAN for synthetic action generation
        # _generate_gan_suggestions may be async (it awaits GAN internals). Await it.
        gan_suggestions = []
        try:
            gan_suggestions = await self._generate_gan_suggestions(
                frame_analysis, available_actions, current_score, game_state
            )
        except AttributeError:
            # If _generate_gan_suggestions is not async for some reason, fall back
            try:
                gan_suggestions = self._generate_gan_suggestions(
                    frame_analysis, available_actions, current_score, game_state
                )
            except Exception as e:
                logger.warning(f"GAN suggestion generation failed: {e}")
        
        # 10. GAME-SPECIFIC KNOWLEDGE - Use knowledge from similar game types
        game_specific_suggestions = self._generate_game_specific_suggestions(
            game_state, available_actions, frame_analysis
        )
        
        # 11. ADVANCED STAGNATION DETECTION - Check for stuck situations
        stagnation_event = None
        if self.stagnation_intervention_system:
            try:
                stagnation_event = await self.stagnation_intervention_system.analyze_frame(frame_data, game_state)
            except Exception as e:
                logger.error(f"Error in stagnation detection: {e}")
        
        # 12. EMERGENCY OVERRIDE CHECK - Check for emergency override conditions using safety mechanisms
        emergency_override = None
        if self.safety_mechanisms:
            try:
                safe_action_history = []
                for a in (self.action_history or []):
                    try:
                        if isinstance(a, dict) and 'id' in a:
                            action_id = int(a.get('id'))
                        elif isinstance(a, (int, float)):
                            action_id = int(a)
                        else:
                            logger.debug(f"Skipping malformed action in history: {a}")
                            continue  # Skip actions without valid IDs

                        safe_action_history.append({
                            'action_id': action_id,
                            'score': float(a.get('score', 0.0)) if isinstance(a, dict) else 0.0
                        })
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Skipping invalid action in history: {a}, error: {e}")
                        continue
                safe_performance_history = [float(x) for x in (self.performance_history or []) if isinstance(x, (int, float))]
                emergency_override = await self.safety_mechanisms.check_game_emergency_override(
                    game_id=str(game_state.get('game_id', 'unknown')),
                    session_id=str(game_state.get('session_id', 'unknown')),
                    current_state={k: v for k, v in game_state.items() if isinstance(k, str)},
                    action_history=safe_action_history,
                    performance_history=safe_performance_history,
                    available_actions=[int(x) for x in available_actions]
                )
            except Exception as e:
                logger.error(f"Error in emergency override check: {e}")
        
        # 13. VISUAL-INTERACTIVE ACTION6 TARGETING - Only when actions 1-4 are NOT available
        visual_targeting_suggestions = []
        if 6 in available_actions and not primary_actions_available and self.visual_interactive_system:
            try:
                visual_analysis = await self.visual_interactive_system.analyze_frame_for_action6_targets(
                    frame_data, game_state.get('game_id', 'unknown'), available_actions
                )
                if visual_analysis.get('recommended_action6_coord'):
                    x, y = visual_analysis['recommended_action6_coord']
                    visual_targeting_suggestions.append({
                        'action': 'ACTION6',
                        'coordinates': (x, y),
                        'confidence': visual_analysis.get('confidence', 0.5),
                        'reason': visual_analysis.get('targeting_reason', 'Visual target detected (fallback)'),
                        'source': 'visual_interactive_targeting'
                    })
            except Exception as e:
                logger.error(f"Error in visual interactive targeting: {e}")
        
        # 14. SYSTEMATIC EXPLORATION - Only when actions 1-4 are NOT available
        exploration_phase_suggestions = []
        if self.exploration_system and not primary_actions_available:
            try:
                grid_dimensions = self._get_grid_dimensions(frame_data)
                x, y, phase_name = await self.exploration_system.get_exploration_coordinates(
                    game_id=game_state.get('game_id', 'unknown'),
                    session_id=game_state.get('session_id', 'unknown'),
                    grid_dimensions=grid_dimensions,
                    available_actions=available_actions
                )
                exploration_phase_suggestions.append({
                    'action': 'ACTION6',
                    'coordinates': (x, y),
                    'confidence': 0.6,
                    'reason': f'Systematic exploration - {phase_name} phase (fallback)',
                    'source': 'systematic_exploration'
                })
            except Exception as e:
                logger.error(f"Error in systematic exploration: {e}")
        
        # 15. STRATEGY DISCOVERY - Check for strategy replication opportunities
        strategy_suggestions = []
        if self.strategy_discovery_system:
            try:
                game_id = game_state.get('game_id', 'unknown')
                should_replicate = await self.strategy_discovery_system.should_attempt_strategy_replication(game_id)
                if should_replicate:
                    best_strategy = await self.strategy_discovery_system.get_best_strategy_for_game(game_id)
                    if best_strategy and best_strategy.action_sequence:
                        # Use the first action from the strategy
                        first_action = best_strategy.action_sequence[0]
                        if first_action in available_actions:
                            strategy_suggestions.append({
                                'action': f'ACTION{first_action}',
                                'confidence': best_strategy.efficiency,
                                'reason': f'Strategy replication - {best_strategy.strategy_id}',
                                'source': 'strategy_discovery'
                            })
            except Exception as e:
                logger.error(f"Error in strategy discovery: {e}")

        # 16. WIN CONDITION INFLUENCE - Apply known win conditions to action selection
        win_condition_suggestions = []
        if self.strategy_discovery_system:
            try:
                game_id = game_state.get('game_id', 'unknown')
                game_type = self.strategy_discovery_system.game_type_classifier.extract_game_type(game_id)

                # Get win conditions for this game type
                win_conditions = await self.strategy_discovery_system.get_win_conditions_for_game_type(game_type)

                if win_conditions:
                    # Process high-success win conditions
                    high_success_conditions = [c for c in win_conditions if c['success_rate'] > 0.7]

                    for condition in high_success_conditions[:3]:  # Top 3 conditions
                        condition_data = condition['condition_data']

                        # Handle action pattern conditions
                        if condition['condition_type'] == 'action_pattern':
                            if 'pattern' in condition_data:
                                pattern = condition_data['pattern']
                                if len(pattern) > 0 and pattern[0] in available_actions:
                                    win_condition_suggestions.append({
                                        'action': f'ACTION{pattern[0]}',
                                        'confidence': condition['success_rate'],
                                        'reason': f'Win condition: action pattern with {condition["success_rate"]:.1%} success rate',
                                        'source': 'win_condition_pattern'
                                    })

                            elif 'dominant_action' in condition_data:
                                dominant_action = condition_data['dominant_action']
                                if dominant_action in available_actions:
                                    win_condition_suggestions.append({
                                        'action': f'ACTION{dominant_action}',
                                        'confidence': condition['success_rate'] * condition_data.get('percentage', 0.5),
                                        'reason': f'Win condition: dominant action {dominant_action} ({condition_data.get("percentage", 0.5):.1%} frequency)',
                                        'source': 'win_condition_dominant'
                                    })

                        # Handle score threshold conditions
                        elif condition['condition_type'] == 'score_threshold':
                            if 'final_score_threshold' in condition_data:
                                threshold = condition_data['final_score_threshold']
                                current_score = game_state.get('score', 0)

                                # If we're below the winning threshold, prioritize score-gaining actions
                                if current_score < threshold:
                                    # Prefer actions that historically lead to score increases
                                    score_gaining_actions = [1, 2, 3, 4, 5]  # Generally score-gaining actions
                                    for action in score_gaining_actions:
                                        if action in available_actions:
                                            win_condition_suggestions.append({
                                                'action': f'ACTION{action}',
                                                'confidence': condition['success_rate'] * 0.8,  # Slightly lower confidence
                                                'reason': f'Win condition: need score {threshold}, current {current_score}',
                                                'source': 'win_condition_score_threshold'
                                            })
                                            break  # Only suggest one action per condition

            except Exception as e:
                logger.error(f"Error applying win conditions: {e}")

        # 16. LOSING STREAK INTERVENTION - Apply intervention guidance when losing streaks detected
        losing_streak_suggestions = []
        try:
            # Try to get losing streak systems from the continuous learning loop
            # This is a bit hacky but necessary since we don't have direct access
            from src.training.core.continuous_learning_loop import ContinuousLearningLoop
            import inspect

            # Look for ContinuousLearningLoop instance in the call stack to get access to losing streak systems
            loop_instance = None
            for frame_info in inspect.stack():
                frame_locals = frame_info.frame.f_locals
                for var_name, var_value in frame_locals.items():
                    if isinstance(var_value, ContinuousLearningLoop):
                        loop_instance = var_value
                        break
                if loop_instance:
                    break

            if (loop_instance and
                hasattr(loop_instance, '_losing_streak_systems_initialized') and
                loop_instance._losing_streak_systems_initialized):

                game_id = game_state.get('game_id', 'unknown')
                game_type = self.strategy_discovery_system.game_type_classifier.extract_game_type(game_id) if self.strategy_discovery_system else 'unknown'

                # Check for active losing streak
                if loop_instance.losing_streak_detector:
                    active_streak = loop_instance.losing_streak_detector.get_streak_for_game(game_type, game_id)

                    if active_streak and active_streak.consecutive_failures >= 3:
                        logger.info(f" LOSING STREAK GUIDANCE: {active_streak.consecutive_failures} consecutive failures detected")

                        # Get active interventions
                        if loop_instance.escalated_intervention_system:
                            active_interventions = loop_instance.escalated_intervention_system.get_active_interventions()

                            for intervention in active_interventions:
                                guidance = loop_instance.escalated_intervention_system.get_intervention_guidance(
                                    intervention['intervention_id']
                                )

                                if guidance:
                                    losing_streak_suggestions.extend(
                                        self._apply_intervention_guidance(guidance, available_actions, game_state)
                                    )

                        # Get anti-pattern avoidance suggestions
                        if loop_instance.anti_pattern_learner:
                            # Simulate proposed actions to check for anti-patterns
                            proposed_action_sequence = [a for a in available_actions[:3]]  # Sample available actions
                            proposed_coordinates = [(100, 100), (200, 200)] if 6 in available_actions else []

                            pattern_suggestions = loop_instance.anti_pattern_learner.get_pattern_suggestions(
                                game_type, proposed_action_sequence, proposed_coordinates
                            )

                            if pattern_suggestions.get('warnings'):
                                logger.warning(f" ANTI-PATTERN WARNING: {len(pattern_suggestions['warnings'])} patterns to avoid")

                            # Convert alternative suggestions to action suggestions
                            for suggestion in pattern_suggestions.get('suggestions', []):
                                if suggestion['type'] == 'action_substitution':
                                    for alt_action in suggestion.get('suggested_actions', []):
                                        if alt_action in available_actions:
                                            losing_streak_suggestions.append({
                                                'action': f'ACTION{alt_action}',
                                                'confidence': 0.8,
                                                'reason': f'Anti-pattern avoidance: alternative to failing pattern',
                                                'source': 'losing_streak_intervention'
                                            })

                                elif suggestion['type'] == 'coordinate_offset' and 6 in available_actions:
                                    coords = suggestion.get('coordinates', [100, 100])
                                    losing_streak_suggestions.append({
                                        'action': 'ACTION6',
                                        'x': coords[0],
                                        'y': coords[1],
                                        'confidence': 0.7,
                                        'reason': f'Anti-pattern avoidance: coordinate offset from failed cluster',
                                        'source': 'losing_streak_intervention'
                                    })

                        if losing_streak_suggestions:
                            logger.info(f" LOSING STREAK SUGGESTIONS: Generated {len(losing_streak_suggestions)} intervention suggestions")

        except Exception as e:
            logger.error(f"Error applying losing streak guidance: {e}")

        # 17. ENHANCED FRAME ANALYSIS - Analyze frame changes for better action selection
        if self.frame_analysis_system and hasattr(self, 'last_frame_data') and self.last_frame_data is not None and isinstance(self.last_frame_data, list) and len(self.last_frame_data) > 0:
            try:
                logger.info(f"[CHECK] DEBUG: About to analyze frame changes with last_frame_data type: {type(self.last_frame_data)}")
                logger.info(f"[CHECK] DEBUG: frame_data type: {type(frame_data)}")
                frame_change_analysis = await self.frame_analysis_system.analyze_frame_changes(
                    before_frame=self.last_frame_data,
                    after_frame=frame_data,
                    game_id=game_state.get('game_id', 'unknown'),
                    action_number=(len(self.action_history) if hasattr(self, 'action_history') else 0),
                    coordinates=getattr(self, 'last_coordinates', None)
                )
                if frame_change_analysis:
                    # Update frame change history for stagnation detection
                    if not hasattr(self, 'frame_change_history'):
                        self.frame_change_history = []
                    self.frame_change_history.append(frame_change_analysis.movement_detected)
                    if len(self.frame_change_history) > 20:
                        self.frame_change_history = self.frame_change_history[-20:]
            except Exception as e:
                logger.error(f"[CHECK] DEBUG: Error in enhanced frame analysis: {e}")
                import traceback
                logger.error(f"[CHECK] DEBUG: Frame analysis traceback: {traceback.format_exc()}")
                raise
        
        # Store current frame for next analysis
        self.last_frame_data = frame_data
        
        # 11. PSEUDO BUTTON SUGGESTIONS - Actions that previously discovered new actions
        pseudo_button_suggestions = self._generate_pseudo_button_suggestions(available_actions)
        
        # 11. ACTION 5 FALLBACK - Detect stuck situations and suggest Action 5
        action5_suggestions = []
        if self._detect_stuck_situation(game_state, available_actions):
            action5_suggestions = self._generate_action5_fallback_suggestions(available_actions)
        
        # 12.5. BUTTON DISCOVERY SUGGESTIONS - Only when actions 1-4 are NOT available
        button_discovery_suggestions = []
        if self.button_discovery_system and 6 in available_actions and not primary_actions_available:
            button_discovery_suggestions = self.button_discovery_system.get_button_suggestions(limit=3)
            if button_discovery_suggestions:
                logger.info(f"[TARGET] BUTTON DISCOVERY: Found {len(button_discovery_suggestions)} button suggestions")
        
        # 13. COMBINE ALL SUGGESTIONS - Multi-source decision making
        all_suggestions = self._combine_suggestions(
            discovered_button_suggestions,
            frame_comparison_suggestions,  # Frame change button candidates
            pattern_suggestions,
            coordinate_suggestions,
            opencv_suggestions,
            learning_suggestions,
            exploration_suggestions,
            advanced_suggestions,
            bayesian_suggestions,
            gan_suggestions,
            game_specific_suggestions,
            pseudo_button_suggestions,
            visual_targeting_suggestions,
            exploration_phase_suggestions,
            strategy_suggestions,
            win_condition_suggestions,  # Win condition-based action recommendations
            losing_streak_suggestions,  # Losing streak intervention guidance
            action5_suggestions,
            button_discovery_suggestions
        )

        # Build an available_buttons list similar to available_actions (only when Action 6 is available and no primary actions)
        try:
            if 6 in available_actions and not primary_actions_available:
                buttons = []
                for s in all_suggestions:
                    if isinstance(s, dict):
                        # Accept suggestions formatted with 'action' or 'id' and coordinates
                        action_name = s.get('action') or f"ACTION{s.get('id', '')}"
                        if action_name == 'ACTION6' or s.get('id') == 6:
                            x = s.get('x') if 'x' in s else (s.get('coordinates')[0] if 'coordinates' in s and s.get('coordinates') else None)
                            y = s.get('y') if 'y' in s else (s.get('coordinates')[1] if 'coordinates' in s and s.get('coordinates') else None)
                            if x is not None and y is not None:
                                coord = (int(x), int(y))
                                if coord not in buttons:
                                    buttons.append(coord)
                self.available_buttons = buttons
                logger.info(f"🎯 Available Buttons detected: {len(self.available_buttons)}")
            else:
                self.available_buttons = []
        except Exception as e:
            logger.debug(f"Failed to build available_buttons: {e}")
        
        # 8. INTELLIGENT SELECTION - Multi-factor scoring
        best_action = self._select_best_action_intelligent(
            all_suggestions, available_actions, current_score, frame_analysis
        )
        
        # Log selected action with reasoning
        action_id = best_action.get('id', 'UNKNOWN')
        action_source = best_action.get('source', 'unknown')
        action_reason = best_action.get('reason', 'no reason')
        action_confidence = best_action.get('confidence', 0.0)
        
        logger.info(f"[TARGET] Selected Action: {action_id} | Source: {action_source} | Confidence: {action_confidence:.2f}")
        logger.info(f"   Reason: {action_reason}")
        
        # Log coordinates for Action 6
        if action_id == 6 and 'x' in best_action and 'y' in best_action:
            logger.info(f"   Coordinates: ({best_action['x']}, {best_action['y']})")
        
        # Store coordinates for next analysis
        if 'coordinates' in best_action:
            self.last_coordinates = best_action['coordinates']
        
        # 9. TRACK ACTION AVAILABILITY CHANGES - Monitor pseudo score increases
        if len(self.performance_history) > 0:
            last_action = self.performance_history[-1]
            availability_changes = self._track_action_availability_changes(available_actions, last_action)
            
            # If we discovered new actions, treat as pseudo score increase
            if availability_changes['pseudo_score_increase']:
                logger.info(f"🎉 PSEUDO SCORE INCREASE: Discovered {len(availability_changes['newly_available'])} new actions!")
        
        # Always log available actions for visibility
        logger.info(f"🎮 Available Actions: {available_actions}")
        
        # Log current game state summary
        current_score = game_state.get('score', 0)
        game_state_status = game_state.get('state', 'UNKNOWN')
        logger.info(f"[STATS] Game State: Score={current_score}, Status={game_state_status}")
        logger.info("=" * 80)  # Separator line for readability
        
        # 10. LEARNING UPDATE - Update all learning systems
        self._update_learning_systems(best_action, game_state, frame_analysis)
        
        # 10. OBJECT TESTING UPDATE - Test objects for interactivity
        self._update_object_testing(best_action, game_state, frame_analysis)
        
        # 11. ACTION REPETITION TRACKING - Track and penalize repeated actions
        repetition_penalty = self._track_action_repetition(best_action)
        if repetition_penalty > 0.5:
            logger.warning(f"High repetition penalty for action {best_action.get('id')}: {repetition_penalty:.2f}")
        
        # 12. PERFORMANCE TRACKING - Track performance
        self._track_performance(best_action, current_score, game_state)
        
        # 13. GAME RESULT TRACKING - Save game-specific knowledge
        self._track_game_result(best_action, game_state, frame_analysis)
        
        # 14. ADD REASONING TO ACTION - Include detailed reasoning for API
        best_action = self._add_reasoning_to_action(best_action, game_state, frame_analysis, all_suggestions)

        # 15. VALIDATE BEST ACTION - Ensure action has valid ID before returning
        if not best_action or best_action.get('id') is None:
            logger.error("Best action is invalid - creating emergency fallback")
            best_action = self._get_fallback_action(available_actions)

        return best_action
    
    async def handle_action_result(self, 
                                 game_state: Dict[str, Any], 
                                 action_result: Dict[str, Any],
                                 previous_score: float,
                                 previous_available_actions: List[int]) -> None:
        """
        Handle the result of an action and update all systems accordingly.
        
        Args:
            game_state: Current game state after action
            action_result: Result of the action taken
            previous_score: Score before the action
            previous_available_actions: Available actions before the action
        """
        try:
            import time
            game_id = game_state.get('game_id', 'unknown')
            session_id = game_state.get('session_id', 'unknown')
            current_score = game_state.get('score', 0)
            current_available_actions = game_state.get('available_actions', [])
            action_number = action_result.get('action', 0)
            coordinates = action_result.get('coordinates')
            
            # Calculate score change
            score_change = current_score - previous_score
            
            # Check for frame changes
            frame_changes = score_change > 0 or len(current_available_actions) != len(previous_available_actions)
            
            # Update button discovery system for Action 6
            if self.button_discovery_system and action_number == 6 and coordinates:
                try:
                    action_unlocks = len(current_available_actions) - len(previous_available_actions)
                    visual_changes = 1 if frame_changes else 0
                    
                    await self.button_discovery_system.record_test_result(
                        coordinate=coordinates,
                        score_change=score_change,
                        action_unlocks=action_unlocks,
                        visual_changes=visual_changes,
                        frame_changed=frame_changes
                    )
                    
                    logger.info(f" BUTTON TEST RESULT: {coordinates} - Score: {score_change:+.2f}, Actions: {action_unlocks:+d}, Visual: {visual_changes}")
                except Exception as e:
                    logger.error(f"Error updating button discovery system: {e}")
            
            # Update stagnation intervention system
            if self.stagnation_intervention_system:
                try:
                    action_data = {
                        'id': action_number,
                        'x': coordinates[0] if coordinates else None,
                        'y': coordinates[1] if coordinates else None
                    }
                    self.stagnation_intervention_system.record_action(action_data, coordinates)
                except Exception as e:
                    logger.error(f"Error updating stagnation intervention system: {e}")
            
            # Update performance history
            self.performance_history.append({
                'score': current_score,
                'action': action_number,
                'coordinates': coordinates,
                'timestamp': time.time(),
                'score_change': score_change,
                'frame_changes': frame_changes
            })

            # Keep only recent history
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]

            # Store frame-action data for GAN training
            try:
                from src.database.system_integration import get_system_integration
                integration = get_system_integration()

                # Get frame data from game state
                current_frame_data = game_state.get('frame', None)

                # Detect frame changes using our new comparison logic
                frame_change_coords = []
                is_button_candidate = False
                button_confidence = 0.0

                if self.previous_frame_cache is not None and current_frame_data is not None:
                    # Use our frame comparison to get actual changes
                    frame_change_coords = self._compare_frames_for_changes(
                        self.previous_frame_cache,
                        self._normalize_frame_data(current_frame_data)
                    )

                    # Determine if this is a button candidate based on frame changes
                    if len(frame_change_coords) >= 3:  # Minimum changes for button candidate
                        is_button_candidate = True
                        button_confidence = min(len(frame_change_coords) / 20.0, 0.9)

                # Store the frame-action pair for GAN training
                await integration.store_frame_action_data_for_gan(
                    game_id=game_id,
                    session_id=session_id,
                    action_number=action_number,
                    coordinates=coordinates,
                    previous_frame=self.previous_frame_cache,
                    current_frame=self._normalize_frame_data(current_frame_data),
                    frame_changes=frame_change_coords,
                    score_change=int(score_change),
                    is_button_candidate=is_button_candidate,
                    button_candidate_confidence=button_confidence
                )

                logger.debug(f"[GAN] Stored frame-action data: action={action_number}, changes={len(frame_change_coords)}, "
                           f"button_candidate={is_button_candidate}, confidence={button_confidence:.2f}")

            except Exception as e:
                logger.error(f"Error storing frame-action data for GAN: {e}")
            
            # Update action history
            self.action_history.append(action_number)
            if len(self.action_history) > 50:
                self.action_history = self.action_history[-50:]
            
            # Update visual interactive system
            if self.visual_interactive_system and coordinates and action_number == 6:
                try:
                    # Record target interaction result
                    await self.visual_interactive_system.record_target_interaction(
                        game_id=game_id,
                        target=type('VisualTarget', (), {
                            'x': coordinates[0],
                            'y': coordinates[1],
                            'target_type': 'interactive_element',
                            'confidence': 0.5,
                            'detection_method': 'action_result'
                        })(),
                        interaction_successful=score_change > 0,
                        frame_changes_detected=frame_changes,
                        score_impact=score_change
                    )
                except Exception as e:
                    logger.error(f"Error updating visual interactive system: {e}")
            
            # Update exploration system
            if self.exploration_system and coordinates and action_number == 6:
                try:
                    await self.exploration_system.record_exploration_result(
                        game_id=game_id,
                        coordinates=coordinates,
                        success=score_change > 0,
                        frame_changes=frame_changes,
                        score_impact=score_change
                    )
                except Exception as e:
                    logger.error(f"Error updating exploration system: {e}")
            
            # Update strategy discovery system
            if self.strategy_discovery_system and score_change > 5:
                try:
                    # Check if we should discover a new strategy
                    recent_actions = self.action_history[-10:] if len(self.action_history) >= 10 else self.action_history
                    recent_scores = [p['score'] for p in self.performance_history[-10:]] if len(self.performance_history) >= 10 else [p['score'] for p in self.performance_history]
                    
                    if len(recent_actions) >= 3 and len(recent_scores) >= 3:
                        await self.strategy_discovery_system.discover_winning_strategy(
                            game_id=game_id,
                            action_sequence=recent_actions,
                            score_progression=recent_scores
                        )
                except Exception as e:
                    logger.error(f"Error updating strategy discovery system: {e}")
            
            logger.debug(f"Action result processed: Action {action_number}, "
                        f"Score change: {score_change}, Frame changes: {frame_changes}")
            
        except Exception as e:
            logger.error(f"Error handling action result: {e}")
    
    def _get_grid_dimensions(self, frame_data: Any) -> Tuple[int, int]:
        """Get grid dimensions from frame data."""
        try:
            if isinstance(frame_data, list) and len(frame_data) > 0:
                if isinstance(frame_data[0], list):
                    return (len(frame_data[0]), len(frame_data))
                else:
                    return (len(frame_data), 1)
            return (32, 32)  # Default fallback
        except Exception as e:
            logger.error(f"Error getting grid dimensions: {e}")
            return (32, 32)

    def _normalize_frame_data(self, frame_data: Any) -> Optional[List[List[Any]]]:
        """Normalize frame data to consistent 2D list format."""
        try:
            if frame_data is None:
                return None

            # Handle different frame data formats
            if isinstance(frame_data, list):
                # If it's a list of frames, take the first frame
                if len(frame_data) > 0 and isinstance(frame_data[0], list):
                    if isinstance(frame_data[0][0], list):
                        # 3D list - take first frame
                        return frame_data[0]
                    else:
                        # Already 2D list
                        return frame_data

            # Convert numpy arrays if needed
            if hasattr(frame_data, 'tolist'):
                frame_list = frame_data.tolist()
                if isinstance(frame_list[0], list):
                    return frame_list

            logger.warning(f"Could not normalize frame data of type: {type(frame_data)}")
            return None

        except Exception as e:
            logger.error(f"Error normalizing frame data: {e}")
            return None

    def _compare_frames_for_changes(self, previous_frame: List[List[Any]], current_frame: List[List[Any]]) -> List[Tuple[int, int]]:
        """Compare two frames and return coordinates of notable changes."""
        try:
            if not previous_frame or not current_frame:
                return []

            changes = []
            height = min(len(previous_frame), len(current_frame))
            width = min(len(previous_frame[0]), len(current_frame[0])) if height > 0 else 0

            for y in range(height):
                for x in range(width):
                    # Compare pixel values
                    prev_pixel = previous_frame[y][x]
                    curr_pixel = current_frame[y][x]

                    # Calculate change based on pixel type
                    change_detected = False

                    if isinstance(prev_pixel, (list, tuple)) and isinstance(curr_pixel, (list, tuple)):
                        # RGB/color pixels - calculate color distance
                        if len(prev_pixel) >= 3 and len(curr_pixel) >= 3:
                            color_distance = sum(abs(a - b) for a, b in zip(prev_pixel[:3], curr_pixel[:3]))
                            if color_distance > (255 * 3 * self.frame_comparison_threshold):
                                change_detected = True
                    else:
                        # Single value pixels - direct comparison
                        if abs(prev_pixel - curr_pixel) > (255 * self.frame_comparison_threshold):
                            change_detected = True

                    if change_detected:
                        changes.append((x, y))

            return changes

        except Exception as e:
            logger.error(f"Error comparing frames: {e}")
            return []

    def _detect_button_candidates_by_frame_changes(self, previous_frame: Any, current_frame: Any) -> List[Dict[str, Any]]:
        """Detect button candidates based on notable changes between frames."""
        try:
            # Normalize frames
            prev_normalized = self._normalize_frame_data(previous_frame)
            curr_normalized = self._normalize_frame_data(current_frame)

            if not prev_normalized or not curr_normalized:
                return []

            # Get change coordinates
            change_coords = self._compare_frames_for_changes(prev_normalized, curr_normalized)

            if not change_coords:
                return []

            # Group nearby changes into button candidates
            candidates = []
            clustered_changes = self._cluster_change_coordinates(change_coords)

            for cluster in clustered_changes:
                if len(cluster) >= 3:  # Minimum cluster size for a button candidate
                    # Calculate centroid of the cluster
                    center_x = sum(coord[0] for coord in cluster) // len(cluster)
                    center_y = sum(coord[1] for coord in cluster) // len(cluster)

                    # Calculate confidence based on cluster size and density
                    confidence = min(len(cluster) / 20.0, 0.9)  # Max confidence of 0.9

                    candidates.append({
                        'action': 'ACTION6',
                        'x': center_x,
                        'y': center_y,
                        'coordinates': (center_x, center_y),
                        'confidence': confidence,
                        'reason': f'Button candidate detected from {len(cluster)} frame changes',
                        'source': 'frame_comparison',
                        'cluster_size': len(cluster)
                    })

            logger.info(f"[FRAME] Detected {len(candidates)} button candidates from {len(change_coords)} frame changes")
            return candidates

        except Exception as e:
            logger.error(f"Error detecting button candidates from frame changes: {e}")
            return []

    def _cluster_change_coordinates(self, coords: List[Tuple[int, int]], cluster_radius: int = 3) -> List[List[Tuple[int, int]]]:
        """Cluster nearby change coordinates into groups."""
        if not coords:
            return []

        clusters = []
        used_coords = set()

        for coord in coords:
            if coord in used_coords:
                continue

            # Start new cluster
            cluster = [coord]
            used_coords.add(coord)

            # Find nearby coordinates
            for other_coord in coords:
                if other_coord in used_coords:
                    continue

                # Calculate distance
                dx = abs(coord[0] - other_coord[0])
                dy = abs(coord[1] - other_coord[1])

                if dx <= cluster_radius and dy <= cluster_radius:
                    cluster.append(other_coord)
                    used_coords.add(other_coord)

            if cluster:
                clusters.append(cluster)

        return clusters

    def _cache_frame_data(self, frame_data: Any, game_id: str) -> None:
        """Cache current frame data as previous frame for next comparison."""
        try:
            normalized_frame = self._normalize_frame_data(frame_data)
            if normalized_frame:
                self.previous_frame_cache = normalized_frame

                # Track frame change history
                if len(self.frame_change_history) >= 10:
                    self.frame_change_history.pop(0)  # Remove oldest entry

                # We'll update this with actual change detection later
                self.frame_change_history.append(True)  # Placeholder

        except Exception as e:
            logger.error(f"Error caching frame data: {e}")
    
    def _analyze_patterns(self, frame_analysis: Dict[str, Any], available_actions: List[int]) -> List[Dict[str, Any]]:
        """Analyze patterns from frame data and historical success."""
        suggestions = []
        
        # Check if primary actions 1-4 are available
        primary_actions_available = any(action in [1, 2, 3, 4] for action in available_actions)
        
        # If primary actions are available, prioritize them over action 6 pattern matching
        if primary_actions_available:
            # Generate suggestions for primary actions based on general patterns
            for action_id in [1, 2, 3, 4]:
                if action_id in available_actions:
                    suggestions.append({
                        'action': f'ACTION{action_id}',
                        'confidence': 0.8,  # High confidence for primary actions
                        'reason': f'Primary action {action_id} available - pattern priority',
                        'source': 'pattern_matching_primary'
                    })
            return suggestions
        
        # Only generate ACTION6 pattern suggestions when primary actions are NOT available
        if 6 not in available_actions:
            return suggestions
            
        # Look for patterns in the frame
        objects = frame_analysis.get('objects', [])
        interactive_elements = frame_analysis.get('interactive_elements', [])
        
        # Prioritize interactive elements
        for element in interactive_elements:
            if element.get('confidence', 0) > 0.6:
                suggestions.append({
                    'action': 'ACTION6',
                    'x': element['centroid'][0],
                    'y': element['centroid'][1],
                    'confidence': element['confidence'],
                    'reason': f"Pattern-matched interactive element at ({element['centroid'][0]}, {element['centroid'][1]}) [fallback]",
                    'source': 'pattern_matching'
                })
        
        # Add large objects as secondary targets
        for obj in objects:
            if obj.get('area', 0) > 30:
                suggestions.append({
                    'action': 'ACTION6',
                    'x': obj['centroid'][0],
                    'y': obj['centroid'][1],
                    'confidence': min(obj['area'] / 100, 1.0),
                    'reason': f"Pattern-matched large object at ({obj['centroid'][0]}, {obj['centroid'][1]}) [fallback]",
                    'source': 'pattern_matching'
                })
        
        return suggestions
    
    async def _get_intelligent_coordinates(self, frame_analysis: Dict[str, Any], available_actions: List[int], game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get intelligent coordinate suggestions based on learning and penalty avoidance."""
        suggestions = []
        
        if 6 not in available_actions:
            return suggestions
        
        # Get penalty-aware avoidance scores from frame analyzer (async)
        try:
            # Build candidate list from known coordinates or exploration grid
            candidate_coords = list(self.coordinate_success_rates.keys())
            # If we have no learned coordinates, use a small exploration grid around center
            if not candidate_coords:
                candidate_coords = [(32, 32), (16, 16), (48, 32), (32, 48)]

            avoidance_scores = await self.frame_analyzer.get_penalty_aware_avoidance_scores(candidate_coords, game_state.get('game_id', 'unknown'))
            if avoidance_scores:
                # Filter out heavily penalized coordinates
                for coord, penalty_score in avoidance_scores.items():
                    try:
                        if penalty_score < 0.5:  # Avoid coordinates with high penalty scores
                            x, y = coord
                            suggestions.append({
                                'action': 'ACTION6',
                                'x': x,
                                'y': y,
                                'confidence': 1.0 - penalty_score,  # Higher confidence for lower penalties
                                'reason': f"Penalty-aware coordinate (penalty: {penalty_score:.2f})",
                                'source': 'penalty_aware_intelligence'
                            })
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"Failed to get penalty-aware coordinates: {e}")
        
        # Fallback to basic coordinate intelligence from past successes
        for coord, data in self.coordinate_success_rates.items():
            if data.get('success_rate', 0) > 0.3:  # 30% success rate threshold
                suggestions.append({
                    'action': 'ACTION6',
                    'x': coord[0],
                    'y': coord[1],
                    'confidence': data['success_rate'],
                    'reason': f"Intelligent coordinate from learning (success rate: {data['success_rate']:.2f})",
                    'source': 'coordinate_intelligence'
                })
        
        return suggestions
    
    def _detect_opencv_targets(self, frame_analysis: Dict[str, Any], available_actions: List[int]) -> List[Dict[str, Any]]:
        """Detect targets using OpenCV analysis."""
        suggestions = []
        
        if 6 not in available_actions:
            return suggestions
        
        # Use the frame analyzer's OpenCV detection
        interactive_elements = frame_analysis.get('interactive_elements', [])
        
        # Lower threshold for ARC puzzles - any detected interactive element is worth trying
        for element in interactive_elements:
            if element.get('confidence', 0) > 0.3:  # Much lower threshold for ARC puzzles
                suggestions.append({
                    'action': 'ACTION6',
                    'x': element['centroid'][0],
                    'y': element['centroid'][1],
                    'confidence': element['confidence'],
                    'reason': f"OpenCV detected target at ({element['centroid'][0]}, {element['centroid'][1]}) [ARC element]",
                    'source': 'opencv_detection'
                })
        
        # Also use regular objects as potential targets
        objects = frame_analysis.get('objects', [])
        for obj in objects:
            # Any non-background colored object could be interactive
            if obj.get('color', 0) != 0 and obj.get('area', 0) >= 1:
                suggestions.append({
                    'action': 'ACTION6',
                    'x': obj['centroid'][0],
                    'y': obj['centroid'][1],
                    'confidence': min(0.8, 0.5 + obj.get('area', 1) / 100),  # Base confidence + size bonus
                    'reason': f"Colored object (color {obj.get('color', '?')}) at ({obj['centroid'][0]}, {obj['centroid'][1]})",
                    'source': 'opencv_object_detection'
                })
        
        logger.info(f"🎯 OpenCV generated {len(suggestions)} target suggestions")
        return suggestions
    
    def _generate_learning_suggestions(self, frame_analysis: Dict[str, Any], available_actions: List[int]) -> List[Dict[str, Any]]:
        """Generate suggestions based on learning from past actions."""
        suggestions = []
        
        # Learn from action effectiveness
        for action_id in available_actions:
            if action_id in self.action_effectiveness:
                effectiveness = self.action_effectiveness[action_id]
                # Calculate success rate from effectiveness data
                if isinstance(effectiveness, dict):
                    attempts = effectiveness.get('attempts', 0)
                    successes = effectiveness.get('successes', 0)
                    success_rate = successes / attempts if attempts > 0 else 0
                else:
                    success_rate = float(effectiveness) if isinstance(effectiveness, (int, float)) else 0
                
                if success_rate > 0.5:
                    if action_id == 6:
                        # For ACTION6, use learned coordinates
                        for coord, data in self.coordinate_success_rates.items():
                            if data.get('success_rate', 0) > 0.4:
                                suggestions.append({
                                    'action': 'ACTION6',
                                    'x': coord[0],
                                    'y': coord[1],
                                    'confidence': effectiveness * data['success_rate'],
                                    'reason': f"Learning-based ACTION6 at ({coord[0]}, {coord[1]})",
                                    'source': 'learning'
                                })
                    else:
                        suggestions.append({
                            'action': f'ACTION{action_id}',
                            'confidence': effectiveness,
                            'reason': f"Learning-based action {action_id}",
                            'source': 'learning'
                        })
        
        return suggestions
    
    def _generate_exploration_suggestions(self, available_actions: List[int]) -> List[Dict[str, Any]]:
        """Generate exploration suggestions for unknown areas."""
        suggestions = []
        
        # Exploration strategy: try different areas of the grid
        exploration_coords = [
            (16, 16), (32, 16), (48, 16),  # Top row
            (16, 32), (32, 32), (48, 32),  # Middle row
            (16, 48), (32, 48), (48, 48),  # Bottom row
        ]
        
        for coord in exploration_coords:
            if 6 in available_actions:
                suggestions.append({
                    'action': 'ACTION6',
                    'x': coord[0],
                    'y': coord[1],
                    'confidence': 0.3,
                    'reason': f"Exploration at ({coord[0]}, {coord[1]})",
                    'source': 'exploration'
                })
        
        # Add movement actions for exploration
        for action_id in [1, 2, 3, 4, 5, 7]:
            if action_id in available_actions:
                suggestions.append({
                    'action': f'ACTION{action_id}',
                    'confidence': 0.2,
                    'reason': f"Exploration movement {action_id}",
                    'source': 'exploration'
                })
        
        return suggestions
    
    def _generate_advanced_suggestions(self, frame_analysis: Dict[str, Any], 
                                     available_actions: List[int], 
                                     current_score: int, 
                                     game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate suggestions using advanced cognitive systems."""
        suggestions = []
        
        try:
            # Tree Evaluation Simulation
            if self.tree_evaluation_engine:
                tree_suggestions = self._get_tree_evaluation_suggestions(
                    frame_analysis, available_actions, current_score
                )
                suggestions.extend(tree_suggestions)
            
            # Action Sequence Optimization
            if self.action_sequence_optimizer:
                sequence_suggestions = self._get_sequence_optimization_suggestions(
                    frame_analysis, available_actions, game_state
                )
                suggestions.extend(sequence_suggestions)
            
            # Enhanced Exploration
            if self.exploration_system:
                exploration_suggestions = self._get_enhanced_exploration_suggestions(
                    frame_analysis, available_actions, current_score
                )
                suggestions.extend(exploration_suggestions)
            
            # Predictive Core Simulation
            if self.predictive_core:
                predictive_suggestions = self._get_predictive_suggestions(
                    frame_analysis, available_actions, game_state
                )
                suggestions.extend(predictive_suggestions)
                
        except Exception as e:
            logger.warning(f"Advanced suggestions generation failed: {e}")
        
        return suggestions
    
    def _get_tree_evaluation_suggestions(self, frame_analysis: Dict[str, Any], 
                                       available_actions: List[int], 
                                       current_score: int) -> List[Dict[str, Any]]:
        """Get suggestions from Tree Evaluation Simulation Engine."""
        suggestions = []
        
        try:
            # Create simulation hypothesis
            hypothesis = {
                'actions': available_actions,
                'frame_data': frame_analysis,
                'current_score': current_score,
                'max_depth': 5
            }
            
            # Run tree evaluation
            result = self.tree_evaluation_engine.evaluate_action_sequences(
                hypothesis, max_depth=5, timeout=0.1
            )
            
            if result and 'best_sequence' in result:
                best_action = result['best_sequence'][0] if result['best_sequence'] else available_actions[0]
                suggestions.append({
                    'action': f'ACTION{best_action}',
                    'id': best_action,
                    'confidence': result.get('confidence', 0.7),
                    'reason': f"Tree evaluation suggests ACTION{best_action}",
                    'source': 'tree_evaluation'
                })
                
        except Exception as e:
            logger.debug(f"Tree evaluation failed: {e}")
        
        return suggestions
    
    def _get_sequence_optimization_suggestions(self, frame_analysis: Dict[str, Any], 
                                             available_actions: List[int], 
                                             game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get suggestions from Action Sequence Optimizer."""
        suggestions = []
        
        try:
            # Convert frame to grid format
            frame_data = frame_analysis.get('frame', [])
            if frame_data is not None and len(frame_data) > 0:
                grid = frame_data[0] if isinstance(frame_data, list) else frame_data
                
                # Optimize action sequence
                result = self.action_sequence_optimizer.optimize_for_action6(
                    current_state=game_state,
                    available_actions=available_actions,
                    grid=grid,
                    game_id=game_state.get('game_id', 'unknown')
                )
                
                action_id, coordinates = result
                if action_id in available_actions:
                    suggestion = {
                        'action': f'ACTION{action_id}',
                        'id': action_id,
                        'confidence': 0.8,
                        'reason': f"Sequence optimizer suggests ACTION{action_id}",
                        'source': 'sequence_optimizer'
                    }
                    
                    if coordinates and action_id == 6:
                        suggestion['x'] = coordinates[0]
                        suggestion['y'] = coordinates[1]
                        suggestion['reason'] += f" at ({coordinates[0]}, {coordinates[1]})"
                    
                    suggestions.append(suggestion)
                    
        except Exception as e:
            logger.debug(f"Sequence optimization failed: {e}")
        
        return suggestions
    
    def _get_enhanced_exploration_suggestions(self, frame_analysis: Dict[str, Any], 
                                            available_actions: List[int], 
                                            current_score: int) -> List[Dict[str, Any]]:
        """Get suggestions from Enhanced Exploration System."""
        suggestions = []
        
        try:
            # Create exploration state
            state = ExplorationState(
                position=(0, 0),  # Default position
                score=current_score,
                frame_data=frame_analysis,
                available_actions=available_actions
            )
            
            # Get exploration result
            result = self.exploration_system.explore(state, available_actions)
            
            if result and result.action in available_actions:
                suggestions.append({
                    'action': f'ACTION{result.action}',
                    'id': result.action,
                    'confidence': result.confidence,
                    'reason': f"Enhanced exploration suggests ACTION{result.action}",
                    'source': 'enhanced_exploration'
                })
                
        except Exception as e:
            logger.debug(f"Enhanced exploration failed: {e}")
        
        return suggestions
    
    def _get_predictive_suggestions(self, frame_analysis: Dict[str, Any], 
                                  available_actions: List[int], 
                                  game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get suggestions from Predictive Core."""
        suggestions = []
        
        try:
            # Create simulation hypothesis
            hypothesis = SimulationHypothesis(
                action_sequence=[(action, None) for action in available_actions[:3]],  # Top 3 actions
                expected_outcome="score_improvement",
                confidence_threshold=0.6
            )
            
            # Run simulation
            result = self.predictive_core.simulate_rollout(
                initial_state=game_state,
                hypothesis=hypothesis,
                max_steps=3,
                timeout=0.1
            )
            
            if result and result.simulation_history:
                # Find best action from simulation
                best_step = max(result.simulation_history, 
                              key=lambda step: step.learning_progress)
                
                if best_step.action in available_actions:
                    suggestions.append({
                        'action': f'ACTION{best_step.action}',
                        'id': best_step.action,
                        'confidence': best_step.confidence,
                        'reason': f"Predictive core suggests ACTION{best_step.action}",
                        'source': 'predictive_core'
                    })
                    
        except Exception as e:
            logger.debug(f"Predictive core failed: {e}")
        
        return suggestions
    
    def _combine_suggestions(self, *suggestion_lists) -> List[Dict[str, Any]]:
        """Combine all suggestion sources into a unified list."""
        all_suggestions = []
        
        for suggestion_list in suggestion_lists:
            if isinstance(suggestion_list, list):
                all_suggestions.extend(suggestion_list)
            elif isinstance(suggestion_list, dict):
                all_suggestions.append(suggestion_list)
        
        # Remove duplicates
        unique_suggestions = []
        seen_actions = set()
        
        for suggestion in all_suggestions:
            if isinstance(suggestion, dict) and 'action' in suggestion:
                action_key = (suggestion.get('action'), suggestion.get('x'), suggestion.get('y'))
                if action_key not in seen_actions:
                    seen_actions.add(action_key)
                    unique_suggestions.append(suggestion)
        
        return unique_suggestions
    
    def _select_best_action_intelligent(self, suggestions: List[Dict[str, Any]], 
                                           available_actions: List[int],
                                           current_score: int,
                                           frame_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best action using intelligent multi-factor analysis with AGI priority."""
        
        if not suggestions:
            return self._get_fallback_action(available_actions)
        
        # Filter suggestions based on available actions
        valid_suggestions = self._filter_valid_actions(suggestions, available_actions)
        
        if not valid_suggestions:
            return self._get_fallback_action(available_actions)
        
        # PRIORITY: AGI HIGH-CONFIDENCE SUGGESTIONS - Human-like rapid puzzle solving
        agi_insights = frame_analysis.get('agi_insights')
        if agi_insights and agi_insights.get('confidence', 0.0) > 0.7:
            logger.info(f"[AGI] High-confidence AGI insights detected (confidence: {agi_insights['confidence']:.2f}) - prioritizing AGI strategy")
            
            # Look for AGI-suggested actions in suggestions
            agi_suggestions = [s for s in valid_suggestions if s.get('agi_insight') == True]
            if agi_suggestions:
                # Sort AGI suggestions by confidence
                agi_suggestions.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
                best_agi = agi_suggestions[0]
                
                # Convert AGI suggestion to action format
                action_id = self._get_action_id(best_agi.get('action', 'ACTION1'))
                action = {
                    'id': action_id,
                    'reason': f"AGI Strategy: {best_agi.get('reason', 'AGI-recommended action')} (confidence: {best_agi.get('confidence', 0.0):.2f})",
                    'confidence': best_agi.get('confidence', 0.8),
                    'source': 'agi_rapid_solver'
                }
                
                # Add coordinates for ACTION6
                if action_id == 6:
                    action['x'] = best_agi.get('x', 32)
                    action['y'] = best_agi.get('y', 32)
                
                logger.info(f"[AGI] Selected AGI action: {action_id} - {action['reason']}")
                return action
        
        # Normalize suggestion coordinate fields: accept 'coordinates' tuples as 'x'/'y'
        for s in valid_suggestions:
            if 'coordinates' in s and ('x' not in s or 'y' not in s):
                try:
                    coord = s.get('coordinates')
                    if coord and isinstance(coord, (list, tuple)) and len(coord) >= 2:
                        s['x'] = int(coord[0])
                        s['y'] = int(coord[1])
                except Exception:
                    pass

        # Multi-factor scoring system with breakdowns
        scored_suggestions = []
        breakdowns = []

        # If a coordinate for ACTION6 has a very high repetition penalty, treat it as temporarily blacklisted
        REPETITION_EXCLUDE_THRESHOLD = 0.6

        for suggestion in valid_suggestions:
            score, breakdown = self._score_with_breakdown(suggestion, current_score, frame_analysis)
            
            # BOOST: AGI suggestions get automatic score boost
            if suggestion.get('agi_insight') == True:
                agi_boost = 0.3  # Significant boost for AGI suggestions
                score = min(1.0, score + agi_boost)
                breakdown['agi_boost'] = agi_boost
                logger.debug(f"[AGI] Applied AGI boost of {agi_boost} to suggestion")
            
            # If breakdown reports a high repetition penalty for a coordinate, exclude it from consideration
            try:
                repetition_penalty = breakdown.get('repetition_penalty', 0.0)
            except Exception:
                repetition_penalty = 0.0

            if repetition_penalty >= REPETITION_EXCLUDE_THRESHOLD:
                # Log exclusion for diagnostics
                try:
                    coord = (suggestion.get('x'), suggestion.get('y')) if 'x' in suggestion and 'y' in suggestion else suggestion.get('coordinates', None)
                    logger.info(f"Excluding suggestion due to high repetition penalty (threshold={REPETITION_EXCLUDE_THRESHOLD}): action={suggestion.get('action')} coord={coord} penalty={repetition_penalty:.2f}")
                except Exception:
                    logger.info(f"Excluding suggestion due to high repetition penalty: action={suggestion.get('action')} penalty={repetition_penalty:.2f}")
                continue

            scored_suggestions.append((score, suggestion))
            breakdowns.append((score, suggestion, breakdown))
        
        # Sort by score (highest first)
        scored_suggestions.sort(key=lambda x: x[0], reverse=True)

        # Filter out suggestions with extremely low score (likely heavily penalized)
        SCORE_CUTOFF = 0.05
        filtered_scored_suggestions = [s for s in scored_suggestions if s[0] > SCORE_CUTOFF]
        if not filtered_scored_suggestions:
            logger.warning(f"All suggestions filtered by SCORE_CUTOFF={SCORE_CUTOFF:.2f}; using fallback action")
            return self._get_fallback_action(available_actions)

        # Use the filtered list for subsequent selection and breakdown logging
        scored_suggestions = filtered_scored_suggestions

        # Log top-5 breakdowns for debugging reasons
        try:
            top_breakdowns = sorted(breakdowns, key=lambda x: x[0], reverse=True)[:5]
            logger.info("--- Top 5 suggestion score breakdowns ---")
            for sc, sug, br in top_breakdowns:
                src = sug.get('source', 'unknown')
                coord = (sug.get('x'), sug.get('y')) if 'x' in sug and 'y' in sug else sug.get('coordinates', None)
                agi_marker = " [AGI]" if sug.get('agi_insight') else ""
                logger.info(f"SCORE={sc:.3f} | action={sug.get('action', sug.get('id'))} | source={src}{agi_marker} | coord={coord}")
                logger.info(f"   breakdown: confidence={br.get('confidence_factor'):.3f}, source={br.get('source_factor'):.3f}, learning={br.get('learning_factor'):.3f}, frame={br.get('frame_factor'):.3f}, penalty={br.get('penalty_factor'):.3f}, repetition_penalty={br.get('repetition_penalty'):.3f}, repetition_factor={br.get('repetition_factor'):.3f}, agi_boost={br.get('agi_boost', 0.0):.3f}")
            logger.info("--- end breakdown ---")
        except Exception:
            pass
        
        # Select the best suggestion
        best_score, best_suggestion = scored_suggestions[0]
        
        # Convert to action format
        action_id = self._get_action_id(best_suggestion.get('action', 'ACTION1'))

        # Validate action_id is not None
        if action_id is None:
            logger.error("Action ID is None after conversion - using fallback")
            action_id = 1

        action = {
            'id': action_id,
            'reason': best_suggestion.get('reason', f'Intelligent selection (score: {best_score:.2f})'),
            'confidence': min(best_score, 1.0),
            'source': best_suggestion.get('source', 'unknown')
        }
        
        # Add coordinates for ACTION6
        if action_id == 6:
            action['x'] = best_suggestion.get('x', 32)
            action['y'] = best_suggestion.get('y', 32)
        
        return action
    
    def _calculate_intelligent_score(self, suggestion: Dict[str, Any], 
                                   current_score: int,
                                   frame_analysis: Dict[str, Any]) -> float:
        """Calculate intelligent score for action suggestion."""
        
        base_score = suggestion.get('confidence', 0.5)
        source = suggestion.get('source', 'unknown')
        
        # Factor 1: Base confidence - ensure it's a float
        if isinstance(base_score, (int, float)):
            confidence_factor = float(base_score)
        else:
            confidence_factor = 0.5
        
        # Factor 2: Source reliability
        source_weights = {
            'confirmed_interactive': 1.0,  # Highest priority for confirmed interactive objects
            'game_specific_knowledge': 0.95,  # Very high priority for game-specific knowledge (lp85, vc33)
            'pseudo_button': 0.9,  # Very high priority for pseudo buttons (discovered new actions)
            'action5_stuck_fallback': 0.95,  # Very high priority for Action 5 when stuck
            'bayesian_pattern': 0.95,  # High priority for Bayesian pattern detection
            'opencv_detection': 0.9,  # High priority for OpenCV detected objects
            'object_testing': 0.85,  # High priority for untested objects (Action 6 primary function)
            'pattern_matching': 0.8,
            'coordinate_intelligence': 0.7,
            'gan_synthetic': 0.75,  # High priority for GAN-generated suggestions
            'action5_fallback': 0.7,  # High priority for Action 5 fallback
            'learning': 0.6,
            'exploration': 0.5,
            'coordinate_exploration': 0.3,  # LOW priority - last resort for Action 6
            'random_fallback': 0.4  # Lower priority for random fallback
        }
        source_factor = source_weights.get(source, 0.5)
        
        # Factor 3: Learning from past success
        learning_factor = 1.0
        if source == 'coordinate_intelligence' and 'x' in suggestion and 'y' in suggestion:
            coord = (suggestion['x'], suggestion['y'])
            if coord in self.coordinate_success_rates:
                success_data = self.coordinate_success_rates[coord]
                if isinstance(success_data, dict):
                    learning_factor = success_data.get('success_rate', 0.5)
                else:
                    learning_factor = float(success_data) if success_data is not None else 0.5
        
        # Factor 4: Frame analysis alignment
        frame_factor = 1.0
        if 'x' in suggestion and 'y' in suggestion:
            # Check if coordinates align with detected objects
            x, y = suggestion['x'], suggestion['y']
            for obj in frame_analysis.get('objects', []):
                obj_x, obj_y = obj.get('centroid', (0, 0))
                distance = np.sqrt((x - obj_x)**2 + (y - obj_y)**2)
                if distance < 10:  # Close to detected object
                    frame_factor = 1.2
                    break
        
        # Factor 5: Penalty avoidance for coordinates
        penalty_factor = 1.0
        if 'x' in suggestion and 'y' in suggestion:
            try:
                # Synchronous fallback: use local avoidance score cache exposed on frame_analyzer
                avoidance_scores = getattr(self.frame_analyzer, 'avoidance_scores', None)
                if not avoidance_scores:
                    avoidance_scores = {}

                coord = (suggestion['x'], suggestion['y'])
                # Keys in avoidance_scores may be strings like 'x,y' or tuples
                penalty_score = 0.0
                if coord in avoidance_scores:
                    penalty_score = avoidance_scores.get(coord, 0.0)
                else:
                    coord_key = f"{coord[0]},{coord[1]}"
                    penalty_score = avoidance_scores.get(coord_key, 0.0)

                # Higher penalty score = lower factor (avoid heavily penalized coordinates)
                penalty_factor = 1.0 - penalty_score
                # Ensure minimum factor to avoid completely blocking coordinates
                penalty_factor = max(penalty_factor, 0.1)
            except Exception as e:
                logger.warning(f"Failed to get penalty scores: {e}")
        
        # Factor 6: Action repetition penalty
        repetition_factor = 1.0
        action_id = self._get_action_id(suggestion.get('action', 'ACTION1'))
        
        # For Action 6, check coordinate-specific penalties
        if action_id == 6 and 'x' in suggestion and 'y' in suggestion:
            coordinate = (suggestion['x'], suggestion['y'])
            repetition_penalty = self._get_action_repetition_penalty(action_id, coordinate)
        else:
            repetition_penalty = self._get_action_repetition_penalty(action_id)
        
        repetition_factor = 1.0 - repetition_penalty
        repetition_factor = max(repetition_factor, 0.1)  # Minimum factor
        
        # Factor 7: Exploration vs exploitation balance
        exploration_factor = 0.8 if source == 'exploration' else 1.0
        
        # Hard cutoff for heavily penalized actions (lowered threshold to be more responsive)
        if repetition_penalty > 0.7:  # 70%+ penalty = almost block
            return 0.05  # Very low score
        
        # Weighted combination with penalty avoidance and repetition penalties
        intelligent_score = (
            confidence_factor * 0.2 +
            source_factor * 0.15 +
            learning_factor * 0.1 +
            frame_factor * 0.1 +
            penalty_factor * 0.1 +  # Coordinate penalty avoidance
            repetition_factor * 0.45  # Increased action repetition penalty weight (more influence)
        ) * exploration_factor
        
        return min(intelligent_score, 1.0)

    def _score_with_breakdown(self, suggestion: Dict[str, Any], current_score: int, frame_analysis: Dict[str, Any]) -> tuple:
        """Return (score, breakdown_dict) similar to _calculate_intelligent_score but with factor breakdowns for debugging."""
        # Reuse the same logic but capture intermediate factors
        base_score = suggestion.get('confidence', 0.5)
        if isinstance(base_score, (int, float)):
            confidence_factor = float(base_score)
        else:
            confidence_factor = 0.5

        source = suggestion.get('source', 'unknown')
        source_weights = {
            'confirmed_interactive': 1.0,
            'game_specific_knowledge': 0.95,
            'pseudo_button': 0.9,
            'action5_stuck_fallback': 0.95,
            'bayesian_pattern': 0.95,
            'opencv_detection': 0.9,
            'object_testing': 0.85,
            'pattern_matching': 0.8,
            'pattern_matching_primary': 0.95,  # Higher priority for primary actions
            'primary_action': 0.95,  # High priority for actions 1-4
            'coordinate_intelligence': 0.7,
            'gan_synthetic': 0.75,
            'action5_fallback': 0.7,
            'learning': 0.6,
            'exploration': 0.5,
            'coordinate_exploration': 0.3,
            'random_fallback': 0.4
        }
        source_factor = source_weights.get(source, 0.5)

        learning_factor = 1.0
        if source == 'coordinate_intelligence' and 'x' in suggestion and 'y' in suggestion:
            coord = (suggestion['x'], suggestion['y'])
            if coord in self.coordinate_success_rates:
                success_data = self.coordinate_success_rates[coord]
                if isinstance(success_data, dict):
                    learning_factor = success_data.get('success_rate', 0.5)
                else:
                    learning_factor = float(success_data) if success_data is not None else 0.5

        frame_factor = 1.0
        if 'x' in suggestion and 'y' in suggestion:
            x, y = suggestion['x'], suggestion['y']
            for obj in frame_analysis.get('objects', []):
                obj_x, obj_y = obj.get('centroid', (0, 0))
                distance = np.sqrt((x - obj_x)**2 + (y - obj_y)**2)
                if distance < 10:
                    frame_factor = 1.2
                    break

        # Penalty avoidance
        penalty_factor = 1.0
        penalty_score = 0.0
        try:
            avoidance_scores = getattr(self.frame_analyzer, 'avoidance_scores', None) or {}
            coord = (suggestion.get('x'), suggestion.get('y')) if 'x' in suggestion and 'y' in suggestion else None
            if coord:
                if coord in avoidance_scores:
                    penalty_score = avoidance_scores.get(coord, 0.0)
                else:
                    coord_key = f"{coord[0]},{coord[1]}"
                    penalty_score = avoidance_scores.get(coord_key, 0.0)
            penalty_factor = max(0.1, 1.0 - penalty_score)
        except Exception:
            penalty_score = 0.0
            penalty_factor = 1.0

        # Modified Action repetition penalty - more forgiving for visual targets
        action_id = self._get_action_id(suggestion.get('action', 'ACTION1'))
        if action_id == 6 and 'x' in suggestion and 'y' in suggestion:
            coordinate = (suggestion['x'], suggestion['y'])
            
            # Check if this coordinate corresponds to a visual object in the current frame
            is_visual_target = False
            for obj in frame_analysis.get('objects', []):
                obj_x, obj_y = obj.get('centroid', (0, 0))
                distance = np.sqrt((coordinate[0] - obj_x)**2 + (coordinate[1] - obj_y)**2)
                if distance < 5:  # Close to a visual object
                    is_visual_target = True
                    break
            
            # Check interactive elements too
            if not is_visual_target:
                for element in frame_analysis.get('interactive_elements', []):
                    elem_x, elem_y = element.get('centroid', (0, 0))
                    distance = np.sqrt((coordinate[0] - elem_x)**2 + (coordinate[1] - elem_y)**2)
                    if distance < 5:  # Close to an interactive element
                        is_visual_target = True
                        break
            
            # Get base repetition penalty
            repetition_penalty = self._get_action_repetition_penalty(action_id, coordinate)
            
            # For visual targets, cap the penalty to allow retries
            if is_visual_target:
                repetition_penalty = min(repetition_penalty, 0.6)  # Cap at 60% penalty for visual objects
                logger.debug(f"🎯 Visual target at {coordinate} - capping repetition penalty at {repetition_penalty:.2f}")
        else:
            repetition_penalty = self._get_action_repetition_penalty(action_id)
            
        repetition_factor = max(0.1, 1.0 - repetition_penalty)

        # More forgiving hard cutoff for visual targets
        max_penalty_threshold = 0.8 if action_id == 6 and is_visual_target else 0.7
        if repetition_penalty > max_penalty_threshold:
            breakdown = {
                'confidence_factor': confidence_factor,
                'source_factor': source_factor,
                'learning_factor': learning_factor,
                'frame_factor': frame_factor,
                'penalty_factor': penalty_factor,
                'penalty_score': penalty_score,
                'repetition_penalty': repetition_penalty,
                'repetition_factor': repetition_factor,
                'is_visual_target': is_visual_target if action_id == 6 else False
            }
            return 0.01, breakdown

        # Weighted combination (match _calculate_intelligent_score weights)
        intelligent_score = (
            confidence_factor * 0.2 +
            source_factor * 0.15 +
            learning_factor * 0.1 +
            frame_factor * 0.1 +
            penalty_factor * 0.1 +
            repetition_factor * 0.45
        )

        intelligent_score = intelligent_score * (0.8 if suggestion.get('source') == 'exploration' else 1.0)
        intelligent_score = min(intelligent_score, 1.0)

        breakdown = {
            'confidence_factor': confidence_factor,
            'source_factor': source_factor,
            'learning_factor': learning_factor,
            'frame_factor': frame_factor,
            'penalty_factor': penalty_factor,
            'penalty_score': penalty_score,
            'repetition_penalty': repetition_penalty,
            'repetition_factor': repetition_factor,
            'is_visual_target': is_visual_target if action_id == 6 else False
        }

        return intelligent_score, breakdown
    
    def _test_object_interactivity(self, coordinate: tuple, frame_before: List, frame_after: List) -> bool:
        """Test if an object at a coordinate is interactive by checking for frame changes."""
        # Avoid ambiguous truth-value checks for numpy arrays by using explicit None/len checks
        if frame_before is None or frame_after is None:
            return False
        try:
            if getattr(frame_before, '__len__', None) is not None and len(frame_before) == 0:
                return False
            if getattr(frame_after, '__len__', None) is not None and len(frame_after) == 0:
                return False
        except Exception:
            # If length checks fail for some exotic type, fall back to non-empty assumption
            pass
        
        # Convert frames to numpy arrays for comparison
        try:
            import numpy as np
            frame_before_array = np.array(frame_before)
            frame_after_array = np.array(frame_after)
            
            # Check if frames are different
            frames_different = not np.array_equal(frame_before_array, frame_after_array)
            
            return frames_different
        except Exception as e:
            logger.warning(f"Failed to compare frames for object testing: {e}")
            return False
    
    def _update_object_testing(self, action: Dict[str, Any], game_state: Dict[str, Any], frame_analysis: Dict[str, Any]):
        """Update object testing based on Action 6 results - focus on visual/frame changes as buttons."""
        if action.get('id') != 6 or 'x' not in action or 'y' not in action:
            return

        coordinate = (action['x'], action['y'])
        current_score = game_state.get('score', 0)
        current_actions = set(game_state.get('available_actions', []))
        
        # Track previous state for comparison
        previous_score = 0
        previous_actions = set()
        if len(self.performance_history) > 0:
            previous_score = self.performance_history[-1].get('score', 0)
            previous_actions = set(self.performance_history[-1].get('available_actions', []))

        # Initialize object testing data if not exists
        if coordinate not in self.tested_objects:
            self.tested_objects[coordinate] = {
                'interactive': False,
                'test_count': 0,
                'frame_changes': 0,
                'score_changes': 0,
                'movement_detected': 0,
                'action_unlocks': 0,
                'button_confidence': 0.0,
                'button_type': 'unknown'  # 'score_button', 'action_button', 'visual_button'
            }

        # Increment test count
        self.tested_objects[coordinate]['test_count'] += 1

        # Check for different types of positive signals
        positive_signal = False
        signal_type = None
        
        # 1. Score increase (strongest signal - direct progress)
        if current_score > previous_score:
            self.tested_objects[coordinate]['score_changes'] += 1
            positive_signal = True
            signal_type = 'score_button'
            logger.info(f" BUTTON DETECTED: Object at {coordinate} caused SCORE INCREASE: {previous_score} -> {current_score}")
        
        # 2. Action unlock (new actions available - strategic progress)
        elif current_actions != previous_actions:
            new_actions = current_actions - previous_actions
            if new_actions:
                self.tested_objects[coordinate]['action_unlocks'] += 1
                positive_signal = True
                signal_type = 'action_button'
                logger.info(f" BUTTON DETECTED: Object at {coordinate} UNLOCKED NEW ACTIONS: {new_actions}")
        
        # 3. Frame movement/visual change (any change indicates interactivity)
        elif current_score != previous_score or current_actions != previous_actions:
            self.tested_objects[coordinate]['movement_detected'] += 1
            positive_signal = True
            signal_type = 'visual_button'
            logger.info(f" BUTTON DETECTED: Object at {coordinate} caused VISUAL/FRAME CHANGE")

        # Update frame changes counter (any positive signal)
        if positive_signal:
            self.tested_objects[coordinate]['frame_changes'] += 1
            
            # Update button type and confidence
            if signal_type:
                self.tested_objects[coordinate]['button_type'] = signal_type
                
                # Calculate confidence based on signal strength
                if signal_type == 'score_button':
                    self.tested_objects[coordinate]['button_confidence'] = min(1.0, 
                        self.tested_objects[coordinate]['button_confidence'] + 0.3)
                elif signal_type == 'action_button':
                    self.tested_objects[coordinate]['button_confidence'] = min(1.0, 
                        self.tested_objects[coordinate]['button_confidence'] + 0.25)
                elif signal_type == 'visual_button':
                    self.tested_objects[coordinate]['button_confidence'] = min(1.0, 
                        self.tested_objects[coordinate]['button_confidence'] + 0.2)

        # Determine if object is a confirmed button
        test_count = self.tested_objects[coordinate]['test_count']
        frame_changes = self.tested_objects[coordinate]['frame_changes']
        score_changes = self.tested_objects[coordinate]['score_changes']
        action_unlocks = self.tested_objects[coordinate]['action_unlocks']
        button_confidence = self.tested_objects[coordinate]['button_confidence']

        if test_count >= self.object_test_threshold:
            # After enough tests, determine if object is a button
            if frame_changes > 0:
                # Object caused positive signals - it's a confirmed button
                self.tested_objects[coordinate]['interactive'] = True
                self.interactive_objects.add(coordinate)
                
                button_type = self.tested_objects[coordinate]['button_type']
                logger.info(f" CONFIRMED BUTTON: Object at {coordinate} is a {button_type.upper()}")
                logger.info(f"   - Confidence: {button_confidence:.2f}")
                logger.info(f"   - Score changes: {score_changes}, Action unlocks: {action_unlocks}")
                logger.info(f"   - Total changes: {frame_changes}/{test_count} tests")
                
                # Mark as high-priority button for Action 6 centric games
                if button_type in ['score_button', 'action_button']:
                    logger.info(f" HIGH-PRIORITY BUTTON: {coordinate} should be prioritized for score/action progress")
            else:
                # Object never caused positive signals - it's not a button
                self.tested_objects[coordinate]['interactive'] = False
                self.non_interactive_objects.add(coordinate)
                logger.info(f" NOT A BUTTON: Object at {coordinate} never caused changes ({frame_changes}/{test_count} tests)")
    
    def _get_interactive_objects(self, frame_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get confirmed interactive objects from frame analysis - prioritize button types."""
        suggestions = []
        
        # Sort interactive objects by button type and confidence
        sorted_objects = sorted(
            self.interactive_objects,
            key=lambda coord: (
                self.tested_objects.get(coord, {}).get('button_type', 'unknown'),
                -self.tested_objects.get(coord, {}).get('button_confidence', 0.0)
            )
        )
        
        # Get objects from frame analysis
        objects = frame_analysis.get('objects', [])
        interactive_elements = frame_analysis.get('interactive_elements', [])
        
        # Check confirmed interactive objects with button prioritization
        for coord in sorted_objects:
            # Check if coordinate is still in current frame
            if self._is_coordinate_in_frame(coord, frame_analysis):
                obj_data = self.tested_objects.get(coord, {})
                button_type = obj_data.get('button_type', 'unknown')
                button_confidence = obj_data.get('button_confidence', 0.0)
                score_changes = obj_data.get('score_changes', 0)
                action_unlocks = obj_data.get('action_unlocks', 0)
                
                # Calculate confidence based on button type and performance
                base_confidence = 0.9
                if button_type == 'score_button':
                    base_confidence = 0.95  # Highest priority for score buttons
                elif button_type == 'action_button':
                    base_confidence = 0.9   # High priority for action unlock buttons
                elif button_type == 'visual_button':
                    base_confidence = 0.8   # Medium priority for visual buttons
                
                # Boost confidence based on performance
                performance_boost = min(0.05, (score_changes + action_unlocks) * 0.01)
                final_confidence = min(1.0, base_confidence + performance_boost)
                
                # Create detailed reason
                reason_parts = [f'Confirmed {button_type.replace("_", " ")} at {coord}']
                if score_changes > 0:
                    reason_parts.append(f'{score_changes} score increases')
                if action_unlocks > 0:
                    reason_parts.append(f'{action_unlocks} action unlocks')
                reason_parts.append(f'confidence: {button_confidence:.2f}')
                
                suggestions.append({
                    'action': 'ACTION6',
                    'x': coord[0],
                    'y': coord[1],
                    'confidence': final_confidence,
                    'reason': ' | '.join(reason_parts),
                    'source': 'confirmed_interactive',
                    'button_type': button_type,
                    'button_confidence': button_confidence
                })
        
        # Log button prioritization
        if suggestions:
            button_types = [s.get('button_type', 'unknown') for s in suggestions]
            logger.info(f" Interactive objects prioritized: {button_types}")
        
        return suggestions
    
    def _is_action6_centric_game(self, game_state: Dict[str, Any], available_actions: List[int]) -> bool:
        """Detect if this is an Action 6 centric game based on available actions and game state."""
        try:
            # Check if Action 6 is available
            if 6 not in available_actions:
                return False
            
            # Count non-Action 6 actions
            non_action6_actions = [a for a in available_actions if a != 6]
            
            # If very few non-Action 6 actions, likely Action 6 centric
            if len(non_action6_actions) <= 2:
                return True
            
            # Check if Action 6 has been used frequently in this game
            action6_usage = 0
            total_actions = 0
            
            for entry in self.performance_history[-20:]:  # Check last 20 actions
                action = entry.get('action', {})
                if action.get('id') == 6:
                    action6_usage += 1
                total_actions += 1
            
            if total_actions > 0:
                action6_ratio = action6_usage / total_actions
                # If Action 6 is used more than 60% of the time, it's centric
                if action6_ratio > 0.6:
                    return True
            
            # Check if we have many interactive objects (suggests button-heavy game)
            if len(self.interactive_objects) > 3:
                return True
            
            # Check if we have many tested objects (suggests exploration-heavy game)
            if len(self.tested_objects) > 5:
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Failed to detect Action 6 centric game: {e}")
            return False
    
    def _is_coordinate_in_frame(self, coord: Tuple[int, int], frame_analysis: Dict[str, Any]) -> bool:
        """Check if a coordinate is still present in the current frame."""
        try:
            x, y = coord
            objects = frame_analysis.get('objects', [])
            
            # Check if coordinate is near any detected object
            for obj in objects:
                centroid = obj.get('centroid', (0, 0))
                if abs(centroid[0] - x) <= 5 and abs(centroid[1] - y) <= 5:
                    return True
            
            return True  # Assume coordinate is valid if no objects detected
        except Exception as e:
            logger.warning(f"Failed to check coordinate in frame: {e}")
            return True
    
    def _get_untested_objects(self, frame_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get objects that haven't been tested yet for interactivity."""
        suggestions = []
        
        # Get objects from frame analysis
        objects = frame_analysis.get('objects', [])
        interactive_elements = frame_analysis.get('interactive_elements', [])
        
        # Check untested objects
        for obj in objects:
            centroid = obj.get('centroid', (0, 0))
            if (centroid not in self.interactive_objects and 
                centroid not in self.non_interactive_objects and
                centroid not in self.tested_objects):
                suggestions.append({
                    'action': 'ACTION6',
                    'x': centroid[0],
                    'y': centroid[1],
                    'confidence': 0.7,  # Medium confidence for untested objects
                    'reason': f"Untested object at ({centroid[0]}, {centroid[1]}) - testing for interactivity",
                    'source': 'object_testing'
                })
        
        # Check untested interactive elements
        for element in interactive_elements:
            centroid = element.get('centroid', (0, 0))
            if (centroid not in self.interactive_objects and 
                centroid not in self.non_interactive_objects and
                centroid not in self.tested_objects):
                suggestions.append({
                    'action': 'ACTION6',
                    'x': centroid[0],
                    'y': centroid[1],
                    'confidence': 0.8,  # Higher confidence for untested interactive elements
                    'reason': f"Untested interactive element at ({centroid[0]}, {centroid[1]}) - testing for interactivity",
                    'source': 'object_testing'
                })
        
        return suggestions

    def _gather_buttons_from_frame(self, frame_analysis: Dict[str, Any]) -> List[tuple]:
        """Return a list of candidate button coordinates found in a frame analysis."""
        coords = []
        try:
            if not frame_analysis:
                return coords

            # Favor interactive_elements, then objects with sufficient area, then centroids
            interactive = frame_analysis.get('interactive_elements', [])
            for el in interactive:
                centroid = el.get('centroid')
                if centroid and isinstance(centroid, (list, tuple)):
                    coords.append((int(centroid[0]), int(centroid[1])))

            # Add large object centroids
            for obj in frame_analysis.get('objects', []):
                if obj.get('area', 0) > 20:
                    centroid = obj.get('centroid')
                    if centroid and isinstance(centroid, (list, tuple)):
                        coords.append((int(centroid[0]), int(centroid[1])))

            # Deduplicate while preserving order
            seen = set()
            unique = []
            for c in coords:
                if c not in seen:
                    seen.add(c)
                    unique.append(c)

            return unique
        except Exception:
            return coords
    
    def _get_exploration_coordinates(self, frame_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get exploration coordinates for Action 6 that grow incrementally."""
        suggestions = []
        
        # Get current exploration radius (starts small, grows over time)
        base_radius = 8  # Start with 8x8 area
        max_radius = 32  # Maximum exploration area
        exploration_radius = min(base_radius + (self.learning_cycles // 10), max_radius)
        
        # Generate exploration coordinates in expanding pattern
        center_x, center_y = 16, 16  # Center of 32x32 grid
        
        # Create exploration pattern: spiral outward
        for radius in range(2, exploration_radius, 2):
            for angle in range(0, 360, 45):  # 8 directions
                import math
                x = int(center_x + radius * math.cos(math.radians(angle)))
                y = int(center_y + radius * math.sin(math.radians(angle)))
                
                # Ensure coordinates are within bounds
                x = max(0, min(31, x))
                y = max(0, min(31, y))
                
                # Skip if already tested and non-interactive
                if (x, y) in self.non_interactive_objects:
                    continue
                
                # Skip if heavily penalized
                coordinate_key = f"6_{x}_{y}"
                if coordinate_key in self.action_repetition_penalties:
                    penalty = self.action_repetition_penalties[coordinate_key]
                    if penalty > 0.7:  # Skip heavily penalized coordinates
                        continue
                
                suggestions.append({
                    'action': 'ACTION6',
                    'x': x,
                    'y': y,
                    'confidence': 0.5 + (0.3 * (1 - radius / max_radius)),  # Higher confidence for closer coordinates
                    'reason': f'Exploration at ({x}, {y}) - radius {radius}',
                    'source': 'coordinate_exploration'
                })
        
        return suggestions
    
    def _generate_bayesian_suggestions(self, frame_analysis: Dict[str, Any], 
                                     available_actions: List[int], 
                                     current_score: int, 
                                     game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate suggestions using Bayesian pattern detection."""
        suggestions = []
        
        if not self.bayesian_scorer:
            return suggestions
        
        try:
            # Create a mock search path for Bayesian scoring
            # Note: SearchPath might not be available, so we'll create a simple mock
            class MockSearchPath:
                def __init__(self, actions, state_features):
                    self.actions = actions
                    self.state_features = state_features
                    self.nodes = []  # Add missing nodes attribute
            
            # Generate suggestions for each available action
            for action_id in available_actions:
                # Create a simple search path for this action
                path = MockSearchPath(
                    actions=[(action_id, None)],
                    state_features={
                        'score': current_score,
                        'frame_complexity': len(frame_analysis.get('objects', [])),
                        'interactive_elements': len(frame_analysis.get('interactive_elements', []))
                    }
                )
                
                # Get Bayesian success probability
                success_prob = self.bayesian_scorer.score_path_success_probability(path)
                
                if success_prob > 0.3:  # Only suggest actions with reasonable success probability
                    suggestion = {
                        'action': f'ACTION{action_id}',
                        'confidence': success_prob,
                        'reason': f'Bayesian pattern detection (success prob: {success_prob:.2f})',
                        'source': 'bayesian_pattern'
                    }
                    
                    # Add coordinates for Action 6
                    if action_id == 6:
                        # Use center coordinates for Bayesian suggestions
                        suggestion['x'] = 16
                        suggestion['y'] = 16
                    
                    suggestions.append(suggestion)
        
        except Exception as e:
            logger.warning(f"Bayesian suggestion generation failed: {e}")
        
        return suggestions
    
    async def _generate_gan_suggestions(self, frame_analysis: Dict[str, Any], 
                                available_actions: List[int], 
                                current_score: int, 
                                game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate suggestions using GAN system. This method is async because
        GAN internals may expose async coroutines (e.g., GPU calls or remote inference).
        """
        suggestions = []

        if not self.gan_system:
            return suggestions

        try:
            # Generate synthetic game states using GAN
            context = {
                'current_score': current_score,
                'frame_objects': len(frame_analysis.get('objects', [])),
                'available_actions': available_actions
            }

            # Generate a few synthetic states
            synthetic_states = self.gan_system.generate_synthetic_states(3, context)

            # If the GAN returned a coroutine/awaitable, await it
            if inspect.isawaitable(synthetic_states):
                synthetic_states = await synthetic_states

            # Now iterate synthetic states safely
            for state in synthetic_states:
                # Some GAN implementations might return dicts; normalize access
                prob = getattr(state, 'success_probability', None)
                if prob is None and isinstance(state, dict):
                    prob = state.get('success_probability', 0)

                if prob and prob > 0.5:  # Only use high-probability synthetic states
                    # Extract action suggestions from synthetic state
                    for action_id in available_actions:
                        suggestion = {
                            'action': f'ACTION{action_id}',
                            'confidence': prob,
                            'reason': f'GAN-generated synthetic state (prob: {prob:.2f})',
                            'source': 'gan_synthetic'
                        }

                        # Add coordinates for Action 6
                        if action_id == 6:
                            # Use coordinates from synthetic state or default
                            if isinstance(state, dict):
                                suggestion['x'] = state.get('x', 16)
                                suggestion['y'] = state.get('y', 16)
                            else:
                                suggestion['x'] = getattr(state, 'x', 16)
                                suggestion['y'] = getattr(state, 'y', 16)

                        suggestions.append(suggestion)

        except Exception as e:
            logger.warning(f"GAN suggestion generation failed: {e}")

        return suggestions
    
    def _detect_stuck_situation(self, game_state: Dict[str, Any], available_actions: List[int]) -> bool:
        """Detect if the system is stuck and needs Action 5 fallback."""
        # Check if we've been stuck for too long
        if len(self.performance_history) < 10:
            return False
        
        # Check recent performance - if no score improvement in last 10 actions
        recent_scores = [entry.get('score', 0) for entry in self.performance_history[-10:]]
        if len(set(recent_scores)) == 1:  # All same score = no progress
            return True
        
        # Check if we're repeating the same action too much
        if len(self.recent_actions) >= 5:
            recent_action_counts = {}
            for action in self.recent_actions[-5:]:
                action_id = action if isinstance(action, int) else int(action.split('_')[0])
                recent_action_counts[action_id] = recent_action_counts.get(action_id, 0) + 1
            
            # If any action is repeated 4+ times in last 5 actions
            if max(recent_action_counts.values()) >= 4:
                return True
        
        # Check if we're stuck on Action 6 with no interactive objects
        if 6 in available_actions and len(self.interactive_objects) == 0:
            # If we've tested many objects and found none interactive
            total_tests = sum(obj.get('test_count', 0) for obj in self.tested_objects.values())
            if total_tests > 20:  # Tested many objects, found none interactive
                return True
        
        return False
    
    def _generate_action5_fallback_suggestions(self, available_actions: List[int]) -> List[Dict[str, Any]]:
        """Generate Action 5 fallback suggestions when system is stuck."""
        suggestions = []
        
        if 5 in available_actions:
            suggestions.append({
                'action': 'ACTION5',
                'confidence': 0.9,  # Very high confidence for Action 5 when stuck
                'reason': 'Action 5 fallback - system detected as stuck',
                'source': 'action5_stuck_fallback'
            })
        
        return suggestions
    
    def _track_action_availability_changes(self, current_available_actions: List[int], 
                                         last_action: Dict[str, Any]) -> Dict[str, Any]:
        """Track changes in available actions and treat them as pseudo score increases."""
        current_actions_set = set(current_available_actions)
        previous_actions_set = self.previous_available_actions
        
        # Calculate changes
        newly_available = current_actions_set - previous_actions_set
        newly_unavailable = previous_actions_set - current_actions_set
        
        change_info = {
            'has_changes': len(newly_available) > 0 or len(newly_unavailable) > 0,
            'newly_available': list(newly_available),
            'newly_unavailable': list(newly_unavailable),
            'pseudo_score_increase': len(newly_available) > 0,  # New actions = positive
            'pseudo_score_decrease': len(newly_unavailable) > 0,  # Lost actions = negative
            'net_change': len(newly_available) - len(newly_unavailable)
        }
        
        # If there were changes, record them
        if change_info['has_changes']:
            action_id = last_action.get('id')
            if action_id:
                if action_id not in self.action_availability_changes:
                    self.action_availability_changes[action_id] = {'count': 0, 'last_change': 0}
                
                self.action_availability_changes[action_id]['count'] += 1
                self.action_availability_changes[action_id]['last_change'] = len(self.performance_history)
                
                # If this action caused new actions to become available, it's a pseudo button
                if change_info['pseudo_score_increase']:
                    self.pseudo_buttons.add(action_id)
                    logger.info(f" Action {action_id} discovered new available actions: {newly_available} - PSEUDO BUTTON!")
                
                logger.info(f" Action availability changed: +{len(newly_available)} -{len(newly_unavailable)} (Action {action_id})")
                logger.info(f"   Newly available: {newly_available}")
                if newly_unavailable:
                    logger.info(f"   Newly unavailable: {newly_unavailable}")
        
        # Update previous available actions
        self.previous_available_actions = current_actions_set.copy()
        
        return change_info
    
    def _generate_pseudo_button_suggestions(self, available_actions: List[int]) -> List[Dict[str, Any]]:
        """Generate suggestions based on actions that previously caused action availability changes."""
        suggestions = []
        
        # Prioritize actions that have previously caused new actions to become available
        for action_id in self.pseudo_buttons:
            if action_id in available_actions:
                # Get effectiveness data for this pseudo button
                effectiveness = self.action_effectiveness.get(action_id, {})
                pseudo_successes = effectiveness.get('pseudo_successes', 0)
                attempts = effectiveness.get('attempts', 1)
                pseudo_success_rate = pseudo_successes / attempts if attempts > 0 else 0
                
                if pseudo_success_rate > 0.1:  # Only suggest if it has some pseudo success rate
                    suggestion = {
                        'action': f'ACTION{action_id}',
                        'confidence': 0.8 + (pseudo_success_rate * 0.2),  # High confidence for pseudo buttons
                        'reason': f'Pseudo button - previously discovered {pseudo_successes} new actions',
                        'source': 'pseudo_button'
                    }
                    
                    # Add coordinates for Action 6
                    if action_id == 6:
                        suggestion['x'] = 16
                        suggestion['y'] = 16
                    
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _add_reasoning_to_action(self, action: Dict[str, Any], game_state: Dict[str, Any], 
                                frame_analysis: Dict[str, Any], all_suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add detailed reasoning to the action for API submission."""
        
        # Create comprehensive reasoning object
        reasoning = {
            "policy": f"π_{action.get('source', 'unknown')}",
            "action_id": action.get('id'),
            "confidence": action.get('confidence', 0.0),
            "source": action.get('source', 'unknown'),
            "reason": action.get('reason', 'no reason'),
            "game_state": {
                "score": game_state.get('score', 0),
                "state": game_state.get('state', 'UNKNOWN'),
                "available_actions": game_state.get('available_actions', [])
            },
            "frame_analysis": {
                "objects_detected": len(frame_analysis.get('objects', [])),
                "interactive_elements": len(frame_analysis.get('interactive_elements', [])),
                "patterns_found": len(frame_analysis.get('patterns', []))
            },
            "learning_data": {
                "action_effectiveness": self.action_effectiveness.get(action.get('id'), {}),
                "coordinate_success_rates": len(self.coordinate_success_rates),
                "pseudo_buttons": list(self.pseudo_buttons),
                "learning_cycles": self.learning_cycles
            },
            "suggestion_analysis": {
                "total_suggestions": len(all_suggestions),
                "sources_used": list(set(s.get('source', 'unknown') for s in all_suggestions)),
                "top_3_suggestions": [
                    {
                        "action": s.get('action', 'UNKNOWN'),
                        "source": s.get('source', 'unknown'),
                        "confidence": s.get('confidence', 0.0),
                        "reason": s.get('reason', 'no reason')
                    }
                    for s in sorted(all_suggestions, key=lambda x: x.get('confidence', 0), reverse=True)[:3]
                ]
            },
            "penalty_data": {
                "action_repetition_penalty": self.action_repetition_penalties.get(action.get('id'), 0.0),
                "coordinate_penalty": 0.0  # Will be calculated if Action 6
            }
        }

        # Ensure action repetition penalty uses coordinate-keyed penalty for ACTION6
        try:
            action_id_val = action.get('id')
            if action_id_val == 6 and 'x' in action and 'y' in action:
                coord_key = f"6_{action['x']}_{action['y']}"
                reasoning["penalty_data"]["action_repetition_penalty"] = self.action_repetition_penalties.get(coord_key, 0.0)
            else:
                reasoning["penalty_data"]["action_repetition_penalty"] = self.action_repetition_penalties.get(action_id_val, 0.0)
        except Exception:
            # If anything goes wrong, keep the existing penalty value
            pass
        
        # Add coordinate-specific data for Action 6
        if action.get('id') == 6 and 'x' in action and 'y' in action:
            coordinate = (action['x'], action['y'])
            coordinate_key = f"6_{coordinate[0]}_{coordinate[1]}"
            reasoning["penalty_data"]["coordinate_penalty"] = self.action_repetition_penalties.get(coordinate_key, 0.0)
            reasoning["coordinate_data"] = {
                "x": action['x'],
                "y": action['y'],
                "coordinate_success_rate": self.coordinate_success_rates.get(coordinate, {}).get('success_rate', 0.0),
                "coordinate_attempts": self.coordinate_success_rates.get(coordinate, {}).get('attempts', 0)
            }
        
        # Add pseudo score data if available
        if hasattr(self, 'action_availability_changes') and self.action_availability_changes:
            reasoning["pseudo_score_data"] = {
                "total_availability_changes": sum(data.get('count', 0) for data in self.action_availability_changes.values()),
                "actions_that_discovered_new_actions": list(self.pseudo_buttons)
            }
        
        # Add the reasoning to the action
        action['reasoning'] = reasoning
        
        # Log the reasoning for debugging
        logger.info(f" Action Reasoning: {reasoning['policy']} | Confidence: {reasoning['confidence']:.2f}")
        logger.info(f"   Sources used: {', '.join(reasoning['suggestion_analysis']['sources_used'])}")
        logger.info(f"   Learning cycles: {reasoning['learning_data']['learning_cycles']}")
        
        return action
    
    def _generate_game_specific_suggestions(self, game_state: Dict[str, Any], 
                                          available_actions: List[int], 
                                          frame_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate suggestions based on game-specific knowledge (lp85, vc33, etc.)."""
        suggestions = []
        
        if not self.game_type_classifier:
            return suggestions
        
        try:
            game_id = game_state.get('game_id', 'unknown')
            
            # Get game-specific recommendations
            recommendations = self.game_type_classifier.get_recommendations_for_game(game_id)
            game_type = recommendations.get('primary_game_type', 'unknown')
            
            logger.info(f" Game Type: {game_type} | Success Rate: {recommendations.get('success_rate', 0):.2f}")
            
            # Log Action 6 centric status
            if recommendations.get('is_action6_centric', False):
                logger.info(f" Action 6 Centric Game: {recommendations.get('action6_centric_count', 0)} previous games")
            
            # Log button priorities available
            button_priorities = recommendations.get('button_priorities', [])
            if button_priorities:
                logger.info(f" Button Priorities Available: {len(button_priorities)} saved buttons")
                score_buttons = [bp for bp in button_priorities if bp.get('button_type') == 'score_button']
                action_buttons = [bp for bp in button_priorities if bp.get('button_type') == 'action_button']
                visual_buttons = [bp for bp in button_priorities if bp.get('button_type') == 'visual_button']
                logger.info(f"   - Score buttons: {len(score_buttons)}, Action buttons: {len(action_buttons)}, Visual buttons: {len(visual_buttons)}")
            
            # Generate suggestions based on button priorities (highest priority)
            for button_priority in recommendations.get('button_priorities', [])[:10]:  # Top 10 button priorities
                if 6 in available_actions and 'coordinate' in button_priority:
                    coord = button_priority['coordinate']
                    button_type = button_priority.get('button_type', 'unknown')
                    confidence = button_priority.get('confidence', 0.0)
                    success_count = button_priority.get('success_count', 1)
                    
                    # Calculate confidence based on button type and historical success
                    base_confidence = 0.8
                    if button_type == 'score_button':
                        base_confidence = 0.9
                    elif button_type == 'action_button':
                        base_confidence = 0.85
                    elif button_type == 'visual_button':
                        base_confidence = 0.8
                    
                    # Boost confidence based on success count
                    success_boost = min(0.1, success_count * 0.02)
                    final_confidence = min(1.0, base_confidence + success_boost)
                    
                    suggestions.append({
                        'action': 'ACTION6',
                        'x': coord[0],
                        'y': coord[1],
                        'confidence': final_confidence,
                        'reason': f'Game-specific {button_type} from {game_type} (confidence: {confidence:.2f}, successes: {success_count})',
                        'source': 'game_specific_knowledge',
                        'button_type': button_type,
                        'button_confidence': confidence
                    })
            
            # Generate suggestions based on successful coordinates (fallback)
            for coord in recommendations.get('recommended_coordinates', [])[:3]:  # Top 3
                if 6 in available_actions:
                    suggestions.append({
                        'action': 'ACTION6',
                        'x': coord[0],
                        'y': coord[1],
                        'confidence': 0.75,  # Medium confidence for coordinates without button data
                        'reason': f'Game-specific coordinate from {game_type} (success rate: {recommendations.get("success_rate", 0):.2f})',
                        'source': 'game_specific_knowledge'
                    })
            
            # Generate suggestions based on interactive objects (fallback)
            for obj in recommendations.get('recommended_objects', [])[:3]:  # Top 3
                if 6 in available_actions and 'coordinate' in obj:
                    coord = obj['coordinate']
                    suggestions.append({
                        'action': 'ACTION6',
                        'x': coord[0],
                        'y': coord[1],
                        'confidence': 0.8,  # High confidence for confirmed interactive objects
                        'reason': f'Game-specific interactive object from {game_type}',
                        'source': 'game_specific_knowledge'
                    })
            
            # Generate suggestions based on winning sequences
            for sequence in recommendations.get('recommended_sequences', [])[:3]:  # Top 3
                if sequence and len(sequence) > 0:
                    first_action = sequence[0]
                    if first_action in available_actions:
                        suggestions.append({
                            'action': f'ACTION{first_action}',
                            'confidence': 0.75,  # High confidence for winning sequences
                            'reason': f'Game-specific winning sequence from {game_type}',
                            'source': 'game_specific_knowledge'
                        })
            
            if suggestions:
                logger.info(f" Generated {len(suggestions)} game-specific suggestions for {game_type}")
        
        except Exception as e:
            logger.warning(f"Failed to generate game-specific suggestions: {e}")
        
        return suggestions
    
    def _track_game_result(self, action: Dict[str, Any], game_state: Dict[str, Any], frame_analysis: Dict[str, Any]):
        """Track game results and save game-specific knowledge."""
        if not self.game_type_classifier:
            return
        
        try:
            game_id = game_state.get('game_id', 'unknown')
            current_score = game_state.get('score', 0)
            game_status = game_state.get('state', 'UNKNOWN')
            
            # Determine if this was a successful action
            success = False
            if len(self.performance_history) > 0:
                last_score = self.performance_history[-1].get('score', 0)
                success = current_score > last_score
            
            # Check if game is complete
            game_complete = game_status in ['WIN', 'GAME_OVER', 'NOT_STARTED']
            
            if success or game_complete:
                # Extract patterns from this action
                patterns = []
                if action.get('source') == 'game_specific_knowledge':
                    patterns.append({
                        'pattern_id': f"{game_id}_{action.get('id')}_{action.get('source')}",
                        'pattern_type': 'action_success',
                        'confidence': action.get('confidence', 0.0),
                        'context': {
                            'action_id': action.get('id'),
                            'source': action.get('source'),
                            'reason': action.get('reason', '')
                        }
                    })
                
                # Extract successful coordinates
                successful_coordinates = []
                if action.get('id') == 6 and 'x' in action and 'y' in action:
                    successful_coordinates.append((action['x'], action['y']))
                
                # Extract interactive objects
                interactive_objects = []
                if action.get('id') == 6 and 'x' in action and 'y' in action:
                    interactive_objects.append({
                        'coordinate': (action['x'], action['y']),
                        'confidence': action.get('confidence', 0.0),
                        'source': action.get('source', 'unknown'),
                        'reason': action.get('reason', '')
                    })
                
                # Extract winning sequence
                winning_sequence = []
                if game_complete and game_status == 'WIN':
                    # Get recent actions as winning sequence
                    recent_actions = [entry.get('action', {}).get('id') for entry in self.performance_history[-10:]]
                    winning_sequence = [a for a in recent_actions if a is not None]
                
                # Extract button priorities from current tested objects
                button_priorities = []
                for coord, obj_data in self.tested_objects.items():
                    if obj_data.get('interactive', False):
                        button_priorities.append({
                            'coordinate': coord,
                            'button_type': obj_data.get('button_type', 'unknown'),
                            'confidence': obj_data.get('button_confidence', 0.0),
                            'score_changes': obj_data.get('score_changes', 0),
                            'action_unlocks': obj_data.get('action_unlocks', 0),
                            'test_count': obj_data.get('test_count', 0)
                        })
                
                # Check if this is an Action 6 centric game
                is_action6_centric = self._is_action6_centric_game(game_state, game_state.get('available_actions', []))
                
                # Update game type classifier
                self.game_type_classifier.update_game_result(
                    game_id=game_id,
                    success=success,
                    score=current_score,
                    patterns=patterns,
                    successful_coordinates=successful_coordinates,
                    interactive_objects=interactive_objects,
                    winning_sequence=winning_sequence,
                    button_priorities=button_priorities,
                    action6_centric=is_action6_centric
                )
                
                if success:
                    logger.info(f" Saved game-specific knowledge for {game_id}: {len(patterns)} patterns, {len(successful_coordinates)} coordinates")
                
                if game_complete:
                    logger.info(f" Game {game_id} completed with status {game_status} - knowledge saved")
        
        except Exception as e:
            logger.warning(f"Failed to track game result: {e}")
    
    def _track_action_repetition(self, action: Dict[str, Any]) -> float:
        """Track action repetition and return penalty score."""
        action_id = action.get('id')
        if not action_id:
            return 0.0
        
        # For Action 6, track coordinate-specific repetition instead of action repetition
        if action_id == 6 and 'x' in action and 'y' in action:
            coordinate = (action['x'], action['y'])
            return self._track_coordinate_repetition(coordinate)
        
        # For other actions, track action-level repetition
        # Add to recent actions
        self.recent_actions.append(action_id)
        if len(self.recent_actions) > self.max_recent_actions:
            self.recent_actions.pop(0)
        
        # Count recent repetitions of this action
        recent_count = self.recent_actions.count(action_id)
        
        # Calculate penalty based on repetition frequency
        if recent_count >= 10:  # 10+ repetitions in last 20 actions
            penalty = 0.9  # Very high penalty
        elif recent_count >= 7:  # 7+ repetitions
            penalty = 0.7  # High penalty
        elif recent_count >= 5:  # 5+ repetitions
            penalty = 0.5  # Medium penalty
        elif recent_count >= 3:  # 3+ repetitions
            penalty = 0.3  # Low penalty
        else:
            penalty = 0.0  # No penalty
        
        # Store penalty for this action
        self.action_repetition_penalties[action_id] = penalty
        
        return penalty
    
    def _track_coordinate_repetition(self, coordinate: tuple) -> float:
        """Track coordinate-specific repetition for Action 6."""
        # Add coordinate to recent actions with coordinate info
        self.recent_actions.append(f"6_{coordinate[0]}_{coordinate[1]}")
        if len(self.recent_actions) > self.max_recent_actions:
            self.recent_actions.pop(0)
        
        # Count recent repetitions of this specific coordinate
        coordinate_key = f"6_{coordinate[0]}_{coordinate[1]}"
        recent_count = self.recent_actions.count(coordinate_key)
        
        # Calculate penalty based on coordinate repetition frequency
        if recent_count >= 8:  # 8+ repetitions of same coordinate
            penalty = 0.9  # Very high penalty
        elif recent_count >= 6:  # 6+ repetitions
            penalty = 0.7  # High penalty
        elif recent_count >= 4:  # 4+ repetitions
            penalty = 0.5  # Medium penalty
        elif recent_count >= 2:  # 2+ repetitions
            penalty = 0.3  # Low penalty
        else:
            penalty = 0.0  # No penalty
        
        # Store penalty for this coordinate
        self.action_repetition_penalties[coordinate_key] = penalty
        
        return penalty

    def _reset_visual_target_penalties(self, frame_analysis: Dict[str, Any]) -> None:
        """Reset repetition penalties for coordinates that correspond to current visual objects."""
        if not hasattr(self, 'action_repetition_penalties'):
            return
            
        # Get current visual targets
        visual_targets = set()
        
        # Add objects as visual targets
        for obj in frame_analysis.get('objects', []):
            if obj.get('color', 0) != 0:  # Non-background colored objects
                cx, cy = obj.get('centroid', (0, 0))
                visual_targets.add((cx, cy))
        
        # Add interactive elements as visual targets
        for element in frame_analysis.get('interactive_elements', []):
            cx, cy = element.get('centroid', (0, 0))
            visual_targets.add((cx, cy))
        
        # Reset penalties for coordinates near visual targets
        penalties_reset = 0
        for coord in visual_targets:
            coordinate_key = f"6_{coord[0]}_{coord[1]}"
            if coordinate_key in self.action_repetition_penalties:
                old_penalty = self.action_repetition_penalties[coordinate_key]
                # Reduce penalty instead of completely removing it
                new_penalty = max(0.0, old_penalty - 0.2)  # Reduce by 20%
                self.action_repetition_penalties[coordinate_key] = new_penalty
                if old_penalty > 0.5 and new_penalty <= 0.5:
                    penalties_reset += 1
                    logger.info(f"🎯 Reset penalty for visual target at {coord}: {old_penalty:.2f} -> {new_penalty:.2f}")
        
        if penalties_reset > 0:
            logger.info(f"🔄 Reset penalties for {penalties_reset} visual targets to allow retries")

    def _should_reset_penalties(self, current_action_count: int) -> bool:
        """Determine if we should reset penalties based on action count and time."""
        if not hasattr(self, '_last_penalty_reset'):
            self._last_penalty_reset = 0
            
        # Reset penalties every 100 actions to give visual targets another chance
        if current_action_count - self._last_penalty_reset >= 100:
            self._last_penalty_reset = current_action_count
            return True
        return False

    def _emergency_penalty_reset(self, game_state: Dict[str, Any], current_action_count: int) -> None:
        """Emergency reset of all coordinate penalties when clearly stuck."""
        current_score = game_state.get('score', 0)
        
        # Define "stuck" conditions
        stuck_conditions = [
            current_action_count > 1000 and current_score == 0,  # 1000+ actions with no score
            current_action_count > 2000,  # 2000+ actions regardless of score
        ]
        
        if any(stuck_conditions):
            if not hasattr(self, '_last_emergency_reset'):
                self._last_emergency_reset = 0
                
            # Only do emergency reset once every 500 actions to avoid spam
            if current_action_count - self._last_emergency_reset >= 500:
                self._last_emergency_reset = current_action_count
                
                # Reset ALL coordinate penalties
                coordinate_penalties_cleared = 0
                for key in list(self.action_repetition_penalties.keys()):
                    if key.startswith('6_'):  # ACTION6 coordinate penalties
                        self.action_repetition_penalties[key] = 0.0
                        coordinate_penalties_cleared += 1
                
                # Also clear recent actions to reset penalty tracking
                self.recent_actions = []
                
                logger.warning(f"🚨 EMERGENCY PENALTY RESET - System appears stuck!")
                logger.warning(f"   Actions taken: {current_action_count}, Score: {current_score}")
                logger.warning(f"   Cleared {coordinate_penalties_cleared} coordinate penalties")
                logger.warning(f"   Giving all visual targets a fresh chance...")
    
    def _get_action_repetition_penalty(self, action_id: int, coordinate: tuple = None) -> float:
        """Get penalty score for action repetition."""
        # For Action 6, check coordinate-specific penalties
        if action_id == 6 and coordinate:
            coordinate_key = f"6_{coordinate[0]}_{coordinate[1]}"
            return self.action_repetition_penalties.get(coordinate_key, 0.0)
        
        # For other actions, check action-level penalties
        return self.action_repetition_penalties.get(action_id, 0.0)
    
    def _decay_action_penalties(self):
        """Decay action repetition penalties over time."""
        for action_id in list(self.action_repetition_penalties.keys()):
            current_penalty = self.action_repetition_penalties[action_id]
            # Decay by 10% each time
            new_penalty = current_penalty * 0.9
            if new_penalty < 0.1:
                # Remove very low penalties
                del self.action_repetition_penalties[action_id]
            else:
                self.action_repetition_penalties[action_id] = new_penalty
    
    def _update_learning_systems(self, action: Dict[str, Any], game_state: Dict[str, Any], frame_analysis: Dict[str, Any]):
        """Update all learning systems with new action and results."""
        
        # Update action history
        self.action_history.append({
            'action': action,
            'game_state': game_state,
            'frame_analysis': frame_analysis,
            'timestamp': game_state.get('timestamp', 0)
        })
        
        # Keep only recent history
        if len(self.action_history) > 1000:
            self.action_history = self.action_history[-1000:]
        
        # Update learning cycles
        self.learning_cycles += 1
        
        # Check for pseudo score increases (action availability changes)
        pseudo_score_increase = False
        if len(self.performance_history) > 0:
            last_action = self.performance_history[-1]
            availability_changes = self._track_action_availability_changes(
                game_state.get('available_actions', []), last_action
            )
            pseudo_score_increase = availability_changes['pseudo_score_increase']
        
        # Update action effectiveness
        action_id = action.get('id')
        if action_id:
            if action_id not in self.action_effectiveness:
                self.action_effectiveness[action_id] = {'successes': 0, 'attempts': 0, 'pseudo_successes': 0}
            
            self.action_effectiveness[action_id]['attempts'] += 1
            
            # Check if action was successful (score increased)
            current_score = game_state.get('score', 0)
            # Handle case where score might be a dictionary or other type
            if isinstance(current_score, dict):
                # If it's a dictionary, try to extract a numeric value
                current_score = current_score.get('value', current_score.get('score', 0))
            if isinstance(current_score, (int, float)):
                current_score = float(current_score)
            else:
                current_score = 0.0
            
            if len(self.performance_history) > 0:
                last_score = self.performance_history[-1].get('score', 0)
                if isinstance(last_score, (int, float)):
                    last_score = float(last_score)
                else:
                    last_score = 0.0
                    
                if current_score > last_score:
                    self.action_effectiveness[action_id]['successes'] += 1
            
            # Track pseudo score increases (action availability changes)
            if pseudo_score_increase:
                self.action_effectiveness[action_id]['pseudo_successes'] += 1
                logger.info(f" Action {action_id} marked as pseudo-successful (discovered new actions)")
        
        # Update coordinate intelligence for ACTION6
        if action.get('id') == 6:
            coordinate = (action.get('x', 0), action.get('y', 0))
            if coordinate not in self.coordinate_success_rates:
                self.coordinate_success_rates[coordinate] = {'successes': 0, 'attempts': 0, 'pseudo_successes': 0}
            
            self.coordinate_success_rates[coordinate]['attempts'] += 1
            
            # Check if coordinate was successful
            current_score = game_state.get('score', 0)
            if isinstance(current_score, (int, float)):
                current_score = float(current_score)
            else:
                current_score = 0.0

            # Initialize last_score before the conditional block
            last_score = 0.0
            if len(self.performance_history) > 0:
                last_score = self.performance_history[-1].get('score', 0)
                if isinstance(last_score, (int, float)):
                    last_score = float(last_score)
                else:
                    last_score = 0.0

                if current_score > last_score:
                    self.coordinate_success_rates[coordinate]['successes'] += 1
            
            # Track pseudo score increases for coordinates
            if pseudo_score_increase:
                self.coordinate_success_rates[coordinate]['pseudo_successes'] += 1
                logger.info(f" Coordinate {coordinate} marked as pseudo-successful (discovered new actions)")
            
            # Calculate success rate (include pseudo successes)
            attempts = self.coordinate_success_rates[coordinate]['attempts']
            successes = self.coordinate_success_rates[coordinate]['successes']
            pseudo_successes = self.coordinate_success_rates[coordinate]['pseudo_successes']
            total_successes = successes + pseudo_successes
            self.coordinate_success_rates[coordinate]['success_rate'] = total_successes / attempts if attempts > 0 else 0
            
            # Update penalty decay system with coordinate attempt
            try:
                from src.core.coordinate_intelligence_system import CoordinateIntelligenceSystem
                if not hasattr(self, 'coordinate_intelligence_system'):
                    self.coordinate_intelligence_system = CoordinateIntelligenceSystem()
                
                # Update coordinate intelligence in database (track task to avoid fire-and-forget)
                import asyncio
                task = asyncio.create_task(self.coordinate_intelligence_system.update_coordinate_intelligence(
                    game_id=game_state.get('game_id', 'unknown'),
                    x=coordinate[0],
                    y=coordinate[1],
                    action_id=6,
                    success=current_score > last_score if len(self.performance_history) > 0 else False,
                    frame_changes=1 if current_score > last_score else 0,
                    score_change=current_score - last_score if len(self.performance_history) > 0 else 0.0,
                    context={'action_selector': True}
                ))
                # Store task to prevent it from being garbage collected
                if not hasattr(self, '_background_tasks'):
                    self._background_tasks = set()
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            except Exception as e:
                logger.error(f"Failed to update coordinate intelligence: {e}")
            try:
                # Record coordinate attempt with penalty system (track tasks)
                task1 = asyncio.create_task(self.frame_analyzer.record_coordinate_attempt(
                    coordinate[0], coordinate[1],
                    (current_score > last_score or pseudo_score_increase) if len(self.performance_history) > 0 else False,
                    game_state.get('game_id', 'unknown')
                ))
                self._background_tasks.add(task1)
                task1.add_done_callback(self._background_tasks.discard)

                # Decay penalties periodically
                if attempts % 10 == 0:  # Decay every 10 attempts
                    task2 = asyncio.create_task(self.frame_analyzer.decay_penalties())
                    self._background_tasks.add(task2)
                    task2.add_done_callback(self._background_tasks.discard)
                    
            except Exception as e:
                logger.warning(f"Failed to update penalty system for coordinate {coordinate}: {e}")
        
        # Save learning patterns periodically
        if self.learning_cycles % 50 == 0:  # Save patterns every 50 learning cycles
            self._save_learning_patterns(action, game_state, frame_analysis)
        
        # Decay action repetition penalties periodically
        if self.learning_cycles % 20 == 0:  # Decay every 20 learning cycles
            self._decay_action_penalties()
    
    def _save_learning_patterns(self, action: Dict[str, Any], game_state: Dict[str, Any], frame_analysis: Dict[str, Any]):
        """Save learning patterns to database."""
        try:
            from src.database.system_integration import get_system_integration
            integration = get_system_integration()
            
            # Save action pattern
            action_pattern = {
                'action_id': action.get('id'),
                'source': action.get('source', 'unknown'),
                'confidence': action.get('confidence', 0.0),
                'coordinates': {'x': action.get('x'), 'y': action.get('y')} if 'x' in action else None,
                'learning_cycles': self.learning_cycles,
                'success_rate': self.action_effectiveness.get(action.get('id', 0), {}).get('success_rate', 0.0)
            }
            
            asyncio.create_task(integration.save_learned_pattern(
                pattern_type='action_pattern',
                pattern_data=action_pattern,
                confidence=action.get('confidence', 0.0),
                frequency=1,
                success_rate=self.action_effectiveness.get(action.get('id', 0), {}).get('success_rate', 0.0),
                game_context=game_state.get('game_id', 'unknown')
            ))
            
            # Save coordinate pattern if coordinates exist
            if 'x' in action and 'y' in action:
                coord_pattern = {
                    'coordinates': {'x': action['x'], 'y': action['y']},
                    'action_id': action.get('id'),
                    'success_rate': self.coordinate_success_rates.get((action['x'], action['y']), 0.0),
                    'learning_cycles': self.learning_cycles
                }
                
                asyncio.create_task(integration.save_learned_pattern(
                    pattern_type='coordinate_pattern',
                    pattern_data=coord_pattern,
                    confidence=action.get('confidence', 0.0),
                    frequency=1,
                    success_rate=self.coordinate_success_rates.get((action['x'], action['y']), 0.0),
                    game_context=game_state.get('game_id', 'unknown')
                ))
            
            logger.info(f" Saved learning patterns at cycle {self.learning_cycles}")
            
        except Exception as e:
            logger.warning(f"Failed to save learning patterns: {e}")
    
    def _track_performance(self, action: Dict[str, Any], current_score: int, game_state: Dict[str, Any]):
        """Track performance and update metrics."""
        
        # Ensure current_score is a number
        # Handle case where score might be a dictionary or other type
        if isinstance(current_score, dict):
            # If it's a dictionary, try to extract a numeric value
            current_score = current_score.get('value', current_score.get('score', 0))
        if isinstance(current_score, (int, float)):
            current_score = float(current_score)
        else:
            current_score = 0.0
        
        # Update performance history
        self.performance_history.append({
            'action': action,
            'score': current_score,
            'timestamp': game_state.get('timestamp', 0)
        })
        
        # Keep only recent performance data
        if len(self.performance_history) > 500:
            self.performance_history = self.performance_history[-500:]
    
    def _validate_and_prioritize_actions(self, available_actions: List[int]) -> List[int]:
        """Validate and prioritize available actions - remove action 0 and prioritize actions 1-4.

        NOTE: Action 6 is treated as a single action type despite having 64x64 coordinate options.
        The coordinate selection is handled separately in suggestion generation.
        """
        # First, filter out action 0 completely - action 0 should never be sent to API
        validated_actions = [action for action in available_actions if action != 0]

        if not validated_actions:
            logger.warning("[PRIORITY] All actions filtered out, including action 0. Using fallback.")
            return [1]  # Emergency fallback

        # Sort actions with priority: 1-4 first (highest priority), then others
        priority_actions = [action for action in validated_actions if 1 <= action <= 4]
        other_actions = [action for action in validated_actions if action < 1 or action > 4]

        # Prioritized order: 1-4 first, then others
        prioritized = priority_actions + other_actions

        if priority_actions:
            logger.debug(f"[PRIORITY] Priority actions 1-4 available: {priority_actions}, others: {other_actions}")

        return prioritized

    def _apply_action_priority_boost(self, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply confidence boost to actions 1-4."""
        for suggestion in suggestions:
            if isinstance(suggestion, dict):
                action_id = self._get_action_id(suggestion.get('action', 'ACTION1'))

                # Boost confidence for actions 1-4
                if 1 <= action_id <= 4:
                    current_confidence = suggestion.get('confidence', 0.5)
                    boosted_confidence = min(current_confidence * 1.5, 0.95)  # 50% boost, max 0.95
                    suggestion['confidence'] = boosted_confidence
                    suggestion['reason'] = f"{suggestion.get('reason', '')} [PRIORITY BOOST]"
                    logger.debug(f"[PRIORITY] Boosted ACTION{action_id} confidence: {current_confidence:.2f} -> {boosted_confidence:.2f}")

        return suggestions

    def _filter_valid_actions(self, suggestions: List[Dict[str, Any]], available_actions: List[int]) -> List[Dict[str, Any]]:
        """Filter suggestions to only include valid and prioritized actions."""
        # First validate and prioritize available actions (removes action 0)
        validated_actions = self._validate_and_prioritize_actions(available_actions)

        valid_suggestions = []
        for suggestion in suggestions:
            action_id = self._get_action_id(suggestion.get('action', 'ACTION1'))
            if action_id in validated_actions:
                valid_suggestions.append(suggestion)

        # Apply priority boost to valid suggestions
        prioritized_suggestions = self._apply_action_priority_boost(valid_suggestions)

        return prioritized_suggestions
    
    def _get_action_id(self, action_name: str) -> int:
        """Convert action name to action ID with validation."""
        try:
            if not action_name or not isinstance(action_name, str):
                logger.warning(f"Invalid action string: {action_name}, using fallback ACTION1")
                return 1

            action_map = {
                'ACTION1': 1, 'ACTION2': 2, 'ACTION3': 3, 'ACTION4': 4,
                'ACTION5': 5, 'ACTION6': 6, 'ACTION7': 7
            }

            # Direct lookup first
            if action_name in action_map:
                return action_map[action_name]

            # Try parsing ACTION format
            if action_name.startswith('ACTION'):
                try:
                    action_id = int(action_name[6:])  # Extract number after 'ACTION'
                    if 1 <= action_id <= 7:
                        return action_id
                except (ValueError, IndexError):
                    pass

            logger.warning(f"Invalid action format: {action_name}, using fallback ACTION1")
            return 1
        except Exception as e:
            logger.warning(f"Failed to parse action: {action_name}, error: {e}, using fallback ACTION1")
            return 1
    
    def _get_fallback_action(self, available_actions: List[int]) -> Dict[str, Any]:
        """Get a fallback action when no intelligent suggestions are available."""

        # Validate and prioritize actions first (removes action 0)
        validated_actions = self._validate_and_prioritize_actions(available_actions)

        if not validated_actions:
            logger.error("[PRIORITY] No valid actions available for fallback!")
            # Emergency fallback
            return {
                'id': 1,
                'reason': 'Emergency fallback to ACTION1 - no valid actions available',
                'confidence': 0.05,
                'source': 'emergency_fallback'
            }

        # Prefer priority actions (1-4) first, then simple actions, then complex ones
        priority_actions = [1, 2, 3, 4]  # Highest priority
        simple_actions = [5, 7]  # Simple actions (excluding ACTION6)
        complex_actions = [6]  # ACTION6

        # Try priority actions first (1-4)
        for action_id in priority_actions:
            if action_id in validated_actions:
                logger.info(f"[PRIORITY] Using priority fallback action: {action_id}")
                return {
                    'id': action_id,
                    'reason': f'Priority fallback to ACTION{action_id}',
                    'confidence': 0.2,  # Higher confidence for priority actions
                    'source': 'priority_fallback'
                }

        # Try simple actions
        for action_id in simple_actions:
            if action_id in validated_actions:
                return {
                    'id': action_id,
                    'reason': f'Fallback to simple action {action_id}',
                    'confidence': 0.1,
                    'source': 'simple_fallback'
                }

        # Try complex actions
        for action_id in complex_actions:
            if action_id in validated_actions:
                if action_id == 6:
                    return {
                        'id': action_id,
                        'x': random.randint(0, 63),
                        'y': random.randint(0, 63),
                        'reason': f'Fallback to complex action {action_id} with random coordinates',
                        'confidence': 0.1,
                        'source': 'fallback'
                    }
                else:
                    return {
                        'id': action_id,
                        'reason': f'Fallback to complex action {action_id}',
                        'confidence': 0.1,
                        'source': 'fallback'
                    }
        
        # Last resort: random action
        action_id = random.choice(available_actions)
        action = {
            'id': action_id,
            'reason': f'Random fallback action {action_id}',
            'confidence': 0.05,
            'source': 'fallback'
        }
        
        if action_id == 6:
            action['x'] = random.randint(0, 63)
            action['y'] = random.randint(0, 63)
        
        return action

    def _apply_intervention_guidance(self, guidance: Dict[str, Any], available_actions: List[int], game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply intervention guidance to generate action suggestions."""
        suggestions = []

        try:
            intervention_type = guidance.get('type', '')

            if intervention_type == 'randomization':
                # Apply randomization to action selection
                randomization_factor = guidance.get('randomization_factor', 0.3)
                if randomization_factor > 0.5:  # High randomization
                    # Suggest less commonly used actions
                    less_used_actions = [a for a in available_actions if a not in [1, 2, 3, 4]]
                    for action in less_used_actions[:2]:
                        suggestions.append({
                            'action': f'ACTION{action}',
                            'confidence': 0.6,
                            'reason': f'Randomization intervention: trying less common action',
                            'source': 'losing_streak_randomization'
                        })

            elif intervention_type == 'exploration_boost':
                # Boost exploration by suggesting diverse actions
                exploration_multiplier = guidance.get('exploration_multiplier', 1.5)
                if 6 in available_actions and exploration_multiplier > 1.3:
                    # Suggest diverse coordinates
                    diverse_coords = [(100, 100), (400, 300), (200, 200), (500, 400)]
                    for x, y in diverse_coords[:2]:
                        suggestions.append({
                            'action': 'ACTION6',
                            'x': x,
                            'y': y,
                            'confidence': 0.7,
                            'reason': f'Exploration boost: diverse coordinate testing',
                            'source': 'losing_streak_exploration'
                        })

            elif intervention_type == 'coordinate_diversification':
                # Force coordinate diversity
                if 6 in available_actions:
                    suggested_regions = guidance.get('suggested_regions', [])
                    minimum_distance = guidance.get('minimum_distance', 50)

                    for region in suggested_regions[:3]:
                        x, y = region
                        # Add some randomness within the region
                        offset = minimum_distance // 2
                        x_rand = x + random.randint(-offset, offset)
                        y_rand = y + random.randint(-offset, offset)

                        # Clamp to screen bounds
                        x_rand = max(0, min(639, x_rand))
                        y_rand = max(0, min(479, y_rand))

                        suggestions.append({
                            'action': 'ACTION6',
                            'x': x_rand,
                            'y': y_rand,
                            'confidence': 0.8,
                            'reason': f'Coordinate diversification: testing region ({x}, {y})',
                            'source': 'losing_streak_diversification'
                        })

            elif intervention_type == 'action_sequence_shuffle':
                # Shuffle action sequence recommendations
                shuffle_probability = guidance.get('shuffle_probability', 0.5)
                shuffled_sequences = guidance.get('shuffled_sequences', [])

                for sequence in shuffled_sequences[:2]:
                    if len(sequence) > 0 and sequence[0] in available_actions:
                        suggestions.append({
                            'action': f'ACTION{sequence[0]}',
                            'confidence': 0.6,
                            'reason': f'Action sequence shuffle: trying shuffled pattern',
                            'source': 'losing_streak_shuffle'
                        })

            elif intervention_type == 'strategy_override':
                # Override normal strategy with experimental approach
                if guidance.get('force_experimental_approach', False):
                    # Suggest actions in reverse order of typical priority
                    reverse_priority_actions = [a for a in available_actions[::-1]]
                    for action in reverse_priority_actions[:2]:
                        suggestions.append({
                            'action': f'ACTION{action}',
                            'confidence': 0.7,
                            'reason': f'Strategy override: experimental action choice',
                            'source': 'losing_streak_override'
                        })

            elif intervention_type == 'timing_adjustment':
                # Add timing considerations (this would typically affect action execution timing)
                minimum_delay = guidance.get('minimum_action_delay', 1.0)
                if minimum_delay > 1.5:
                    # Suggest deliberate actions with timing consideration
                    for action in available_actions[:2]:
                        suggestions.append({
                            'action': f'ACTION{action}',
                            'confidence': 0.6,
                            'reason': f'Timing adjustment: deliberate action with {minimum_delay:.1f}s delay',
                            'source': 'losing_streak_timing'
                        })

            elif intervention_type == 'complete_reset':
                # Complete reset - suggest fresh exploration
                if guidance.get('start_fresh_exploration', False):
                    # Start with basic actions if available
                    basic_actions = [a for a in available_actions if a in [1, 2, 3, 4]]
                    if basic_actions:
                        suggestions.append({
                            'action': f'ACTION{basic_actions[0]}',
                            'confidence': 0.8,
                            'reason': f'Complete reset: fresh start with basic action',
                            'source': 'losing_streak_reset'
                        })
                    elif 6 in available_actions:
                        # Try center coordinate for fresh start
                        suggestions.append({
                            'action': 'ACTION6',
                            'x': 320,
                            'y': 240,
                            'confidence': 0.8,
                            'reason': f'Complete reset: fresh start at center coordinate',
                            'source': 'losing_streak_reset'
                        })

        except Exception as e:
            logger.error(f"Error applying intervention guidance: {e}")

        return suggestions

    def get_action_statistics(self) -> Dict[str, Any]:
        """Get statistics about action selection."""
        if not self.action_history:
            return {'total_actions': 0, 'action_distribution': {}}
        
        action_counts = {}
        for record in self.action_history:
            action_id = record['action']['id']
            action_counts[action_id] = action_counts.get(action_id, 0) + 1
        
        total_actions = len(self.action_history)
        action_distribution = {
            action_id: count / total_actions 
            for action_id, count in action_counts.items()
        }
        
        return {
            'total_actions': total_actions,
            'action_distribution': action_distribution,
            'recent_actions': self.action_history[-10:] if self.action_history else [],
            'coordinate_success_rates': self.coordinate_success_rates,
            'action_effectiveness': self.action_effectiveness
        }

def create_action_selector(api_manager: APIManager, bayesian_inference_system=None, graph_traversal_system=None) -> ActionSelector:
    """Create an action selector instance with optional advanced systems."""
    return ActionSelector(api_manager, bayesian_inference_system, graph_traversal_system)
