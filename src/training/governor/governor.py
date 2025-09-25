"""
Training Governor

Manages meta-cognitive processes, resource allocation, and system state management
for the training system.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

class TrainingGovernor:
    """Manages meta-cognitive processes and resource allocation."""
    
    def __init__(self, persistence_dir: Optional[str] = None):
        self.persistence_dir = persistence_dir
        self.enhanced_governor = None
        self.architect = None
        self.governor_decisions = []
        self.resource_allocation = {}
        self.system_state = {
            'training_mode': 'active',
            'resource_usage': 0.0,
            'performance_level': 'normal',
            'learning_rate': 1.0
        }
        self._initialize_governor()
    
    def _initialize_governor(self) -> None:
        """Initialize the enhanced space-time governor."""
        try:
            from src.core.enhanced_space_time_governor import create_enhanced_space_time_governor
            if self.persistence_dir:
                self.enhanced_governor = create_enhanced_space_time_governor(
                    persistence_dir=self.persistence_dir
                )
                logger.info("Enhanced space-time governor initialized")
            else:
                logger.warning("No persistence directory provided for governor")
        except ImportError as e:
            logger.warning(f"Enhanced space-time governor not available: {e}")
            self.enhanced_governor = None
        except Exception as e:
            logger.error(f"Error initializing governor: {e}")
            self.enhanced_governor = None
    
    def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make a meta-cognitive decision based on context."""
        try:
            decision = {
                'timestamp': datetime.now(),
                'context': context,
                'decision_type': 'resource_allocation',
                'action': 'continue',
                'reasoning': 'Default decision',
                'confidence': 0.5
            }
            
            if self.enhanced_governor:
                # Use enhanced governor for decision making
                decision = self._make_enhanced_decision(context)
            else:
                # Fallback to simple decision making
                decision = self._make_simple_decision(context)
            
            # Record the decision
            self.governor_decisions.append(decision)
            # Persist decision to DB (best-effort, non-blocking)
            try:
                from src.database.persistence_helpers import persist_governor_decision
                import asyncio
                asyncio.create_task(persist_governor_decision(
                    session_id=str(decision.get('context', {}).get('session_id', 'unknown')),
                    decision_type=decision.get('decision_type', 'unknown'),
                    context=decision.get('context', {}),
                    confidence=float(decision.get('confidence', 0.0)),
                    outcome={'action': decision.get('action')}
                ))
            except Exception:
                logger.debug('Failed to schedule DB persist for governor decision')
            
            # Keep only recent decisions
            if len(self.governor_decisions) > 1000:
                self.governor_decisions = self.governor_decisions[-500:]
            
            return decision
            
        except Exception as e:
            logger.error(f"Error making governor decision: {e}")
            return {
                'timestamp': datetime.now(),
                'context': context,
                'decision_type': 'error',
                'action': 'continue',
                'reasoning': f'Error in decision making: {e}',
                'confidence': 0.0
            }
    
    def _make_enhanced_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make decision using enhanced governor."""
        try:
            # This would integrate with the actual enhanced governor
            # For now, return a structured decision
            return {
                'timestamp': datetime.now(),
                'context': context,
                'decision_type': 'enhanced_governor',
                'action': 'continue',
                'reasoning': 'Enhanced governor decision',
                'confidence': 0.8
            }
        except Exception as e:
            logger.error(f"Error in enhanced decision making: {e}")
            return self._make_simple_decision(context)
    
    def _make_simple_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make simple fallback decision."""
        # Simple decision logic based on context
        memory_usage = context.get('memory_usage', 0)
        performance_level = context.get('performance_level', 'normal')
        
        if memory_usage > 1000:  # High memory usage
            return {
                'timestamp': datetime.now(),
                'context': context,
                'decision_type': 'simple',
                'action': 'reduce_memory',
                'reasoning': 'High memory usage detected',
                'confidence': 0.7
            }
        elif performance_level == 'poor':
            return {
                'timestamp': datetime.now(),
                'context': context,
                'decision_type': 'simple',
                'action': 'optimize',
                'reasoning': 'Poor performance detected',
                'confidence': 0.6
            }
        else:
            return {
                'timestamp': datetime.now(),
                'context': context,
                'decision_type': 'simple',
                'action': 'continue',
                'reasoning': 'Normal operation',
                'confidence': 0.5
            }
    
    def allocate_resources(self, resource_type: str, amount: float) -> bool:
        """Allocate resources for a specific purpose."""
        try:
            current_allocation = self.resource_allocation.get(resource_type, 0.0)
            max_allocation = self._get_max_allocation(resource_type)
            
            if current_allocation + amount <= max_allocation:
                self.resource_allocation[resource_type] = current_allocation + amount
                logger.info(f"Allocated {amount} {resource_type} resources")
                return True
            else:
                logger.warning(f"Cannot allocate {amount} {resource_type} resources (max: {max_allocation})")
                return False
                
        except Exception as e:
            logger.error(f"Error allocating resources: {e}")
            return False
    
    def deallocate_resources(self, resource_type: str, amount: float) -> bool:
        """Deallocate resources."""
        try:
            current_allocation = self.resource_allocation.get(resource_type, 0.0)
            new_allocation = max(0.0, current_allocation - amount)
            self.resource_allocation[resource_type] = new_allocation
            logger.info(f"Deallocated {amount} {resource_type} resources")
            return True
        except Exception as e:
            logger.error(f"Error deallocating resources: {e}")
            return False
    
    def _get_max_allocation(self, resource_type: str) -> float:
        """Get maximum allocation for a resource type."""
        max_allocations = {
            'memory': 2000.0,  # 2GB
            'cpu': 100.0,      # 100% CPU
            'network': 100.0,  # 100% network
            'storage': 1000.0  # 1GB storage
        }
        return max_allocations.get(resource_type, 100.0)
    
    def update_system_state(self, state_updates: Dict[str, Any]) -> None:
        """Update system state."""
        try:
            self.system_state.update(state_updates)
            logger.debug(f"System state updated: {state_updates}")
        except Exception as e:
            logger.error(f"Error updating system state: {e}")
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get current system state."""
        return self.system_state.copy()
    
    def get_governor_status(self) -> Dict[str, Any]:
        """Get comprehensive governor status."""
        return {
            'enhanced_governor_available': self.enhanced_governor is not None,
            'total_decisions': len(self.governor_decisions),
            'resource_allocation': self.resource_allocation.copy(),
            'system_state': self.system_state.copy(),
            'recent_decisions': self.governor_decisions[-10:] if self.governor_decisions else []
        }
    
    def get_decision_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get decision history."""
        if limit is None:
            return self.governor_decisions.copy()
        return self.governor_decisions[-limit:]
    
    def reset_governor(self) -> None:
        """Reset governor state."""
        self.governor_decisions.clear()
        self.resource_allocation.clear()
        self.system_state = {
            'training_mode': 'active',
            'resource_usage': 0.0,
            'performance_level': 'normal',
            'learning_rate': 1.0
        }
        logger.info("Governor reset")
    
    def is_healthy(self) -> bool:
        """Check if governor is healthy."""
        try:
            # Check if we have too many recent errors
            recent_decisions = self.governor_decisions[-10:]
            error_decisions = [d for d in recent_decisions if d.get('decision_type') == 'error']
            
            if len(error_decisions) > 5:
                logger.warning("High number of error decisions")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error checking governor health: {e}")
            return False
    
    async def get_advanced_action_systems_status(self) -> Dict[str, Any]:
        """Get status of all advanced action systems."""
        try:
            status = {
                'visual_interactive_system': {
                    'available': True,
                    'description': 'Visual-Interactive Action6 Targeting System',
                    'features': ['touchscreen_paradigm', 'opencv_detection', 'button_prioritization']
                },
                'stagnation_system': {
                    'available': True,
                    'description': 'Advanced Stagnation Detection System',
                    'features': ['score_regression_detection', 'action_repetition_detection', 'recovery_strategies']
                },
                'strategy_discovery_system': {
                    'available': True,
                    'description': 'Strategy Discovery & Replication System',
                    'features': ['winning_sequence_discovery', 'strategy_refinement', 'replication_testing']
                },
                'frame_analysis_system': {
                    'available': True,
                    'description': 'Enhanced Frame Change Analysis System',
                    'features': ['movement_detection', 'change_classification', 'pattern_analysis']
                },
                'exploration_system': {
                    'available': True,
                    'description': 'Systematic Exploration Phases System',
                    'features': ['corner_center_edge_random', 'coordinate_tracking', 'success_rate_analysis']
                },
                'emergency_override_system': {
                    'available': True,
                    'description': 'Emergency Override Systems',
                    'features': ['action_loop_break', 'coordinate_stuck_break', 'stagnation_break']
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting advanced action systems status: {e}")
            return {'error': str(e)}
    
    async def control_advanced_action_system(self, system_name: str, action: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Control advanced action systems through the Governor."""
        try:
            if parameters is None:
                parameters = {}
            
            result = {
                'system': system_name,
                'action': action,
                'parameters': parameters,
                'success': False,
                'message': '',
                'timestamp': datetime.now()
            }
            
            if system_name == 'visual_interactive_system':
                result = await self._control_visual_interactive_system(action, parameters)
            elif system_name == 'stagnation_system':
                result = await self._control_stagnation_system(action, parameters)
            elif system_name == 'strategy_discovery_system':
                result = await self._control_strategy_discovery_system(action, parameters)
            elif system_name == 'frame_analysis_system':
                result = await self._control_frame_analysis_system(action, parameters)
            elif system_name == 'exploration_system':
                result = await self._control_exploration_system(action, parameters)
            elif system_name == 'emergency_override_system':
                result = await self._control_emergency_override_system(action, parameters)
            else:
                result['message'] = f'Unknown system: {system_name}'
            
            return result
            
        except Exception as e:
            logger.error(f"Error controlling advanced action system: {e}")
            return {
                'system': system_name,
                'action': action,
                'success': False,
                'message': f'Error: {str(e)}',
                'timestamp': datetime.now()
            }
    
    async def _control_visual_interactive_system(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Control the visual interactive system."""
        try:
            if action == 'get_targets':
                game_id = parameters.get('game_id', 'unknown')
                from src.database.system_integration import get_system_integration
                integration = get_system_integration()
                targets = await integration.get_visual_targets_for_game(game_id)
                return {
                    'system': 'visual_interactive_system',
                    'action': action,
                    'success': True,
                    'data': targets,
                    'message': f'Retrieved {len(targets)} visual targets'
                }
            elif action == 'reset_targets':
                return {
                    'system': 'visual_interactive_system',
                    'action': action,
                    'success': True,
                    'message': 'Visual targets reset'
                }
            else:
                return {
                    'system': 'visual_interactive_system',
                    'action': action,
                    'success': False,
                    'message': f'Unknown action: {action}'
                }
        except Exception as e:
            return {
                'system': 'visual_interactive_system',
                'action': action,
                'success': False,
                'message': f'Error: {str(e)}'
            }
    
    async def _control_stagnation_system(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Control the stagnation detection system."""
        try:
            if action == 'get_events':
                game_id = parameters.get('game_id', 'unknown')
                from src.database.system_integration import get_system_integration
                integration = get_system_integration()
                events = await integration.get_stagnation_events_for_game(game_id)
                return {
                    'system': 'stagnation_system',
                    'action': action,
                    'success': True,
                    'data': events,
                    'message': f'Retrieved {len(events)} stagnation events'
                }
            else:
                return {
                    'system': 'stagnation_system',
                    'action': action,
                    'success': False,
                    'message': f'Unknown action: {action}'
                }
        except Exception as e:
            return {
                'system': 'stagnation_system',
                'action': action,
                'success': False,
                'message': f'Error: {str(e)}'
            }
    
    async def _control_strategy_discovery_system(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Control the strategy discovery system."""
        try:
            if action == 'get_strategies':
                game_type = parameters.get('game_type', 'unknown')
                from src.database.system_integration import get_system_integration
                integration = get_system_integration()
                strategies = await integration.get_winning_strategies_for_game_type(game_type)
                return {
                    'system': 'strategy_discovery_system',
                    'action': action,
                    'success': True,
                    'data': strategies,
                    'message': f'Retrieved {len(strategies)} strategies for game type {game_type}'
                }
            else:
                return {
                    'system': 'strategy_discovery_system',
                    'action': action,
                    'success': False,
                    'message': f'Unknown action: {action}'
                }
        except Exception as e:
            return {
                'system': 'strategy_discovery_system',
                'action': action,
                'success': False,
                'message': f'Error: {str(e)}'
            }
    
    async def _control_frame_analysis_system(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Control the frame analysis system."""
        try:
            if action == 'get_analysis':
                game_id = parameters.get('game_id', 'unknown')
                from src.database.system_integration import get_system_integration
                integration = get_system_integration()
                analysis = await integration.get_frame_change_analysis_for_game(game_id)
                return {
                    'system': 'frame_analysis_system',
                    'action': action,
                    'success': True,
                    'data': analysis,
                    'message': f'Retrieved {len(analysis)} frame change analyses'
                }
            else:
                return {
                    'system': 'frame_analysis_system',
                    'action': action,
                    'success': False,
                    'message': f'Unknown action: {action}'
                }
        except Exception as e:
            return {
                'system': 'frame_analysis_system',
                'action': action,
                'success': False,
                'message': f'Error: {str(e)}'
            }
    
    async def _control_exploration_system(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Control the exploration system."""
        try:
            if action == 'get_phases':
                game_id = parameters.get('game_id', 'unknown')
                from src.database.system_integration import get_system_integration
                integration = get_system_integration()
                phases = await integration.get_exploration_phases_for_game(game_id)
                return {
                    'system': 'exploration_system',
                    'action': action,
                    'success': True,
                    'data': phases,
                    'message': f'Retrieved {len(phases)} exploration phases'
                }
            else:
                return {
                    'system': 'exploration_system',
                    'action': action,
                    'success': False,
                    'message': f'Unknown action: {action}'
                }
        except Exception as e:
            return {
                'system': 'exploration_system',
                'action': action,
                'success': False,
                'message': f'Error: {str(e)}'
            }
    
    async def _control_emergency_override_system(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Control the emergency override system."""
        try:
            if action == 'get_overrides':
                game_id = parameters.get('game_id', 'unknown')
                from src.database.system_integration import get_system_integration
                integration = get_system_integration()
                overrides = await integration.get_emergency_overrides_for_game(game_id)
                return {
                    'system': 'emergency_override_system',
                    'action': action,
                    'success': True,
                    'data': overrides,
                    'message': f'Retrieved {len(overrides)} emergency overrides'
                }
            else:
                return {
                    'system': 'emergency_override_system',
                    'action': action,
                    'success': False,
                    'message': f'Unknown action: {action}'
                }
        except Exception as e:
            return {
                'system': 'emergency_override_system',
                'action': action,
                'success': False,
                'message': f'Error: {str(e)}'
            }