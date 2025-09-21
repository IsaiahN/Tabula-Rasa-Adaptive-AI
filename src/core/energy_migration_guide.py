"""
Energy Management Migration Guide

This module provides utilities to help migrate existing training loops
to use the unified energy management system.
"""

import logging
from typing import Dict, Any, Optional, Callable
from .unified_energy_system import UnifiedEnergySystem, EnergyConfig, EnergySystemIntegration

logger = logging.getLogger(__name__)


class EnergyMigrationHelper:
    """
    Helper class to migrate existing energy systems to the unified system.
    """
    
    def __init__(self, energy_system: Optional[UnifiedEnergySystem] = None):
        self.energy_system = energy_system or UnifiedEnergySystem()
        self.integration = EnergySystemIntegration(self.energy_system)
        self.migration_complete = False
    
    def migrate_training_loop(self, training_loop) -> bool:
        """
        Migrate a training loop to use the unified energy system.
        
        Args:
            training_loop: The training loop to migrate
            
        Returns:
            True if migration was successful
        """
        try:
            # Step 1: Replace existing energy system
            if hasattr(training_loop, 'energy_system'):
                logger.info("Replacing existing energy system with unified system")
                training_loop.old_energy_system = training_loop.energy_system
            
            training_loop.energy_system = self.energy_system
            training_loop.current_energy = self.energy_system.current_energy
            
            # Step 2: Replace energy consumption methods
            if hasattr(training_loop, 'consume_energy'):
                training_loop.old_consume_energy = training_loop.consume_energy
                training_loop.consume_energy = self._create_energy_consumer()
            
            # Step 3: Replace energy checking methods
            if hasattr(training_loop, 'should_sleep'):
                training_loop.old_should_sleep = training_loop.should_sleep
                training_loop.should_sleep = self._create_sleep_checker()
            
            # Step 4: Add energy restoration methods
            if not hasattr(training_loop, 'trigger_sleep'):
                training_loop.trigger_sleep = self._create_sleep_trigger()
            
            if not hasattr(training_loop, 'handle_death'):
                training_loop.handle_death = self._create_death_handler()
            
            # Step 5: Add energy status methods
            if not hasattr(training_loop, 'get_energy_status'):
                training_loop.get_energy_status = self._create_status_getter()
            
            # Step 6: Integrate with existing sleep system if available
            if hasattr(training_loop, 'sleep_system') and training_loop.sleep_system:
                self._integrate_with_sleep_system(training_loop)
            
            self.migration_complete = True
            logger.info("Training loop successfully migrated to unified energy system")
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate training loop: {e}")
            return False
    
    def _create_energy_consumer(self) -> Callable:
        """Create a unified energy consumer function."""
        def consume_energy(action_id: int, success: bool = False, 
                          learning_progress: float = 0.0, **kwargs) -> Dict[str, Any]:
            """Consume energy for an action using the unified system."""
            return self.energy_system.consume_energy_for_action(
                action_id, success, learning_progress
            )
        return consume_energy
    
    def _create_sleep_checker(self) -> Callable:
        """Create a unified sleep checker function."""
        def should_sleep() -> tuple:
            """Check if the agent should sleep using the unified system."""
            return self.energy_system.should_sleep()
        return should_sleep
    
    def _create_sleep_trigger(self) -> Callable:
        """Create a unified sleep trigger function."""
        def trigger_sleep() -> Dict[str, Any]:
            """Trigger sleep using the unified system."""
            return self.energy_system.trigger_sleep()
        return trigger_sleep
    
    def _create_death_handler(self) -> Callable:
        """Create a unified death handler function."""
        def handle_death() -> Dict[str, Any]:
            """Handle death using the unified system."""
            return self.energy_system.handle_death()
        return handle_death
    
    def _create_status_getter(self) -> Callable:
        """Create a unified status getter function."""
        def get_energy_status() -> Dict[str, Any]:
            """Get energy status using the unified system."""
            return self.energy_system.get_status()
        return get_energy_status
    
    def _integrate_with_sleep_system(self, training_loop):
        """Integrate with existing sleep system."""
        try:
            sleep_system = training_loop.sleep_system
            
            # Update sleep trigger threshold
            if hasattr(sleep_system, 'sleep_trigger_energy'):
                sleep_system.sleep_trigger_energy = self.energy_system.config.sleep_trigger_threshold
            
            # Add energy system reference
            sleep_system.energy_system = self.energy_system
            
            logger.info("Integrated unified energy system with existing sleep system")
            
        except Exception as e:
            logger.warning(f"Failed to integrate with sleep system: {e}")
    
    def create_energy_aware_action_wrapper(self, original_action_method: Callable) -> Callable:
        """
        Create a wrapper for action methods that includes energy consumption.
        
        Args:
            original_action_method: The original action method to wrap
            
        Returns:
            Wrapped action method with energy consumption
        """
        def energy_aware_action(*args, **kwargs):
            # Extract action information from arguments
            action_id = kwargs.get('action_id', 1)
            success = kwargs.get('success', False)
            learning_progress = kwargs.get('learning_progress', 0.0)
            
            # Consume energy before action
            consumption_record = self.energy_system.consume_energy_for_action(
                action_id, success, learning_progress
            )
            
            # Execute original action
            result = original_action_method(*args, **kwargs)
            
            # Update result with energy information
            if isinstance(result, dict):
                result['energy_consumption'] = consumption_record
                result['current_energy'] = self.energy_system.current_energy
                result['energy_state'] = self.energy_system.energy_state.value
            
            return result
        
        return energy_aware_action
    
    def get_migration_summary(self) -> Dict[str, Any]:
        """Get a summary of the migration status."""
        return {
            'migration_complete': self.migration_complete,
            'energy_system_type': type(self.energy_system).__name__,
            'current_energy': self.energy_system.current_energy,
            'energy_state': self.energy_system.energy_state.value,
            'total_actions_taken': self.energy_system.total_actions_taken,
            'total_deaths': self.energy_system.total_deaths,
            'integration_active': self.integration.integration_active
        }


def migrate_existing_training_loop(training_loop, config: Optional[EnergyConfig] = None) -> bool:
    """
    Convenience function to migrate an existing training loop.
    
    Args:
        training_loop: The training loop to migrate
        config: Optional energy configuration
        
    Returns:
        True if migration was successful
    """
    helper = EnergyMigrationHelper(UnifiedEnergySystem(config))
    return helper.migrate_training_loop(training_loop)


def create_energy_aware_training_loop(base_training_loop, config: Optional[EnergyConfig] = None):
    """
    Create an energy-aware version of a training loop.
    
    Args:
        base_training_loop: The base training loop class
        config: Optional energy configuration
        
    Returns:
        Energy-aware training loop class
    """
    class EnergyAwareTrainingLoop(base_training_loop):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
            # Initialize unified energy system
            self.energy_system = UnifiedEnergySystem(config)
            self.current_energy = self.energy_system.current_energy
            
            # Create energy integration
            self.energy_integration = EnergySystemIntegration(self.energy_system)
            self.energy_integration.integrate_with_training_loop(self)
        
        def consume_energy(self, action_id: int, success: bool = False, 
                          learning_progress: float = 0.0, **kwargs) -> Dict[str, Any]:
            """Consume energy for an action."""
            return self.energy_system.consume_energy_for_action(
                action_id, success, learning_progress
            )
        
        def should_sleep(self) -> tuple:
            """Check if the agent should sleep."""
            return self.energy_system.should_sleep()
        
        def trigger_sleep(self) -> Dict[str, Any]:
            """Trigger sleep cycle."""
            return self.energy_system.trigger_sleep()
        
        def handle_death(self) -> Dict[str, Any]:
            """Handle agent death."""
            return self.energy_system.handle_death()
        
        def get_energy_status(self) -> Dict[str, Any]:
            """Get energy system status."""
            return self.energy_system.get_status()
        
        def is_dead(self) -> bool:
            """Check if agent is dead."""
            return self.energy_system.is_dead()
    
    return EnergyAwareTrainingLoop


# Example usage and migration patterns
def example_migration_pattern():
    """
    Example of how to migrate an existing training loop.
    """
    # Example 1: Direct migration
    from .unified_energy_system import UnifiedEnergySystem, EnergyConfig
    
    # Create energy system with custom config
    config = EnergyConfig(
        max_energy=100.0,
        sleep_trigger_threshold=30.0,
        action_costs={1: 0.5, 6: 2.0}
    )
    energy_system = UnifiedEnergySystem(config)
    
    # Migrate existing training loop
    helper = EnergyMigrationHelper(energy_system)
    # helper.migrate_training_loop(your_training_loop)
    
    # Example 2: Create energy-aware training loop
    # EnergyAwareLoop = create_energy_aware_training_loop(YourBaseTrainingLoop)
    # training_loop = EnergyAwareLoop()
    
    # Example 3: Manual integration
    # training_loop.energy_system = energy_system
    # training_loop.current_energy = energy_system.current_energy
    # training_loop.consume_energy = energy_system.consume_energy_for_action
    # training_loop.should_sleep = energy_system.should_sleep
    # training_loop.trigger_sleep = energy_system.trigger_sleep
    # training_loop.handle_death = energy_system.handle_death


if __name__ == "__main__":
    # Run example migration
    example_migration_pattern()
    print("Energy migration guide loaded successfully")
