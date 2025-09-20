"""
Master ARC Trainer

This module provides the main training interface for ARC-AGI-3.

This file has been modularized. The main functionality is now in src/training/.
"""

# Import from the new modular structure
from src.training import (
    MasterARCTrainer,
    MasterTrainingConfig,
    ActionLimits
)

# Re-export for backward compatibility
__all__ = [
    'MasterARCTrainer',
    'MasterTrainingConfig',
    'ActionLimits'
]

# For direct execution, create a simple wrapper
if __name__ == "__main__":
    import logging
    import asyncio
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    async def main():
        """Main entry point for master ARC trainer."""
        try:
            # Create master trainer
            trainer = MasterARCTrainer()
            logger.info("Master ARC trainer created successfully")
            return trainer
        except Exception as e:
            logger.error(f"Error creating master ARC trainer: {e}")
            raise
    
    # Run the main function
    asyncio.run(main())
