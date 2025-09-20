"""
Continuous Learning Loop for ARC-AGI-3 Training

This module implements a continuous learning system that runs the Adaptive Learning Agent
against ARC-AGI-3 tasks, collecting insights and improving performance over time.

This file has been modularized. The main functionality is now in src/training/.
"""

# Import from the new modular structure
from src.training import ContinuousLearningLoop

# Re-export for backward compatibility
__all__ = ['ContinuousLearningLoop']

# For direct execution, create a simple wrapper
if __name__ == "__main__":
    import asyncio
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    async def main():
        """Main entry point for continuous learning loop."""
        try:
            # Create and run the continuous learning loop
            loop = ContinuousLearningLoop()
            await loop.run_continuous_learning()
        except KeyboardInterrupt:
            logger.info("Continuous learning loop interrupted by user")
        except Exception as e:
            logger.error(f"Error in continuous learning loop: {e}")
            raise
    
    # Run the main function
    asyncio.run(main())
