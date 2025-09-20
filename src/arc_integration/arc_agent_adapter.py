"""
ARC Agent Adapter for ARC-AGI-3

This module provides integration between the ARC API and the Adaptive Learning Agent.

This file has been modularized. The main functionality is now in src/adapters/.
"""

# Import from the new modular structure
from src.adapters import (
    ARCVisualProcessor,
    ARCActionMapper,
    AdaptiveLearningARCAgent
)

# Re-export for backward compatibility
__all__ = [
    'ARCVisualProcessor',
    'ARCActionMapper',
    'AdaptiveLearningARCAgent'
]

# For direct execution, create a simple wrapper
if __name__ == "__main__":
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    def main():
        """Main entry point for ARC agent adapter."""
        try:
            # Create adaptive learning agent
            agent = AdaptiveLearningARCAgent()
            logger.info("ARC agent adapter created successfully")
            return agent
        except Exception as e:
            logger.error(f"Error creating ARC agent adapter: {e}")
            raise
    
    # Run the main function
    main()
