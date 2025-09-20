"""
ARC Meta Learning System

This module implements meta-learning capabilities for ARC-AGI-3.

This file has been modularized. The main functionality is now in src/learning/.
"""

# Import from the new modular structure
from src.learning import (
    ARCPatternRecognizer, ARCPattern,
    ARCInsightExtractor, ARCInsight,
    KnowledgeTransfer,
    ARCMetaLearningSystem
)

# Re-export for backward compatibility
__all__ = [
    'ARCPatternRecognizer', 'ARCPattern',
    'ARCInsightExtractor', 'ARCInsight',
    'KnowledgeTransfer',
    'ARCMetaLearningSystem'
]

# For direct execution, create a simple wrapper
if __name__ == "__main__":
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    def main():
        """Main entry point for ARC meta learning system."""
        try:
            # Create meta learning system
            system = ARCMetaLearningSystem()
            logger.info("ARC meta learning system created successfully")
            return system
        except Exception as e:
            logger.error(f"Error creating ARC meta learning system: {e}")
            raise
    
    # Run the main function
    main()
