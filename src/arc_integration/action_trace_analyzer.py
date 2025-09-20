"""
Action Trace Analyzer for ARC-AGI-3

This module analyzes action traces to identify successful patterns and sequences.

This file has been modularized. The main functionality is now in src/analysis/.
"""

# Import from the new modular structure
from src.analysis import (
    PatternAnalyzer,
    SequenceDetector,
    PerformanceTracker,
    InsightGenerator
)

# Re-export for backward compatibility
__all__ = [
    'PatternAnalyzer',
    'SequenceDetector',
    'PerformanceTracker',
    'InsightGenerator'
]

# For direct execution, create a simple wrapper
if __name__ == "__main__":
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    def main():
        """Main entry point for action trace analyzer."""
        try:
            # Create pattern analyzer
            analyzer = PatternAnalyzer()
            logger.info("Action trace analyzer created successfully")
            return analyzer
        except Exception as e:
            logger.error(f"Error creating action trace analyzer: {e}")
            raise
    
    # Run the main function
    main()
