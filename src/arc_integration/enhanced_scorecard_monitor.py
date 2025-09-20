"""
Enhanced Scorecard Monitor for ARC-AGI-3

This module provides monitoring and analytics capabilities.

This file has been modularized. The main functionality is now in src/monitoring/.
"""

# Import from the new modular structure
from src.monitoring import (
    PerformanceTracker,
    TrendAnalyzer,
    ReportGenerator,
    DataCollector
)

# Re-export for backward compatibility
__all__ = [
    'PerformanceTracker',
    'TrendAnalyzer',
    'ReportGenerator',
    'DataCollector'
]

# For direct execution, create a simple wrapper
if __name__ == "__main__":
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    def main():
        """Main entry point for enhanced scorecard monitor."""
        try:
            # Create performance tracker
            tracker = PerformanceTracker()
            logger.info("Enhanced scorecard monitor created successfully")
            return tracker
        except Exception as e:
            logger.error(f"Error creating enhanced scorecard monitor: {e}")
            raise
    
    # Run the main function
    main()
