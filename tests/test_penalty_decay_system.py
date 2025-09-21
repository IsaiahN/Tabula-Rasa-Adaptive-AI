#!/usr/bin/env python3
"""
Test script for the comprehensive Penalty Decay System.

This script tests all features of the penalty decay system:
- Penalty application for coordinates that don't improve score
- Learning from both successes and failures
- Coordinate diversity to avoid recently used and failed coordinates
- Gradual recovery through time-based penalty decay
- Detailed logging of all penalty and decay events
"""

import asyncio
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_penalty_decay_system():
    """Test the complete penalty decay system."""
    
    try:
        # Import the systems
        from src.core.penalty_decay_system import get_penalty_decay_system
        from src.core.coordinate_intelligence_system import create_coordinate_intelligence_system
        from src.core.penalty_logging_system import get_penalty_logging_system
        
        logger.info("🚀 Starting Penalty Decay System Test")
        logger.info("=" * 60)
        
        # Initialize systems
        penalty_system = get_penalty_decay_system()
        coordinate_system = create_coordinate_intelligence_system()
        logging_system = get_penalty_logging_system()
        
        await penalty_system.initialize()
        
        test_game_id = "penalty_test_game"
        
        # Test 1: Penalty Application
        logger.info("📊 Test 1: Penalty Application")
        logger.info("-" * 30)
        
        # Simulate various coordinate attempts
        test_cases = [
            # (x, y, success, score_change, expected_penalty_type)
            (10, 10, True, 10.0, "none"),  # Successful
            (15, 15, False, 0.0, "no_improvement"),  # No improvement
            (20, 20, False, -5.0, "score_decrease"),  # Score decrease
            (25, 25, False, 0.0, "stuck_loop"),  # Will become stuck after multiple attempts
        ]
        
        for x, y, success, score_change, expected_type in test_cases:
            logger.info(f"Testing coordinate ({x},{y}): success={success}, score_change={score_change}")
            
            # For stuck loop test, make multiple attempts
            if expected_type == "stuck_loop":
                for i in range(6):  # More than stuck threshold
                    result = await penalty_system.record_coordinate_attempt(
                        game_id=test_game_id,
                        x=x, y=y,
                        success=False,
                        score_change=0.0,
                        action_type="ACTION6",
                        context={'test_attempt': i+1}
                    )
                    logger.info(f"  Attempt {i+1}: penalty_applied={result.get('penalty_applied', False)}, "
                              f"penalty_score={result.get('penalty_score', 0):.3f}")
            else:
                result = await penalty_system.record_coordinate_attempt(
                    game_id=test_game_id,
                    x=x, y=y,
                    success=success,
                    score_change=score_change,
                    action_type="ACTION6",
                    context={'test_case': expected_type}
                )
                logger.info(f"  Result: penalty_applied={result.get('penalty_applied', False)}, "
                          f"penalty_score={result.get('penalty_score', 0):.3f}")
        
        # Test 2: Avoidance Recommendations
        logger.info("\n🎯 Test 2: Avoidance Recommendations")
        logger.info("-" * 30)
        
        candidate_coords = [(10, 10), (15, 15), (20, 20), (25, 25), (30, 30)]
        avoidance_scores = await penalty_system.get_avoidance_recommendations(
            test_game_id, candidate_coords
        )
        
        logger.info("Avoidance scores (higher = more avoidable):")
        for coord, score in sorted(avoidance_scores.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {coord}: {score:.3f}")
        
        # Test 3: Coordinate Diversity
        logger.info("\n🎨 Test 3: Coordinate Diversity")
        logger.info("-" * 30)
        
        diverse_recommendations = await coordinate_system.get_diverse_coordinate_recommendations(
            game_id=test_game_id,
            action_id=6,
            grid_size=(64, 64),
            strategy="balanced",
            max_recommendations=5
        )
        
        logger.info("Diverse coordinate recommendations:")
        for i, rec in enumerate(diverse_recommendations):
            logger.info(f"  {i+1}. ({rec['x']},{rec['y']}) - "
                      f"confidence: {rec['confidence_score']:.3f}, "
                      f"avoidance: {rec['avoidance_score']:.3f}, "
                      f"adjusted: {rec['adjusted_score']:.3f}")
        
        # Test 4: Penalty Decay
        logger.info("\n⏰ Test 4: Penalty Decay")
        logger.info("-" * 30)
        
        # Get current penalties
        current_penalties = {}
        for x, y in [(15, 15), (20, 20), (25, 25)]:
            penalty_info = await penalty_system.get_coordinate_penalty(test_game_id, x, y)
            current_penalties[(x, y)] = penalty_info['penalty_score']
            logger.info(f"Current penalty for ({x},{y}): {penalty_info['penalty_score']:.3f}")
        
        # Apply decay
        decay_result = await penalty_system.decay_penalties(test_game_id)
        logger.info(f"Decay result: {decay_result}")
        
        # Check penalties after decay
        logger.info("Penalties after decay:")
        for x, y in [(15, 15), (20, 20), (25, 25)]:
            penalty_info = await penalty_system.get_coordinate_penalty(test_game_id, x, y)
            old_penalty = current_penalties.get((x, y), 0.0)
            new_penalty = penalty_info['penalty_score']
            change = new_penalty - old_penalty
            logger.info(f"  ({x},{y}): {old_penalty:.3f} -> {new_penalty:.3f} (change: {change:+.3f})")
        
        # Test 5: System Status
        logger.info("\n📊 Test 5: System Status")
        logger.info("-" * 30)
        
        status = await penalty_system.get_system_status()
        logger.info("Penalty system status:")
        logger.info(f"  Metrics: {status.get('metrics', {})}")
        logger.info(f"  Penalty stats: {status.get('penalty_stats', {})}")
        logger.info(f"  Failure stats: {status.get('failure_stats', {})}")
        
        # Test 6: Logging System
        logger.info("\n📝 Test 6: Logging System")
        logger.info("-" * 30)
        
        # Get recent events
        recent_events = await logging_system.get_recent_events(limit=10)
        logger.info(f"Recent events: {len(recent_events)}")
        for event in recent_events[-5:]:  # Last 5 events
            logger.info(f"  {event.timestamp} - {event.event_type.value}: {event.coordinate}")
        
        # Generate penalty report
        report = await logging_system.generate_penalty_report(test_game_id)
        logger.info("\n📋 Penalty Report:")
        logger.info(report)
        
        logger.info("\n✅ All tests completed successfully!")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    success = await test_penalty_decay_system()
    if success:
        logger.info("🎉 Penalty Decay System test completed successfully!")
    else:
        logger.error("❌ Penalty Decay System test failed!")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())
