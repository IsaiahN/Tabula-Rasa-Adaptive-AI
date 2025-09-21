#!/usr/bin/env python3
"""
Test script for the simplified Penalty Decay System.

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
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_simple_penalty_decay_system():
    """Test the simplified penalty decay system."""
    
    try:
        # Import the simplified system
        from src.core.penalty_decay_system_simple import get_simple_penalty_decay_system
        
        logger.info("🚀 Starting Simplified Penalty Decay System Test")
        logger.info("=" * 60)
        
        # Initialize system
        penalty_system = get_simple_penalty_decay_system()
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
        
        # Test 3: Penalty Decay
        logger.info("\n⏰ Test 3: Penalty Decay")
        logger.info("-" * 30)
        
        # Get current penalties
        current_penalties = {}
        for x, y in [(15, 15), (20, 20), (25, 25)]:
            penalty_info = await penalty_system.get_coordinate_penalty(test_game_id, x, y)
            current_penalties[(x, y)] = penalty_info['penalty_score']
            logger.info(f"Current penalty for ({x},{y}): {penalty_info['penalty_score']:.3f}")
        
        # Simulate time passing by modifying timestamps
        logger.info("⏳ Simulating time passage for penalty decay...")
        
        # Manually age the penalties for testing
        for coord_key, penalty_data in penalty_system.penalty_cache.items():
            if coord_key[0] == test_game_id:
                # Set last penalty to 2 hours ago
                penalty_data['last_penalty'] = datetime.now() - timedelta(hours=2)
        
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
        
        # Test 4: System Status
        logger.info("\n📊 Test 4: System Status")
        logger.info("-" * 30)
        
        status = await penalty_system.get_system_status()
        logger.info("Penalty system status:")
        logger.info(f"  Metrics: {status.get('metrics', {})}")
        logger.info(f"  Penalty stats: {status.get('penalty_stats', {})}")
        logger.info(f"  Failure stats: {status.get('failure_stats', {})}")
        logger.info(f"  Cache sizes: {status.get('cache_sizes', {})}")
        
        # Test 5: Coordinate Diversity
        logger.info("\n🎨 Test 5: Coordinate Diversity")
        logger.info("-" * 30)
        
        # Test multiple attempts on same coordinate to increase diversity penalty
        logger.info("Testing coordinate diversity by making multiple attempts on (30, 30)...")
        for i in range(3):
            result = await penalty_system.record_coordinate_attempt(
                game_id=test_game_id,
                x=30, y=30,
                success=False,
                score_change=0.0,
                action_type="ACTION6",
                context={'diversity_test': i+1}
            )
            logger.info(f"  Attempt {i+1}: penalty_applied={result.get('penalty_applied', False)}")
        
        # Check avoidance scores again
        avoidance_scores_after = await penalty_system.get_avoidance_recommendations(
            test_game_id, candidate_coords
        )
        
        logger.info("Avoidance scores after diversity test:")
        for coord, score in sorted(avoidance_scores_after.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {coord}: {score:.3f}")
        
        logger.info("\n✅ All tests completed successfully!")
        logger.info("=" * 60)
        
        # Final system status
        final_status = await penalty_system.get_system_status()
        logger.info("📈 Final System Status:")
        logger.info(f"  Total penalties applied: {final_status.get('metrics', {}).get('penalties_applied', 0)}")
        logger.info(f"  Total penalties decayed: {final_status.get('metrics', {}).get('penalties_decayed', 0)}")
        logger.info(f"  Coordinates avoided: {final_status.get('metrics', {}).get('coordinates_avoided', 0)}")
        logger.info(f"  Successful recoveries: {final_status.get('metrics', {}).get('successful_recoveries', 0)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    success = await test_simple_penalty_decay_system()
    if success:
        logger.info("🎉 Simplified Penalty Decay System test completed successfully!")
    else:
        logger.error("❌ Simplified Penalty Decay System test failed!")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())
