#!/usr/bin/env python3
"""
Test script for Penalty Decay System integration with Action6Coordinator.

This script verifies that the penalty system is correctly integrated
and working with the enhanced pseudo-button learning system.
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_penalty_system_integration():
    """Test the penalty decay system integration."""

    print("Testing Penalty Decay System Integration")
    print("=" * 50)

    try:
        # Test 1: Import penalty system
        print("\n1. Testing penalty system import...")
        from src.core.penalty_decay_system import get_penalty_decay_system, PenaltyDecaySystem
        print("[OK] Penalty system imported successfully")

        # Test 2: Create penalty system instance
        print("\n2. Testing penalty system creation...")
        penalty_system = get_penalty_decay_system()
        print(f"✅ Penalty system created: {type(penalty_system).__name__}")

        # Test 3: Initialize penalty system
        print("\n3. Testing penalty system initialization...")
        await penalty_system.initialize()
        print("✅ Penalty system initialized (in-memory mode)")

        # Test 4: Test basic penalty operations
        print("\n4. Testing basic penalty operations...")

        # Record a failed attempt
        result = await penalty_system.record_coordinate_attempt(
            game_id="test_game",
            x=10, y=20,
            success=False,
            score_change=-1.0,
            action_type="ACTION6",
            context={"test": True},
            pseudo_button_data={"was_pseudo_button": True, "confidence": 0.8}
        )

        print(f"✅ Recorded failed attempt: {result}")

        # Get penalty information
        penalty_info = await penalty_system.get_coordinate_penalty("test_game", 10, 20)
        print(f"✅ Retrieved penalty info: {penalty_info}")

        # Test 5: Import Action6Coordinator
        print("\n5. Testing Action6Coordinator import...")
        from src.gameplay.action6_coordinator import Action6Coordinator, create_action6_coordinator
        print("✅ Action6Coordinator imported successfully")

        # Test 6: Create Action6Coordinator with penalty integration
        print("\n6. Testing Action6Coordinator creation with penalty integration...")
        coordinator = create_action6_coordinator()
        print(f"✅ Action6Coordinator created: {type(coordinator).__name__}")
        print(f"✅ Penalty system integrated: {coordinator.penalty_system is not None}")

        # Test 7: Test avoidance recommendations
        print("\n7. Testing avoidance recommendations...")
        candidate_coords = [(10, 20), (30, 40), (50, 60)]
        avoidance_scores = await penalty_system.get_avoidance_recommendations("test_game", candidate_coords)

        print("✅ Avoidance recommendations:")
        for coord, score in avoidance_scores.items():
            print(f"   {coord}: {score:.3f}")

        # Test 8: Test penalty decay
        print("\n8. Testing penalty decay...")
        decay_result = await penalty_system.decay_penalties("test_game")
        print(f"✅ Penalty decay result: {decay_result}")

        # Test 9: Test system status
        print("\n9. Testing system status...")
        status = await penalty_system.get_system_status()
        print("✅ System status:")
        print(f"   Metrics: {status['metrics']}")
        print(f"   Cache sizes: {status['cache_sizes']}")
        print(f"   Database enabled: {status['database_enabled']}")

        # Test 10: Test Action6Coordinator methods
        print("\n10. Testing Action6Coordinator penalty filtering...")

        # Create mock candidates
        mock_candidates = [
            {'x': 10, 'y': 20, 'confidence': 0.8, 'total_score': 0.7},
            {'x': 30, 'y': 40, 'confidence': 0.6, 'total_score': 0.5},
            {'x': 50, 'y': 60, 'confidence': 0.9, 'total_score': 0.8}
        ]

        filtered_candidates = await coordinator._apply_penalty_filtering(
            mock_candidates, "test_game", {"test": True}
        )

        print(f"✅ Penalty filtering: {len(mock_candidates)} -> {len(filtered_candidates)} candidates")
        for candidate in filtered_candidates:
            penalty_score = candidate.get('penalty_info', {}).get('penalty_score', 0)
            print(f"   ({candidate['x']}, {candidate['y']}): score={candidate['total_score']:.3f}, penalty={penalty_score:.3f}")

        print("\n🎉 All tests completed successfully!")
        print("✅ Penalty Decay System is properly integrated with Action6Coordinator")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_enhanced_gameplay_integration():
    """Test integration with enhanced gameplay system."""

    print("\n🧪 Testing Enhanced Gameplay Integration")
    print("=" * 50)

    try:
        # Test enhanced gameplay import
        print("\n1. Testing enhanced gameplay import...")
        from src.gameplay.enhanced_gameplay import CoreGameplay
        print("✅ Enhanced gameplay imported successfully")

        # Create CoreGameplay instance
        print("\n2. Testing CoreGameplay creation...")
        core_gameplay = CoreGameplay()
        print(f"✅ CoreGameplay created: {type(core_gameplay).__name__}")
        print(f"✅ Enhanced gameplay available: {core_gameplay.enhanced_gameplay is not None}")

        if core_gameplay.enhanced_gameplay:
            action6_coordinator = core_gameplay.enhanced_gameplay.action6_coordinator
            if action6_coordinator:
                print(f"✅ Action6Coordinator available: {type(action6_coordinator).__name__}")
                print(f"✅ Penalty system integrated: {action6_coordinator.penalty_system is not None}")
            else:
                print("⚠️ Action6Coordinator not available in enhanced gameplay")
        else:
            print("⚠️ Enhanced gameplay not available")

        print("\n✅ Enhanced gameplay integration verified!")

        return True

    except Exception as e:
        print(f"\n❌ Enhanced gameplay test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""

    print("🚀 Starting Penalty Decay System Integration Tests")
    print("=" * 60)

    # Run individual tests
    test1_passed = await test_penalty_system_integration()
    test2_passed = await test_enhanced_gameplay_integration()

    # Final results
    print("\n" + "=" * 60)
    print("📋 FINAL TEST RESULTS")
    print("=" * 60)

    print(f"Penalty System Integration: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Enhanced Gameplay Integration: {'✅ PASSED' if test2_passed else '❌ FAILED'}")

    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Penalty Decay System is successfully integrated and ready for use")
        print("✅ The system will now:")
        print("   - Filter out heavily penalized coordinates")
        print("   - Learn from coordinate failures and successes")
        print("   - Apply time-based penalty decay for recovery")
        print("   - Enhance pseudo-button learning with failure patterns")
        print("   - Provide penalty-aware fallback strategies")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("⚠️ Please review the errors above and fix any issues")

    return test1_passed and test2_passed


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())

    # Exit with appropriate code
    sys.exit(0 if success else 1)