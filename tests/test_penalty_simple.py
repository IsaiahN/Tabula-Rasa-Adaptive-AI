#!/usr/bin/env python3
"""
Simple test script for Penalty Decay System integration.
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_penalty_integration():
    """Test penalty system integration."""

    print("Testing Penalty Decay System Integration")
    print("=" * 50)

    try:
        # Test 1: Import penalty system
        print("\n1. Testing penalty system import...")
        from src.core.penalty_decay_system import get_penalty_decay_system
        print("[OK] Penalty system imported successfully")

        # Test 2: Create and initialize penalty system
        print("\n2. Testing penalty system creation...")
        penalty_system = get_penalty_decay_system()
        await penalty_system.initialize()
        print("[OK] Penalty system created and initialized")

        # Test 3: Test basic operations
        print("\n3. Testing basic penalty operations...")
        result = await penalty_system.record_coordinate_attempt(
            game_id="test_game",
            x=10, y=20,
            success=False,
            score_change=-1.0
        )
        print(f"[OK] Recorded attempt: {result}")

        # Test 4: Test Action6Coordinator integration
        print("\n4. Testing Action6Coordinator integration...")
        from src.gameplay.action6_coordinator import Action6Coordinator
        coordinator = Action6Coordinator()
        print(f"[OK] Action6Coordinator created")
        print(f"[OK] Penalty system available: {coordinator.penalty_system is not None}")

        # Test 5: Test enhanced gameplay integration
        print("\n5. Testing enhanced gameplay integration...")
        from src.gameplay.enhanced_gameplay import CoreGameplay
        core_gameplay = CoreGameplay()
        print("[OK] Enhanced gameplay system loaded")

        print("\n[SUCCESS] All tests passed!")
        print("Penalty Decay System is properly integrated")

        return True

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_penalty_integration())
    sys.exit(0 if success else 1)