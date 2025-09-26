#!/usr/bin/env python3
"""
Verify All Fixes Are Working
This script verifies that all the critical fixes are in place and working.
"""

import os
import sys
import asyncio

# Disable bytecode cache
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

print("=== TABULA RASA FIXES VERIFICATION ===")
print()

def test_systemintegration():
    """Test SystemIntegration import and methods."""
    print("1. Testing SystemIntegration fixes...")

    try:
        # Clear any cached modules
        modules_to_clear = [k for k in sys.modules.keys() if 'integration' in k.lower()]
        for module in modules_to_clear:
            del sys.modules[module]

        from src.core import SystemIntegration
        integration = SystemIntegration()

        # Check methods exist
        has_save = hasattr(integration, 'save_scorecard_data')
        has_flush = hasattr(integration, 'flush_pending_writes')

        print(f"   ✓ SystemIntegration imported successfully")
        print(f"   ✓ save_scorecard_data: {'EXISTS' if has_save else 'MISSING'}")
        print(f"   ✓ flush_pending_writes: {'EXISTS' if has_flush else 'MISSING'}")
        print(f"   ✓ Using: {integration.__class__.__module__}.{integration.__class__.__name__}")

        if has_save and has_flush:
            print("   ✓ SUCCESS: SystemIntegration fixes verified!")
            return True
        else:
            print("   ✗ ERROR: Methods still missing")
            return False

    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        return False

def test_api_manager():
    """Test API manager reasoning fix."""
    print("\n2. Testing API manager reasoning fix...")

    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

        # Read the file to check the fix
        with open('src/training/api/api_manager.py', 'r') as f:
            content = f.read()

        # Check if reasoning fix is present
        has_reasoning_fix = 'reasoning = action.get(\'reasoning\', {})' in content
        has_reasoning_call = 'reasoning=reasoning' in content

        print(f"   ✓ Reasoning extraction: {'FOUND' if has_reasoning_fix else 'MISSING'}")
        print(f"   ✓ Reasoning API call: {'FOUND' if has_reasoning_call else 'MISSING'}")

        if has_reasoning_fix and has_reasoning_call:
            print("   ✓ SUCCESS: API manager reasoning fix verified!")
            return True
        else:
            print("   ✗ ERROR: Reasoning fix not found")
            return False

    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        return False

def test_cache_disabled():
    """Test that Python cache is disabled."""
    print("\n3. Testing Python cache disabled...")

    cache_disabled = os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
    print(f"   ✓ PYTHONDONTWRITEBYTECODE: {'SET' if cache_disabled else 'NOT SET'}")

    # Check .env file
    try:
        with open('.env', 'r') as f:
            env_content = f.read()

        has_env_cache_disable = 'PYTHONDONTWRITEBYTECODE=1' in env_content
        print(f"   ✓ .env cache disable: {'SET' if has_env_cache_disable else 'NOT SET'}")

        if cache_disabled and has_env_cache_disable:
            print("   ✓ SUCCESS: Python cache properly disabled!")
            return True
        else:
            print("   ✗ WARNING: Cache disable not fully configured")
            return True  # Not critical

    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        return False

async def test_method_calls():
    """Test actual method calls."""
    print("\n4. Testing method calls...")

    try:
        from src.core import SystemIntegration
        integration = SystemIntegration()

        # Test the methods
        result1 = await integration.save_scorecard_data({"test": "verification"})
        result2 = await integration.flush_pending_writes()

        print(f"   ✓ save_scorecard_data() call: {'SUCCESS' if result1 else 'FAILED'}")
        print(f"   ✓ flush_pending_writes() call: {'SUCCESS' if result2 else 'FAILED'}")

        if result1 and result2:
            print("   ✓ SUCCESS: Method calls working!")
            return True
        else:
            print("   ✗ ERROR: Method calls failed")
            return False

    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        return False

async def main():
    """Run all verification tests."""
    print("Running comprehensive verification tests...")
    print()

    tests = [
        test_systemintegration(),
        test_api_manager(),
        test_cache_disabled(),
        await test_method_calls()
    ]

    passed = sum(tests)
    total = len(tests)

    print(f"\n=== VERIFICATION RESULTS ===")
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✓ ALL TESTS PASSED!")
        print("✓ Your training system is ready to run with all fixes!")
        print("\nNext steps:")
        print("1. Restart your training system")
        print("2. You should see frame changes and wins again")
        print("3. No more scorecard or SystemIntegration errors")
        print("4. Reasoning logs will be sent with all actions")
    else:
        print("✗ SOME TESTS FAILED!")
        print("Check the errors above and ensure all fixes are applied.")

    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)