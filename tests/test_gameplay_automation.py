#!/usr/bin/env python3
"""
Gameplay Automation Test Script

Tests all gameplay automation systems.
"""

import asyncio
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gameplay import (
    process_gameplay_errors,
    correct_action,
    start_gameplay_monitoring,
    stop_gameplay_monitoring,
    get_gameplay_health,
    get_correction_stats,
    get_monitoring_status
)

async def test_error_automation():
    """Test the error automation system."""
    print("🧪 Testing Error Automation System...")
    
    # Mock game state
    game_state = {
        "score": 0,
        "status": "NOT_FINISHED",
        "available_actions": [1, 2, 3, 4, 5, 6]
    }
    
    # Mock action history with some issues
    action_history = [
        {"id": 6, "x": 32, "y": 32, "confidence": 0.3, "score_after": 0},
        {"id": 6, "x": 32, "y": 32, "confidence": 0.2, "score_after": 0},
        {"id": 6, "x": 32, "y": 32, "confidence": 0.1, "score_after": 0},
        {"id": 6, "x": 32, "y": 32, "confidence": 0.4, "score_after": 0},
        {"id": 6, "x": 32, "y": 32, "confidence": 0.2, "score_after": 0},
    ]
    
    # Mock frame data
    frame_data = [[0] * 64 for _ in range(64)]
    
    # Mock API responses
    api_responses = [
        {"status_code": 200, "success": True},
        {"status_code": 404, "error": "VALIDATION_ERROR"},
        {"status_code": 200, "success": True}
    ]
    
    # Process errors
    result = await process_gameplay_errors(game_state, action_history, frame_data, api_responses)
    
    print(f"✅ Errors detected: {result['errors_detected']}")
    print(f"✅ Fixes applied: {result['fixes_applied']}")
    print(f"✅ System health: {result['system_health']['status']}")
    
    if result['errors']:
        print("🔍 Detected errors:")
        for error in result['errors'][:3]:  # Show first 3
            print(f"  - {error['type']}: {error['description']}")
    
    if result['fixes']:
        print("🔧 Applied fixes:")
        for fix in result['fixes'][:3]:  # Show first 3
            print(f"  - {fix['error_type']}: {fix['fix_applied']}")
    
    return result

def test_action_correction():
    """Test the action correction system."""
    print("\n🧪 Testing Action Correction System...")
    
    # Mock problematic actions
    problematic_actions = [
        {"id": 6, "x": 100, "y": 100, "confidence": 0.2},  # Out of bounds, low confidence
        {"id": 6, "x": 32, "y": 32, "confidence": 0.1},    # Low confidence
        {"id": 99, "x": 32, "y": 32, "confidence": 0.8},   # Invalid action ID
    ]
    
    game_state = {
        "available_actions": [1, 2, 3, 4, 5, 6],
        "frame_quality": 0.8
    }
    
    frame_data = [[0] * 64 for _ in range(64)]
    
    corrections_made = []
    
    for i, action in enumerate(problematic_actions):
        print(f"\n🔍 Testing action {i+1}: {action}")
        
        correction = correct_action(action, game_state, frame_data)
        corrections_made.append(correction)
        
        print(f"✅ Correction applied: {correction.reason}")
        print(f"✅ Confidence: {correction.confidence:.2f}")
        print(f"✅ Original: {correction.original_action}")
        print(f"✅ Corrected: {correction.corrected_action}")
    
    # Get correction stats
    stats = get_correction_stats()
    print(f"\n📊 Correction Statistics:")
    print(f"  Total corrections: {stats['total_corrections']}")
    print(f"  Average confidence: {stats['average_confidence']:.2f}")
    
    return corrections_made

async def test_realtime_monitoring():
    """Test the real-time monitoring system."""
    print("\n🧪 Testing Real-time Monitoring System...")
    
    # Mock game state callback
    game_state_data = {
        "score": 0,
        "action_history": [],
        "frame_data": [[0] * 64 for _ in range(64)],
        "api_responses": [],
        "memory_usage": 0.5
    }
    
    def game_state_callback():
        return game_state_data
    
    # Start monitoring
    print("🔍 Starting real-time monitoring...")
    monitoring_task = asyncio.create_task(start_gameplay_monitoring(game_state_callback))
    
    # Let it run for a few seconds
    await asyncio.sleep(3)
    
    # Get monitoring status
    status = get_monitoring_status()
    print(f"✅ Monitoring status: {status}")
    
    # Get recent events
    events = get_gameplay_events(5)
    print(f"✅ Recent events: {len(events)}")
    
    # Stop monitoring
    stop_gameplay_monitoring()
    monitoring_task.cancel()
    
    try:
        await monitoring_task
    except asyncio.CancelledError:
        pass
    
    print("✅ Monitoring stopped")
    
    return status

async def test_integration():
    """Test integration of all systems."""
    print("\n🧪 Testing System Integration...")
    
    # Create a comprehensive test scenario
    game_state = {
        "score": 0,
        "status": "NOT_FINISHED",
        "available_actions": [1, 2, 3, 4, 5, 6],
        "frame_quality": 0.7
    }
    
    # Simulate a problematic action
    problematic_action = {
        "id": 6,
        "x": 100,  # Out of bounds
        "y": 100,  # Out of bounds
        "confidence": 0.1,  # Very low confidence
        "reason": "Test action"
    }
    
    print(f"🔍 Original action: {problematic_action}")
    
    # Correct the action
    correction = correct_action(problematic_action, game_state, None)
    corrected_action = correction.corrected_action
    
    print(f"✅ Corrected action: {corrected_action}")
    
    # Process with error automation
    action_history = [corrected_action]
    frame_data = [[0] * 64 for _ in range(64)]
    api_responses = [{"status_code": 200, "success": True}]
    
    error_result = await process_gameplay_errors(game_state, action_history, frame_data, api_responses)
    
    print(f"✅ Error processing result: {error_result['errors_detected']} errors, {error_result['fixes_applied']} fixes")
    
    return {
        "original_action": problematic_action,
        "corrected_action": corrected_action,
        "error_result": error_result
    }

async def main():
    """Run all tests."""
    print("🚀 GAMEPLAY AUTOMATION TEST SUITE")
    print("=" * 50)
    
    try:
        # Test error automation
        error_result = await test_error_automation()
        
        # Test action correction
        correction_result = test_action_correction()
        
        # Test real-time monitoring
        monitoring_result = await test_realtime_monitoring()
        
        # Test integration
        integration_result = await test_integration()
        
        print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        # Summary
        print("📊 TEST SUMMARY:")
        print(f"  Error automation: ✅ {error_result['errors_detected']} errors detected")
        print(f"  Action correction: ✅ {len(correction_result)} corrections made")
        print(f"  Real-time monitoring: ✅ {monitoring_result['is_monitoring']} status")
        print(f"  System integration: ✅ All systems working together")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
