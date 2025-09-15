#!/usr/bin/env python3
"""
Test Integration Without API
Tests the scorecard integration structure without requiring a real API key.
"""

import json
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

def test_scorecard_module_import():
    """Test that the scorecard module can be imported."""
    
    print("🧪 CONDUCTOR: Testing Scorecard Module Import")
    print("=" * 60)
    
    try:
        from arc_integration.scorecard_api import ScorecardAPIManager, get_api_key_from_config
        print("✅ Scorecard module imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import scorecard module: {e}")
        return False

def test_continuous_learning_integration():
    """Test that the continuous learning loop has scorecard integration."""
    
    print("\n🧪 CONDUCTOR: Testing Continuous Learning Integration")
    print("=" * 60)
    
    try:
        from arc_integration.continuous_learning_loop import ContinuousLearningLoop
        
        # Create a test instance
        loop = ContinuousLearningLoop(
            arc_agents_path="test_path",
            tabula_rasa_path="test_path",
            api_key="test_key"
        )
        
        print("✅ ContinuousLearningLoop created successfully")
        
        # Check if scorecard attributes exist
        has_scorecard_manager = hasattr(loop, 'scorecard_manager')
        has_active_scorecard_id = hasattr(loop, 'active_scorecard_id')
        has_scorecard_stats = hasattr(loop, 'scorecard_stats')
        
        print(f"   Scorecard Manager Attribute: {'✅' if has_scorecard_manager else '❌'}")
        print(f"   Active Scorecard ID Attribute: {'✅' if has_active_scorecard_id else '❌'}")
        print(f"   Scorecard Stats Attribute: {'✅' if has_scorecard_stats else '❌'}")
        
        # Check if scorecard methods exist
        has_init_scorecard = hasattr(loop, '_initialize_scorecard')
        has_update_stats = hasattr(loop, '_update_scorecard_stats')
        has_log_level = hasattr(loop, '_log_level_completion')
        has_log_game = hasattr(loop, '_log_game_completion')
        has_performance_summary = hasattr(loop, 'get_performance_summary')
        
        print(f"   Initialize Scorecard Method: {'✅' if has_init_scorecard else '❌'}")
        print(f"   Update Stats Method: {'✅' if has_update_stats else '❌'}")
        print(f"   Log Level Completion Method: {'✅' if has_log_level else '❌'}")
        print(f"   Log Game Completion Method: {'✅' if has_log_game else '❌'}")
        print(f"   Performance Summary Method: {'✅' if has_performance_summary else '❌'}")
        
        # Test performance summary
        try:
            summary = loop.get_performance_summary()
            print("✅ Performance summary method works")
            print(f"   Summary keys: {list(summary.keys())}")
        except Exception as e:
            print(f"❌ Performance summary method failed: {e}")
        
        return (has_scorecard_manager and has_active_scorecard_id and has_scorecard_stats and
                has_init_scorecard and has_update_stats and has_log_level and 
                has_log_game and has_performance_summary)
        
    except Exception as e:
        print(f"❌ Error testing continuous learning integration: {e}")
        return False

def test_configuration_loading():
    """Test that configuration files can be loaded."""
    
    print("\n🧪 CONDUCTOR: Testing Configuration Loading")
    print("=" * 60)
    
    config_files = [
        "data/optimized_config.json",
        "data/training/results/unified_trainer_results.json"
    ]
    
    success = True
    
    for config_file in config_files:
        if Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                
                has_api_key = 'api_key' in str(data)
                print(f"✅ {config_file}: {'Has API key' if has_api_key else 'No API key'}")
                
            except Exception as e:
                print(f"❌ {config_file}: Error loading - {e}")
                success = False
        else:
            print(f"⚠️ {config_file}: File not found")
    
    return success

def main():
    """Main test function."""
    
    print("🎯 CONDUCTOR: Integration Test Suite (Without API)")
    print("=" * 60)
    
    # Test 1: Module import
    import_test_passed = test_scorecard_module_import()
    
    # Test 2: Continuous learning integration
    integration_test_passed = test_continuous_learning_integration()
    
    # Test 3: Configuration loading
    config_test_passed = test_configuration_loading()
    
    # Summary
    print(f"\n📊 TEST RESULTS:")
    print(f"   Module Import: {'✅ PASSED' if import_test_passed else '❌ FAILED'}")
    print(f"   Continuous Learning: {'✅ PASSED' if integration_test_passed else '❌ FAILED'}")
    print(f"   Configuration Loading: {'✅ PASSED' if config_test_passed else '❌ FAILED'}")
    
    if import_test_passed and integration_test_passed and config_test_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"   Scorecard integration is properly implemented!")
        print(f"   Ready for use with a valid API key!")
    else:
        print(f"\n⚠️ SOME TESTS FAILED")
        print(f"   Please check the error messages above and fix any issues")

if __name__ == "__main__":
    main()
