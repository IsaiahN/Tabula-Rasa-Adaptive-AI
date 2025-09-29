#!/usr/bin/env python3
"""
Test Scorecard Integration
Tests the new scorecard API integration for Tabula Rasa.
"""

import json
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from src.arc_integration.scorecard_api import ScorecardAPIManager, get_api_key_from_config

def test_scorecard_api():
    """Test the scorecard API integration."""
    
    print(" DIRECTOR: Testing Scorecard API Integration")
    print("=" * 60)
    
    # Get API key
    api_key = get_api_key_from_config()
    if not api_key:
        print(" No API key found")
        print("   Please ensure API key is configured in:")
        print("   - data/optimized_config.json")
        print("   - config.json")
        print("   - Environment variable ARC_API_KEY")
        print("   NOTE: unified_trainer_results.json now stored in database")
        return False
    
    print(f" API key found: {api_key[:8]}...")
    
    # Create scorecard manager
    try:
        manager = ScorecardAPIManager(api_key)
        print(" Scorecard manager created successfully")
    except Exception as e:
        print(f" Failed to create scorecard manager: {e}")
        return False
    
    # Test opening a scorecard
    print("\n Testing scorecard creation...")
    source_url = "https://github.com/your-org/tabula-rasa"
    tags = ["tabula_rasa", "test", "integration"]
    opaque = {
        "test": True,
        "timestamp": 1234567890
    }
    
    card_id = manager.open_scorecard(source_url, tags, opaque)
    if card_id:
        print(f" Scorecard opened successfully: {card_id}")
        
        # Test retrieving scorecard data
        print("\n Testing scorecard data retrieval...")
        scorecard_data = manager.get_scorecard_data(card_id)
        if scorecard_data:
            print(" Scorecard data retrieved successfully")
            print(f"   Card ID: {scorecard_data.get('card_id')}")
            print(f"   Games Won: {scorecard_data.get('won', 0)}")
            print(f"   Games Played: {scorecard_data.get('played', 0)}")
            print(f"   Total Actions: {scorecard_data.get('total_actions', 0)}")
            print(f"   Total Score: {scorecard_data.get('score', 0)}")
            
            # Test analysis
            print("\n Testing level completion analysis...")
            analysis = manager.analyze_level_completions(scorecard_data)
            print(f" Analysis completed:")
            print(f"   Level Completions: {analysis['level_completions']}")
            print(f"   Games Completed: {analysis['games_completed']}")
            print(f"   Win Rate: {analysis['win_rate']:.1f}%")
            
            # Test saving data
            print("\n Testing data saving...")
            filepath = manager.save_scorecard_data(card_id, scorecard_data, analysis)
            if filepath:
                print(f" Data saved to: {filepath}")
            else:
                print(" Failed to save data")
            
        else:
            print(" Failed to retrieve scorecard data")
            return False
    else:
        print(" Failed to open scorecard")
        return False
    
    print("\n All tests passed! Scorecard integration is working correctly.")
    return True

def test_continuous_learning_integration():
    """Test integration with continuous learning loop."""
    
    print("\n DIRECTOR: Testing Continuous Learning Integration")
    print("=" * 60)
    
    try:
        from training import ContinuousLearningLoop
        
        # Create a test instance
        loop = ContinuousLearningLoop(
            arc_agents_path="test_path",
            tabula_rasa_path="test_path",
            api_key=get_api_key_from_config()
        )
        
        print(" ContinuousLearningLoop created successfully")
        
        # Test scorecard initialization
        if loop.scorecard_manager:
            print(" Scorecard manager initialized in ContinuousLearningLoop")
            
            # Test performance summary
            summary = loop.get_performance_summary()
            print(" Performance summary generated:")
            print(f"   Scorecard Available: {summary['scorecard_available']}")
            print(f"   Active Scorecard ID: {summary['active_scorecard_id']}")
            print(f"   Local Stats: {summary['local_stats']}")
        else:
            print(" Scorecard manager not initialized (no API key?)")
        
    except Exception as e:
        print(f" Error testing continuous learning integration: {e}")
        return False
    
    return True

def main():
    """Main test function."""
    
    print(" DIRECTOR: Scorecard Integration Test Suite")
    print("=" * 60)
    
    # Test 1: Basic API functionality
    api_test_passed = test_scorecard_api()
    
    # Test 2: Continuous learning integration
    integration_test_passed = test_continuous_learning_integration()
    
    # Summary
    print(f"\n TEST RESULTS:")
    print(f"   API Integration: {' PASSED' if api_test_passed else ' FAILED'}")
    print(f"   Continuous Learning: {' PASSED' if integration_test_passed else ' FAILED'}")
    
    if api_test_passed and integration_test_passed:
        print(f"\n ALL TESTS PASSED!")
        print(f"   Tabula Rasa is ready for scorecard-based progress tracking!")
    else:
        print(f"\n SOME TESTS FAILED")
        print(f"   Please check the error messages above and fix any issues")

if __name__ == "__main__":
    main()
