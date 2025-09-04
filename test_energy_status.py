#!/usr/bin/env python3

import sys
sys.path.append('src')

from arc_integration.continuous_learning_loop import ContinuousLearningLoop

def test_energy_status_display():
    """Test the adaptive energy status display system"""
    
    # Create a test instance
    loop = ContinuousLearningLoop(
        arc_agents_path='dummy',
        tabula_rasa_path='dummy',
        api_key='dummy'
    )
    
    # Test scenario: Early Learning phase (25% win rate)
    loop.training_state = {
        'total_episodes': 20,
        'total_wins': 5,
        'games_played': {
            'test_game': {
                'episodes': [
                    {'success': True}, {'success': False}, 
                    {'success': True}, {'success': False},
                    {'success': False}, {'success': False}
                ]
            }
        }
    }
    
    # Set test energy level
    loop.current_energy = 85.0
    
    # Test calculations
    current_win_rate = loop._calculate_current_win_rate()
    energy_params = loop._calculate_win_rate_adaptive_energy_parameters()
    
    print("=== Adaptive Energy Status Test ===")
    print(f"Current Win Rate: {current_win_rate:.1%}")
    print(f"Skill Phase: {energy_params['skill_phase']}")
    print(f"Action Energy Cost: {energy_params['action_energy_cost']:.1f}")
    print(f"Sleep Threshold: {energy_params['sleep_trigger_threshold']:.0f}%")
    
    actions_until_sleep = int(loop.current_energy / energy_params['action_energy_cost'])
    print(f"Actions until sleep: ~{actions_until_sleep}")
    print(f"Current Energy: {loop.current_energy:.1f}%")
    
    # Test different skill phases
    print("\n=== Testing Different Skill Phases ===")
    
    test_scenarios = [
        (0.0, "Beginner"),
        (0.15, "Early Learning"), 
        (0.35, "Developing"),
        (0.55, "Competent"),
        (0.70, "Skilled"),
        (0.85, "Expert")
    ]
    
    for win_rate, expected_phase in test_scenarios:
        # Mock win rate in global performance metrics
        loop.global_performance_metrics = {
            'total_episodes': 100,
            'total_wins': int(100 * win_rate),
            'win_rate': win_rate  # This is the key field
        }
        
        current_win_rate = loop._calculate_current_win_rate()
        energy_params = loop._calculate_win_rate_adaptive_energy_parameters()
        actions_until_sleep = int(85.0 / energy_params['action_energy_cost'])
        
        print(f"{expected_phase}: {current_win_rate:.1%} win rate | {energy_params['action_energy_cost']:.1f} energy/action | ~{actions_until_sleep} actions until sleep")
    
    print("=== Test Complete ===")

if __name__ == "__main__":
    test_energy_status_display()
