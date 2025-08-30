#!/usr/bin/env python3
"""
Test script for enhanced sleep system with automatic object encoding and consolidation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import numpy as np
import time
from core.sleep_system import SleepCycle
from core.predictive_core import PredictiveCore
from core.meta_learning import MetaLearningSystem
from core.data_models import Experience, AgentState, SensoryInput

def create_mock_experience(visual_pattern: str = "random") -> Experience:
    """Create a mock experience for testing."""
    # Create visual tensor with different patterns
    if visual_pattern == "circle":
        visual = torch.zeros(3, 64, 64)
        center = 32
        radius = 15
        y, x = torch.meshgrid(torch.arange(64), torch.arange(64), indexing='ij')
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        visual[0, mask] = 1.0  # Red circle
    elif visual_pattern == "square":
        visual = torch.zeros(3, 64, 64)
        visual[1, 20:44, 20:44] = 1.0  # Green square
    else:
        visual = torch.randn(3, 64, 64)
    
    # Create proprioception
    proprioception = torch.randn(12)
    
    # Create sensory inputs
    sensory_input = SensoryInput(
        visual=visual,
        proprioception=proprioception,
        energy_level=50.0,
        timestamp=int(time.time())
    )
    
    next_sensory_input = SensoryInput(
        visual=visual + torch.randn(3, 64, 64) * 0.1,  # Slight variation
        proprioception=proprioception + torch.randn(12) * 0.1,
        energy_level=49.0,
        timestamp=int(time.time()) + 1
    )
    
    return Experience(
        state=sensory_input,
        action=torch.randn(4),
        next_state=next_sensory_input,
        learning_progress=0.3,
        energy_change=-1.0,
        timestamp=int(time.time())
    )

def test_enhanced_sleep_system():
    """Test the enhanced sleep system with object encoding."""
    print("Testing Enhanced Sleep System...")
    
    # Initialize components
    predictive_core = PredictiveCore(
        visual_size=(3, 64, 64),
        proprioception_size=12,
        hidden_size=128,
        memory_config={
            'memory_size': 512,
            'word_size': 64,
            'num_read_heads': 4,
            'num_write_heads': 1,
            'controller_size': 256
        }
    )
    
    meta_learning = MetaLearningSystem(
        memory_capacity=500,
        insight_threshold=0.1,
        consolidation_interval=10
    )
    
    sleep_system = SleepCycle(
        predictive_core=predictive_core,
        meta_learning=meta_learning,
        sleep_trigger_energy=20.0,
        object_encoding_threshold=0.05
    )
    
    print("Sleep system initialized with meta-learning integration")
    
    # Create diverse experiences with different visual patterns
    experiences = []
    patterns = ["circle", "square", "random", "circle", "square"]
    
    for i, pattern in enumerate(patterns * 4):  # 20 experiences total
        exp = create_mock_experience(pattern)
        experiences.append(exp)
        sleep_system.add_experience(exp)
    
    print(f"Added {len(experiences)} experiences to sleep system")
    
    # Test sleep trigger conditions
    low_energy_state = AgentState(
        position=torch.randn(3),
        orientation=torch.randn(4),
        energy=15.0,  # Below sleep trigger
        hidden_state=torch.randn(128),
        active_goals=[],
        timestamp=int(time.time())
    )
    
    should_sleep = sleep_system.should_sleep(low_energy_state, 500, 0.5)
    print(f"Should sleep (low energy): {should_sleep}")
    
    # Enter sleep mode
    sleep_system.enter_sleep(low_energy_state)
    print(f"Sleep mode active: {sleep_system.is_sleeping}")
    
    # Execute sleep cycle
    print("Executing sleep cycle...")
    sleep_results = sleep_system.execute_sleep_cycle(experiences)
    
    print("Sleep cycle results:")
    for key, value in sleep_results.items():
        print(f"  - {key}: {value}")
    
    # Check object encodings
    object_encodings = sleep_system.get_object_encodings()
    encoding_quality = sleep_system.get_encoding_quality()
    
    print(f"Object encodings learned: {len(object_encodings)}")
    print(f"Encoding quality score: {encoding_quality:.3f}")
    
    for obj_id, encoding in object_encodings.items():
        print(f"  - {obj_id}: confidence={encoding['confidence']:.3f}")
    
    # Test wake up
    wake_results = sleep_system.wake_up()
    print(f"Wake up results: {wake_results}")
    
    # Get final metrics
    metrics = sleep_system.get_sleep_metrics()
    print("Final sleep metrics:")
    for key, value in metrics.items():
        print(f"  - {key}: {value}")
    
    return True

if __name__ == "__main__":
    try:
        test_enhanced_sleep_system()
        print("\nEnhanced sleep system test completed successfully!")
    except Exception as e:
        print(f"\nEnhanced sleep system test failed: {e}")
        import traceback
        traceback.print_exc()
