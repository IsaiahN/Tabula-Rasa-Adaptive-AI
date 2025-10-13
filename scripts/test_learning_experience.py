import sys
from pathlib import Path
sys.dont_write_bytecode = True

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from src.core.enhanced_learning_integration import create_enhanced_learning_integration

async def test_learning_experience():
    integration = create_enhanced_learning_integration()
    await integration.initialize()
    
    # Test with string game score
    await integration.process_learning_experience('game_score')
    
    # Test with numeric game score
    await integration.process_learning_experience(75.5)
    
    # Test with dictionary
    await integration.process_learning_experience({
        'final_score': 85.0,
        'game_won': True,
        'total_actions': 10,
        'game_duration': 120.0,
        'lifecycle_patterns': {
            'failure_risk_score': 0.2,
            'oscillation_detected': False,
            'action_effectiveness': {'action1': 0.9}
        }
    })

if __name__ == '__main__':
    asyncio.run(test_learning_experience())