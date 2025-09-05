#!/usr/bin/env python3
"""
Test Meta-Cognitive Integration

This script tests the integration of MetaCognitiveGovernor and Architect
with the existing Tabula Rasa system to ensure everything works correctly.
"""

import sys
import asyncio
import logging
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.core.meta_cognitive_governor import (
        MetaCognitiveGovernor, CognitiveCost, CognitiveBenefit,
        GovernorRecommendationType, ArchitectRequest
    )
    from src.core.architect import Architect, SystemGenome, MutationType
    from src.core.salience_system import SalienceMode
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the tabula-rasa root directory")
    
    # Try installing missing dependencies
    print("Attempting to install missing dependencies...")
    import subprocess
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "GitPython"], check=True)
        print("✅ GitPython installed, retrying imports...")
        from src.core.meta_cognitive_governor import (
            MetaCognitiveGovernor, CognitiveCost, CognitiveBenefit,
            GovernorRecommendationType, ArchitectRequest
        )
        from src.core.architect import Architect, SystemGenome, MutationType
        from src.core.salience_system import SalienceMode
    except Exception as install_error:
        print(f"❌ Could not install dependencies: {install_error}")
        sys.exit(1)

def setup_logging():
    """Set up logging for testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('meta_cognitive_test.log')
        ]
    )

def test_governor_basic_functionality():
    """Test basic Governor functionality."""
    print("🧠 Testing MetaCognitiveGovernor...")
    
    # Create Governor
    governor = MetaCognitiveGovernor("test_governor.log")
    
    # Test system activation recording
    test_cost = CognitiveCost(
        compute_units=15.0,
        memory_operations=8,
        decision_complexity=2.5,
        coordination_overhead=1.2
    )
    
    test_benefit = CognitiveBenefit(
        win_rate_improvement=0.15,
        score_improvement=8.0,
        learning_efficiency=0.3,
        knowledge_transfer=0.1
    )
    
    # Record some system activations
    systems_to_test = [
        "swarm_intelligence",
        "dnc_memory", 
        "meta_learning_system",
        "action_intelligence",
        "coordinate_intelligence"
    ]
    
    for system in systems_to_test:
        governor.record_system_activation(system, test_cost, test_benefit)
    
    # Test recommendation system
    current_performance = {
        'win_rate': 0.25,  # Low win rate to trigger recommendations
        'average_score': 35.2,
        'learning_speed': 0.4
    }
    
    current_config = {
        'enable_swarm': True,
        'salience_mode': 'decay_compression',
        'max_actions_per_session': 500,
        'enable_contrarian_mode': False
    }
    
    recommendation = governor.get_recommended_configuration(
        puzzle_type="spatial_reasoning",
        current_performance=current_performance,
        current_config=current_config
    )
    
    if recommendation:
        print(f"✅ Governor recommendation received:")
        print(f"   Type: {recommendation.type.value}")
        print(f"   Confidence: {recommendation.confidence:.1%}")
        print(f"   Rationale: {recommendation.rationale}")
        print(f"   Changes: {recommendation.configuration_changes}")
        return True
    else:
        print("❌ No Governor recommendation generated")
        return False

def test_architect_basic_functionality():
    """Test basic Architect functionality."""
    print("🔬 Testing Architect...")
    
    try:
        # Create Architect
        base_path = str(Path(__file__).parent)
        architect = Architect(
            base_path=base_path,
            repo_path=base_path
        )
        
        # Test genome system
        genome = SystemGenome()
        print(f"✅ Created genome with hash: {genome.get_hash()}")
        
        # Test mutation generation
        mutation = architect.mutation_engine.generate_exploratory_mutation()
        print(f"✅ Generated mutation:")
        print(f"   ID: {mutation.id}")
        print(f"   Type: {mutation.type.value}")
        print(f"   Changes: {mutation.changes}")
        print(f"   Rationale: {mutation.rationale}")
        
        # Test genome application
        mutated_genome = mutation.apply_to_genome(genome)
        print(f"✅ Applied mutation, new hash: {mutated_genome.get_hash()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Architect test failed: {e}")
        return False

async def test_governor_architect_integration():
    """Test Governor-Architect communication."""
    print("🤝 Testing Governor-Architect Integration...")
    
    try:
        # Create both systems
        governor = MetaCognitiveGovernor("integration_test.log")
        base_path = str(Path(__file__).parent)
        architect = Architect(base_path=base_path, repo_path=base_path)
        
        # Simulate Governor creating an Architect request
        performance_data = {
            'win_rate': 0.45,
            'average_score': 40.0,
            'stagnation_cycles': 8
        }
        
        request = governor.create_architect_request(
            issue_type="low_efficiency",
            problem_description="Win rate stuck at 45% for 8 cycles despite multiple strategy attempts",
            performance_data=performance_data
        )
        
        print(f"✅ Governor created Architect request:")
        print(f"   Issue: {request.issue_type}")
        print(f"   Priority: {request.priority:.2f}")
        
        # Test Architect processing the request
        response = await architect.process_governor_request(request)
        
        print(f"✅ Architect processed request:")
        print(f"   Success: {response['success']}")
        if response['success']:
            print(f"   Improvement: {response['improvement']:.3f}")
            print(f"   Recommendation: {response['recommendation']}")
            if response.get('branch_created'):
                print(f"   Branch created: {response['branch_created']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

async def test_autonomous_evolution():
    """Test Architect autonomous evolution cycle."""
    print("🧬 Testing Autonomous Evolution...")
    
    try:
        base_path = str(Path(__file__).parent)
        architect = Architect(base_path=base_path, repo_path=base_path)
        
        # Run one evolution cycle
        result = await architect.autonomous_evolution_cycle()
        
        print(f"✅ Evolution cycle completed:")
        print(f"   Success: {result['success']}")
        print(f"   Generation: {result['generation']}")
        if result.get('improvement'):
            print(f"   Improvement: {result['improvement']:.3f}")
        
        # Show evolution status
        status = architect.get_evolution_status()
        print(f"✅ Evolution status:")
        print(f"   Total mutations tested: {status['total_mutations_tested']}")
        print(f"   Success rate: {status['success_rate']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evolution test failed: {e}")
        return False

def test_integration_with_existing_system():
    """Test integration with existing UnifiedTrainer architecture."""
    print("🔗 Testing Integration with Existing System...")
    
    try:
        # Create a mock args object similar to what UnifiedTrainer receives
        class MockArgs:
            def __init__(self):
                self.mode = "sequential"
                self.salience = "decay"
                self.verbose = True
                self.mastery_sessions = 5
                self.games = 2
                self.target_win_rate = 0.8
                self.target_score = 70.0
                self.max_learning_cycles = 3
                self.max_actions_per_session = 1000
                self.enable_contrarian_mode = True
                self.enable_meta_cognitive = True
                self.disable_meta_cognitive = False
                self.governor_only = False
                self.architect_autonomous_evolution = False
        
        # Import and test UnifiedTrainer initialization
        from train_arc_agent import UnifiedTrainer
        
        args = MockArgs()
        trainer = UnifiedTrainer(args)
        
        # Verify meta-cognitive systems were initialized
        if trainer.governor:
            print("✅ Governor initialized in UnifiedTrainer")
            
            # Test Governor status
            status = trainer.governor.get_system_status()
            print(f"   Governor systems monitored: {len(status['system_efficiencies'])}")
        else:
            print("❌ Governor not initialized")
            return False
        
        if trainer.architect:
            print("✅ Architect initialized in UnifiedTrainer")
            
            # Test Architect status
            arch_status = trainer.architect.get_evolution_status()
            print(f"   Architect generation: {arch_status['generation']}")
        else:
            print("❌ Architect not initialized")
            return False
        
        print("✅ Integration with existing system successful")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

async def run_all_tests():
    """Run all meta-cognitive integration tests."""
    setup_logging()
    
    print("🧪 META-COGNITIVE INTEGRATION TESTS")
    print("=" * 60)
    print()
    
    tests = [
        ("Governor Basic Functionality", test_governor_basic_functionality),
        ("Architect Basic Functionality", test_architect_basic_functionality), 
        ("Governor-Architect Integration", test_governor_architect_integration),
        ("Autonomous Evolution", test_autonomous_evolution),
        ("Existing System Integration", test_integration_with_existing_system)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running: {test_name}")
        print("-" * 40)
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                print(f"✅ PASSED: {test_name}")
                passed += 1
            else:
                print(f"❌ FAILED: {test_name}")
        
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
        
        print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Meta-cognitive integration successful.")
        print()
        print("🚀 READY TO USE:")
        print("   # Enable meta-cognitive features (default)")
        print("   python train_arc_agent.py --mode sequential --verbose")
        print()
        print("   # Governor-only mode")
        print("   python train_arc_agent.py --governor-only --verbose")
        print()
        print("   # Disable meta-cognitive features")
        print("   python train_arc_agent.py --disable-meta-cognitive")
        print()
        print("   # Enable autonomous evolution")
        print("   python train_arc_agent.py --architect-autonomous-evolution --verbose")
    else:
        print("❌ Some tests failed. Check logs for details.")
    
    return passed == total

if __name__ == "__main__":
    print("🧠 Starting Meta-Cognitive Integration Tests...")
    print("   This will test the Governor and Architect systems")
    print("   with the existing Tabula Rasa architecture.")
    print()
    
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
