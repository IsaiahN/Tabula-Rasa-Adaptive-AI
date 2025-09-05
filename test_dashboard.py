#!/usr/bin/env python3
"""
Test and demonstration of the Meta-Cognitive Visualization Dashboard
"""

import sys
import time
import threading
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.meta_cognitive_dashboard import MetaCognitiveDashboard, DashboardMode
from src.core.meta_cognitive_governor import MetaCognitiveGovernor

def test_console_dashboard():
    """Test the console-based dashboard."""
    print("🖥️ Testing Console Dashboard...")
    
    # Create dashboard in console mode
    dashboard = MetaCognitiveDashboard(
        mode=DashboardMode.CONSOLE,
        update_interval=1.0
    )
    
    # Start monitoring
    dashboard.start("test_console_session")
    
    print("📊 Dashboard started - will simulate 10 seconds of meta-cognitive activity")
    print("Press Ctrl+C to stop early")
    
    try:
        # Simulate meta-cognitive activity
        simulate_meta_cognitive_activity(dashboard, duration=10)
        
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    
    # Stop dashboard
    dashboard.stop()
    
    # Export session data
    export_path = Path("test_dashboard_session.json")
    if dashboard.export_session_data(export_path):
        print(f"📁 Session data exported to {export_path}")
        
        # Clean up
        if export_path.exists():
            export_path.unlink()
    
    print("✅ Console dashboard test completed!")

def test_gui_dashboard():
    """Test the GUI-based dashboard."""
    print("🖼️ Testing GUI Dashboard...")
    
    try:
        # Create dashboard in GUI mode
        dashboard = MetaCognitiveDashboard(
            mode=DashboardMode.GUI,
            update_interval=1.0
        )
        
        # Start monitoring
        dashboard.start("test_gui_session")
        
        print("🎮 GUI Dashboard opened - simulating activity...")
        
        # Start simulation in background
        sim_thread = threading.Thread(
            target=simulate_meta_cognitive_activity,
            args=(dashboard, 30),  # 30 seconds of simulation
            daemon=True
        )
        sim_thread.start()
        
        # Run GUI (blocking)
        print("🖥️ GUI window should be open now...")
        dashboard.run_gui()
        
        # Cleanup
        dashboard.stop()
        
    except Exception as e:
        print(f"❌ GUI dashboard test failed: {e}")
        print("This is likely because tkinter is not available in your environment")

def simulate_meta_cognitive_activity(dashboard, duration=30):
    """Simulate meta-cognitive system activity for testing."""
    import random
    
    start_time = time.time()
    iteration = 0
    
    while time.time() - start_time < duration:
        iteration += 1
        
        # Simulate Governor decisions
        if iteration % 3 == 0:
            rec_types = ["mode_switch", "parameter_adjustment", "consolidation_trigger"]
            rec_type = random.choice(rec_types)
            
            recommendation = {
                "type": rec_type,
                "configuration_changes": {
                    "max_actions_per_game": random.randint(400, 800),
                    "threshold": random.uniform(0.3, 0.9)
                }
            }
            
            confidence = random.uniform(0.4, 0.95)
            context = {
                "puzzle_type": random.choice(["transformation", "pattern", "spatial"]),
                "current_performance": {
                    "win_rate": random.uniform(0.1, 0.8),
                    "score": random.uniform(5, 25)
                }
            }
            
            dashboard.log_governor_decision(recommendation, confidence, context)
        
        # Simulate Architect evolutions
        if iteration % 5 == 0:
            mutation_types = ["neural_architecture_search", "attention_modification", "parameter_adjustment"]
            mutation_type = random.choice(mutation_types)
            
            mutation = {
                "type": mutation_type,
                "changes": {
                    "learning_rate": random.uniform(0.001, 0.01),
                    "hidden_dims": [256, 512, 1024]
                }
            }
            
            test_result = {
                "success_score": random.uniform(0.0, 1.0),
                "performance_improvement": random.uniform(-0.1, 0.2)
            }
            
            dashboard.log_architect_evolution(mutation, test_result)
        
        # Simulate performance updates
        if iteration % 2 == 0:
            performance_metrics = {
                "win_rate": random.uniform(0.1, 0.8),
                "average_score": random.uniform(5, 30),
                "learning_efficiency": random.uniform(0.3, 0.9),
                "computational_efficiency": random.uniform(0.6, 1.0)
            }
            
            dashboard.log_performance_update(performance_metrics)
        
        # Simulate learning updates
        if iteration % 7 == 0:
            patterns_learned = random.randint(1, 5)
            success_rate = random.uniform(0.4, 0.9)
            insights = [
                "Parameter adjustment strategy showing improvement",
                "Mode switching effective for low-performance contexts",
                "Neural architecture search yielding positive results"
            ]
            
            dashboard.log_learning_update(patterns_learned, success_rate, 
                                        random.sample(insights, random.randint(1, 2)))
        
        time.sleep(0.8)  # Simulate processing time

def test_integrated_dashboard():
    """Test dashboard integration with actual Governor."""
    print("🔗 Testing Integrated Dashboard with Governor...")
    
    # Create test directories
    test_dir = Path("test_integrated_dashboard")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Create dashboard
        dashboard = MetaCognitiveDashboard(
            mode=DashboardMode.CONSOLE,
            update_interval=1.5
        )
        
        # Create Governor with outcome tracking and learning
        governor = MetaCognitiveGovernor(
            log_file="test_integrated_governor.log",
            outcome_tracking_dir=str(test_dir / "outcomes"),
            persistence_dir=str(test_dir / "learning")
        )
        
        # Start dashboard monitoring
        dashboard.start("integrated_test_session")
        
        # Start Governor learning session
        if governor.learning_manager:
            governor.start_learning_session({
                "training_type": "dashboard_integration_test",
                "environment": "test"
            })
        
        print("🧠 Running integrated test for 15 seconds...")
        
        # Simulate training with real Governor
        training_scenarios = [
            {
                'puzzle_type': 'transformation',
                'performance': {'win_rate': 0.2, 'average_score': 10.0, 'learning_efficiency': 0.4},
                'config': {'max_actions_per_game': 500}
            },
            {
                'puzzle_type': 'pattern',
                'performance': {'win_rate': 0.4, 'average_score': 18.0, 'learning_efficiency': 0.6},
                'config': {'max_actions_per_game': 600}
            },
            {
                'puzzle_type': 'spatial',
                'performance': {'win_rate': 0.15, 'average_score': 8.0, 'learning_efficiency': 0.3},
                'config': {'max_actions_per_game': 400}
            }
        ]
        
        for i, scenario in enumerate(training_scenarios):
            print(f"\n   🎯 Scenario {i + 1}: {scenario['puzzle_type']}")
            
            # Get Governor recommendation
            recommendation = governor.get_recommended_configuration(
                puzzle_type=scenario['puzzle_type'],
                current_performance=scenario['performance'],
                current_config=scenario['config']
            )
            
            # Log to dashboard
            if recommendation:
                rec_dict = {
                    "type": recommendation.type.value,
                    "configuration_changes": recommendation.configuration_changes
                }
                
                dashboard.log_governor_decision(
                    rec_dict, 
                    recommendation.confidence, 
                    scenario
                )
                
                # Simulate performance improvement
                improved_performance = scenario['performance'].copy()
                improved_performance['win_rate'] += 0.05
                improved_performance['average_score'] += 2.0
                
                dashboard.log_performance_update(improved_performance, "system")
                
                print(f"      📊 Recommendation: {recommendation.type.value}")
                print(f"      🎯 Confidence: {recommendation.confidence:.2f}")
            
            time.sleep(3)  # Wait between scenarios
        
        # Test dashboard summary
        print("\n📈 Getting Performance Summary...")
        summary = dashboard.get_performance_summary(hours=1)
        
        print(f"   • Governor Decisions: {summary['decisions']['governor']}")
        print(f"   • Average Confidence: {summary['decisions']['average_confidence']:.2f}")
        print(f"   • Total Events: {summary['events']['total']}")
        print(f"   • Metrics Tracked: {len(summary['metrics'])}")
        
        # End sessions
        if governor.learning_manager:
            governor.end_learning_session()
        
        dashboard.stop()
        
        print("✅ Integrated dashboard test completed!")
        
    finally:
        # Cleanup
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
        
        test_files = ["test_integrated_governor.log"]
        for file in test_files:
            if Path(file).exists():
                Path(file).unlink()

def main():
    """Main test function."""
    print("🧠 Meta-Cognitive Dashboard Test Suite")
    print("=" * 50)
    
    # Test console dashboard
    print("\n1️⃣ Console Dashboard Test")
    test_console_dashboard()
    
    # Test GUI dashboard (if available)
    print(f"\n2️⃣ GUI Dashboard Test")
    try:
        import tkinter
        test_gui_dashboard()
    except ImportError:
        print("⚠️ Skipping GUI test - tkinter not available")
    
    # Test integrated dashboard
    print(f"\n3️⃣ Integrated Dashboard Test")
    test_integrated_dashboard()
    
    print("\n" + "=" * 50)
    print("✅ All dashboard tests completed!")
    print("🔍 Key capabilities demonstrated:")
    print("   • Real-time monitoring of meta-cognitive decisions")
    print("   • Performance metrics visualization") 
    print("   • Event logging and analysis")
    print("   • Integration with Governor and Architect systems")
    print("   • Session data export capabilities")
    print("   • Both console and GUI interfaces")

if __name__ == "__main__":
    main()
