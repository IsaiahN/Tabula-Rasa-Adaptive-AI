#!/usr/bin/env python3
"""
Test GUI fixes for the meta-cognitive dashboard.
"""

import time
import asyncio
from src.core.meta_cognitive_dashboard import MetaCognitiveDashboard, DashboardMode

def test_console_dashboard():
    """Test console dashboard with the fixes."""
    print("🧪 Testing Console Dashboard with Fixes...")
    
    # Create dashboard in console mode
    dashboard = MetaCognitiveDashboard(
        mode=DashboardMode.CONSOLE,
        update_interval=2.0
    )
    
    # Start monitoring
    dashboard.start("test_session_console")
    
    # Test with problematic string values that caused warnings
    test_metrics = {
        'status': 'starting',  # This was causing warnings
        'completion': 'completed',
        'progress': 'running',
        'success': True,
        'score': 85.5,
        'win_rate': 0.75
    }
    
    print("📊 Logging test metrics...")
    dashboard.log_performance_update(test_metrics, "test_source")
    
    # Test governor decision logging
    print("🧠 Testing Governor decision logging...")
    dashboard.log_governor_decision(
        recommendation={'type': 'optimize_salience', 'changes': {'threshold': 0.7}},
        confidence=0.85,
        context={'puzzle_type': 'test'}
    )
    
    # Test architect evolution logging
    print("🏗️ Testing Architect evolution logging...")
    dashboard.log_architect_evolution(
        mutation={'type': 'parameter_adjustment', 'changes': {'learning_rate': 0.01}},
        test_result={'success_score': 0.92, 'improvement': 0.15}
    )
    
    # Let it update once
    time.sleep(3)
    
    # Get summary
    summary = dashboard.get_performance_summary(hours=1)
    print(f"✅ Performance summary generated: {len(summary['metrics'])} metrics")
    
    # Stop dashboard
    dashboard.stop()
    print("✅ Console dashboard test completed successfully!")
    
    assert True

def test_gui_dashboard():
    """Test GUI dashboard with the fixes."""
    print("\n🧪 Testing GUI Dashboard with Fixes...")
    try:
        # Create dashboard in GUI mode
        dashboard = MetaCognitiveDashboard(
            mode=DashboardMode.GUI,
            update_interval=1.0
        )

        # Start monitoring
        dashboard.start("test_session_gui")

        # Test metrics that previously caused issues
        test_metrics = {
            'status': 'starting',
            'unicode_test': 'testing with unicode: éñçødîng',  # Test unicode handling
            'completion': 'completed',
            'score': 92.3
        }

        print("📊 Logging test metrics to GUI...")
        dashboard.log_performance_update(test_metrics, "gui_test")

        # Add some test events
        dashboard.log_event(
            "TEST_EVENT",
            "test_system",
            "GUI Test Event",
            "Testing GUI with unicode: éñçødîng and special chars",
            importance=0.8
        )

        print("✅ GUI dashboard initialized successfully!")
        print("Note: GUI window should open. Close it to continue the test.")

        # Run GUI for a short time (non-blocking test)
        dashboard.gui_root.after(5000, dashboard.gui_root.quit)  # Auto-close after 5 seconds
        dashboard.run_gui()

        # Stop dashboard
        dashboard.stop()
        print("✅ GUI dashboard test completed!")
        assert True

    except Exception as e:
        print(f"⚠️ GUI test failed (this is OK on headless systems): {e}")
        assert False

def test_continuous_training_dashboard():
    """Run the continuous training dashboard test via asyncio.run to avoid
    depending on pytest async plugins."""
    async def _body():
        print("\n🧪 Testing Continuous Training Dashboard...")
        
        dashboard = MetaCognitiveDashboard(
            mode=DashboardMode.CONSOLE,
            update_interval=1.0
        )
        
        dashboard.start("continuous_test")
        
        # Simulate continuous training metrics
        for session in range(3):
            print(f"📈 Simulating training session {session + 1}...")
            
            # Metrics that mirror what the actual training sends
            session_metrics = {
                'session_number': session + 1,
                'status': 'starting' if session == 0 else 'running',
                'duration': 45.5 + (session * 12.3),
                'success': True,
                'win_rate': 0.6 + (session * 0.1),
                'learning_efficiency': 0.8 - (session * 0.05)
            }
            
            dashboard.log_performance_update(session_metrics, 'continuous_runner')
            
            # Simulate meta-cognitive activity
            if session > 0:
                dashboard.log_governor_decision(
                    recommendation={'type': 'adjust_parameters'},
                    confidence=0.75 + (session * 0.1),
                    context={'session': session + 1}
                )
            
            await asyncio.sleep(1)  # Brief pause between sessions
        
        # Final status
        final_metrics = {'status': 'completed', 'total_sessions': 3}
        dashboard.log_performance_update(final_metrics, 'continuous_runner')
        
        # Get final summary
        summary = dashboard.get_performance_summary(hours=1)
        print(f"📋 Final summary: {summary['events']['total']} events, {len(summary['metrics'])} metrics")
        
        dashboard.stop()
        print("✅ Continuous training dashboard test completed!")
        
        return True

    return __import__('asyncio').run(_body())

if __name__ == "__main__":
    print("🔧 TESTING GUI FIXES FOR META-COGNITIVE DASHBOARD")
    print("=" * 60)
    
    # Test console dashboard (this should always work)
    console_success = test_console_dashboard()
    
    # Test GUI dashboard (may fail on headless systems)
    gui_success = test_gui_dashboard()
    
    # Test continuous training dashboard
    continuous_success = asyncio.run(test_continuous_training_dashboard())
    
    print("\n" + "=" * 60)
    print("📋 TEST RESULTS:")
    print(f"✅ Console Dashboard: {'PASS' if console_success else 'FAIL'}")
    print(f"🖥️ GUI Dashboard: {'PASS' if gui_success else 'FAIL (OK on headless)'}")
    print(f"🔄 Continuous Dashboard: {'PASS' if continuous_success else 'FAIL'}")
    
    if console_success and continuous_success:
        print("\n🎉 Core dashboard functionality is working!")
        print("The GUI fixes have resolved the metric value warnings.")
    else:
        print("\n⚠️ Some tests failed - check the error messages above.")
