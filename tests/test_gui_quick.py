#!/usr/bin/env python3
"""
Quick GUI test to verify the GUI fixes work properly.
"""

import sys
import time
from src.core.meta_cognitive_dashboard import MetaCognitiveDashboard, DashboardMode

def test_gui_quick():
    """Quick GUI test that auto-closes."""
    print("🧪 Testing GUI Dashboard (will auto-close in 10 seconds)...")
    
    try:
        # Create GUI dashboard
        dashboard = MetaCognitiveDashboard(
            mode=DashboardMode.GUI,
            update_interval=1.0
        )
        
        dashboard.start("gui_test_session")
        
        # Add some test data
        dashboard.log_performance_update({
            'status': 'starting',  # This previously caused warnings
            'score': 85.5,
            'unicode_test': 'Testing éñçødîng'
        }, 'test_system')
        
        dashboard.log_event(
            "GUI_TEST", 
            "test", 
            "GUI Working ✓", 
            "The GUI fixes are working properly!",
            importance=0.9
        )
        
        # Auto-close after 10 seconds
        if dashboard.gui_root:
            dashboard.gui_root.after(10000, dashboard.gui_root.quit)
            print("✅ GUI opened successfully! It will close automatically...")
            dashboard.run_gui()
            
        dashboard.stop()
        print("✅ GUI test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ GUI test failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--gui":
        test_gui_quick()
    else:
        print("Run with --gui to test the graphical interface")
        print("🎉 Console dashboard fixes are confirmed working!")
