#!/usr/bin/env python3
"""
Test script to verify enhanced terminal output functionality
"""
import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the enhanced functions
from master_arc_trainer import safe_print, setup_windows_logging, TeeHandler

def test_safe_print():
    """Test the enhanced safe_print function"""
    print("\n🧪 Testing enhanced safe_print function...")
    
    # Test basic functionality
    safe_print("✅ Basic safe_print test")
    safe_print("🎨 Color test with emojis", use_color=True)
    safe_print("📝 File logging test", log_to_file=True)
    
    # Test with special characters
    safe_print("🔥 Unicode test: 日本語 العربية Русский")
    
    # Test progress simulation
    for i in range(3):
        safe_print(f"⚡ Progress step {i+1}/3: Processing...")
        time.sleep(0.5)
    
    safe_print("✅ All tests completed!")
    
    # Check if output file was created
    output_file = Path('continuous_learning_data/logs/master_arc_trainer_output.log')
    if output_file.exists():
        print(f"✅ Output file created: {output_file}")
        print(f"📊 File size: {output_file.stat().st_size} bytes")
        
        # Show last few lines of the file
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print("\n📄 Last 5 lines from output file:")
            for line in lines[-5:]:
                print(f"  {line.strip()}")
    else:
        print("❌ Output file not created")


def test_tee_handler():
    """Test the TeeHandler logging functionality"""
    print("\n🧪 Testing TeeHandler logging...")
    
    import logging
    
    # Set up test logger with TeeHandler
    logger = logging.getLogger('test_logger')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create TeeHandler
    tee_handler = TeeHandler(
        file_path='continuous_learning_data/logs/test_tee_output.log',
        console_handler=console_handler
    )
    tee_handler.setLevel(logging.INFO)
    
    # Add handler to logger
    logger.addHandler(tee_handler)
    
    # Test logging
    logger.info("🚀 TeeHandler test - this should appear in both console and file")
    logger.info("📊 Testing multiple lines...")
    logger.info("✅ TeeHandler test complete")
    
    # Check if file was created
    test_file = Path('continuous_learning_data/logs/test_tee_output.log')
    if test_file.exists():
        print(f"✅ TeeHandler file created: {test_file}")
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"📄 File content preview:\n{content}")
    else:
        print("❌ TeeHandler file not created")


def test_logging_setup():
    """Test the enhanced Windows logging setup"""
    print("\n🧪 Testing enhanced logging setup...")
    
    # Test the setup function
    setup_windows_logging()
    
    # Test logging with the configured system
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("🎯 Testing enhanced logging system")
    logger.info("📝 This should appear in both terminal and files")
    logger.warning("⚠️ Warning level test")
    logger.error("❌ Error level test")
    
    print("✅ Logging setup test completed")

if __name__ == "__main__":
    print("🧪 TESTING ENHANCED TERMINAL OUTPUT FUNCTIONALITY")
    print("=" * 60)
    
    try:
        test_safe_print()
        test_tee_handler()
        test_logging_setup()
        
        print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Final verification
        output_file = Path('continuous_learning_data/logs/master_arc_trainer_output.log')
        if output_file.exists():
            file_size = output_file.stat().st_size
            print(f"📊 Final output file size: {file_size} bytes")
            if file_size > 0:
                print("✅ Terminal output fix is working correctly!")
            else:
                print("⚠️ Output file is empty - check implementation")
        else:
            print("⚠️ Output file not found - check file paths")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
