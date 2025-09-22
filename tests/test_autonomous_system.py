#!/usr/bin/env python3
"""
Test script for the Autonomous System

This script demonstrates the autonomous Governor and Architect capabilities
and their collaboration through the communication bridge.
"""

import asyncio
import logging
import time
from src.core.autonomous_system_manager import (
    start_autonomous_system,
    stop_autonomous_system,
    get_autonomous_system_status,
    get_autonomy_summary,
    execute_director_command
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_autonomous_system():
    """Test the autonomous system functionality."""
    try:
        logger.info("🧪 Starting Autonomous System Test")
        
        # Test 1: Start autonomous system
        logger.info("\n📋 Test 1: Starting Autonomous System")
        await start_autonomous_system("autonomous")
        
        # Wait for system to initialize
        await asyncio.sleep(5)
        
        # Test 2: Check system status
        logger.info("\n📊 Test 2: Checking System Status")
        status = await get_autonomous_system_status()
        logger.info(f"System Active: {status.get('system_active')}")
        logger.info(f"Current Mode: {status.get('current_mode')}")
        logger.info(f"Overall Health: {status.get('overall_health', 0):.2f}")
        logger.info(f"Autonomy Level: {status.get('autonomy_level', 0):.2f}")
        
        # Test 3: Get autonomy summary
        logger.info("\n🎯 Test 3: Getting Autonomy Summary")
        summary = get_autonomy_summary()
        logger.info(f"Governor Active: {summary['governor_autonomy']['active']}")
        logger.info(f"Architect Active: {summary['architect_autonomy']['active']}")
        logger.info(f"Collaboration Active: {summary['collaboration']['active']}")
        logger.info(f"Governor Decisions: {summary['governor_autonomy']['decisions_made']}")
        logger.info(f"Architect Evolutions: {summary['architect_autonomy']['evolutions_made']}")
        logger.info(f"Collaborative Decisions: {summary['collaboration']['collaborative_decisions']}")
        
        # Test 4: Let system run autonomously
        logger.info("\n⏱️ Test 4: Running System Autonomously for 30 seconds")
        start_time = time.time()
        while time.time() - start_time < 30:
            # Check status every 5 seconds
            await asyncio.sleep(5)
            current_status = await get_autonomous_system_status()
            logger.info(f"Health: {current_status.get('overall_health', 0):.2f}, "
                       f"Governor Decisions: {current_status.get('governor_status', {}).get('decisions_made', 0)}, "
                       f"Architect Evolutions: {current_status.get('architect_status', {}).get('evolutions_made', 0)}")
        
        # Test 5: Test mode switching
        logger.info("\n🔄 Test 5: Testing Mode Switching")
        
        # Switch to collaborative mode
        logger.info("Switching to collaborative mode...")
        await execute_director_command("switch_mode", {"mode": "collaborative"})
        await asyncio.sleep(2)
        
        # Switch to directed mode
        logger.info("Switching to directed mode...")
        await execute_director_command("switch_mode", {"mode": "directed"})
        await asyncio.sleep(2)
        
        # Switch back to autonomous mode
        logger.info("Switching back to autonomous mode...")
        await execute_director_command("switch_mode", {"mode": "autonomous"})
        await asyncio.sleep(2)
        
        # Test 6: Test emergency mode
        logger.info("\n🚨 Test 6: Testing Emergency Mode")
        await execute_director_command("emergency_mode")
        await asyncio.sleep(2)
        
        # Switch back to autonomous
        await execute_director_command("switch_mode", {"mode": "autonomous"})
        await asyncio.sleep(2)
        
        # Test 7: Final status check
        logger.info("\n📈 Test 7: Final Status Check")
        final_status = await get_autonomous_system_status()
        final_summary = get_autonomy_summary()
        
        logger.info(f"Final System Health: {final_status.get('overall_health', 0):.2f}")
        logger.info(f"Final Autonomy Level: {final_status.get('autonomy_level', 0):.2f}")
        logger.info(f"Total Governor Decisions: {final_summary['governor_autonomy']['decisions_made']}")
        logger.info(f"Total Architect Evolutions: {final_summary['architect_autonomy']['evolutions_made']}")
        logger.info(f"Total Collaborative Decisions: {final_summary['collaboration']['collaborative_decisions']}")
        logger.info(f"Total Messages Exchanged: {final_summary['collaboration']['messages_exchanged']}")
        
        # Test 8: Stop system
        logger.info("\n🛑 Test 8: Stopping System")
        await stop_autonomous_system()
        
        # Final status after stop
        final_status = await get_autonomous_system_status()
        logger.info(f"System Active After Stop: {final_status.get('system_active')}")
        
        logger.info("\n✅ Autonomous System Test Completed Successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

async def test_individual_components():
    """Test individual components separately."""
    try:
        logger.info("\n🔧 Testing Individual Components")
        
        # Test Governor
        logger.info("Testing Autonomous Governor...")
        from src.core.autonomous_governor import start_autonomous_governor, stop_autonomous_governor, get_autonomous_governor_status
        
        await start_autonomous_governor()
        await asyncio.sleep(3)
        
        governor_status = get_autonomous_governor_status()
        logger.info(f"Governor Active: {governor_status.get('autonomous_cycle_active')}")
        logger.info(f"Governor Decisions: {governor_status.get('decisions_made')}")
        
        await stop_autonomous_governor()
        
        # Test Architect
        logger.info("Testing Autonomous Architect...")
        from src.core.autonomous_architect import start_autonomous_architect, stop_autonomous_architect, get_autonomous_architect_status
        
        await start_autonomous_architect()
        await asyncio.sleep(3)
        
        architect_status = get_autonomous_architect_status()
        logger.info(f"Architect Active: {architect_status.get('autonomous_cycle_active')}")
        logger.info(f"Architect Evolutions: {architect_status.get('evolutions_made')}")
        
        await stop_autonomous_architect()
        
        # Test Bridge
        logger.info("Testing Governor-Architect Bridge...")
        from src.core.governor_architect_bridge import start_governor_architect_communication, stop_governor_architect_communication, get_bridge_status
        
        await start_governor_architect_communication()
        await asyncio.sleep(3)
        
        bridge_status = get_bridge_status()
        logger.info(f"Bridge Active: {bridge_status.get('communication_active')}")
        logger.info(f"Messages Exchanged: {bridge_status.get('messages_exchanged')}")
        
        await stop_governor_architect_communication()
        
        logger.info("✅ Individual Component Tests Completed!")
        
    except Exception as e:
        logger.error(f"❌ Individual component test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

async def main():
    """Main test function."""
    logger.info("🚀 Starting Autonomous System Test Suite")
    
    # Test individual components first
    await test_individual_components()
    
    # Test integrated system
    await test_autonomous_system()
    
    logger.info("🎉 All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
