#!/usr/bin/env python3
"""
Integration Script for Autonomous System

This script shows how to integrate the autonomous Governor and Architect
into the existing Tabula Rasa system.
"""

import asyncio
import logging
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

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

async def integrate_with_existing_system():
    """Integrate autonomous system with existing Tabula Rasa system."""
    try:
        logger.info("🔗 Integrating Autonomous System with Tabula Rasa")
        
        # 1. Start autonomous system in collaborative mode
        logger.info("📋 Starting autonomous system in collaborative mode...")
        await start_autonomous_system("collaborative")
        
        # 2. Check integration status
        logger.info("📊 Checking integration status...")
        status = await get_autonomous_system_status()
        
        logger.info(f"✅ System Active: {status.get('system_active')}")
        logger.info(f"✅ Current Mode: {status.get('current_mode')}")
        logger.info(f"✅ Overall Health: {status.get('overall_health', 0):.2f}")
        logger.info(f"✅ Autonomy Level: {status.get('autonomy_level', 0):.2f}")
        
        # 3. Show autonomy capabilities
        logger.info("🎯 Autonomy Capabilities:")
        summary = get_autonomy_summary()
        
        logger.info(f"  🧠 Governor: {summary['governor_autonomy']['active']} "
                   f"(Decisions: {summary['governor_autonomy']['decisions_made']})")
        logger.info(f"  🏗️ Architect: {summary['architect_autonomy']['active']} "
                   f"(Evolutions: {summary['architect_autonomy']['evolutions_made']})")
        logger.info(f"  🌉 Collaboration: {summary['collaboration']['active']} "
                   f"(Decisions: {summary['collaboration']['collaborative_decisions']})")
        
        # 4. Demonstrate autonomous operation
        logger.info("⏱️ Running autonomous operation for 60 seconds...")
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < 60:
            await asyncio.sleep(10)
            
            # Get current status
            current_status = await get_autonomous_system_status()
            current_summary = get_autonomy_summary()
            
            logger.info(f"📈 Health: {current_status.get('overall_health', 0):.2f}, "
                       f"Governor Decisions: {current_summary['governor_autonomy']['decisions_made']}, "
                       f"Architect Evolutions: {current_summary['architect_autonomy']['evolutions_made']}, "
                       f"Collaborations: {current_summary['collaboration']['collaborative_decisions']}")
        
        # 5. Demonstrate mode switching
        logger.info("🔄 Demonstrating mode switching...")
        
        # Switch to full autonomous mode
        logger.info("  Switching to full autonomous mode...")
        await execute_director_command("switch_mode", {"mode": "autonomous"})
        await asyncio.sleep(5)
        
        # Switch to directed mode
        logger.info("  Switching to directed mode...")
        await execute_director_command("switch_mode", {"mode": "directed"})
        await asyncio.sleep(5)
        
        # Switch back to collaborative
        logger.info("  Switching back to collaborative mode...")
        await execute_director_command("switch_mode", {"mode": "collaborative"})
        await asyncio.sleep(5)
        
        # 6. Final status report
        logger.info("📊 Final Integration Report:")
        final_status = await get_autonomous_system_status()
        final_summary = get_autonomy_summary()
        
        logger.info(f"  🎯 System Health: {final_status.get('overall_health', 0):.2f}")
        logger.info(f"  🎯 Autonomy Level: {final_status.get('autonomy_level', 0):.2f}")
        logger.info(f"  🎯 Total Governor Decisions: {final_summary['governor_autonomy']['decisions_made']}")
        logger.info(f"  🎯 Total Architect Evolutions: {final_summary['architect_autonomy']['evolutions_made']}")
        logger.info(f"  🎯 Total Collaborative Decisions: {final_summary['collaboration']['collaborative_decisions']}")
        logger.info(f"  🎯 Total Messages Exchanged: {final_summary['collaboration']['messages_exchanged']}")
        
        # 7. Stop autonomous system
        logger.info("🛑 Stopping autonomous system...")
        await stop_autonomous_system()
        
        logger.info("✅ Integration completed successfully!")
        logger.info("🎉 The Director can now focus on high-level strategy while Governor and Architect handle tactical operations!")
        
    except Exception as e:
        logger.error(f"❌ Integration failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

async def demonstrate_director_interface():
    """Demonstrate the Director interface for controlling the autonomous system."""
    try:
        logger.info("🎯 Demonstrating Director Interface")
        
        # Start system
        await start_autonomous_system("directed")
        
        # Show Director commands
        logger.info("📋 Available Director Commands:")
        
        commands = [
            ("start_autonomous", "Start full autonomous mode"),
            ("switch_mode", "Switch between modes (autonomous/collaborative/directed/emergency)"),
            ("emergency_mode", "Switch to emergency mode"),
            ("get_status", "Get comprehensive system status"),
            ("stop_autonomous", "Stop autonomous operation")
        ]
        
        for command, description in commands:
            logger.info(f"  🔧 {command}: {description}")
        
        # Demonstrate commands
        logger.info("\n🧪 Testing Director Commands:")
        
        # Get status
        logger.info("  Getting system status...")
        status = await execute_director_command("get_status")
        logger.info(f"    Status: {status.get('status')}")
        
        # Switch to autonomous
        logger.info("  Switching to autonomous mode...")
        result = await execute_director_command("switch_mode", {"mode": "autonomous"})
        logger.info(f"    Result: {result.get('status')} - New mode: {result.get('new_mode')}")
        
        # Wait a bit
        await asyncio.sleep(5)
        
        # Switch to emergency
        logger.info("  Switching to emergency mode...")
        result = await execute_director_command("emergency_mode")
        logger.info(f"    Result: {result.get('status')} - Mode: {result.get('mode')}")
        
        # Stop system
        logger.info("  Stopping autonomous system...")
        result = await execute_director_command("stop_autonomous")
        logger.info(f"    Result: {result.get('status')} - Mode: {result.get('mode')}")
        
        logger.info("✅ Director interface demonstration completed!")
        
    except Exception as e:
        logger.error(f"❌ Director interface demonstration failed: {e}")

async def main():
    """Main integration function."""
    logger.info("🚀 Starting Tabula Rasa Autonomous System Integration")
    
    # Demonstrate Director interface
    await demonstrate_director_interface()
    
    # Integrate with existing system
    await integrate_with_existing_system()
    
    logger.info("🎉 Integration demonstration completed!")

if __name__ == "__main__":
    asyncio.run(main())
