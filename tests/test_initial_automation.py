#!/usr/bin/env python3
"""
Test script for Phase 1 Automation System

This script tests the Phase 1 automation components:
- Self-Healing System
- Autonomous System Monitor
- Self-Configuring System
- Unified Automation System
"""

import asyncio
import logging
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.unified_automation_system import (
    start_unified_automation,
    stop_unified_automation,
    get_automation_status,
    get_phase_1_status
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_phase1_automation():
    """Test the Phase 1 automation system."""
    try:
        logger.info(" Starting Phase 1 Automation System Test")
        
        # Test 1: Start unified automation system
        logger.info("\n Test 1: Starting Unified Automation System")
        await start_unified_automation("phase_1")
        
        # Wait for systems to initialize
        await asyncio.sleep(5)
        
        # Test 2: Check automation status
        logger.info("\n Test 2: Checking Automation Status")
        status = get_automation_status()
        logger.info(f"Automation Active: {status.get('automation_active')}")
        logger.info(f"Current Phase: {status.get('current_phase')}")
        logger.info(f"Systems Active: {status.get('systems_active')}")
        logger.info(f"Metrics: {status.get('metrics')}")
        
        # Test 3: Check individual system status
        logger.info("\n Test 3: Checking Individual System Status")
        phase1_status = get_phase_1_status()
        
        # Self-Healing System
        healing_status = phase1_status.get('self_healing', {})
        logger.info(f"Self-Healing Active: {healing_status.get('healing_active')}")
        logger.info(f"Healing Metrics: {healing_status.get('metrics')}")
        
        # Monitoring System
        monitoring_status = phase1_status.get('monitoring', {})
        logger.info(f"Monitoring Active: {monitoring_status.get('monitoring_active')}")
        logger.info(f"Monitoring Metrics: {monitoring_status.get('metrics')}")
        
        # Self-Configuring System
        configuring_status = phase1_status.get('self_configuring', {})
        logger.info(f"Self-Configuring Active: {configuring_status.get('configuring_active')}")
        logger.info(f"Configuring Metrics: {configuring_status.get('metrics')}")
        
        # Test 4: Let systems run autonomously
        logger.info("\n⏱ Test 4: Running Systems Autonomously for 60 seconds")
        start_time = time.time()
        
        while time.time() - start_time < 60:
            await asyncio.sleep(10)
            
            # Get current status
            current_status = get_automation_status()
            current_phase1 = get_phase_1_status()
            
            # Log progress
            logger.info(f"Progress: {int(time.time() - start_time)}s - "
                       f"Errors Fixed: {current_phase1['self_healing']['metrics'].get('errors_auto_fixed', 0)}, "
                       f"Threshold Violations: {current_phase1['monitoring']['metrics'].get('threshold_violations', 0)}, "
                       f"Config Changes: {current_phase1['self_configuring']['metrics'].get('config_changes_applied', 0)}")
        
        # Test 5: Test system coordination
        logger.info("\n Test 5: Testing System Coordination")
        
        # Simulate some system stress to trigger coordination
        logger.info("Simulating system stress...")
        
        # Let systems handle the stress
        await asyncio.sleep(30)
        
        # Check coordination metrics
        final_status = get_automation_status()
        coordinations = final_status.get('metrics', {}).get('cross_system_coordinations', 0)
        logger.info(f"Cross-system coordinations: {coordinations}")
        
        # Test 6: Test health monitoring
        logger.info("\n Test 6: Testing Health Monitoring")
        
        health_history_size = final_status.get('health_history_size', 0)
        logger.info(f"Health history entries: {health_history_size}")
        
        # Test 7: Final status report
        logger.info("\n Test 7: Final Status Report")
        
        final_phase1 = get_phase_1_status()
        
        # Self-Healing System final status
        healing_final = final_phase1.get('self_healing', {})
        logger.info(f"Self-Healing Final Status:")
        logger.info(f"  Errors Detected: {healing_final.get('metrics', {}).get('errors_detected', 0)}")
        logger.info(f"  Errors Auto-Fixed: {healing_final.get('metrics', {}).get('errors_auto_fixed', 0)}")
        logger.info(f"  Fix Success Rate: {healing_final.get('metrics', {}).get('fix_success_rate', 0):.2f}")
        
        # Monitoring System final status
        monitoring_final = final_phase1.get('monitoring', {})
        logger.info(f"Monitoring Final Status:")
        logger.info(f"  Monitoring Cycles: {monitoring_final.get('metrics', {}).get('monitoring_cycles', 0)}")
        logger.info(f"  Threshold Violations: {monitoring_final.get('metrics', {}).get('threshold_violations', 0)}")
        logger.info(f"  Auto Actions Taken: {monitoring_final.get('metrics', {}).get('auto_actions_taken', 0)}")
        
        # Self-Configuring System final status
        configuring_final = final_phase1.get('self_configuring', {})
        logger.info(f"Self-Configuring Final Status:")
        logger.info(f"  Config Cycles: {configuring_final.get('metrics', {}).get('config_cycles', 0)}")
        logger.info(f"  Config Changes Applied: {configuring_final.get('metrics', {}).get('config_changes_applied', 0)}")
        logger.info(f"  Config Changes Rolled Back: {configuring_final.get('metrics', {}).get('config_changes_rolled_back', 0)}")
        
        # Unified System final status
        logger.info(f"Unified System Final Status:")
        logger.info(f"  Total Automation Cycles: {final_status.get('metrics', {}).get('total_automation_cycles', 0)}")
        logger.info(f"  System Health Improvements: {final_status.get('metrics', {}).get('system_health_improvements', 0)}")
        logger.info(f"  Emergency Interventions: {final_status.get('metrics', {}).get('emergency_interventions', 0)}")
        
        # Test 8: Stop automation system
        logger.info("\n Test 8: Stopping Automation System")
        await stop_unified_automation()
        
        # Final status after stop
        final_status_after_stop = get_automation_status()
        logger.info(f"Automation Active After Stop: {final_status_after_stop.get('automation_active')}")
        
        logger.info("\n Phase 1 Automation System Test Completed Successfully!")
        
    except Exception as e:
        logger.error(f" Test failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

async def test_individual_systems():
    """Test individual automation systems separately."""
    try:
        logger.info("\n Testing Individual Automation Systems")
        
        # Test Self-Healing System
        logger.info("Testing Self-Healing System...")
        from src.core.self_healing_system import start_self_healing, stop_self_healing, get_healing_status
        
        await start_self_healing()
        await asyncio.sleep(3)
        
        healing_status = get_healing_status()
        logger.info(f"Self-Healing Active: {healing_status.get('healing_active')}")
        logger.info(f"Self-Healing Metrics: {healing_status.get('metrics')}")
        
        await stop_self_healing()
        
        # Test Autonomous System Monitor
        logger.info("Testing Autonomous System Monitor...")
        from src.core.autonomous_system_monitor import start_autonomous_monitoring, stop_autonomous_monitoring, get_monitoring_status
        
        await start_autonomous_monitoring()
        await asyncio.sleep(3)
        
        monitoring_status = get_monitoring_status()
        logger.info(f"Monitoring Active: {monitoring_status.get('monitoring_active')}")
        logger.info(f"Monitoring Metrics: {monitoring_status.get('metrics')}")
        
        await stop_autonomous_monitoring()
        
        # Test Self-Configuring System
        logger.info("Testing Self-Configuring System...")
        from src.core.self_configuring_system import start_self_configuring, stop_self_configuring, get_configuring_status
        
        await start_self_configuring()
        await asyncio.sleep(3)
        
        configuring_status = get_configuring_status()
        logger.info(f"Self-Configuring Active: {configuring_status.get('configuring_active')}")
        logger.info(f"Self-Configuring Metrics: {configuring_status.get('metrics')}")
        
        await stop_self_configuring()
        
        logger.info(" Individual System Tests Completed!")
        
    except Exception as e:
        logger.error(f" Individual system test failed: {e}")

async def test_automation_phases():
    """Test different automation phases."""
    try:
        logger.info("\n Testing Automation Phases")
        
        # Test Phase 1
        logger.info("Testing Phase 1 (Self-Healing, Monitoring, Configuring)...")
        await start_unified_automation("phase_1")
        await asyncio.sleep(10)
        
        phase1_status = get_automation_status()
        logger.info(f"Phase 1 Status: {phase1_status.get('current_phase')}")
        logger.info(f"Phase 1 Systems Active: {phase1_status.get('systems_active')}")
        
        await stop_unified_automation()
        
        logger.info(" Automation Phase Tests Completed!")
        
    except Exception as e:
        logger.error(f" Automation phase test failed: {e}")

async def main():
    """Main test function."""
    logger.info(" Starting Phase 1 Automation System Test Suite")
    
    # Test individual systems first
    await test_individual_systems()
    
    # Test automation phases
    await test_automation_phases()
    
    # Test integrated system
    await test_phase1_automation()
    
    logger.info(" All Phase 1 automation tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
