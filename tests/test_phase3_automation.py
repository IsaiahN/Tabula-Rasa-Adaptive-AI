#!/usr/bin/env python3
"""
Test script for Phase 3 Automation System

This script tests the Phase 3 automation components:
- Self-Evolving Code System
- Self-Improving Architecture System
- Autonomous Knowledge Management System
- Phase 3 Integration System
"""

import asyncio
import logging
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.phase3_automation_system import (
    start_phase3_automation,
    stop_phase3_automation,
    get_phase3_status,
    get_phase3_code_evolution_status,
    get_phase3_architecture_status,
    get_phase3_knowledge_status
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_phase3_automation():
    """Test the Phase 3 automation system."""
    try:
        logger.info("🧪 Starting Phase 3 Automation System Test")
        
        # Test 1: Start Phase 3 automation system
        logger.info("\n📋 Test 1: Starting Phase 3 Automation System")
        await start_phase3_automation("full_active")
        
        # Wait for systems to initialize
        await asyncio.sleep(5)
        
        # Test 2: Check Phase 3 status
        logger.info("\n📊 Test 2: Checking Phase 3 Status")
        status = get_phase3_status()
        logger.info(f"Phase 3 Active: {status.get('phase3_active')}")
        logger.info(f"Current Status: {status.get('current_status')}")
        logger.info(f"Code Evolution Active: {status.get('code_evolution_active')}")
        logger.info(f"Architecture Active: {status.get('architecture_active')}")
        logger.info(f"Knowledge Active: {status.get('knowledge_active')}")
        logger.info(f"Safety Mode: {status.get('safety_mode')}")
        logger.info(f"Emergency Stop: {status.get('emergency_stop')}")
        logger.info(f"Self-Sufficiency Level: {status.get('self_sufficiency_level')}")
        logger.info(f"Autonomy Score: {status.get('autonomy_score')}")
        logger.info(f"Safety Score: {status.get('safety_score')}")
        logger.info(f"Metrics: {status.get('metrics')}")
        
        # Test 3: Check individual system status
        logger.info("\n🔍 Test 3: Checking Individual System Status")
        
        # Code Evolution System
        code_status = get_phase3_code_evolution_status()
        logger.info(f"Code Evolution Status:")
        logger.info(f"  Evolution Active: {code_status.get('evolution_active')}")
        logger.info(f"  Total Changes: {code_status.get('metrics', {}).get('total_changes', 0)}")
        logger.info(f"  Successful Changes: {code_status.get('metrics', {}).get('successful_changes', 0)}")
        logger.info(f"  Games Since Last Architectural Change: {code_status.get('metrics', {}).get('games_since_last_architectural_change', 0)}")
        logger.info(f"  Safety Violations: {code_status.get('metrics', {}).get('safety_violations', 0)}")
        logger.info(f"  Cooldown Violations: {code_status.get('metrics', {}).get('cooldown_violations', 0)}")
        
        # Architecture System
        arch_status = get_phase3_architecture_status()
        logger.info(f"Architecture Status:")
        logger.info(f"  Architecture Active: {arch_status.get('architecture_active')}")
        logger.info(f"  Total Changes: {arch_status.get('metrics', {}).get('total_changes', 0)}")
        logger.info(f"  Successful Changes: {arch_status.get('metrics', {}).get('successful_changes', 0)}")
        logger.info(f"  Minor Changes: {arch_status.get('metrics', {}).get('minor_changes', 0)}")
        logger.info(f"  Major Changes: {arch_status.get('metrics', {}).get('major_changes', 0)}")
        logger.info(f"  Critical Changes: {arch_status.get('metrics', {}).get('critical_changes', 0)}")
        logger.info(f"  Performance Improvements: {arch_status.get('metrics', {}).get('performance_improvements', 0):.3f}")
        logger.info(f"  Scalability Improvements: {arch_status.get('metrics', {}).get('scalability_improvements', 0):.3f}")
        logger.info(f"  Stability Improvements: {arch_status.get('metrics', {}).get('stability_improvements', 0):.3f}")
        
        # Knowledge Management System
        knowledge_status = get_phase3_knowledge_status()
        logger.info(f"Knowledge Management Status:")
        logger.info(f"  Knowledge Active: {knowledge_status.get('knowledge_active')}")
        logger.info(f"  Total Knowledge Items: {knowledge_status.get('metrics', {}).get('total_knowledge_items', 0)}")
        logger.info(f"  Validated Items: {knowledge_status.get('metrics', {}).get('validated_items', 0)}")
        logger.info(f"  Conflicted Items: {knowledge_status.get('metrics', {}).get('conflicted_items', 0)}")
        logger.info(f"  Obsolete Items: {knowledge_status.get('metrics', {}).get('obsolete_items', 0)}")
        logger.info(f"  Knowledge Quality: {knowledge_status.get('metrics', {}).get('knowledge_quality', 0):.3f}")
        logger.info(f"  Knowledge Coverage: {knowledge_status.get('metrics', {}).get('knowledge_coverage', 0):.3f}")
        logger.info(f"  Knowledge Consistency: {knowledge_status.get('metrics', {}).get('knowledge_consistency', 0):.3f}")
        
        # Test 4: Let systems run autonomously
        logger.info("\n⏱️ Test 4: Running Systems Autonomously for 180 seconds")
        start_time = time.time()
        
        while time.time() - start_time < 180:
            await asyncio.sleep(20)
            
            # Get current status
            current_status = get_phase3_status()
            current_code = get_phase3_code_evolution_status()
            current_arch = get_phase3_architecture_status()
            current_knowledge = get_phase3_knowledge_status()
            
            # Log progress
            logger.info(f"Progress: {int(time.time() - start_time)}s - "
                       f"Code Changes: {current_code.get('metrics', {}).get('total_changes', 0)}, "
                       f"Arch Changes: {current_arch.get('metrics', {}).get('total_changes', 0)}, "
                       f"Knowledge Items: {current_knowledge.get('metrics', {}).get('total_knowledge_items', 0)}, "
                       f"Self-Sufficiency: {current_status.get('self_sufficiency_level')}, "
                       f"Safety Score: {current_status.get('safety_score', 0):.2f}")
        
        # Test 5: Test different modes
        logger.info("\n🔄 Test 5: Testing Different Modes")
        
        # Test code evolution only mode
        logger.info("Testing code evolution only mode...")
        await stop_phase3_automation()
        await asyncio.sleep(2)
        await start_phase3_automation("code_evolution_only")
        await asyncio.sleep(10)
        
        code_only_status = get_phase3_status()
        logger.info(f"Code Evolution Only Mode: {code_only_status.get('current_status')}")
        logger.info(f"Code Evolution Active: {code_only_status.get('code_evolution_active')}")
        logger.info(f"Architecture Active: {code_only_status.get('architecture_active')}")
        logger.info(f"Knowledge Active: {code_only_status.get('knowledge_active')}")
        
        # Test architecture only mode
        logger.info("Testing architecture only mode...")
        await stop_phase3_automation()
        await asyncio.sleep(2)
        await start_phase3_automation("architecture_only")
        await asyncio.sleep(10)
        
        arch_only_status = get_phase3_status()
        logger.info(f"Architecture Only Mode: {arch_only_status.get('current_status')}")
        logger.info(f"Code Evolution Active: {arch_only_status.get('code_evolution_active')}")
        logger.info(f"Architecture Active: {arch_only_status.get('architecture_active')}")
        logger.info(f"Knowledge Active: {arch_only_status.get('knowledge_active')}")
        
        # Test knowledge only mode
        logger.info("Testing knowledge only mode...")
        await stop_phase3_automation()
        await asyncio.sleep(2)
        await start_phase3_automation("knowledge_only")
        await asyncio.sleep(10)
        
        knowledge_only_status = get_phase3_status()
        logger.info(f"Knowledge Only Mode: {knowledge_only_status.get('current_status')}")
        logger.info(f"Code Evolution Active: {knowledge_only_status.get('code_evolution_active')}")
        logger.info(f"Architecture Active: {knowledge_only_status.get('architecture_active')}")
        logger.info(f"Knowledge Active: {knowledge_only_status.get('knowledge_active')}")
        
        # Test safety mode
        logger.info("Testing safety mode...")
        await stop_phase3_automation()
        await asyncio.sleep(2)
        await start_phase3_automation("safety_mode")
        await asyncio.sleep(10)
        
        safety_status = get_phase3_status()
        logger.info(f"Safety Mode: {safety_status.get('current_status')}")
        logger.info(f"Safety Mode Active: {safety_status.get('safety_mode')}")
        logger.info(f"Code Evolution Active: {safety_status.get('code_evolution_active')}")
        logger.info(f"Architecture Active: {safety_status.get('architecture_active')}")
        logger.info(f"Knowledge Active: {safety_status.get('knowledge_active')}")
        
        # Test 6: Test safety mechanisms
        logger.info("\n🛡️ Test 6: Testing Safety Mechanisms")
        
        # Restart full mode for safety testing
        await stop_phase3_automation()
        await asyncio.sleep(2)
        await start_phase3_automation("full_active")
        await asyncio.sleep(5)
        
        safety_status = get_phase3_status()
        logger.info(f"Safety Mechanisms:")
        logger.info(f"  Safety Score: {safety_status.get('safety_score', 0):.3f}")
        logger.info(f"  Safety Violations: {safety_status.get('metrics', {}).get('safety_violations', 0)}")
        logger.info(f"  Cooldown Violations: {safety_status.get('metrics', {}).get('cooldown_violations', 0)}")
        logger.info(f"  Emergency Stop: {safety_status.get('emergency_stop')}")
        
        # Test 7: Final status report
        logger.info("\n📈 Test 7: Final Status Report")
        
        final_status = get_phase3_status()
        final_code = get_phase3_code_evolution_status()
        final_arch = get_phase3_architecture_status()
        final_knowledge = get_phase3_knowledge_status()
        
        # Phase 3 System final status
        logger.info(f"Phase 3 Final Status:")
        logger.info(f"  Self-Sufficiency Level: {final_status.get('self_sufficiency_level')}")
        logger.info(f"  Autonomy Score: {final_status.get('autonomy_score', 0):.3f}")
        logger.info(f"  Safety Score: {final_status.get('safety_score', 0):.3f}")
        logger.info(f"  Code Evolution Changes: {final_status.get('metrics', {}).get('code_evolution_changes', 0)}")
        logger.info(f"  Architecture Improvements: {final_status.get('metrics', {}).get('architecture_improvements', 0)}")
        logger.info(f"  Knowledge Items Managed: {final_status.get('metrics', {}).get('knowledge_items_managed', 0)}")
        
        # Code Evolution System final status
        code_final = final_code
        logger.info(f"Code Evolution Final Status:")
        logger.info(f"  Total Changes: {code_final.get('metrics', {}).get('total_changes', 0)}")
        logger.info(f"  Successful Changes: {code_final.get('metrics', {}).get('successful_changes', 0)}")
        logger.info(f"  Failed Changes: {code_final.get('metrics', {}).get('failed_changes', 0)}")
        logger.info(f"  Rolled Back Changes: {code_final.get('metrics', {}).get('rolled_back_changes', 0)}")
        logger.info(f"  Games Since Last Architectural Change: {code_final.get('metrics', {}).get('games_since_last_architectural_change', 0)}")
        logger.info(f"  Safety Violations: {code_final.get('metrics', {}).get('safety_violations', 0)}")
        logger.info(f"  Cooldown Violations: {code_final.get('metrics', {}).get('cooldown_violations', 0)}")
        
        # Architecture System final status
        arch_final = final_arch
        logger.info(f"Architecture Final Status:")
        logger.info(f"  Total Changes: {arch_final.get('metrics', {}).get('total_changes', 0)}")
        logger.info(f"  Successful Changes: {arch_final.get('metrics', {}).get('successful_changes', 0)}")
        logger.info(f"  Minor Changes: {arch_final.get('metrics', {}).get('minor_changes', 0)}")
        logger.info(f"  Major Changes: {arch_final.get('metrics', {}).get('major_changes', 0)}")
        logger.info(f"  Critical Changes: {arch_final.get('metrics', {}).get('critical_changes', 0)}")
        logger.info(f"  Performance Improvements: {arch_final.get('metrics', {}).get('performance_improvements', 0):.3f}")
        logger.info(f"  Scalability Improvements: {arch_final.get('metrics', {}).get('scalability_improvements', 0):.3f}")
        logger.info(f"  Stability Improvements: {arch_final.get('metrics', {}).get('stability_improvements', 0):.3f}")
        
        # Knowledge Management System final status
        knowledge_final = final_knowledge
        logger.info(f"Knowledge Management Final Status:")
        logger.info(f"  Total Knowledge Items: {knowledge_final.get('metrics', {}).get('total_knowledge_items', 0)}")
        logger.info(f"  Validated Items: {knowledge_final.get('metrics', {}).get('validated_items', 0)}")
        logger.info(f"  Conflicted Items: {knowledge_final.get('metrics', {}).get('conflicted_items', 0)}")
        logger.info(f"  Obsolete Items: {knowledge_final.get('metrics', {}).get('obsolete_items', 0)}")
        logger.info(f"  Knowledge Quality: {knowledge_final.get('metrics', {}).get('knowledge_quality', 0):.3f}")
        logger.info(f"  Knowledge Coverage: {knowledge_final.get('metrics', {}).get('knowledge_coverage', 0):.3f}")
        logger.info(f"  Knowledge Consistency: {knowledge_final.get('metrics', {}).get('knowledge_consistency', 0):.3f}")
        
        # Test 8: Stop Phase 3 automation system
        logger.info("\n🛑 Test 8: Stopping Phase 3 Automation System")
        await stop_phase3_automation()
        
        # Final status after stop
        final_status_after_stop = get_phase3_status()
        logger.info(f"Phase 3 Active After Stop: {final_status_after_stop.get('phase3_active')}")
        
        logger.info("\n✅ Phase 3 Automation System Test Completed Successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

async def test_individual_phase3_systems():
    """Test individual Phase 3 systems separately."""
    try:
        logger.info("\n🔧 Testing Individual Phase 3 Systems")
        
        # Test Self-Evolving Code System
        logger.info("Testing Self-Evolving Code System...")
        from src.core.self_evolving_code_system import start_self_evolving_code, stop_self_evolving_code, get_evolution_status
        
        await start_self_evolving_code()
        await asyncio.sleep(5)
        
        code_status = get_evolution_status()
        logger.info(f"Code Evolution Active: {code_status.get('evolution_active')}")
        logger.info(f"Code Evolution Metrics: {code_status.get('metrics')}")
        logger.info(f"Pending Changes: {code_status.get('pending_changes_count', 0)}")
        logger.info(f"Applied Changes: {code_status.get('applied_changes_count', 0)}")
        logger.info(f"Games Since Last Architectural Change: {code_status.get('metrics', {}).get('games_since_last_architectural_change', 0)}")
        
        await stop_self_evolving_code()
        
        # Test Self-Improving Architecture System
        logger.info("Testing Self-Improving Architecture System...")
        from src.core.self_improving_architecture_system import start_self_improving_architecture, stop_self_improving_architecture, get_architecture_status
        
        await start_self_improving_architecture()
        await asyncio.sleep(5)
        
        arch_status = get_architecture_status()
        logger.info(f"Architecture Active: {arch_status.get('architecture_active')}")
        logger.info(f"Architecture Metrics: {arch_status.get('metrics')}")
        logger.info(f"Pending Changes: {arch_status.get('pending_changes_count', 0)}")
        logger.info(f"Applied Changes: {arch_status.get('applied_changes_count', 0)}")
        logger.info(f"Current Architecture: {arch_status.get('current_architecture')}")
        
        await stop_self_improving_architecture()
        
        # Test Autonomous Knowledge Management System
        logger.info("Testing Autonomous Knowledge Management System...")
        from src.core.autonomous_knowledge_management_system import start_autonomous_knowledge_management, stop_autonomous_knowledge_management, get_knowledge_status
        
        await start_autonomous_knowledge_management()
        await asyncio.sleep(5)
        
        knowledge_status = get_knowledge_status()
        logger.info(f"Knowledge Active: {knowledge_status.get('knowledge_active')}")
        logger.info(f"Knowledge Metrics: {knowledge_status.get('metrics')}")
        logger.info(f"Pending Knowledge: {knowledge_status.get('pending_knowledge_count', 0)}")
        logger.info(f"Validated Knowledge: {knowledge_status.get('validated_knowledge_count', 0)}")
        logger.info(f"Knowledge Items: {knowledge_status.get('knowledge_items_count', 0)}")
        
        await stop_autonomous_knowledge_management()
        
        logger.info("✅ Individual Phase 3 System Tests Completed!")
        
    except Exception as e:
        logger.error(f"❌ Individual Phase 3 system test failed: {e}")

async def test_phase3_safety_mechanisms():
    """Test Phase 3 safety mechanisms."""
    try:
        logger.info("\n🛡️ Testing Phase 3 Safety Mechanisms")
        
        # Start full Phase 3 system
        await start_phase3_automation("full_active")
        await asyncio.sleep(5)
        
        # Monitor safety mechanisms for 60 seconds
        logger.info("Monitoring safety mechanisms for 60 seconds...")
        start_time = time.time()
        
        while time.time() - start_time < 60:
            await asyncio.sleep(10)
            
            status = get_phase3_status()
            code_status = get_phase3_code_evolution_status()
            arch_status = get_phase3_architecture_status()
            
            logger.info(f"Safety Progress: {int(time.time() - start_time)}s - "
                       f"Safety Score: {status.get('safety_score', 0):.3f}, "
                       f"Safety Violations: {status.get('metrics', {}).get('safety_violations', 0)}, "
                       f"Cooldown Violations: {code_status.get('metrics', {}).get('cooldown_violations', 0)}, "
                       f"Frequency Violations: {arch_status.get('metrics', {}).get('change_frequency_violations', 0)}")
        
        # Final safety report
        final_status = get_phase3_status()
        logger.info(f"Final Safety Report:")
        logger.info(f"  Safety Score: {final_status.get('safety_score', 0):.3f}")
        logger.info(f"  Safety Violations: {final_status.get('metrics', {}).get('safety_violations', 0)}")
        logger.info(f"  Cooldown Violations: {final_status.get('metrics', {}).get('cooldown_violations', 0)}")
        logger.info(f"  Emergency Stop: {final_status.get('emergency_stop')}")
        
        await stop_phase3_automation()
        
        logger.info("✅ Phase 3 Safety Mechanisms Test Completed!")
        
    except Exception as e:
        logger.error(f"❌ Phase 3 safety mechanisms test failed: {e}")

async def main():
    """Main test function."""
    logger.info("🚀 Starting Phase 3 Automation System Test Suite")
    
    # Test individual systems first
    await test_individual_phase3_systems()
    
    # Test safety mechanisms
    await test_phase3_safety_mechanisms()
    
    # Test integrated system
    await test_phase3_automation()
    
    logger.info("🎉 All Phase 3 automation tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
