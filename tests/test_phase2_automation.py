#!/usr/bin/env python3
"""
Test script for Phase 2 Automation System

This script tests the Phase 2 automation components:
- Meta-Learning System
- Autonomous Testing & Validation System
- Phase 2 Integration System
"""

import asyncio
import logging
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.phase2_automation_system import (
    start_phase2_automation,
    stop_phase2_automation,
    get_phase2_status,
    get_phase2_learning_status,
    get_phase2_testing_status
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_phase2_automation():
    """Test the Phase 2 automation system."""
    try:
        logger.info("🧪 Starting Phase 2 Automation System Test")
        
        # Test 1: Start Phase 2 automation system
        logger.info("\n📋 Test 1: Starting Phase 2 Automation System")
        await start_phase2_automation("full_active")
        
        # Wait for systems to initialize
        await asyncio.sleep(5)
        
        # Test 2: Check Phase 2 status
        logger.info("\n📊 Test 2: Checking Phase 2 Status")
        status = get_phase2_status()
        logger.info(f"Phase 2 Active: {status.get('phase2_active')}")
        logger.info(f"Current Status: {status.get('current_status')}")
        logger.info(f"Learning Active: {status.get('learning_active')}")
        logger.info(f"Testing Active: {status.get('testing_active')}")
        logger.info(f"Integration Mode: {status.get('integration_mode')}")
        logger.info(f"Metrics: {status.get('metrics')}")
        
        # Test 3: Check individual system status
        logger.info("\n🔍 Test 3: Checking Individual System Status")
        
        # Meta-Learning System
        learning_status = get_phase2_learning_status()
        logger.info(f"Meta-Learning Active: {learning_status.get('learning_active')}")
        logger.info(f"Learning Metrics: {learning_status.get('metrics')}")
        logger.info(f"Strategies: {learning_status.get('strategies')}")
        logger.info(f"Domain Learning: {learning_status.get('domain_learning')}")
        
        # Testing System
        testing_status = get_phase2_testing_status()
        logger.info(f"Testing Active: {testing_status.get('testing_active')}")
        logger.info(f"Testing Metrics: {testing_status.get('metrics')}")
        logger.info(f"Test Suites: {testing_status.get('test_suites_count')}")
        logger.info(f"Test Cases: {testing_status.get('test_cases_count')}")
        
        # Test 4: Let systems run autonomously
        logger.info("\n⏱️ Test 4: Running Systems Autonomously for 120 seconds")
        start_time = time.time()
        
        while time.time() - start_time < 120:
            await asyncio.sleep(15)
            
            # Get current status
            current_status = get_phase2_status()
            current_learning = get_phase2_learning_status()
            current_testing = get_phase2_testing_status()
            
            # Log progress
            logger.info(f"Progress: {int(time.time() - start_time)}s - "
                       f"Learning Experiences: {current_learning['metrics'].get('total_experiences', 0)}, "
                       f"Tests Executed: {current_testing['metrics'].get('total_tests_executed', 0)}, "
                       f"Quality Score: {current_status['metrics'].get('quality_score', 0):.2f}")
        
        # Test 5: Test learning-test coordination
        logger.info("\n🤝 Test 5: Testing Learning-Test Coordination")
        
        # Check coordination metrics
        final_status = get_phase2_status()
        coordinations = final_status.get('metrics', {}).get('optimization_cycles', 0)
        mutual_improvements = final_status.get('metrics', {}).get('mutual_improvements', 0)
        adaptive_adjustments = final_status.get('metrics', {}).get('adaptive_adjustments', 0)
        
        logger.info(f"Optimization Cycles: {coordinations}")
        logger.info(f"Mutual Improvements: {mutual_improvements}")
        logger.info(f"Adaptive Adjustments: {adaptive_adjustments}")
        
        # Test 6: Test different integration modes
        logger.info("\n🔄 Test 6: Testing Different Integration Modes")
        
        # Test learning-only mode
        logger.info("Testing learning-only mode...")
        await stop_phase2_automation()
        await asyncio.sleep(2)
        await start_phase2_automation("learning_only")
        await asyncio.sleep(10)
        
        learning_only_status = get_phase2_status()
        logger.info(f"Learning-Only Mode: {learning_only_status.get('current_status')}")
        logger.info(f"Learning Active: {learning_only_status.get('learning_active')}")
        logger.info(f"Testing Active: {learning_only_status.get('testing_active')}")
        
        # Test testing-only mode
        logger.info("Testing testing-only mode...")
        await stop_phase2_automation()
        await asyncio.sleep(2)
        await start_phase2_automation("testing_only")
        await asyncio.sleep(10)
        
        testing_only_status = get_phase2_status()
        logger.info(f"Testing-Only Mode: {testing_only_status.get('current_status')}")
        logger.info(f"Learning Active: {testing_only_status.get('learning_active')}")
        logger.info(f"Testing Active: {testing_only_status.get('testing_active')}")
        
        # Test 7: Final status report
        logger.info("\n📈 Test 7: Final Status Report")
        
        # Restart full mode for final report
        await stop_phase2_automation()
        await asyncio.sleep(2)
        await start_phase2_automation("full_active")
        await asyncio.sleep(5)
        
        final_status = get_phase2_status()
        final_learning = get_phase2_learning_status()
        final_testing = get_phase2_testing_status()
        
        # Phase 2 System final status
        logger.info(f"Phase 2 Final Status:")
        logger.info(f"  Learning Effectiveness: {final_status['metrics'].get('learning_effectiveness', 0):.3f}")
        logger.info(f"  Test Coverage: {final_status['metrics'].get('test_coverage', 0):.3f}")
        logger.info(f"  Learning-Test Sync: {final_status['metrics'].get('learning_test_sync', 0):.3f}")
        logger.info(f"  Quality Score: {final_status['metrics'].get('quality_score', 0):.3f}")
        logger.info(f"  Performance Trend: {final_status['metrics'].get('performance_trend', 0):.3f}")
        
        # Meta-Learning System final status
        learning_final = final_learning
        logger.info(f"Meta-Learning Final Status:")
        logger.info(f"  Total Experiences: {learning_final.get('metrics', {}).get('total_experiences', 0)}")
        logger.info(f"  Successful Learnings: {learning_final.get('metrics', {}).get('successful_learnings', 0)}")
        logger.info(f"  Patterns Discovered: {learning_final.get('patterns_count', 0)}")
        logger.info(f"  Meta Insights: {learning_final.get('meta_insights_count', 0)}")
        logger.info(f"  Learning Acceleration: {learning_final.get('metrics', {}).get('learning_acceleration', 0):.3f}")
        
        # Testing System final status
        testing_final = final_testing
        logger.info(f"Testing Final Status:")
        logger.info(f"  Total Tests Executed: {testing_final.get('metrics', {}).get('total_tests_executed', 0)}")
        logger.info(f"  Tests Passed: {testing_final.get('metrics', {}).get('tests_passed', 0)}")
        logger.info(f"  Tests Failed: {testing_final.get('metrics', {}).get('tests_failed', 0)}")
        logger.info(f"  Test Coverage: {testing_final.get('metrics', {}).get('test_coverage', 0):.3f}")
        logger.info(f"  Performance Regressions: {testing_final.get('metrics', {}).get('performance_regressions', 0)}")
        logger.info(f"  Bugs Detected: {testing_final.get('metrics', {}).get('bugs_detected', 0)}")
        
        # Test 8: Stop Phase 2 automation system
        logger.info("\n🛑 Test 8: Stopping Phase 2 Automation System")
        await stop_phase2_automation()
        
        # Final status after stop
        final_status_after_stop = get_phase2_status()
        logger.info(f"Phase 2 Active After Stop: {final_status_after_stop.get('phase2_active')}")
        
        logger.info("\n✅ Phase 2 Automation System Test Completed Successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

async def test_individual_phase2_systems():
    """Test individual Phase 2 systems separately."""
    try:
        logger.info("\n🔧 Testing Individual Phase 2 Systems")
        
        # Test Meta-Learning System
        logger.info("Testing Meta-Learning System...")
        from src.core.meta_learning_system import start_meta_learning, stop_meta_learning, get_meta_learning_status
        
        await start_meta_learning()
        await asyncio.sleep(5)
        
        learning_status = get_meta_learning_status()
        logger.info(f"Meta-Learning Active: {learning_status.get('learning_active')}")
        logger.info(f"Learning Metrics: {learning_status.get('metrics')}")
        logger.info(f"Strategies: {learning_status.get('strategies')}")
        logger.info(f"Domain Learning: {learning_status.get('domain_learning')}")
        
        await stop_meta_learning()
        
        # Test Autonomous Testing System
        logger.info("Testing Autonomous Testing System...")
        from src.core.autonomous_testing_system import start_autonomous_testing, stop_autonomous_testing, get_testing_status
        
        await start_autonomous_testing()
        await asyncio.sleep(5)
        
        testing_status = get_testing_status()
        logger.info(f"Testing Active: {testing_status.get('testing_active')}")
        logger.info(f"Testing Metrics: {testing_status.get('metrics')}")
        logger.info(f"Test Suites: {testing_status.get('test_suites_count')}")
        logger.info(f"Test Cases: {testing_status.get('test_cases_count')}")
        
        await stop_autonomous_testing()
        
        logger.info("✅ Individual Phase 2 System Tests Completed!")
        
    except Exception as e:
        logger.error(f"❌ Individual Phase 2 system test failed: {e}")

async def test_phase2_integration_modes():
    """Test different Phase 2 integration modes."""
    try:
        logger.info("\n🎯 Testing Phase 2 Integration Modes")
        
        # Test Full Active Mode
        logger.info("Testing Full Active Mode...")
        await start_phase2_automation("full_active")
        await asyncio.sleep(10)
        
        full_status = get_phase2_status()
        logger.info(f"Full Active Status: {full_status.get('current_status')}")
        logger.info(f"Learning Active: {full_status.get('learning_active')}")
        logger.info(f"Testing Active: {full_status.get('testing_active')}")
        
        await stop_phase2_automation()
        
        # Test Learning Only Mode
        logger.info("Testing Learning Only Mode...")
        await start_phase2_automation("learning_only")
        await asyncio.sleep(10)
        
        learning_status = get_phase2_status()
        logger.info(f"Learning Only Status: {learning_status.get('current_status')}")
        logger.info(f"Learning Active: {learning_status.get('learning_active')}")
        logger.info(f"Testing Active: {learning_status.get('testing_active')}")
        
        await stop_phase2_automation()
        
        # Test Testing Only Mode
        logger.info("Testing Testing Only Mode...")
        await start_phase2_automation("testing_only")
        await asyncio.sleep(10)
        
        testing_status = get_phase2_status()
        logger.info(f"Testing Only Status: {testing_status.get('current_status')}")
        logger.info(f"Learning Active: {testing_status.get('learning_active')}")
        logger.info(f"Testing Active: {testing_status.get('testing_active')}")
        
        await stop_phase2_automation()
        
        logger.info("✅ Phase 2 Integration Mode Tests Completed!")
        
    except Exception as e:
        logger.error(f"❌ Phase 2 integration mode test failed: {e}")

async def test_phase2_coordination():
    """Test Phase 2 coordination between learning and testing."""
    try:
        logger.info("\n🤝 Testing Phase 2 Coordination")
        
        # Start full Phase 2 system
        await start_phase2_automation("full_active")
        await asyncio.sleep(5)
        
        # Monitor coordination for 60 seconds
        logger.info("Monitoring coordination for 60 seconds...")
        start_time = time.time()
        
        while time.time() - start_time < 60:
            await asyncio.sleep(10)
            
            status = get_phase2_status()
            learning_status = get_phase2_learning_status()
            testing_status = get_phase2_testing_status()
            
            logger.info(f"Coordination Progress: {int(time.time() - start_time)}s - "
                       f"Optimization Cycles: {status['metrics'].get('optimization_cycles', 0)}, "
                       f"Mutual Improvements: {status['metrics'].get('mutual_improvements', 0)}, "
                       f"Adaptive Adjustments: {status['metrics'].get('adaptive_adjustments', 0)}")
        
        # Final coordination report
        final_status = get_phase2_status()
        logger.info(f"Final Coordination Metrics:")
        logger.info(f"  Optimization Cycles: {final_status['metrics'].get('optimization_cycles', 0)}")
        logger.info(f"  Mutual Improvements: {final_status['metrics'].get('mutual_improvements', 0)}")
        logger.info(f"  Adaptive Adjustments: {final_status['metrics'].get('adaptive_adjustments', 0)}")
        logger.info(f"  Learning-Test Sync: {final_status['metrics'].get('learning_test_sync', 0):.3f}")
        logger.info(f"  Quality Score: {final_status['metrics'].get('quality_score', 0):.3f}")
        
        await stop_phase2_automation()
        
        logger.info("✅ Phase 2 Coordination Test Completed!")
        
    except Exception as e:
        logger.error(f"❌ Phase 2 coordination test failed: {e}")

async def main():
    """Main test function."""
    logger.info("🚀 Starting Phase 2 Automation System Test Suite")
    
    # Test individual systems first
    await test_individual_phase2_systems()
    
    # Test integration modes
    await test_phase2_integration_modes()
    
    # Test coordination
    await test_phase2_coordination()
    
    # Test integrated system
    await test_phase2_automation()
    
    logger.info("🎉 All Phase 2 automation tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
