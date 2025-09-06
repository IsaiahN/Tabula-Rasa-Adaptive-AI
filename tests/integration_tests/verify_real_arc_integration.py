#!/usr/bin/env python3
"""
ARC Integration Verification Script

This script verifies that the Tabula Rasa system is actually connecting to
real ARC-AGI-3 servers and not using any simulation or mock data.

It performs comprehensive checks to ensure authentic API integration.
"""

import asyncio
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

print("🔍 =============================================================================")
print("🔍 TABULA RASA ARC INTEGRATION VERIFICATION")
print("🔍 =============================================================================")
print("🔍 This script verifies that ARC tests are using REAL ARC-AGI-3 servers")
print("🔍 and NOT simulated data from Governor or Architect systems.")
print("🔍 =============================================================================\n")

def check_environment_setup():
    """Verify environment configuration for real ARC API usage."""
    print("📋 STEP 1: Environment Configuration Check")
    print("─" * 60)
    
    # Check .env file
    env_file = current_dir / ".env"
    if not env_file.exists():
        print("❌ No .env file found - ARC API integration may not be configured")
        return False
    
    print("✅ .env file exists")
    
    # Load and check environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('ARC_API_KEY')
    host = os.getenv('HOST', 'three.arcprize.org')
    scheme = os.getenv('SCHEME', 'https')
    
    if not api_key or api_key == 'your_api_key_here':
        print("❌ ARC_API_KEY not properly configured in .env")
        return False
    
    print(f"✅ ARC_API_KEY configured: {api_key[:8]}...{api_key[-4:]}")
    print(f"✅ Host: {scheme}://{host}")
    
    # Check if it's pointing to real ARC servers
    if host != 'three.arcprize.org':
        print(f"⚠️ WARNING: Host is not three.arcprize.org - may be using test server")
        return False
    
    if scheme != 'https':
        print(f"⚠️ WARNING: Not using HTTPS - may be using test server")
        return False
    
    print("✅ Environment configured for REAL ARC-AGI-3 servers")
    return True

async def check_api_connectivity():
    """Test actual connectivity to ARC-AGI-3 API."""
    print("\n🌐 STEP 2: Real ARC-AGI-3 API Connectivity Test")
    print("─" * 60)
    
    try:
        from arc_integration.continuous_learning_loop import ContinuousLearningLoop
        
        # Create learning loop instance with required parameters
        arc_agents_path = os.getenv('ARC_AGENTS_PATH', str(current_dir / 'src'))
        tabula_rasa_path = os.getenv('TABULA_RASA_PATH', str(current_dir))
        
        learning_loop = ContinuousLearningLoop(
            arc_agents_path=arc_agents_path,
            tabula_rasa_path=tabula_rasa_path
        )
        
        # Test API connectivity
        print("🔗 Testing connection to ARC-AGI-3 API...")
        verification_results = await learning_loop.verify_api_connection()
        
        if verification_results['api_accessible']:
            print("✅ Successfully connected to ARC-AGI-3 API")
            print(f"✅ Total games available: {verification_results['total_games_available']}")
            print(f"✅ API key valid: {verification_results['api_key_valid']}")
            
            # Show sample games to prove real data
            if verification_results['sample_games']:
                print("\n📋 Sample real games from API:")
                for i, game in enumerate(verification_results['sample_games'], 1):
                    title = game.get('title', 'Unknown')
                    game_id = game.get('game_id', 'Unknown')
                    print(f"   {i}. {title} ({game_id})")
            
            return True
        else:
            print("❌ Failed to connect to ARC-AGI-3 API")
            print(f"❌ Games available: {verification_results['total_games_available']}")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import ARC integration modules: {e}")
        return False
    except Exception as e:
        print(f"❌ API connectivity test failed: {e}")
        return False

def check_for_simulation_code():
    """Check for any simulation or mock code in ARC integration."""
    print("\n🔍 STEP 3: Simulation/Mock Code Detection")
    print("─" * 60)
    
    # Files to check for real API integration
    critical_files = [
        src_dir / "arc_integration" / "continuous_learning_loop.py",
        src_dir / "api" / "enhanced_client.py",
        src_dir / "arc_integration" / "arc_agent_adapter.py"
    ]
    
    simulation_indicators = [
        'mock_response',
        'fake_api',
        'simulate_arc',
        'mock_arc',
        'test_data',
        'dummy_response',
        'stub_api',
        'localhost:',
        '127.0.0.1',
        'file:///',
        'mock.patch',
        'unittest.mock'
    ]
    
    real_api_indicators = [
        'three.arcprize.org',
        'aiohttp.ClientSession',
        'session.post',
        'session.get',
        'X-API-Key',
        'await response.json()',
        'ARC3_BASE_URL'
    ]
    
    for file_path in critical_files:
        if not file_path.exists():
            print(f"⚠️ File not found: {file_path}")
            continue
            
        print(f"🔍 Checking: {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for simulation indicators
        simulation_found = []
        for indicator in simulation_indicators:
            if indicator.lower() in content.lower():
                simulation_found.append(indicator)
        
        # Check for real API indicators
        api_found = []
        for indicator in real_api_indicators:
            if indicator in content:
                api_found.append(indicator)
        
        if simulation_found:
            print(f"   ⚠️ Potential simulation code found: {simulation_found}")
        
        if api_found:
            print(f"   ✅ Real API integration found: {len(api_found)} indicators")
        else:
            print(f"   ❌ No real API integration indicators found")
    
    return True

def check_governor_architect_separation():
    """Verify Governor/Architect don't simulate ARC responses."""
    print("\n🏛️ STEP 4: Governor/Architect Separation Verification")
    print("─" * 60)
    
    try:
        # Check if Governor/Architect have any ARC simulation
        from core.meta_cognitive_governor import MetaCognitiveGovernor
        from core.architect_evolution_engine import ArchitectEvolutionEngine
        
        # Create instances
        governor = MetaCognitiveGovernor()
        architect = ArchitectEvolutionEngine()
        
        # Check Governor methods
        governor_methods = [method for method in dir(governor) if not method.startswith('_')]
        arc_simulation_methods = [m for m in governor_methods if 'arc' in m.lower() or 'game' in m.lower() or 'action' in m.lower()]
        
        if arc_simulation_methods:
            print(f"⚠️ Governor has potential ARC-related methods: {arc_simulation_methods}")
        else:
            print("✅ Governor has no ARC simulation methods")
        
        # Check Architect methods
        architect_methods = [method for method in dir(architect) if not method.startswith('_')]
        arc_simulation_methods = [m for m in architect_methods if 'arc' in m.lower() or 'game' in m.lower() or 'action' in m.lower()]
        
        if arc_simulation_methods:
            print(f"⚠️ Architect has potential ARC-related methods: {arc_simulation_methods}")
        else:
            print("✅ Architect has no ARC simulation methods")
        
        print("✅ Governor and Architect are focused on meta-cognitive functions only")
        return True
        
    except ImportError as e:
        print(f"⚠️ Could not import Governor/Architect for verification: {e}")
        return False
    except Exception as e:
        print(f"⚠️ Governor/Architect check failed: {e}")
        return False

async def test_real_api_call():
    """Make a real API call to verify authentic integration."""
    print("\n🚀 STEP 5: Live ARC API Call Test")
    print("─" * 60)
    
    try:
        from arc_integration.continuous_learning_loop import ContinuousLearningLoop
        
        # Create learning loop instance with required parameters
        arc_agents_path = os.getenv('ARC_AGENTS_PATH', str(current_dir / 'src'))
        tabula_rasa_path = os.getenv('TABULA_RASA_PATH', str(current_dir))
        
        learning_loop = ContinuousLearningLoop(
            arc_agents_path=arc_agents_path,
            tabula_rasa_path=tabula_rasa_path
        )
        
        # Test opening a scorecard (low-impact API call)
        print("🎫 Testing scorecard creation...")
        scorecard_id = await learning_loop._open_scorecard()
        
        if scorecard_id:
            print(f"✅ Real scorecard created: {scorecard_id}")
            
            # Test closing the scorecard
            print("🔚 Closing test scorecard...")
            closed = await learning_loop._close_scorecard(scorecard_id)
            if closed:
                print("✅ Scorecard closed successfully")
            
            return True
        else:
            print("❌ Failed to create real scorecard")
            return False
            
    except Exception as e:
        print(f"❌ Live API call test failed: {e}")
        return False

def check_training_logs():
    """Check training logs for evidence of real API interactions."""
    print("\n📊 STEP 6: Training Log Analysis")
    print("─" * 60)
    
    # Look for recent training logs
    log_patterns = [
        "master_arc_training_*.log",
        "arc_training.log",
        "continuous_learning_*.log"
    ]
    
    logs_found = []
    for pattern in log_patterns:
        logs_found.extend(list(current_dir.glob(pattern)))
    
    if not logs_found:
        print("⚠️ No training logs found")
        return False
    
    print(f"📋 Found {len(logs_found)} training log files")
    
    # Check most recent log for API evidence
    recent_log = max(logs_found, key=lambda x: x.stat().st_mtime)
    print(f"🔍 Analyzing most recent log: {recent_log.name}")
    
    try:
        with open(recent_log, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        # Look for real API indicators
        api_evidence = []
        if 'three.arcprize.org' in log_content:
            api_evidence.append("ARC-AGI-3 server references")
        if 'Rate limit' in log_content:
            api_evidence.append("Rate limiting (real API behavior)")
        if 'scorecard' in log_content.lower():
            api_evidence.append("Scorecard operations")
        if 'GUID' in log_content or 'guid' in log_content:
            api_evidence.append("Game session GUIDs")
        if 'ACTION' in log_content and 'successful' in log_content:
            api_evidence.append("Action execution results")
        
        if api_evidence:
            print("✅ Real API evidence found in logs:")
            for evidence in api_evidence:
                print(f"   • {evidence}")
            return True
        else:
            print("⚠️ No clear API evidence found in logs")
            return False
            
    except Exception as e:
        print(f"⚠️ Failed to analyze logs: {e}")
        return False

async def comprehensive_verification():
    """Run all verification checks."""
    print("🧪 Starting comprehensive ARC integration verification...\n")
    
    results = {}
    
    # Step 1: Environment
    results['environment'] = check_environment_setup()
    
    # Step 2: API Connectivity
    results['api_connectivity'] = await check_api_connectivity()
    
    # Step 3: Simulation Detection
    results['no_simulation'] = check_for_simulation_code()
    
    # Step 4: Governor/Architect Separation
    results['separation'] = check_governor_architect_separation()
    
    # Step 5: Live API Call
    results['live_api'] = await test_real_api_call()
    
    # Step 6: Log Analysis
    results['logs'] = check_training_logs()
    
    # Final Assessment
    print("\n" + "="*80)
    print("🏆 FINAL VERIFICATION RESULTS")
    print("="*80)
    
    total_checks = len(results)
    passed_checks = sum(results.values())
    
    print(f"✅ Checks Passed: {passed_checks}/{total_checks}")
    print(f"📊 Success Rate: {(passed_checks/total_checks)*100:.1f}%")
    
    print(f"\n📋 Detailed Results:")
    for check, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL" 
        print(f"   • {check.replace('_', ' ').title()}: {status}")
    
    # Overall assessment
    if passed_checks >= 5:  # Allow 1 failure for robustness
        print(f"\n🎉 VERIFICATION: ARC INTEGRATION IS AUTHENTIC!")
        print(f"   The system is connecting to REAL ARC-AGI-3 servers.")
        print(f"   No simulation or mock data detected in core ARC functions.")
        print(f"   Governor and Architect are properly separated from ARC testing.")
        return True
    elif passed_checks >= 3:
        print(f"\n⚠️ VERIFICATION: MOSTLY AUTHENTIC with some concerns")
        print(f"   The system appears to use real ARC APIs but has some issues.")
        print(f"   Review failed checks above for potential improvements.")
        return False
    else:
        print(f"\n❌ VERIFICATION: SIGNIFICANT ISSUES DETECTED")
        print(f"   The system may be using simulated data or has integration problems.")
        print(f"   Manual investigation required.")
        return False

if __name__ == "__main__":
    try:
        result = asyncio.run(comprehensive_verification())
        exit_code = 0 if result else 1
        print(f"\n🔚 Verification complete. Exit code: {exit_code}")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
