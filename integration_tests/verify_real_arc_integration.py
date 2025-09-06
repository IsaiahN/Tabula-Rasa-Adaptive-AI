#!/usr/bin/env python3
"""
ARC Integration Verification Script

This script verifies that the Tabula Rasa system is actually connecting to
real ARC-AGI-3 servers and not using any simulation or mock data.

It performs comprehensive checks to ensure authentic API integration.
"""
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
            
        else:
            
            
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
            
            
        print(f"🔍 Checking: {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            
        
        # Check for simulation indicators
        simulation_found = []
        for indicator in simulation_indicators:
            
        
        # Check for real API indicators
        api_found = []
        for indicator in real_api_indicators:
            
        
        if simulation_found:
            
        
        if api_found:
            
        else:
            
    
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
            
        else:
            
        
        # Check Architect methods
        architect_methods = [method for method in dir(architect) if not method.startswith('_')]
        arc_simulation_methods = [m for m in architect_methods if 'arc' in m.lower() or 'game' in m.lower() or 'action' in m.lower()]
        
        if arc_simulation_methods:
            
        else:
            
        
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

            
    except Exception as e:
        

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
        
    
    if not logs_found:
        
    
    print(f"📋 Found {len(logs_found)} training log files")
    
    # Check most recent log for API evidence
    recent_log = max(logs_found, key=lambda x: x.stat().st_mtime)
    print(f"🔍 Analyzing most recent log: {recent_log.name}")
    
    try:
        
            
    except Exception as e:
        

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
        
    
    # Overall assessment
    if passed_checks >= 5:        
    elif passed_checks >= 3:
        
    else:
        

if __name__ == "__main__":
    try:
        
    except KeyboardInterrupt:
        
    except Exception as e:
