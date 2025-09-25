#!/usr/bin/env python3
"""
Simple ARC API Reality Check

A minimal test to verify ARC integration is using real API calls.
"""

import asyncio
import aiohttp
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

async def simple_arc_api_test():
    """Test direct connection to ARC-AGI-3 API."""
    api_key = os.getenv('ARC_API_KEY')
    
    if not api_key or api_key == 'your_api_key_here':
        print(" ARC_API_KEY not configured properly")
        return False
    
    # Test direct API connection
    url = "https://three.arcprize.org/api/games"
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }
    
    print(" Testing direct connection to ARC-AGI-3 API...")
    print(f"   URL: {url}")
    print(f"   API Key: {api_key[:8]}...{api_key[-4:]}")
    
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f" SUCCESS: Connected to real ARC-AGI-3 API")
                    print(f"   Response: {len(data)} games available")
                    if data:
                        print(f"   Sample game: {data[0].get('title', 'Unknown')} ({data[0].get('game_id', 'Unknown')})")
                    return True
                elif response.status == 401:
                    print(f" AUTHENTICATION FAILED: Invalid API key")
                    return False
                elif response.status == 429:
                    print(f" RATE LIMITED: Too many requests")
                    return True  # Still proves real API
                else:
                    print(f" API ERROR: Status {response.status}")
                    text = await response.text()
                    print(f"   Response: {text[:200]}")
                    return False
    except Exception as e:
        print(f" CONNECTION FAILED: {e}")
        return False

def check_code_for_real_api():
    """Check if the code is actually using real API endpoints."""
    print("\n Checking code for real API integration...")
    
    continuous_learning_file = Path("src/arc_integration/continuous_learning_loop.py")
    
    if not continuous_learning_file.exists():
        print(" continuous_learning_loop.py not found")
        return False
    
    with open(continuous_learning_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for real API indicators
    real_api_evidence = []
    
    if 'three.arcprize.org' in content:
        real_api_evidence.append(" Uses three.arcprize.org endpoint")
    
    if 'aiohttp' in content and 'session.post' in content:
        real_api_evidence.append(" Makes real HTTP requests with aiohttp")
    
    if 'X-API-Key' in content:
        real_api_evidence.append(" Uses API key authentication")
    
    if '/api/cmd/ACTION' in content:
        real_api_evidence.append(" Calls real ARC action endpoints")
    
    if '/api/cmd/RESET' in content:
        real_api_evidence.append(" Calls real ARC reset endpoints")
    
    if 'guid' in content.lower() and 'scorecard' in content.lower():
        real_api_evidence.append(" Uses real ARC session management")
    
    # Look for simulation indicators (but ignore harmless internal usage)
    simulation_evidence = []
    
    if 'mock_response' in content.lower():
        simulation_evidence.append(" Contains 'mock_response' references")
    
    if 'fake_api' in content.lower():
        simulation_evidence.append(" Contains 'fake_api' references")
    
    if 'simulate_arc' in content.lower():
        simulation_evidence.append(" Contains 'simulate_arc' references")
    
    if 'localhost' in content or '127.0.0.1' in content:
        simulation_evidence.append(" Contains localhost references")
    
    if 'unittest.mock' in content:
        simulation_evidence.append(" Contains unittest.mock references")
    
    print(f"\n Code Analysis Results:")
    print(f"   Real API Evidence: {len(real_api_evidence)}")
    for evidence in real_api_evidence:
        print(f"      {evidence}")
    
    if simulation_evidence:
        print(f"   Simulation Evidence: {len(simulation_evidence)}")
        for evidence in simulation_evidence:
            print(f"      {evidence}")
    else:
        print("   Simulation Evidence: 0 (Good!)")
    
    return len(real_api_evidence) >= 4 and len(simulation_evidence) == 0

async def main():
    print(" =============================================================")
    print(" SIMPLE ARC API REALITY CHECK")
    print(" =============================================================")
    
    # Test 1: Direct API call
    api_works = await simple_arc_api_test()
    
    # Test 2: Code analysis
    code_good = check_code_for_real_api()
    
    # Summary
    print(f"\n FINAL ASSESSMENT:")
    print(f"   Direct API Test: {' PASS' if api_works else ' FAIL'}")
    print(f"   Code Analysis: {' PASS' if code_good else ' FAIL'}")
    
    if api_works and code_good:
        print(f"\n CONCLUSION: ARC INTEGRATION IS REAL!")
        print(f"   The system connects to authentic ARC-AGI-3 servers.")
        print(f"   No simulation or mock behavior detected.")
        return True
    elif api_works or code_good:
        print(f"\n CONCLUSION: PARTIALLY VERIFIED")
        print(f"   Some evidence of real API usage, but issues detected.")
        return False
    else:
        print(f"\n CONCLUSION: ISSUES DETECTED")
        print(f"   System may not be using real ARC APIs.")
        return False

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
