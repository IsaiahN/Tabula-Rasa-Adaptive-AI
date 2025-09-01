#!/usr/bin/env python3
"""
Diagnostic script for ARC-AGI-3 API connectivity issues.
Run this to identify and fix API connection problems.
"""

import os
import sys
import json
import requests
from pathlib import Path

def load_env_vars():
    """Load environment variables from .env files."""
    env_vars = {}
    
    # Check for .env in ARC-AGI-3-Agents
    arc_agents_path = Path.home() / "Documents" / "GitHub" / "ARC-AGI-3-Agents"
    arc_env = arc_agents_path / ".env"
    
    if arc_env.exists():
        print(f"📁 Found .env file: {arc_env}")
        try:
            with open(arc_env, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip().strip('"')
            print(f"✅ Loaded {len(env_vars)} environment variables from .env")
        except Exception as e:
            print(f"❌ Error reading .env file: {e}")
    else:
        print(f"⚠️ No .env file found at: {arc_env}")
    
    return env_vars

def test_api_connection():
    """Test ARC-AGI-3 API connectivity."""
    print("🚀 TESTING ARC-AGI-3 API CONNECTION")
    print("=" * 50)
    
    # Get API key
    env_vars = load_env_vars()
    api_key = env_vars.get('ARC_API_KEY') or os.environ.get('ARC_API_KEY')
    
    if not api_key:
        print("❌ ARC_API_KEY not found!")
        print("💡 Solutions:")
        print("   1. Set ARC_API_KEY in your environment")
        print("   2. Create .env file in ARC-AGI-3-Agents directory")
        print("   3. Get API key from: https://three.arcprize.org")
        return False
    
    print(f"🔑 API Key found: {api_key[:8]}...{api_key[-4:]}")
    
    # Test connection
    try:
        print("🌐 Testing API connection...")
        url = "https://three.arcprize.org/api/games"
        headers = {
            "X-API-Key": api_key,
            "Accept": "application/json"
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        print(f"📡 Response status: {response.status_code}")
        
        if response.status_code == 200:
            games = response.json()
            print(f"✅ SUCCESS: {len(games)} games available")
            
            if len(games) > 0:
                print("🎮 Sample games:")
                for game in games[:3]:
                    print(f"   - {game.get('game_id', 'Unknown')}")
                return True
            else:
                print("⚠️ WARNING: Game list is empty")
                print("💡 This might be a temporary API issue")
                return False
        
        elif response.status_code == 401:
            print("❌ UNAUTHORIZED: Invalid API key")
            print("💡 Solutions:")
            print("   1. Verify API key is correct")
            print("   2. Check if key has expired")
            print("   3. Generate new key at: https://three.arcprize.org")
            return False
        
        elif response.status_code == 403:
            print("❌ FORBIDDEN: API access denied")
            print("💡 Check if your account has API access")
            return False
        
        else:
            print(f"❌ HTTP ERROR: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
    
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT: API request timed out")
        print("💡 Check your internet connection")
        return False
    
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR: Cannot reach API server")
        print("💡 Check your internet connection")
        return False
    
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        return False

def test_game_execution():
    """Test actual game execution."""
    print("\n🎮 TESTING GAME EXECUTION")
    print("=" * 50)
    
    arc_agents_path = Path.home() / "Documents" / "GitHub" / "ARC-AGI-3-Agents"
    if not arc_agents_path.exists():
        print(f"❌ ARC-AGI-3-Agents not found at: {arc_agents_path}")
        return False
    
    print(f"📁 ARC-AGI-3-Agents found: {arc_agents_path}")
    
    # Check for main.py
    main_py = arc_agents_path / "main.py"
    if not main_py.exists():
        print(f"❌ main.py not found at: {main_py}")
        return False
    
    print("✅ main.py found")
    
    # Test help command
    try:
        import subprocess
        result = subprocess.run(
            ["python", "main.py", "--help"],
            cwd=str(arc_agents_path),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ main.py --help works")
            if "adaptivelearning" in result.stdout:
                print("✅ adaptivelearning agent available")
                return True
            else:
                print("⚠️ adaptivelearning agent not found in help")
                print("Available agents in output:")
                print(result.stdout[:300])
                return False
        else:
            print(f"❌ main.py --help failed: {result.stderr}")
            return False
    
    except subprocess.TimeoutExpired:
        print("❌ main.py --help timed out")
        return False
    
    except Exception as e:
        print(f"❌ Error testing main.py: {e}")
        return False

def main():
    """Run all diagnostics."""
    print("🔍 ARC-AGI-3 API DIAGNOSTIC TOOL")
    print("=" * 50)
    
    api_ok = test_api_connection()
    game_ok = test_game_execution()
    
    print("\n📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    print(f"API Connection: {'✅ OK' if api_ok else '❌ FAILED'}")
    print(f"Game Execution: {'✅ OK' if game_ok else '❌ FAILED'}")
    
    if api_ok and game_ok:
        print("\n🎉 All systems operational!")
        print("The NoneType errors are likely due to temporary issues.")
        print("Try running the training again.")
    else:
        print("\n🚨 Issues detected!")
        print("Fix the above problems before running training.")

if __name__ == "__main__":
    main()
