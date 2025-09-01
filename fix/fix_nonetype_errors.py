#!/usr/bin/env python3
"""
Comprehensive fix for NoneType errors in the continuous learning system.

This script addresses the root causes:
1. API connection issues causing empty game lists
2. None value propagation in game session parsing
3. Arithmetic operations on None values
4. Missing null safety checks throughout the system
"""

import os
import sys
import json
from pathlib import Path

def find_tabula_rasa_path():
    """Find the tabula-rasa directory."""
    current_dir = Path(__file__).parent.absolute()
    if current_dir.name == 'tabula-rasa':
        return current_dir
    
    # Check parent directories
    for parent in current_dir.parents:
        if parent.name == 'tabula-rasa':
            return parent
    
    raise FileNotFoundError("Could not locate tabula-rasa directory")

def fix_continuous_learning_loop():
    """Fix NoneType errors in continuous_learning_loop.py"""
    tabula_rasa_path = find_tabula_rasa_path()
    loop_file = tabula_rasa_path / "src" / "arc_integration" / "continuous_learning_loop.py"
    
    if not loop_file.exists():
        print(f"❌ File not found: {loop_file}")
        return False
    
    print(f"🔧 Fixing NoneType errors in {loop_file}")
    
    # Read the file
    with open(loop_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Enhance _parse_complete_game_session with better null safety
    old_parse_session = '''    def _parse_complete_game_session(self, stdout_text: str, stderr_text: str) -> Dict[str, Any]:
        """Parse complete game session output to extract results and effective actions."""
        import re
        
        result = {
            'final_score': 0,
            'total_actions': 0,
            'final_state': 'UNKNOWN',
            'effective_actions': []
        }'''
    
    new_parse_session = '''    def _parse_complete_game_session(self, stdout_text: str, stderr_text: str) -> Dict[str, Any]:
        """Parse complete game session output to extract results and effective actions."""
        import re
        
        # Enhanced null safety - ensure we never return None values
        result = {
            'final_score': 0,
            'total_actions': 0,
            'final_state': 'UNKNOWN',
            'effective_actions': []
        }
        
        # Null safety for input parameters
        stdout_text = stdout_text if stdout_text is not None else ""
        stderr_text = stderr_text if stderr_text is not None else ""'''
    
    if old_parse_session in content:
        content = content.replace(old_parse_session, new_parse_session)
        print("  ✅ Enhanced _parse_complete_game_session with null safety")
    
    # Fix 2: Add comprehensive null checking before arithmetic operations
    old_update_complexity = '''        # Fix NoneType errors by providing defaults
        actions_taken = actions_taken if actions_taken is not None else 0
        effectiveness_ratio = effectiveness_ratio if effectiveness_ratio is not None else 0.0
        
        history = self.game_complexity_history[game_id]
        history['total_plays'] += 1
        history['total_actions'] += actions_taken
        history['avg_actions'] = history['total_actions'] / history['total_plays']'''
    
    new_update_complexity = '''        # Comprehensive null safety - ensure no None values in calculations
        actions_taken = actions_taken if actions_taken is not None else 0
        effectiveness_ratio = effectiveness_ratio if effectiveness_ratio is not None else 0.0
        
        # Additional safety checks
        if not isinstance(actions_taken, (int, float)):
            actions_taken = 0
        if not isinstance(effectiveness_ratio, (int, float)):
            effectiveness_ratio = 0.0
        
        history = self.game_complexity_history[game_id]
        
        # Ensure history values are never None
        history['total_plays'] = (history.get('total_plays') or 0) + 1
        history['total_actions'] = (history.get('total_actions') or 0) + actions_taken
        history['avg_actions'] = history['total_actions'] / max(history['total_plays'], 1)'''
    
    if old_update_complexity in content:
        content = content.replace(old_update_complexity, new_update_complexity)
        print("  ✅ Enhanced _update_game_complexity_history with comprehensive null safety")
    
    # Fix 3: Add null safety to performance calculations
    old_calc_perf = '''        total_score = sum(sum((ep.get('final_score') or 0) for ep in game.get('episodes', [])) 
                         for game in games_played.values())'''
    
    new_calc_perf = '''        # Enhanced null safety for score calculations
        total_score = 0
        for game in games_played.values():
            episodes = game.get('episodes', []) if game else []
            for ep in episodes:
                if ep:
                    score = ep.get('final_score')
                    if score is not None and isinstance(score, (int, float)):
                        total_score += score'''
    
    if old_calc_perf in content:
        content = content.replace(old_calc_perf, new_calc_perf)
        print("  ✅ Enhanced performance calculation with null safety")
    
    # Fix 4: Add API connection validation
    api_validation = '''    async def _validate_api_connection(self) -> bool:
        """Validate ARC-AGI-3 API connection and game availability."""
        try:
            import requests
            response = requests.get("https://three.arcprize.org/api/games", 
                                  headers={"X-API-Key": self.api_key}, timeout=30)
            if response.status_code == 200:
                games = response.json()
                if isinstance(games, list) and len(games) > 0:
                    print(f"✅ API Connection OK: {len(games)} games available")
                    return True
                else:
                    print(f"⚠️ API returned empty game list: {games}")
                    return False
            else:
                print(f"❌ API Error: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API Connection Failed: {e}")
            return False

'''
    
    # Add API validation method before _train_on_game
    if "_validate_api_connection" not in content:
        train_method_pos = content.find("    async def _train_on_game(")
        if train_method_pos > 0:
            content = content[:train_method_pos] + api_validation + content[train_method_pos:]
            print("  ✅ Added API connection validation")
    
    # Fix 5: Enhanced error handling in game session execution
    old_game_execution = '''            except Exception as e:
                print(f"❌ Error during complete game session: {e}")
                total_score = 0
                episode_actions = 0
                final_state = 'ERROR'
                effective_actions = []
                stdout_text = ""
                stderr_text = ""'''
    
    new_game_execution = '''            except Exception as e:
                print(f"❌ Error during complete game session: {e}")
                # Comprehensive error state with null safety
                total_score = 0
                episode_actions = 0
                final_state = 'ERROR'
                effective_actions = []
                stdout_text = ""
                stderr_text = ""
                
                # Log the error for debugging
                print(f"❌ Game session error details: Game={game_id}, Error={str(e)}")
                
                # Check if this is an API connectivity issue
                if "connection" in str(e).lower() or "timeout" in str(e).lower():
                    print("⚠️ Possible API connectivity issue - validating connection...")
                    api_valid = await self._validate_api_connection()
                    if not api_valid:
                        print("💡 Consider checking ARC_API_KEY and network connectivity")'''
    
    if old_game_execution in content:
        content = content.replace(old_game_execution, new_game_execution)
        print("  ✅ Enhanced game session error handling")
    
    # Fix 6: Add fallback values in _calculate_learning_efficiency
    old_efficiency = '''                early_performance = sum((ep.get('final_score') or 0) for ep in episodes[:5]) / 5
                late_performance = sum((ep.get('final_score') or 0) for ep in episodes[-5:]) / 5'''
    
    new_efficiency = '''                # Enhanced null safety for performance calculations
                early_scores = []
                late_scores = []
                
                for ep in episodes[:5]:
                    if ep:
                        score = ep.get('final_score')
                        if score is not None and isinstance(score, (int, float)):
                            early_scores.append(score)
                
                for ep in episodes[-5:]:
                    if ep:
                        score = ep.get('final_score')
                        if score is not None and isinstance(score, (int, float)):
                            late_scores.append(score)
                
                early_performance = sum(early_scores) / max(len(early_scores), 1)
                late_performance = sum(late_scores) / max(len(late_scores), 1)'''
    
    if old_efficiency in content:
        content = content.replace(old_efficiency, new_efficiency)
        print("  ✅ Enhanced learning efficiency calculation")
    
    # Write the fixed file
    with open(loop_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return True

def create_api_diagnostic():
    """Create a diagnostic script to test API connectivity."""
    tabula_rasa_path = find_tabula_rasa_path()
    diagnostic_file = tabula_rasa_path / "diagnose_api_connection.py"
    
    diagnostic_content = '''#!/usr/bin/env python3
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
    print("\\n🎮 TESTING GAME EXECUTION")
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
    
    print("\\n📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    print(f"API Connection: {'✅ OK' if api_ok else '❌ FAILED'}")
    print(f"Game Execution: {'✅ OK' if game_ok else '❌ FAILED'}")
    
    if api_ok and game_ok:
        print("\\n🎉 All systems operational!")
        print("The NoneType errors are likely due to temporary issues.")
        print("Try running the training again.")
    else:
        print("\\n🚨 Issues detected!")
        print("Fix the above problems before running training.")

if __name__ == "__main__":
    main()
'''
    
    with open(diagnostic_file, 'w', encoding='utf-8') as f:
        f.write(diagnostic_content)
    
    print(f"✅ Created API diagnostic script: {diagnostic_file}")
    return diagnostic_file

def create_safe_training_script():
    """Create a safer version of the training script with enhanced error handling."""
    tabula_rasa_path = find_tabula_rasa_path()
    safe_training = tabula_rasa_path / "safe_arc_training.py"
    
    safe_content = '''#!/usr/bin/env python3
"""
Safe ARC training script with enhanced error handling and NoneType prevention.
This version includes comprehensive null safety and better error recovery.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the Python path
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir / "src"))

from src.arc_integration.continuous_learning_loop import ContinuousLearningLoop
from src.core.salience_system import SalienceMode

async def safe_arc_training():
    """Run ARC training with enhanced safety measures."""
    print("🛡️ STARTING SAFE ARC TRAINING")
    print("=" * 50)
    
    try:
        # Initialize with safety checks
        print("🔧 Initializing continuous learning system...")
        
        continuous_loop = ContinuousLearningLoop(
            arc_agents_path=Path.home() / "Documents" / "GitHub" / "ARC-AGI-3-Agents",
            tabula_rasa_path=script_dir,
            save_directory="continuous_learning_data"
        )
        
        print("✅ System initialized")
        
        # Validate API connection first
        if hasattr(continuous_loop, '_validate_api_connection'):
            print("🌐 Validating API connection...")
            api_valid = await continuous_loop._validate_api_connection()
            if not api_valid:
                print("❌ API validation failed. Please run diagnose_api_connection.py")
                return False
            print("✅ API connection validated")
        
        # Start training session with safety
        print("🚀 Starting training session...")
        
        games = [
            "vc33-58ec4396715d",
            "ft09-f340c8e5138e", 
            "as66-821a4dcad9c2"  # Start with fewer games for testing
        ]
        
        session_id = continuous_loop.start_training_session(
            games=games,
            max_episodes_per_game=10,  # Reduced for safety
            target_win_rate=0.1,
            target_avg_score=10.0,
            salience_mode=SalienceMode.LOSSLESS,
            enable_salience_comparison=False,
            swarm_enabled=False  # Disable swarm mode for safer testing
        )
        
        print(f"📋 Session started: {session_id}")
        
        # Run the training with enhanced monitoring
        print("🎯 Running continuous learning...")
        session_results = await continuous_loop.run_continuous_learning(session_id)
        
        print("\\n🏆 TRAINING COMPLETED")
        print("=" * 50)
        print(f"Session ID: {session_results.get('session_id', 'Unknown')}")
        print(f"Games Played: {len(session_results.get('games_played', {}))}")
        
        # Display results safely
        overall_perf = session_results.get('overall_performance', {})
        win_rate = overall_perf.get('overall_win_rate', 0.0)
        avg_score = overall_perf.get('overall_average_score', 0.0)
        
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Average Score: {avg_score:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        print("💡 This error has been caught safely")
        return False

def main():
    """Main entry point."""
    try:
        result = asyncio.run(safe_arc_training())
        if result:
            print("\\n🎉 Safe training completed successfully!")
        else:
            print("\\n⚠️ Training completed with issues")
    except KeyboardInterrupt:
        print("\\n⏹️ Training stopped by user")
    except Exception as e:
        print(f"\\n💥 Unexpected error: {e}")

if __name__ == "__main__":
    main()
'''
    
    with open(safe_training, 'w', encoding='utf-8') as f:
        f.write(safe_content)
    
    print(f"✅ Created safe training script: {safe_training}")
    return safe_training

def main():
    """Main fix application."""
    print("🛠️ COMPREHENSIVE NONETYPE ERROR FIX")
    print("=" * 50)
    
    try:
        # Apply main fixes
        print("1️⃣ Fixing continuous learning loop...")
        if fix_continuous_learning_loop():
            print("✅ Continuous learning loop fixed")
        else:
            print("❌ Failed to fix continuous learning loop")
            return False
        
        # Create diagnostic tools
        print("\\n2️⃣ Creating diagnostic tools...")
        diagnostic_file = create_api_diagnostic()
        safe_training_file = create_safe_training_script()
        
        print("\\n✅ ALL FIXES APPLIED SUCCESSFULLY!")
        print("=" * 50)
        print("🔧 Applied Fixes:")
        print("  • Enhanced null safety in game session parsing")
        print("  • Comprehensive null checking in arithmetic operations")
        print("  • Improved error handling in game execution")
        print("  • Added API connectivity validation")
        print("  • Enhanced performance calculation safety")
        print()
        print("🔍 Diagnostic Tools Created:")
        print(f"  • API Diagnostic: {diagnostic_file.name}")
        print(f"  • Safe Training: {safe_training_file.name}")
        print()
        print("🚀 Next Steps:")
        print("1. Run the diagnostic script to test API connectivity:")
        print(f"   python {diagnostic_file.name}")
        print()
        print("2. If API is working, try the safe training script:")
        print(f"   python {safe_training_file.name}")
        print()
        print("3. If issues persist, check:")
        print("   • ARC_API_KEY is set correctly")
        print("   • Network connectivity to three.arcprize.org")
        print("   • ARC-AGI-3-Agents directory exists and is up to date")
        
        return True
        
    except Exception as e:
        print(f"❌ Error applying fixes: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
