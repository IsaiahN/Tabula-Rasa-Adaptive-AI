#!/usr/bin/env python3
"""
Setup script to configure ARC-3 integration

This script sets up the proper import paths and configurations
to enable seamless integration between tabula-rasa and ARC-AGI-3-Agents.
"""

import os
import sys
from pathlib import Path
import shutil

def setup_arc3_integration():
    """Set up ARC-3 integration by configuring import paths."""
    print("🔧 Setting up ARC-3 Integration...")
    
    # Find ARC-AGI-3-Agents path
    arc_agents_path = None
    possible_paths = [
        Path("C:/Users/Admin/Documents/GitHub/ARC-AGI-3-Agents"),
        Path.cwd().parent / "ARC-AGI-3-Agents",
        Path.cwd() / "ARC-AGI-3-Agents"
    ]
    
    for path in possible_paths:
        if path.exists() and (path / "main.py").exists():
            arc_agents_path = path
            break
    
    if not arc_agents_path:
        print("❌ ARC-AGI-3-Agents repository not found")
        print("💡 Please clone it first:")
        print("   git clone https://github.com/arc-prize/ARC-AGI-3-Agents")
        return False
    
    print(f"✅ Found ARC-AGI-3-Agents at: {arc_agents_path}")
    
    # Create a simple import helper script for the adaptive learning agent
    adaptive_agent_path = arc_agents_path / "agents" / "templates" / "adaptive_learning_agent.py"
    
    if adaptive_agent_path.exists():
        print("✅ Adaptive learning agent found")
        
        # Create a backup of the original
        backup_path = adaptive_agent_path.with_suffix('.py.backup')
        if not backup_path.exists():
            shutil.copy2(adaptive_agent_path, backup_path)
            print("✅ Created backup of original agent")
        
        # Update the import section to be more robust
        tabula_rasa_src = Path.cwd() / "src"
        
        import_fix = f'''
# Fixed import section for tabula-rasa integration
import sys
import os
from pathlib import Path

# Add tabula-rasa src to path
TABULA_RASA_SRC = Path(r"{tabula_rasa_src}")
if TABULA_RASA_SRC.exists() and str(TABULA_RASA_SRC) not in sys.path:
    sys.path.insert(0, str(TABULA_RASA_SRC))
    print(f"Added tabula-rasa src to path: {{TABULA_RASA_SRC}}")
'''
        
        # Read the current file
        with open(adaptive_agent_path, 'r') as f:
            content = f.read()
        
        # Check if our fix is already applied
        if "TABULA_RASA_SRC" not in content:
            # Find where to insert the import fix (after the initial imports)
            lines = content.split('\n')
            insert_index = 0
            
            # Find the end of the initial imports
            for i, line in enumerate(lines):
                if line.startswith('# Add tabula-rasa src to path'):
                    insert_index = i
                    break
                elif 'from agents.structs import' in line:
                    insert_index = i + 1
                    break
            
            if insert_index > 0:
                # Insert the import fix
                lines.insert(insert_index, import_fix)
                
                # Write back the updated file
                with open(adaptive_agent_path, 'w') as f:
                    f.write('\n'.join(lines))
                
                print("✅ Updated adaptive learning agent with import fix")
            else:
                print("⚠️  Could not automatically update agent - manual fix needed")
        else:
            print("✅ Import fix already applied")
    
    # Update the .env file to include the ARC_AGENTS_PATH
    env_file = Path.cwd() / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            env_content = f.read()
        
        if "ARC_AGENTS_PATH" not in env_content:
            with open(env_file, 'a') as f:
                f.write(f'\n# ARC-AGI-3-Agents path (added by setup)\nARC_AGENTS_PATH={arc_agents_path}\n')
            print("✅ Added ARC_AGENTS_PATH to .env file")
        else:
            print("✅ ARC_AGENTS_PATH already in .env file")
    
    print("\n🎉 ARC-3 Integration Setup Complete!")
    print("📋 Next steps:")
    print("   1. Test connection: python arc3.py status")
    print("   2. Run demo: python arc3.py demo")
    print("   3. Check scoreboard: https://arcprize.org/leaderboard")
    
    return True

if __name__ == "__main__":
    setup_arc3_integration()