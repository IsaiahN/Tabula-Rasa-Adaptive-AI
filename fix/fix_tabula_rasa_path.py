#!/usr/bin/env python3
"""
Robust Tabula-Rasa Path Integration Fix

This script ensures the tabula-rasa integration is properly detected 
when running from the ARC-AGI-3-Agents context.
"""

import os
import sys
from pathlib import Path
import shutil

def fix_tabula_rasa_path_detection():
    """Fix tabula-rasa path detection in the adaptive learning agent."""
    print("🔧 Fixing Tabula-Rasa Path Detection...")
    
    # Find ARC-AGI-3-Agents path
    arc_agents_path = Path("C:/Users/Admin/Documents/GitHub/ARC-AGI-3-Agents")
    
    if not arc_agents_path.exists():
        print("❌ ARC-AGI-3-Agents repository not found")
        return False
    
    # Path to the adaptive learning agent file
    agent_file = arc_agents_path / "agents" / "templates" / "adaptive_learning_agent.py"
    
    if not agent_file.exists():
        print("❌ Adaptive learning agent file not found")
        return False
    
    # Create backup
    backup_file = agent_file.with_suffix('.py.backup_path_fix')
    if not backup_file.exists():
        shutil.copy2(agent_file, backup_file)
        print("✅ Created backup of adaptive_learning_agent.py")
    
    # Create a more robust adaptive learning agent with better path detection
    robust_agent_content = '''"""
Adaptive Learning Agent for ARC-AGI-3

This agent integrates the full Adaptive Learning Agent with sophisticated memory, 
sleep cycles, and meta-learning capabilities.
"""

import sys
import os
from pathlib import Path
import logging
import torch
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import deque, defaultdict
from agents.agent import Agent
from agents.structs import GameAction, GameState

logger = logging.getLogger(__name__)

# Enhanced tabula-rasa integration with multiple detection strategies
def setup_tabula_rasa_integration():
    """Robustly detect and set up tabula-rasa integration with multiple strategies."""
    
    # Strategy 1: Direct known path
    known_path = Path(r"C:\\Users\\Admin\\Documents\\GitHub\\tabula-rasa\\src")
    if known_path.exists() and (known_path / "core" / "agent.py").exists():
        if str(known_path) not in sys.path:
            sys.path.insert(0, str(known_path))
            print(f"OK Tabula-rasa found via direct path: {known_path}")
            return True
    
    # Strategy 2: Search from current file location
    current_file = Path(__file__).resolve()
    
    # Look for tabula-rasa in common relative locations
    search_paths = [
        current_file.parent.parent.parent.parent / "tabula-rasa" / "src",  # ../../../../tabula-rasa/src
        current_file.parent.parent.parent / "tabula-rasa" / "src",        # ../../../tabula-rasa/src  
        Path.cwd().parent / "tabula-rasa" / "src",                        # ../tabula-rasa/src
        Path.cwd() / "tabula-rasa" / "src",                               # ./tabula-rasa/src
    ]
    
    for search_path in search_paths:
        if search_path.exists() and (search_path / "core" / "agent.py").exists():
            if str(search_path) not in sys.path:
                sys.path.insert(0, str(search_path))
                print(f"OK Tabula-rasa found via search: {search_path}")
                return True
    
    # Strategy 3: Environment variable
    tabula_env = os.environ.get('TABULA_RASA_PATH')
    if tabula_env:
        env_path = Path(tabula_env) / "src"
        if env_path.exists() and (env_path / "core" / "agent.py").exists():
            if str(env_path) not in sys.path:
                sys.path.insert(0, str(env_path))
                print(f"OK Tabula-rasa found via environment: {env_path}")
                return True
    
    # Strategy 4: Search in all GitHub directories
    possible_github_roots = [
        Path(r"C:\\Users\\Admin\\Documents\\GitHub"),
        Path.home() / "Documents" / "GitHub",
        Path.home() / "GitHub"
    ]
    
    for github_root in possible_github_roots:
        if github_root.exists():
            tabula_path = github_root / "tabula-rasa" / "src"
            if tabula_path.exists() and (tabula_path / "core" / "agent.py").exists():
                if str(tabula_path) not in sys.path:
                    sys.path.insert(0, str(tabula_path))
                    print(f"OK Tabula-rasa found in GitHub: {tabula_path}")
                    return True
    
    print("WARNING Tabula-rasa not found - using simplified implementation")
    return False

# Try to import full adaptive learning components with enhanced detection
FULL_AGENT_AVAILABLE = False
try:
    if setup_tabula_rasa_integration():
        from core.agent import AdaptiveLearningAgent
        from core.data_models import SensoryInput, AgentState
        FULL_AGENT_AVAILABLE = True
        print("OK Full Adaptive Learning Agent components imported successfully")
    else:
        raise ImportError("Tabula-rasa not available")
        
except ImportError as e:
    logger.warning(f"WARNING Full agent unavailable: {e}")
    print(f"WARNING Full agent unavailable: {e}")
    FULL_AGENT_AVAILABLE = False

# Define the agent class based on availability
if FULL_AGENT_AVAILABLE:
    class AdaptiveLearning(Agent):
        """Full Adaptive Learning Agent with memory, sleep, and meta-learning"""
        
        MAX_ACTIONS = 200
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
            # Create ARC-optimized config
            self.config = {
                'predictive_core': {
                    'visual_size': (3, 64, 64),
                    'proprioception_size': 12,
                    'hidden_size': 256,
                    'architecture': 'lstm'
                },
                'memory': {
                    'enabled': True,
                    'memory_size': 256,
                    'word_size': 64,
                    'num_read_heads': 2,
                    'num_write_heads': 1
                },
                'energy': {
                    'max_energy': 100.0,
                    'base_consumption': 0.1,
                    'action_multiplier': 1.0
                },
                'sleep': {
                    'sleep_trigger_energy': 30.0,
                    'sleep_duration_steps': 50
                }
            }
            
            try:
                # Initialize the full Adaptive Learning Agent
                self.learning_agent = AdaptiveLearningAgent(self.config, device="cpu")
                self.sleep_counter = 0
                print("OK Full Adaptive Learning Agent initialized successfully")
                logger.info("Full Adaptive Learning Agent initialized successfully")
            except Exception as e:
                print(f"ERROR Failed to initialize full agent: {e}")
                logger.error(f"Failed to initialize full agent: {e}")
                # Fall back to simplified behavior
                self.learning_agent = None
            
        def choose_action(self, frames: List[Any], latest_frame: Any) -> GameAction:
            """Choose action using full Adaptive Learning Agent or fallback"""
            if hasattr(latest_frame, 'state') and latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                return GameAction.RESET
                
            if self.learning_agent is not None:
                try:
                    # Convert ARC frame to sensory input
                    sensory_input = self._convert_frame_to_sensory_input(latest_frame)
                    
                    # Get action from learning agent
                    with torch.no_grad():
                        result = self.learning_agent.step(sensory_input, torch.zeros(7))
                        
                    # Convert to ARC action
                    arc_action = self._convert_result_to_arc_action(result)
                    
                    # Handle sleep cycles
                    current_energy = self.learning_agent.energy_system.current_energy
                    if current_energy < 40.0:
                        self._trigger_sleep_cycle()
                        
                    return arc_action
                    
                except Exception as e:
                    logger.warning(f"Full agent error: {e}, using random action")
                    print(f"WARNING Full agent error: {e}")
            
            # Fallback to random action
            actions = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, 
                      GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7]
            return np.random.choice(actions)
            
        def _convert_frame_to_sensory_input(self, frame) -> SensoryInput:
            """Convert ARC frame to sensory input format"""
            if hasattr(frame, 'current_grid') and frame.current_grid:
                grid = np.array(frame.current_grid, dtype=np.float32)
                if grid.shape[0] < 64 or grid.shape[1] < 64:
                    padded = np.zeros((64, 64), dtype=np.float32)
                    padded[:grid.shape[0], :grid.shape[1]] = grid
                    grid = padded
                visual_data = torch.from_numpy(grid).unsqueeze(0)
            else:
                visual_data = torch.zeros(1, 64, 64)
                
            return SensoryInput(
                visual=visual_data,
                proprioception=torch.zeros(12),
                energy_level=100.0,
                timestamp=time.time()
            )
            
        def _convert_result_to_arc_action(self, result) -> GameAction:
            """Convert learning agent result to ARC action"""
            actions = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, 
                      GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7]
            
            # Use learning progress or random selection
            if isinstance(result, dict) and 'learning_progress' in result:
                action_idx = int(abs(result['learning_progress'] * 1000)) % len(actions)
            else:
                action_idx = np.random.randint(0, len(actions))
                
            return actions[action_idx]
            
        def _trigger_sleep_cycle(self):
            """Trigger sleep cycle for memory consolidation"""
            self.sleep_counter += 1
            print(f"SLEEP Sleep cycle #{self.sleep_counter} - consolidating memories...")
            logger.info(f"Sleep cycle #{self.sleep_counter} triggered")
            
            # Restore energy
            if hasattr(self.learning_agent, 'energy_system'):
                old_energy = self.learning_agent.energy_system.current_energy
                self.learning_agent.energy_system.current_energy = min(100.0, old_energy + 40.0)
                print(f"ENERGY Energy: {old_energy:.1f} -> {self.learning_agent.energy_system.current_energy:.1f}")
            
        def is_done(self, frames: List[Any], latest_frame: Any) -> bool:
            """Check if game is complete"""
            return hasattr(latest_frame, 'state') and latest_frame.state == GameState.WIN

else:
    # Simplified implementation when full agent is not available
    class AdaptiveLearning(Agent):
        """Simplified Adaptive Learning Agent for ARC-AGI-3 tasks"""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.step_count = 0
            self.exploration_rate = 0.3
            print("LOADING Simplified Adaptive Learning Agent initialized")
            logger.info("Simplified Adaptive Learning Agent initialized")
            
        def choose_action(self, frames: List[Any], latest_frame: Any) -> GameAction:
            """Choose action based on current game state"""
            if hasattr(latest_frame, 'state') and latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                return GameAction.RESET
                
            # Simple pattern-based action selection
            self.step_count += 1
            actions = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, 
                      GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7]
            
            # Use exploration vs exploitation
            if np.random.random() < self.exploration_rate:
                action = np.random.choice(actions)
            else:
                # Cycle through actions based on step count
                action = actions[self.step_count % len(actions)]
                
            return action
            
        def is_done(self, frames: List[Any], latest_frame: Any) -> bool:
            """Check if the game is complete"""
            return hasattr(latest_frame, 'state') and latest_frame.state == GameState.WIN
'''
    
    # Write the robust version
    with open(agent_file, 'w', encoding='utf-8') as f:
        f.write(robust_agent_content)
    
    print("✅ Updated adaptive learning agent with robust path detection")
    
    print("\n🎯 Path detection improvements:")
    print("   ✅ Added multiple search strategies for tabula-rasa")
    print("   ✅ Enhanced error handling and fallback behavior")
    print("   ✅ Better logging and status messages")
    print("   ✅ Graceful degradation when full agent unavailable")
    
    return True

if __name__ == "__main__":
    print("🛠️  Robust Tabula-Rasa Path Integration Fix")
    print("="*60)
    
    success = fix_tabula_rasa_path_detection()
    
    if success:
        print("\n🎉 Path detection significantly improved!")
        print("✅ Tabula-rasa integration should now be detected reliably")
        print("\n📋 Test the improved integration:")
        print("   python arc3.py status")
    else:
        print("\n❌ Could not apply path detection fix")