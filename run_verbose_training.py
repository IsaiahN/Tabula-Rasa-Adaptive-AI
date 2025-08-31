#!/usr/bin/env python3
"""
VERBOSE ARC-AGI-3 Training System

Shows detailed information during training:
- Individual moves and actions taken
- Memory file creation and management  
- Decay and consolidation operations
- Real-time system status
"""

import asyncio
import sys
import time
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from arc_integration.continuous_learning_loop import ContinuousLearningLoop
    from core.salience_system import SalienceMode
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Make sure you're in the tabula-rasa directory and src/ exists")
    sys.exit(1)

class VerboseTrainer:
    """Trainer with detailed logging and visibility into internal operations."""
    
    def __init__(self):
        # Configuration - non-SWARM, decay compression as requested
        self.swarm_enabled = False
        self.salience_mode_str = 'decay_compression'
        self.target_win_rate = 0.85
        self.target_avg_score = 75.0
        self.max_episodes_per_game = 10  # Reduced for verbose monitoring
        self.game_count = 2  # Start with fewer games for detailed observation
        
        self.continuous_loop = None
        self.training_iterations = 0
        self.memory_files_created = []
        
    def display_config(self):
        """Display current training configuration."""
        print("🎯 VERBOSE TRAINING CONFIGURATION")
        print("="*50)
        print(f"Training Mode: SEQUENTIAL (non-SWARM)")
        print(f"Salience Mode: {self.salience_mode_str.upper()}")
        print(f"Target Win Rate: {self.target_win_rate:.1%}")
        print(f"Max Episodes/Game: {self.max_episodes_per_game}")
        print(f"Game Count: {self.game_count} (reduced for verbose monitoring)")
        print(f"Verbose Logging: ENABLED")
        print()
        
    async def monitor_memory_files(self):
        """Monitor memory and checkpoint files being created."""
        workspace_path = Path("C:/Users/Admin/Documents/GitHub/tabula-rasa")
        
        # Check various memory locations
        memory_locations = [
            workspace_path / "checkpoints",
            workspace_path / "meta_learning_data", 
            workspace_path / "continuous_learning_data",
            workspace_path / "test_meta_learning_data"
        ]
        
        memory_info = {}
        
        for location in memory_locations:
            if location.exists():
                files = list(location.rglob("*"))
                memory_info[location.name] = {
                    'total_files': len(files),
                    'recent_files': []
                }
                
                # Get recent files (last 5)
                for file in sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)[:5]:
                    memory_info[location.name]['recent_files'].append({
                        'name': file.name,
                        'size': file.stat().st_size,
                        'modified': time.ctime(file.stat().st_mtime)
                    })
        
        return memory_info
    
    def print_memory_status(self, memory_info: dict):
        """Print detailed memory status."""
        print("\n💾 MEMORY & FILES STATUS")
        print("="*40)
        
        total_files = 0
        for location, info in memory_info.items():
            file_count = info['total_files']
            total_files += file_count
            print(f"{location}: {file_count} files")
            
            for file_info in info['recent_files'][:3]:  # Show top 3
                size_kb = file_info['size'] / 1024
                print(f"  📄 {file_info['name']} ({size_kb:.1f}KB)")
        
        print(f"📊 Total Memory Files: {total_files}")
        return total_files
    
    async def verbose_episode_with_monitoring(self, game_id: str, episode_num: int):
        """Run a single episode with detailed monitoring and logging."""
        
        print(f"\n🎮 VERBOSE EPISODE {episode_num} - {game_id}")
        print("="*60)
        
        # Pre-episode memory check
        print("📊 PRE-EPISODE MEMORY STATUS:")
        pre_memory = await self.monitor_memory_files()
        pre_file_count = self.print_memory_status(pre_memory)
        
        # Set up environment for verbose output
        env = os.environ.copy()
        env['ARC_API_KEY'] = "3405f9ba-f632-48e6-ac2b-73ed62056b24"
        env['DEBUG'] = 'True'  # Enable debug output
        env['VERBOSE'] = 'True'
        
        episode_start_time = time.time()
        actions_taken = 0
        max_actions = 200  # Reasonable limit for verbose monitoring
        
        print(f"🎯 Starting episode with max {max_actions} actions")
        print("🎮 ACTIONS & MOVES:")
        
        while actions_taken < max_actions:
            actions_taken += 1
            action_start_time = time.time()
            
            # Run single action with verbose output
            cmd = [
                'uv', 'run', 'main.py',
                '--agent=adaptivelearning',
                f'--game={game_id}'
            ]
            
            if actions_taken == 1:
                cmd.append('--reset')  # Reset at start of episode
                print(f"🔄 RESET game {game_id}")
            
            print(f"🎮 Action {actions_taken:3d}: ", end="", flush=True)
            
            try:
                # Create subprocess with verbose output
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd="C:/Users/Admin/Documents/GitHub/ARC-AGI-3-Agents",
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                # Wait for completion
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=30.0
                )
                
                stdout_text = stdout.decode() if stdout else ""
                stderr_text = stderr.decode() if stderr else ""
                
                # Parse results
                score = self._extract_score_from_output(stdout_text, stderr_text)
                state = self._extract_state_from_output(stdout_text, stderr_text)
                move_info = self._extract_move_info(stdout_text, stderr_text)
                
                action_duration = time.time() - action_start_time
                
                # Print action result
                print(f"Score:{score:2d} State:{state:12s} Time:{action_duration:.1f}s", end="")
                if move_info:
                    print(f" Move:{move_info}", end="")
                print()
                
                # Check for terminal states
                if state in ['WIN', 'GAME_OVER']:
                    print(f"🏁 Episode completed: {state} after {actions_taken} actions")
                    break
                
                # Monitor memory every 25 actions
                if actions_taken % 25 == 0:
                    print(f"\n📊 MEMORY CHECK after {actions_taken} actions:")
                    current_memory = await self.monitor_memory_files()
                    current_file_count = self.print_memory_status(current_memory)
                    
                    # Check if new files were created
                    new_files = current_file_count - pre_file_count
                    if new_files > 0:
                        print(f"✅ {new_files} new memory files created!")
                    else:
                        print("⚠️  No new memory files detected")
                    
                    # Check for decay operations
                    if self._check_for_decay_operations():
                        print("🔥 DECAY operations detected!")
                    
                    # Check for consolidation
                    if self._check_for_consolidation_operations():
                        print("🧠 MEMORY CONSOLIDATION detected!")
                    
                    print()
                
                # Small delay between actions
                await asyncio.sleep(0.1)
                
            except asyncio.TimeoutError:
                print(f"TIMEOUT after {action_duration:.1f}s")
                break
            except Exception as e:
                print(f"ERROR: {e}")
                break
        
        # Post-episode memory check
        episode_duration = time.time() - episode_start_time
        print(f"\n📊 POST-EPISODE MEMORY STATUS:")
        post_memory = await self.monitor_memory_files()
        post_file_count = self.print_memory_status(post_memory)
        
        # Summary
        files_created_this_episode = post_file_count - pre_file_count
        print(f"\n🎯 EPISODE {episode_num} SUMMARY:")
        print(f"Actions Taken: {actions_taken}")
        print(f"Duration: {episode_duration/60:.1f} minutes")
        print(f"Memory Files Created: {files_created_this_episode}")
        print(f"Final State: {state}")
        print(f"Final Score: {score}")
        
        return {
            'actions_taken': actions_taken,
            'final_state': state,
            'final_score': score,
            'duration': episode_duration,
            'memory_files_created': files_created_this_episode
        }
    
    def _extract_score_from_output(self, stdout: str, stderr: str) -> int:
        """Extract score from output."""
        # Look for score patterns in output
        import re
        combined = stdout + stderr
        score_patterns = [
            r'score[:\s]+(\d+)',
            r'Score[:\s]+(\d+)',
            r'SCORE[:\s]+(\d+)'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, combined, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return 0
    
    def _extract_state_from_output(self, stdout: str, stderr: str) -> str:
        """Extract game state from output."""
        import re
        combined = stdout + stderr
        state_pattern = r'\b(WIN|GAME_OVER|NOT_FINISHED|NOT_STARTED)\b'
        match = re.search(state_pattern, combined)
        return match.group(1) if match else 'UNKNOWN'
    
    def _extract_move_info(self, stdout: str, stderr: str) -> str:
        """Extract move/action information from output."""
        import re
        combined = stdout + stderr
        
        # Look for coordinate patterns
        coord_pattern = r'(\d+),\s*(\d+)'
        matches = re.findall(coord_pattern, combined)
        if matches:
            return f"({matches[-1][0]},{matches[-1][1]})"
        
        # Look for action patterns
        action_patterns = [r'action[:\s]+(\w+)', r'ACTION[:\s]+(\w+)']
        for pattern in action_patterns:
            match = re.search(pattern, combined, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return ""
    
    def _check_for_decay_operations(self) -> bool:
        """Check if decay operations are happening."""
        # This would check for decay-related log files or operations
        decay_indicators = [
            Path("continuous_learning_data"),
            Path("meta_learning_data")
        ]
        
        for path in decay_indicators:
            if path.exists():
                recent_files = [f for f in path.rglob("*") 
                               if f.stat().st_mtime > time.time() - 300]  # Last 5 minutes
                if recent_files:
                    return True
        return False
    
    def _check_for_consolidation_operations(self) -> bool:
        """Check if memory consolidation is happening."""
        # Check for consolidation files
        consolidation_paths = [
            Path("checkpoints"),
            Path("persistent_learning_state.json")
        ]
        
        for path in consolidation_paths:
            if path.exists():
                if path.is_file():
                    if path.stat().st_mtime > time.time() - 60:  # Last minute
                        return True
                else:
                    recent_files = [f for f in path.rglob("*.pt") 
                                   if f.stat().st_mtime > time.time() - 60]
                    if recent_files:
                        return True
        return False

async def main():
    """Run verbose training to see internal operations."""
    
    trainer = VerboseTrainer()
    trainer.display_config()
    
    print("🔍 VERBOSE TRAINING - MONITORING INTERNAL OPERATIONS")
    print("Will show: moves, memory files, decay, consolidation")
    print("="*60)
    
    # Run a few episodes with detailed monitoring
    try:
        for episode in range(1, 4):  # Just 3 episodes for detailed observation
            print(f"\n🚀 STARTING VERBOSE EPISODE {episode}")
            
            # Use a single game for detailed observation
            game_id = "lp85-2e205bbe3622"  # Use a known game
            
            episode_result = await trainer.verbose_episode_with_monitoring(game_id, episode)
            
            print(f"\n📋 Episode {episode} completed:")
            print(f"   Actions: {episode_result['actions_taken']}")
            print(f"   State: {episode_result['final_state']}")
            print(f"   Score: {episode_result['final_score']}")
            print(f"   Memory Files: {episode_result['memory_files_created']}")
            
            # Brief pause between episodes
            await asyncio.sleep(5)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Verbose training stopped")
    
    print(f"\n✅ Verbose training completed - you can see exactly what's happening!")

if __name__ == "__main__":
    print("🔬 VERBOSE ARC-AGI-3 TRAINING")
    print("Shows moves, memory operations, decay, consolidation")
    print("="*60)
    
    asyncio.run(main())
