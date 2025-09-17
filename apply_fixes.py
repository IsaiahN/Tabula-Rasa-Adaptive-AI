#!/usr/bin/env python3
"""
Script to apply memory leak fixes and code deduplication to existing files.

This script applies the fixes identified in the analysis to master_arc_trainer.py
and continuous_learning_loop.py to eliminate memory leaks and code duplication.
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime


def backup_file(file_path: Path) -> Path:
    """Create a backup of the original file."""
    backup_path = file_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    shutil.copy2(file_path, backup_path)
    print(f"✅ Backed up {file_path} to {backup_path}")
    return backup_path


def apply_master_trainer_fixes():
    """Apply fixes to master_arc_trainer.py."""
    file_path = Path("master_arc_trainer.py")
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    # Create backup
    backup_path = backup_file(file_path)
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply fixes
        fixes_applied = []
        
        # Fix 1: Replace duplicated ActionLimits with centralized import
        if "from action_limits_config import ActionLimits" in content:
            content = content.replace(
                "from action_limits_config import ActionLimits",
                "from src.config.centralized_config import action_limits as ActionLimits"
            )
            fixes_applied.append("Replaced ActionLimits import with centralized version")
        
        # Fix 2: Add memory leak fixes import
        if "from src.config.centralized_config import action_limits as ActionLimits" in content:
            import_line = "from src.config.centralized_config import action_limits as ActionLimits"
            memory_fixes_import = "from src.arc_integration.memory_leak_fixes import BoundedList, BoundedDict, apply_memory_leak_fixes"
            
            if memory_fixes_import not in content:
                content = content.replace(
                    import_line,
                    f"{import_line}\n{memory_fixes_import}"
                )
                fixes_applied.append("Added memory leak fixes import")
        
        # Fix 3: Add memory management to MasterARCTrainer.__init__
        if "self.performance_history = []" in content:
            content = content.replace(
                "self.performance_history = []",
                "self.performance_history = BoundedList(max_size=100)"
            )
            fixes_applied.append("Replaced performance_history with BoundedList")
        
        if "self.governor_decisions = []" in content:
            content = content.replace(
                "self.governor_decisions = []",
                "self.governor_decisions = BoundedList(max_size=100)"
            )
            fixes_applied.append("Replaced governor_decisions with BoundedList")
        
        if "self.architect_evolutions = []" in content:
            content = content.replace(
                "self.architect_evolutions = []",
                "self.architect_evolutions = BoundedList(max_size=100)"
            )
            fixes_applied.append("Replaced architect_evolutions with BoundedList")
        
        # Fix 4: Add memory cleanup to run_training method
        if "async def run_training(self):" in content and "apply_memory_leak_fixes(self)" not in content:
            content = content.replace(
                "async def run_training(self):",
                "async def run_training(self):\n        # Apply memory leak fixes\n        apply_memory_leak_fixes(self)"
            )
            fixes_applied.append("Added memory leak fixes to run_training method")
        
        # Write the fixed file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Applied {len(fixes_applied)} fixes to {file_path}")
        for fix in fixes_applied:
            print(f"   - {fix}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error applying fixes to {file_path}: {e}")
        # Restore backup
        shutil.copy2(backup_path, file_path)
        print(f"🔄 Restored original file from {backup_path}")
        return False


def apply_continuous_loop_fixes():
    """Apply fixes to continuous_learning_loop.py."""
    file_path = Path("src/arc_integration/continuous_learning_loop.py")
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    # Create backup
    backup_path = backup_file(file_path)
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply fixes
        fixes_applied = []
        
        # Fix 1: Replace duplicated ActionLimits with centralized import
        if "from action_limits_config import ActionLimits" in content:
            content = content.replace(
                "from action_limits_config import ActionLimits",
                "from src.config.centralized_config import action_limits as ActionLimits"
            )
            fixes_applied.append("Replaced ActionLimits import with centralized version")
        
        # Fix 2: Add memory leak fixes import
        if "from src.config.centralized_config import action_limits as ActionLimits" in content:
            import_line = "from src.config.centralized_config import action_limits as ActionLimits"
            memory_fixes_import = "from src.arc_integration.memory_leak_fixes import BoundedList, BoundedDict, apply_memory_leak_fixes"
            
            if memory_fixes_import not in content:
                content = content.replace(
                    import_line,
                    f"{import_line}\n{memory_fixes_import}"
                )
                fixes_applied.append("Added memory leak fixes import")
        
        # Fix 3: Add database integration import
        if "from src.arc_integration.memory_leak_fixes import" in content:
            db_import = "from src.database.memory_safe_operations import get_data_manager"
            if db_import not in content:
                content = content.replace(
                    "from src.arc_integration.memory_leak_fixes import BoundedList, BoundedDict, apply_memory_leak_fixes",
                    "from src.arc_integration.memory_leak_fixes import BoundedList, BoundedDict, apply_memory_leak_fixes\nfrom src.database.memory_safe_operations import get_data_manager"
                )
                fixes_applied.append("Added database integration import")
        
        # Fix 4: Replace performance_history with BoundedList
        if "self.performance_history = []" in content:
            content = content.replace(
                "self.performance_history = []",
                "self.performance_history = BoundedList(max_size=100)"
            )
            fixes_applied.append("Replaced performance_history with BoundedList")
        
        # Fix 5: Replace session_history with BoundedList
        if "self.session_history = []" in content:
            content = content.replace(
                "self.session_history = []",
                "self.session_history = BoundedList(max_size=100)"
            )
            fixes_applied.append("Replaced session_history with BoundedList")
        
        # Fix 6: Add memory cleanup to __init__
        if "def __init__(" in content and "apply_memory_leak_fixes(self)" not in content:
            # Find the end of __init__ method and add memory cleanup
            init_end = content.find("self._initialized = False")
            if init_end != -1:
                content = content[:init_end] + "        # Apply memory leak fixes\n        apply_memory_leak_fixes(self)\n        " + content[init_end:]
                fixes_applied.append("Added memory leak fixes to __init__ method")
        
        # Write the fixed file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Applied {len(fixes_applied)} fixes to {file_path}")
        for fix in fixes_applied:
            print(f"   - {fix}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error applying fixes to {file_path}: {e}")
        # Restore backup
        shutil.copy2(backup_path, file_path)
        print(f"🔄 Restored original file from {backup_path}")
        return False


def main():
    """Main function to apply all fixes."""
    print("🔧 Applying memory leak fixes and code deduplication...")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("master_arc_trainer.py").exists():
        print("❌ Please run this script from the project root directory")
        return False
    
    # Apply fixes
    success_count = 0
    
    print("\n📝 Applying fixes to master_arc_trainer.py...")
    if apply_master_trainer_fixes():
        success_count += 1
    
    print("\n📝 Applying fixes to continuous_learning_loop.py...")
    if apply_continuous_loop_fixes():
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"✅ Applied fixes to {success_count}/2 files")
    
    if success_count == 2:
        print("🎉 All fixes applied successfully!")
        print("\n📋 Next steps:")
        print("1. Run the tests: python -m pytest tests/test_memory_leak_fixes.py -v")
        print("2. Run the comprehensive tests: python -m pytest tests/test_comprehensive_fixes.py -v")
        print("3. Test the training system to ensure everything works correctly")
    else:
        print("⚠️ Some fixes failed. Check the error messages above.")
    
    return success_count == 2


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
