#!/usr/bin/env python3
"""
Comprehensive fix for import issues in the continuous learning system.

This script fixes all relative import issues throughout the codebase.
"""

import os
import sys
from pathlib import Path
import re

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

def fix_imports_in_file(file_path: Path, fixes: list):
    """Apply import fixes to a specific file."""
    if not file_path.exists():
        print(f"⚠️  File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = 0
        
        for old_import, new_import in fixes:
            if old_import in content:
                content = content.replace(old_import, new_import)
                changes_made += 1
                print(f"  ✅ Fixed: {old_import} -> {new_import}")
        
        if changes_made > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  📝 Applied {changes_made} fixes to {file_path.name}")
            return True
        else:
            print(f"  ℹ️  No changes needed in {file_path.name}")
            return False
    
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False

def main():
    """Main fix application."""
    print("🔧 COMPREHENSIVE IMPORT FIXES")
    print("=" * 50)
    
    try:
        tabula_rasa_path = find_tabula_rasa_path()
        src_path = tabula_rasa_path / "src"
        
        # Define all the import fixes needed
        import_fixes = [
            # Core module fixes
            {
                'file': src_path / "core" / "predictive_core.py",
                'fixes': [
                    ("from memory.dnc import DNCMemory", "from ..memory.dnc import DNCMemory")
                ]
            },
            {
                'file': src_path / "core" / "agent.py",
                'fixes': [
                    ("from memory.dnc import DNCMemory", "from ..memory.dnc import DNCMemory"),
                    ("from dnc import DNCMemory", "from ..memory.dnc import DNCMemory")
                ]
            },
            {
                'file': src_path / "core" / "sleep_system.py",
                'fixes': [
                    ("from memory.", "from ..memory."),
                    ("from goals.", "from ..goals."),
                    ("from monitoring.", "from ..monitoring.")
                ]
            },
            {
                'file': src_path / "core" / "salience_system.py",
                'fixes': [
                    ("from memory.", "from ..memory."),
                    ("from utils.", "from ..utils.")
                ]
            },
            {
                'file': src_path / "core" / "meta_learning.py",
                'fixes': [
                    ("from memory.", "from ..memory."),
                    ("from utils.", "from ..utils.")
                ]
            },
            # ARC integration fixes
            {
                'file': src_path / "arc_integration" / "continuous_learning_loop.py",
                'fixes': [
                    ("from arc_integration.arc_meta_learning import ARCMetaLearningSystem", "from .arc_meta_learning import ARCMetaLearningSystem"),
                    ("from core.meta_learning import MetaLearningSystem", "from ..core.meta_learning import MetaLearningSystem"),
                    ("from core.salience_system import", "from ..core.salience_system import"),
                    ("from core.sleep_system import SleepCycle", "from ..core.sleep_system import SleepCycle"),
                    ("from core.agent import AdaptiveLearningAgent", "from ..core.agent import AdaptiveLearningAgent"),
                    ("from core.predictive_core import PredictiveCore", "from ..core.predictive_core import PredictiveCore"),
                    ("from goals.goal_system import", "from ..goals.goal_system import"),
                    ("from core.energy_system import EnergySystem", "from ..core.energy_system import EnergySystem"),
                    ("from memory.dnc import DNCMemory", "from ..memory.dnc import DNCMemory")
                ]
            },
            {
                'file': src_path / "arc_integration" / "arc_meta_learning.py",
                'fixes': [
                    ("from core.meta_learning import MetaLearningSystem", "from ..core.meta_learning import MetaLearningSystem")
                ]
            },
            {
                'file': src_path / "arc_integration" / "arc_agent_adapter.py",
                'fixes': [
                    ("from core.agent import AdaptiveLearningAgent", "from ..core.agent import AdaptiveLearningAgent"),
                    ("from core.data_models import", "from ..core.data_models import"),
                    ("from core.meta_learning import MetaLearningSystem", "from ..core.meta_learning import MetaLearningSystem")
                ]
            },
            # Goals system fixes
            {
                'file': src_path / "goals" / "goal_system.py",
                'fixes': [
                    ("from core.", "from ..core."),
                    ("from memory.", "from ..memory."),
                    ("from utils.", "from ..utils.")
                ]
            },
            # Memory system fixes
            {
                'file': src_path / "memory" / "dnc.py",
                'fixes': [
                    ("from utils.", "from ..utils.")
                ]
            }
        ]
        
        # Apply all fixes
        total_changes = 0
        for fix_config in import_fixes:
            file_path = fix_config['file']
            fixes = fix_config['fixes']
            
            print(f"\\n🔧 Processing: {file_path.relative_to(tabula_rasa_path)}")
            
            if fix_imports_in_file(file_path, fixes):
                total_changes += 1
        
        print(f"\\n✅ IMPORT FIXES COMPLETE!")
        print(f"📊 Modified {total_changes} files")
        
        # Test the imports
        print("\\n🧪 TESTING IMPORTS...")
        test_result = test_imports(src_path)
        
        if test_result:
            print("✅ All imports working correctly!")
        else:
            print("⚠️  Some imports may still have issues")
        
        return True
        
    except Exception as e:
        print(f"❌ Error applying fixes: {e}")
        return False

def test_imports(src_path: Path):
    """Test that key imports work after fixes."""
    try:
        # Add src to path for testing
        import sys
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        # Test key imports
        print("  Testing core imports...")
        from src.core.agent import AdaptiveLearningAgent
        print("    ✅ AdaptiveLearningAgent")
        
        from src.core.predictive_core import PredictiveCore
        print("    ✅ PredictiveCore")
        
        print("  Testing ARC integration imports...")
        from src.arc_integration.continuous_learning_loop import ContinuousLearningLoop
        print("    ✅ ContinuousLearningLoop")
        
        from src.arc_integration.arc_meta_learning import ARCMetaLearningSystem
        print("    ✅ ARCMetaLearningSystem")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Import test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
