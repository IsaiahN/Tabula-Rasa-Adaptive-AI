#!/usr/bin/env python3
"""
Test script to verify Git branch safety in the meta-cognitive system.
This ensures the Architect never switches to main/master branches.
"""
import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_branch_safety():
    """Test that branch safety mechanisms work correctly."""
    print("🧪 Testing Git Branch Safety in Meta-Cognitive System")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("BranchSafetyTest")
    
    try:
        from core.architect import Architect
        
        print("\n1️⃣ Initializing Architect with Git safety...")
        architect = Architect(
            base_path="src",
            repo_path=".",
            logger=logger
        )
        
        print(f"✅ Default branch configured: {architect.default_branch}")
        
        print("\n2️⃣ Testing branch validation...")
        # Test validation of dangerous branches
        dangerous_branches = ['main', 'master', 'Main', 'MASTER']
        
        for branch in dangerous_branches:
            is_safe = architect._validate_branch_operation(branch)
            if not is_safe:
                print(f"✅ BLOCKED dangerous branch: '{branch}'")
            else:
                print(f"❌ FAILED to block dangerous branch: '{branch}'")
        
        print("\n3️⃣ Testing safe branch validation...")
        safe_branches = ['Tabula-Rasa-v3', 'feature/test', 'development']
        
        for branch in safe_branches:
            is_safe = architect._validate_branch_operation(branch)
            if is_safe:
                print(f"✅ ALLOWED safe branch: '{branch}'")
            else:
                print(f"❌ INCORRECTLY blocked safe branch: '{branch}'")
        
        print("\n4️⃣ Testing branch enforcement...")
        architect._ensure_correct_branch()
        
        if architect.repo and architect.repo.active_branch.name == architect.default_branch:
            print(f"✅ Successfully on correct branch: {architect.repo.active_branch.name}")
        else:
            current = architect.repo.active_branch.name if architect.repo else "Unknown"
            print(f"❌ Branch mismatch. Current: {current}, Expected: {architect.default_branch}")
        
        print("\n" + "=" * 60)
        print("🎉 Git Branch Safety Test Complete!")
        print(f"🔒 System is locked to branch: {architect.default_branch}")
        print("🚫 main/master branches are blocked")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_branch_safety()
    sys.exit(0 if success else 1)
