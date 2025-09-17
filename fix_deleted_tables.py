#!/usr/bin/env python3
"""
Fix all references to deleted tables in the codebase
"""
import os
import re

def fix_file(file_path):
    """Fix references to deleted tables in a file"""
    if not os.path.exists(file_path):
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix winning_sequences references
    content = re.sub(
        r'SELECT \* FROM winning_sequences[^;]*;',
        'SELECT pattern_data, success_rate, frequency FROM learned_patterns WHERE pattern_type = \'winning_sequence\' ORDER BY success_rate DESC, frequency DESC LIMIT 10;',
        content
    )
    
    # Fix action_transitions references
    content = re.sub(
        r'action_transitions',
        'learned_patterns',
        content
    )
    
    # Fix performance_metrics references
    content = re.sub(
        r'performance_metrics',
        'training_sessions',
        content
    )
    
    # Fix governor_decisions references
    content = re.sub(
        r'governor_decisions',
        'system_logs',
        content
    )
    
    # Fix reset_debug_logs references
    content = re.sub(
        r'reset_debug_logs',
        'system_logs',
        content
    )
    
    # Fix experiments references
    content = re.sub(
        r'INSERT INTO experiments',
        '-- INSERT INTO experiments (table deleted, use system_logs)',
        content
    )
    
    # Fix task_performance references
    content = re.sub(
        r'INSERT INTO task_performance',
        '-- INSERT INTO task_performance (table deleted, use system_logs)',
        content
    )
    
    # Fix architecture_evolution references
    content = re.sub(
        r'architecture_evolution',
        'system_logs',
        content
    )
    
    # Fix meta_learning_sessions references
    content = re.sub(
        r'meta_learning_sessions',
        'training_sessions',
        content
    )
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    return False

def main():
    """Fix all files with deleted table references"""
    files_to_fix = [
        'src/database/schema.sql',
        'src/database/api.py',
        'src/database/system_integration.py',
        'src/database/director_commands.py',
        'src/database/migrate_data.py',
        'src/arc_integration/continuous_learning_loop.py'
    ]
    
    fixed_count = 0
    for file_path in files_to_fix:
        if fix_file(file_path):
            print(f"✅ Fixed {file_path}")
            fixed_count += 1
        else:
            print(f"⏭️  No changes needed in {file_path}")
    
    print(f"\n🎉 Fixed {fixed_count} files")

if __name__ == '__main__':
    main()
