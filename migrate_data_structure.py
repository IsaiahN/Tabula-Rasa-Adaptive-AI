#!/usr/bin/env python3
"""
Data Structure Migration Script for Tabula Rasa

This script:
1. Migrates files from continuous_learning_data to the new organized structure
2. Updates all code references to use the new paths
3. Creates backups and validates the migration
"""

import os
import sys
import shutil
import json
from pathlib import Path
import re
from typing import Dict, List, Tuple
import logging
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from config.data_paths import DataPaths, FilePaths, get_migrated_path
except ImportError:
    print("Error: Could not import data_paths. Make sure src/config/data_paths.py exists.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataMigrator:
    """Handles migration of continuous_learning_data to new organized structure."""
    
    def __init__(self):
        self.old_base = Path("continuous_learning_data")
        self.new_base = Path("data")
        self.backup_dir = Path(f"backup_continuous_learning_data_{int(datetime.now().timestamp())}")
        self.dry_run = False
        
    def create_backup(self):
        """Create a backup of the current continuous_learning_data directory."""
        if self.old_base.exists():
            logger.info(f"Creating backup: {self.backup_dir}")
            shutil.copytree(self.old_base, self.backup_dir)
            logger.info("Backup completed successfully")
        else:
            logger.warning("No continuous_learning_data directory found to backup")
    
    def categorize_files(self) -> Dict[str, List[Path]]:
        """Categorize files based on their content and purpose."""
        if not self.old_base.exists():
            return {}
            
        categories = {
            'training_sessions': [],
            'training_results': [],
            'training_intelligence': [],
            'training_meta_learning': [],
            'logs_training': [],
            'logs_governor': [],
            'logs_system': [],
            'memory_backups': [],
            'memory_patterns': [],
            'architecture_evolution': [],
            'architecture_mutations': [],
            'experiments_research': [],
            'experiments_phase0': [],
            'experiments_evaluations': [],
            'config_states': [],
            'config_counters': [],
            'other': []\n        }\n        
        # Walk through all files in the old directory\n        for root, dirs, files in os.walk(self.old_base):\n            root_path = Path(root)\n            \n            for file in files:\n                file_path = root_path / file\n                relative_path = file_path.relative_to(self.old_base)\n                filename = file.lower()\n                \n                # Categorize based on directory and filename patterns\n                if 'sessions' in str(relative_path).lower():\n                    categories['training_sessions'].append(file_path)\n                elif 'session_session_' in filename:\n                    categories['training_sessions'].append(file_path)\n                elif 'action_intelligence' in filename:\n                    categories['training_intelligence'].append(file_path)\n                elif 'meta_learning_session' in filename:\n                    categories['training_meta_learning'].append(file_path)\n                elif filename.endswith('.log'):\n                    if 'governor' in filename:\n                        categories['logs_governor'].append(file_path)\n                    elif 'master_arc_trainer_output' in filename:\n                        categories['logs_training'].append(file_path)\n                    elif 'master_arc_trainer_error' in filename:\n                        categories['logs_training'].append(file_path)\n                    elif any(x in filename for x in ['master_arc_training', 'meta_cognitive']):\n                        categories['logs_system'].append(file_path)\n                    else:\n                        categories['logs_training'].append(file_path)\n                elif 'backup' in filename or 'backups' in str(relative_path).lower():\n                    categories['memory_backups'].append(file_path)\n                elif filename.endswith('.pkl') or 'pattern' in filename:\n                    categories['memory_patterns'].append(file_path)\n                elif 'architect' in filename or 'evolution' in filename:\n                    categories['architecture_evolution'].append(file_path)\n                elif 'mutation' in filename:\n                    categories['architecture_mutations'].append(file_path)\n                elif 'research' in filename:\n                    categories['experiments_research'].append(file_path)\n                elif 'phase0' in filename or 'phase0' in str(relative_path).lower():\n                    categories['experiments_phase0'].append(file_path)\n                elif 'evaluation' in filename or 'agi_evaluation' in str(relative_path).lower():\n                    categories['experiments_evaluations'].append(file_path)\n                elif 'state' in filename or 'counters' in filename:\n                    if 'global_counters' in filename:\n                        categories['config_counters'].append(file_path)\n                    else:\n                        categories['config_states'].append(file_path)\n                elif filename.endswith('.json') and any(x in filename for x in ['results', 'training']):\n                    categories['training_results'].append(file_path)\n                else:\n                    categories['other'].append(file_path)\n        \n        return categories\n    \n    def migrate_files(self, categories: Dict[str, List[Path]]):\n        """Migrate files to their new locations.""" \n        destination_mapping = {\n            'training_sessions': DataPaths.TRAINING_SESSIONS,\n            'training_results': DataPaths.TRAINING_RESULTS,\n            'training_intelligence': DataPaths.TRAINING_INTELLIGENCE,\n            'training_meta_learning': DataPaths.TRAINING_META_LEARNING,\n            'logs_training': DataPaths.LOGS_TRAINING,\n            'logs_governor': DataPaths.LOGS_GOVERNOR, \n            'logs_system': DataPaths.LOGS_SYSTEM,\n            'memory_backups': DataPaths.MEMORY_BACKUPS,\n            'memory_patterns': DataPaths.MEMORY_PATTERNS,\n            'architecture_evolution': DataPaths.ARCHITECTURE_EVOLUTION,\n            'architecture_mutations': DataPaths.ARCHITECTURE_MUTATIONS,\n            'experiments_research': DataPaths.EXPERIMENTS_RESEARCH,\n            'experiments_phase0': DataPaths.EXPERIMENTS_PHASE0,\n            'experiments_evaluations': DataPaths.EXPERIMENTS_EVALUATIONS,\n            'config_states': DataPaths.CONFIG_STATES,\n            'config_counters': DataPaths.CONFIG_COUNTERS\n        }\n        \n        total_files = sum(len(files) for files in categories.values())\n        processed = 0\n        \n        logger.info(f\"Starting migration of {total_files} files...\")\n        \n        for category, files in categories.items():\n            if not files:\n                continue\n                \n            dest_dir = destination_mapping.get(category, DataPaths.BASE)\n            logger.info(f\"Migrating {len(files)} {category} files to {dest_dir}\")\n            \n            for file_path in files:\n                try:\n                    dest_file = dest_dir / file_path.name\n                    \n                    # Handle filename conflicts\n                    counter = 1\n                    original_dest = dest_file\n                    while dest_file.exists():\n                        stem = original_dest.stem\n                        suffix = original_dest.suffix\n                        dest_file = dest_dir / f\"{stem}_{counter}{suffix}\"\n                        counter += 1\n                    \n                    if not self.dry_run:\n                        shutil.move(str(file_path), str(dest_file))\n                    \n                    processed += 1\n                    if processed % 10 == 0:\n                        logger.info(f\"Progress: {processed}/{total_files} files migrated\")\n                        \n                except Exception as e:\n                    logger.error(f\"Failed to migrate {file_path}: {e}\")\n        \n        # Handle 'other' category files\n        if categories.get('other'):\n            logger.info(f\"Moving {len(categories['other'])} uncategorized files to base directory\")\n            for file_path in categories['other']:\n                try:\n                    dest_file = DataPaths.BASE / file_path.name\n                    if not dest_file.exists() and not self.dry_run:\n                        shutil.move(str(file_path), str(dest_file))\n                        processed += 1\n                except Exception as e:\n                    logger.error(f\"Failed to migrate uncategorized file {file_path}: {e}\")\n        \n        logger.info(f\"Migration completed: {processed}/{total_files} files processed\")\n    \n    def update_code_references(self):\n        \"\"\"Update all code references to use new data paths.\"\"\"\n        logger.info(\"Updating code references...\")\n        \n        # Find all Python files\n        python_files = list(Path('.').rglob('*.py'))\n        updated_files = 0\n        \n        # Path replacement patterns\n        replacements = [\n            # Direct path replacements\n            (r'\"continuous_learning_data\"', '\"data\"'),\n            (r\"'continuous_learning_data'\", \"'data'\"),\n            (r'continuous_learning_data/', 'data/'),\n            \n            # Specific directory replacements\n            (r'continuous_learning_data/logs', 'data/logs/training'),\n            (r'continuous_learning_data/sessions', 'data/training/sessions'), \n            (r'continuous_learning_data/backups', 'data/memory/backups'),\n            (r'continuous_learning_data/meta_learning_data', 'data/training/meta_learning'),\n            (r'continuous_learning_data/architect_evolution_data', 'data/architecture/evolution'),\n            (r'continuous_learning_data/mutations', 'data/architecture/mutations'),\n            \n            # Add imports for new data_paths module\n            (r'(import.*\\n)(class|def|[A-Z_])', r'\\1from config.data_paths import DataPaths, FilePaths\\n\\2')\n        ]\n        \n        for py_file in python_files:\n            if 'migrate_data_structure.py' in str(py_file) or '__pycache__' in str(py_file):\n                continue\n                \n            try:\n                with open(py_file, 'r', encoding='utf-8') as f:\n                    content = f.read()\n                \n                original_content = content\n                \n                # Apply replacements\n                for pattern, replacement in replacements:\n                    content = re.sub(pattern, replacement, content)\n                \n                # Only write if content changed\n                if content != original_content and not self.dry_run:\n                    with open(py_file, 'w', encoding='utf-8') as f:\n                        f.write(content)\n                    updated_files += 1\n                    logger.info(f\"Updated: {py_file}\")\n                    \n            except Exception as e:\n                logger.error(f\"Failed to update {py_file}: {e}\")\n        \n        logger.info(f\"Code update completed: {updated_files} files updated\")\n    \n    def validate_migration(self):\n        \"\"\"Validate that the migration was successful.\"\"\"\n        logger.info(\"Validating migration...\")\n        \n        # Check that new directories exist and have files\n        directories_to_check = [\n            DataPaths.TRAINING_SESSIONS,\n            DataPaths.LOGS_TRAINING,\n            DataPaths.MEMORY_BACKUPS,\n            DataPaths.ARCHITECTURE_EVOLUTION\n        ]\n        \n        for directory in directories_to_check:\n            if directory.exists():\n                file_count = len(list(directory.glob('*')))\n                logger.info(f\"✅ {directory}: {file_count} files\")\n            else:\n                logger.warning(f\"❌ {directory}: Directory not found\")\n        \n        # Check if old directory is empty or gone\n        if self.old_base.exists():\n            remaining_files = len(list(self.old_base.rglob('*')))\n            if remaining_files == 0:\n                logger.info(\"✅ Old directory is empty - migration successful\")\n            else:\n                logger.warning(f\"❌ Old directory still has {remaining_files} files\")\n        else:\n            logger.info(\"✅ Old directory removed - migration successful\")\n    \n    def run_migration(self, dry_run: bool = False):\n        \"\"\"Run the complete migration process.\"\"\"\n        self.dry_run = dry_run\n        \n        if dry_run:\n            logger.info(\"DRY RUN MODE - No files will be moved or modified\")\n        \n        logger.info(\"Starting data structure migration...\")\n        \n        # Step 1: Create backup\n        if not dry_run:\n            self.create_backup()\n        \n        # Step 2: Categorize files\n        categories = self.categorize_files()\n        \n        logger.info(\"File categorization summary:\")\n        for category, files in categories.items():\n            if files:\n                logger.info(f\"  {category}: {len(files)} files\")\n        \n        # Step 3: Migrate files\n        if not dry_run:\n            self.migrate_files(categories)\n        \n        # Step 4: Update code references\n        if not dry_run:\n            self.update_code_references()\n        \n        # Step 5: Validate\n        if not dry_run:\n            self.validate_migration()\n        \n        logger.info(\"Migration process completed!\")\n\ndef main():\n    \"\"\"Main function to run the migration.\"\"\"\n    import argparse\n    \n    parser = argparse.ArgumentParser(description='Migrate continuous_learning_data to organized structure')\n    parser.add_argument('--dry-run', action='store_true', help='Run without making any changes')\n    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')\n    \n    args = parser.parse_args()\n    \n    if args.verbose:\n        logging.getLogger().setLevel(logging.DEBUG)\n    \n    migrator = DataMigrator()\n    \n    try:\n        migrator.run_migration(dry_run=args.dry_run)\n    except KeyboardInterrupt:\n        logger.info(\"Migration interrupted by user\")\n        sys.exit(1)\n    except Exception as e:\n        logger.error(f\"Migration failed: {e}\")\n        sys.exit(1)\n\nif __name__ == \"__main__\":\n    main()
