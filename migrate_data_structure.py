#!/usr/bin/env python3
"""
Data Structure Migration Script for Tabula Rasa

#!/usr/bin/env python3
"""
Data Structure Migration Script for Tabula Rasa

This script migrates files from continuous_learning_data to a new organized
"data/" directory and updates code references. It supports a dry-run mode
that logs planned actions without modifying files.
"""

import os
import sys
import shutil
from pathlib import Path
import re
from typing import Dict, List
import logging
from datetime import datetime


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from config.data_paths import DataPaths
except ImportError:
    print("Error: Could not import data_paths. Make sure src/config/data_paths.py exists.")
    sys.exit(1)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataMigrator:
    def __init__(self):
        self.old_base = Path("continuous_learning_data")
        self.new_base = Path("data")
        self.backup_dir = Path(f"backup_continuous_learning_data_{int(datetime.now().timestamp())}")
        self.dry_run = False

    def create_backup(self):
        if self.old_base.exists():
            logger.info(f"Creating backup: {self.backup_dir}")
            shutil.copytree(self.old_base, self.backup_dir)
            logger.info("Backup completed successfully")
        else:
            logger.warning("No continuous_learning_data directory found to backup")

    def categorize_files(self) -> Dict[str, List[Path]]:
        #!/usr/bin/env python3
        """
        Data Structure Migration Script for Tabula Rasa

        This script migrates files from continuous_learning_data into a new organized
        data/ directory and updates code references. It supports a dry-run mode that
        logs planned actions without modifying files.
        """

        import os
        import sys
        import shutil
        from pathlib import Path
        import re
        from typing import Dict, List
        import logging
        from datetime import datetime

        # Add src to path for imports
        sys.path.insert(0, str(Path(__file__).parent / "src"))

        try:
            from config.data_paths import DataPaths
        except ImportError:
            print("Error: Could not import data_paths. Make sure src/config/data_paths.py exists.")
            sys.exit(1)

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)


        class DataMigrator:
            def __init__(self):
                self.old_base = Path("continuous_learning_data")
                self.new_base = Path("data")
                self.backup_dir = Path(f"backup_continuous_learning_data_{int(datetime.now().timestamp())}")
                self.dry_run = False

            def create_backup(self):
                if self.old_base.exists():
                    logger.info(f"Creating backup: {self.backup_dir}")
                    shutil.copytree(self.old_base, self.backup_dir)
                    logger.info("Backup completed successfully")
                else:
                    logger.warning("No continuous_learning_data directory found to backup")
#!/usr/bin/env python3
"""
Data Structure Migration Script for Tabula Rasa

This script migrates files from continuous_learning_data into a new organized
data/ directory and updates code references. It supports a dry-run mode that
logs planned actions without modifying files.
"""

import os
import sys
import shutil
from pathlib import Path
import re
from typing import Dict, List
import logging
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from config.data_paths import DataPaths
except ImportError:
    print("Error: Could not import data_paths. Make sure src/config/data_paths.py exists.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataMigrator:
    def __init__(self):
        self.old_base = Path("continuous_learning_data")
        self.new_base = Path("data")
        self.backup_dir = Path(f"backup_continuous_learning_data_{int(datetime.now().timestamp())}")
        self.dry_run = False

    def create_backup(self):
        if self.old_base.exists():
            logger.info(f"Creating backup: {self.backup_dir}")
            shutil.copytree(self.old_base, self.backup_dir)
            logger.info("Backup completed successfully")
        else:
            logger.warning("No continuous_learning_data directory found to backup")

    def categorize_files(self) -> Dict[str, List[Path]]:
        if not self.old_base.exists():
            return {}

        categories = {
            'training_sessions': [],
            #!/usr/bin/env python3
            """
            Data Structure Migration Script for Tabula Rasa

            This script migrates files from continuous_learning_data into a new organized
            data/ directory and updates code references. It supports a dry-run mode that
            logs planned actions without modifying files.
            """

            import os
            import sys
            import shutil
            from pathlib import Path
            import re
            from typing import Dict, List
            import logging
            from datetime import datetime

            # Add src to path for imports
            sys.path.insert(0, str(Path(__file__).parent / "src"))

            try:
                from config.data_paths import DataPaths
            except ImportError:
                print("Error: Could not import data_paths. Make sure src/config/data_paths.py exists.")
                sys.exit(1)

            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            logger = logging.getLogger(__name__)


            class DataMigrator:
                def __init__(self):
                    self.old_base = Path("continuous_learning_data")
                    self.new_base = Path("data")
                    self.backup_dir = Path(f"backup_continuous_learning_data_{int(datetime.now().timestamp())}")
                    self.dry_run = False

                def create_backup(self):
                    if self.old_base.exists():
                        logger.info(f"Creating backup: {self.backup_dir}")
                        shutil.copytree(self.old_base, self.backup_dir)
                        logger.info("Backup completed successfully")
                    else:
                        logger.warning("No continuous_learning_data directory found to backup")

                def categorize_files(self) -> Dict[str, List[Path]]:
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
                        'other': []
                    }

                    for root, dirs, files in os.walk(self.old_base):
                        root_path = Path(root)
                        for file in files:
                            file_path = root_path / file
                            relative_path = file_path.relative_to(self.old_base)
                            filename = file.lower()

                            if 'sessions' in str(relative_path).lower():
                                categories['training_sessions'].append(file_path)
                            elif 'action_intelligence' in filename:
                                categories['training_intelligence'].append(file_path)
                            elif 'meta_learning_session' in filename:
                                categories['training_meta_learning'].append(file_path)
                            elif filename.endswith('.log'):
                                if 'governor' in filename:
                                    categories['logs_governor'].append(file_path)
                                else:
                                    categories['logs_training'].append(file_path)
                            elif 'backup' in filename or 'backups' in str(relative_path).lower():
                                categories['memory_backups'].append(file_path)
                            elif filename.endswith('.pkl') or 'pattern' in filename:
                                categories['memory_patterns'].append(file_path)
                            elif 'architect' in filename or 'evolution' in filename:
                                categories['architecture_evolution'].append(file_path)
                            elif 'mutation' in filename:
                                categories['architecture_mutations'].append(file_path)
                            elif 'research' in filename:
                                categories['experiments_research'].append(file_path)
                            elif 'phase0' in filename or 'phase0' in str(relative_path).lower():
                                categories['experiments_phase0'].append(file_path)
                            elif 'evaluation' in filename or 'agi_evaluation' in str(relative_path).lower():
                                categories['experiments_evaluations'].append(file_path)
                            elif 'state' in filename or 'counters' in filename:
                                if 'global_counters' in filename:
                                    categories['config_counters'].append(file_path)
                                else:
                                    categories['config_states'].append(file_path)
                            elif filename.endswith('.json') and any(x in filename for x in ['results', 'training']):
                                categories['training_results'].append(file_path)
                            else:
                                categories['other'].append(file_path)

                    return categories

                def migrate_files(self, categories: Dict[str, List[Path]]):
                    destination_mapping = {
                        'training_sessions': DataPaths.TRAINING_SESSIONS,
                        'training_results': DataPaths.TRAINING_RESULTS,
                        'training_intelligence': DataPaths.TRAINING_INTELLIGENCE,
                        'training_meta_learning': DataPaths.TRAINING_META_LEARNING,
                        'logs_training': DataPaths.LOGS_TRAINING,
                        'logs_governor': DataPaths.LOGS_GOVERNOR,
                        'logs_system': DataPaths.LOGS_SYSTEM,
                        'memory_backups': DataPaths.MEMORY_BACKUPS,
                        'memory_patterns': DataPaths.MEMORY_PATTERNS,
                        'architecture_evolution': DataPaths.ARCHITECTURE_EVOLUTION,
                        'architecture_mutations': DataPaths.ARCHITECTURE_MUTATIONS,
                        'experiments_research': DataPaths.EXPERIMENTS_RESEARCH,
                        'experiments_phase0': DataPaths.EXPERIMENTS_PHASE0,
                        'experiments_evaluations': DataPaths.EXPERIMENTS_EVALUATIONS,
                        'config_states': DataPaths.CONFIG_STATES,
                        'config_counters': DataPaths.CONFIG_COUNTERS
                    }

                    total_files = sum(len(files) for files in categories.values())
                    processed = 0

                    logger.info(f"Starting migration preview of {total_files} files (dry_run={self.dry_run})...")

                    for category, files in categories.items():
                        if not files:
                            continue

                        dest_dir = destination_mapping.get(category, DataPaths.BASE)
                        logger.info(f"Preparing to migrate {len(files)} items from category '{category}' -> {dest_dir}")

                        for file_path in files:
                            try:
                                dest_dir.mkdir(parents=True, exist_ok=True)
                                dest_file = dest_dir / file_path.name

                                counter = 1
                                original_dest = dest_file
                                while dest_file.exists():
                                    stem = original_dest.stem
                                    suffix = original_dest.suffix
                                    dest_file = dest_dir / f"{stem}_{counter}{suffix}"
                                    counter += 1

                                if self.dry_run:
                                    logger.info(f"DRY RUN: Would move {file_path} -> {dest_file}")
                                else:
                                    shutil.move(str(file_path), str(dest_file))
                                    logger.info(f"Moved {file_path} -> {dest_file}")

                                processed += 1
                                if processed % 10 == 0:
                                    logger.info(f"Progress: {processed}/{total_files} files processed")

                            except Exception as e:
                                logger.error(f"Failed to migrate {file_path}: {e}")

                    if categories.get('other'):
                        logger.info(f"Handling {len(categories['other'])} uncategorized files")
                        for file_path in categories['other']:
                            try:
                                dest_file = DataPaths.BASE / file_path.name
                                dest_file.parent.mkdir(parents=True, exist_ok=True)
                                if self.dry_run:
                                    logger.info(f"DRY RUN: Would move {file_path} -> {dest_file}")
                                else:
                                    if not dest_file.exists():
                                        shutil.move(str(file_path), str(dest_file))
                                        logger.info(f"Moved {file_path} -> {dest_file}")
                                processed += 1
                            except Exception as e:
                                logger.error(f"Failed to migrate uncategorized file {file_path}: {e}")

                    logger.info(f"Migration preview completed: {processed}/{total_files} files processed (dry_run={self.dry_run})")

                def update_code_references(self):
                    logger.info("Preparing to update code references (dry_run=%s)" % self.dry_run)

                    python_files = list(Path('.').rglob('*.py'))
                    updated_files = 0

                    replacements = [
                        (r'"continuous_learning_data"', '"data"'),
                        (r"'continuous_learning_data'", "'data'"),
                        (r'continuous_learning_data/', 'data/'),
                        (r'continuous_learning_data/logs', 'data/logs/training'),
                        (r'continuous_learning_data/sessions', 'data/training/sessions'),
                        (r'continuous_learning_data/backups', 'data/memory/backups'),
                        (r'continuous_learning_data/meta_learning_data', 'data/training/meta_learning'),
                        (r'continuous_learning_data/architect_evolution_data', 'data/architecture/evolution'),
                        (r'continuous_learning_data/mutations', 'data/architecture/mutations'),
                    ]

                    skip_files = {Path('migrate_data_structure.py'), Path('src/config/data_paths.py')}

                    for py_file in python_files:
                        if any(str(py_file).startswith(str(s)) for s in skip_files):
                            continue

                        try:
                            with open(py_file, 'r', encoding='utf-8') as f:
                                content = f.read()

                            original_content = content

                            for pattern, replacement in replacements:
                                content = re.sub(pattern, replacement, content)

                            if content != original_content:
                                if self.dry_run:
                                    logger.info(f"DRY RUN: Would update references in {py_file}")
                                else:
                                    backup_path = py_file.with_suffix(py_file.suffix + '.bak')
                                    shutil.copy2(py_file, backup_path)
                                    with open(py_file, 'w', encoding='utf-8') as f:
                                        f.write(content)
                                    updated_files += 1
                                    logger.info(f"Updated: {py_file} (backup: {backup_path})")

                        except Exception as e:
                            logger.error(f"Failed to update {py_file}: {e}")

                    logger.info(f"Code update completed: {updated_files} files updated (dry_run={self.dry_run})")

                def validate_migration(self):
                    logger.info("Validating migration...")

                    directories_to_check = [
                        DataPaths.TRAINING_SESSIONS,
                        DataPaths.LOGS_TRAINING,
                        DataPaths.MEMORY_BACKUPS,
                        DataPaths.ARCHITECTURE_EVOLUTION
                    ]

                    for directory in directories_to_check:
                        if directory.exists():
                            file_count = len(list(directory.glob('*')))
                            logger.info(f"✅ {directory}: {file_count} files")
                        else:
                            logger.warning(f"❌ {directory}: Directory not found")

                    if self.old_base.exists():
                        remaining_files = len(list(self.old_base.rglob('*')))
                        if remaining_files == 0:
                            logger.info("✅ Old directory is empty - migration successful")
                        else:
                            logger.warning(f"❌ Old directory still has {remaining_files} files")
                    else:
                        logger.info("✅ Old directory removed - migration successful")

                def run_migration(self, dry_run: bool = False):
                    self.dry_run = dry_run

                    if dry_run:
                        logger.info("DRY RUN MODE - No files will be moved or modified")

                    logger.info("Starting data structure migration...")

                    if not dry_run:
                        self.create_backup()

                    categories = self.categorize_files()

                    logger.info("File categorization summary:")
                    for category, files in categories.items():
                        if files:
                            logger.info(f"  {category}: {len(files)} files")

                    self.migrate_files(categories)
                    self.update_code_references()

                    if not dry_run:
                        self.validate_migration()

                    logger.info("Migration process completed!")


            def main():
                import argparse

                parser = argparse.ArgumentParser(description='Migrate continuous_learning_data to organized structure')
                parser.add_argument('--dry-run', action='store_true', help='Run without making any changes')
                parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

                args = parser.parse_args()

                if args.verbose:
                    logging.getLogger().setLevel(logging.DEBUG)

                migrator = DataMigrator()

                try:
                    migrator.run_migration(dry_run=args.dry_run)
                except KeyboardInterrupt:
                    logger.info("Migration interrupted by user")
                    sys.exit(1)
                except Exception as e:
                    logger.error(f"Migration failed: {e}")
                    sys.exit(1)


            if __name__ == "__main__":
                main()
