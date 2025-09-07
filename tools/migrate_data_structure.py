#!/usr/bin/env python3
"""
Migration helper (clean copy).
This is a safe, dry-run-capable migration script that uses src.config.data_paths.DataPaths
for canonical destination directories and logs planned actions.
"""

import sys
from pathlib import Path
import logging
import argparse

# ensure src on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from config.data_paths import DataPaths
except Exception as e:
    print("Failed to import DataPaths:", e)
    raise

import shutil
import os
import re
from typing import Dict, List
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataMigrator:
    def __init__(self, dry_run: bool = True, create_backup: bool = False):
        self.old_base = Path("continuous_learning_data")
        self.dry_run = dry_run
        self.create_backup = create_backup
        self.backup_dir = Path(f"backup_continuous_learning_data_{int(datetime.now().timestamp())}")

    def categorize_files(self) -> Dict[str, List[Path]]:
        if not self.old_base.exists():
            logger.warning("No continuous_learning_data directory found; nothing to do.")
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

        for fp in self.old_base.rglob('*'):
            if fp.is_file():
                name = fp.name.lower()
                rel = str(fp.relative_to(self.old_base)).lower()
                if 'sessions' in rel or 'session' in name:
                    categories['training_sessions'].append(fp)
                elif 'action_intelligence' in name:
                    categories['training_intelligence'].append(fp)
                elif 'meta_learning' in name or 'meta_learning_session' in name:
                    categories['training_meta_learning'].append(fp)
                elif name.endswith('.log'):
                    if 'governor' in name:
                        categories['logs_governor'].append(fp)
                    else:
                        categories['logs_training'].append(fp)
                elif 'backup' in name or 'backups' in rel:
                    categories['memory_backups'].append(fp)
                elif name.endswith('.pkl') or 'pattern' in name:
                    categories['memory_patterns'].append(fp)
                elif 'architect' in name or 'evolution' in name:
                    categories['architecture_evolution'].append(fp)
                elif 'mutation' in name:
                    categories['architecture_mutations'].append(fp)
                elif 'research' in name:
                    categories['experiments_research'].append(fp)
                elif 'phase0' in name or rel.startswith('phase0'):
                    categories['experiments_phase0'].append(fp)
                elif 'evaluation' in name:
                    categories['experiments_evaluations'].append(fp)
                elif 'global_counters' in name or 'counters' in name:
                    categories['config_counters'].append(fp)
                elif name.endswith('.json') and ('results' in name or 'training' in name):
                    categories['training_results'].append(fp)
                else:
                    categories['other'].append(fp)
        return categories

    def migrate_files(self, categories: Dict[str, List[Path]]):
        mapping = DataPaths.get_legacy_mapping()

        total = sum(len(v) for v in categories.values())
        logger.info(f"Found {total} files to consider for migration")

        for cat, files in categories.items():
            if not files:
                continue
            # choose destination dir by category
            dest = getattr(DataPaths, cat.upper(), None)
            if dest is None:
                # fallback to DataPaths.BASE
                dest = DataPaths.BASE
            for f in files:
                dest_dir = Path(dest)
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_file = dest_dir / f.name
                # avoid clobber
                if dest_file.exists():
                    dest_file = dest_dir / (f.stem + '_' + datetime.now().strftime('%Y%m%d%H%M%S') + f.suffix)
                if self.dry_run:
                    logger.info(f"DRY RUN: would move {f} -> {dest_file}")
                else:
                    shutil.move(str(f), str(dest_file))
                    logger.info(f"Moved {f} -> {dest_file}")

    def update_code_references(self):
        replacements = [(r'"continuous_learning_data"', '"data"'),
                        (r"'continuous_learning_data'", "'data'"),
                        (r'continuous_learning_data/', 'data/')]

        python_files = [p for p in Path('.').rglob('*.py') if 'venv' not in str(p)]
        for py in python_files:
            if py.name == Path(__file__).name:
                continue
            try:
                text = py.read_text(encoding='utf-8')
            except Exception:
                continue
            new = text
            for pat, rep in replacements:
                new = re.sub(pat, rep, new)
            if new != text:
                if self.dry_run:
                    logger.info(f"DRY RUN: would update {py}")
                else:
                    bak = py.with_suffix(py.suffix + '.bak')
                    shutil.copy2(py, bak)
                    py.write_text(new, encoding='utf-8')
                    logger.info(f"Updated {py} (backup {bak})")

    def run(self):
        cats = self.categorize_files()
        for k, v in cats.items():
            logger.info(f"Category {k}: {len(v)} files")
        if not cats:
            return
        if not self.dry_run and self.create_backup:
            # create backup of the old folder before moving
            backup = self.backup_dir
            logger.info(f"Creating backup of {self.old_base} -> {backup}")
            shutil.copytree(self.old_base, backup)
        self.migrate_files(cats)
        self.update_code_references()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--execute', action='store_true', help='Perform migration (opposite of --dry-run)')
    ap.add_argument('--backup', action='store_true', help='Create a backup copy of the legacy folder before moving')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    # Determine dry_run: default is True unless --execute is provided
    dry_run = not args.execute if (hasattr(args, 'execute')) else args.dry_run
    # If user passed --dry-run explicitly, respect it (but --execute overrides)
    if args.dry_run and not args.execute:
        dry_run = True

    migrator = DataMigrator(dry_run=dry_run, create_backup=getattr(args, 'backup', False))
    migrator.run()

if __name__ == '__main__':
    main()
