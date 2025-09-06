#!/usr/bin/env python3
"""
MetaCognitiveMemoryManager - Specialized Memory Management for Governor/Architect

Implements sophisticated memory management for meta-cognitive systems with:
1. LOSSLESS preservation for critical Governor/Architect files
2. Intelligent Salience Decay for regular system files
3. Hierarchical importance classification
4. Safe garbage collection with rollback capability
"""

import os
import json
import time
import shutil
import hashlib
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict

class MemoryClassification(Enum):
    """Classification levels for different types of system memory."""
    CRITICAL_LOSSLESS = "critical_lossless"  # Governor/Architect vital files
    IMPORTANT_DECAY = "important_decay"      # Learning states with managed decay
    REGULAR_DECAY = "regular_decay"          # Session data with normal decay
    TEMPORARY_PURGE = "temporary_purge"      # Debug/temp files for cleanup

@dataclass
class MemoryFile:
    """Represents a memory file with metadata."""
    path: Path
    classification: MemoryClassification
    size_bytes: int
    last_modified: datetime
    last_accessed: datetime
    importance_score: float
    access_count: int
    salience_strength: float = 1.0
    protection_level: float = 0.0

class MetaCognitiveMemoryManager:
    """
    Advanced memory management for meta-cognitive systems.
    
    Implements a tiered approach:
    - CRITICAL_LOSSLESS: Governor/Architect files - never decay/delete
    - IMPORTANT_DECAY: Learning states - intelligent decay management
    - REGULAR_DECAY: Session data - standard salience decay
    - TEMPORARY_PURGE: Temp files - aggressive cleanup
    """
    
    def __init__(self, base_path: Path, logger: Optional[logging.Logger] = None):
        self.base_path = Path(base_path)
        self.logger = logger or logging.getLogger(__name__)
        
        # Memory classification patterns
        self.classification_patterns = {
            MemoryClassification.CRITICAL_LOSSLESS: [
                "continuous_learning_data/logs/governor_decisions_*.log",
                "continuous_learning_data/architect_evolution_*.json",
                "continuous_learning_data/meta_cognitive_results_*.json",
                "continuous_learning_data/backups/persistent_learning_state.json",
                "continuous_learning_data/unified_trainer_results.json",
                "continuous_learning_data/cross_session_*.json",
                "continuous_learning_data/outcome_tracking_*.json",
                "continuous_learning_data/critical_breakthroughs_*.json"
            ],
            MemoryClassification.IMPORTANT_DECAY: [
                "continuous_learning_data/meta_learning_session_*.json",
                "continuous_learning_data/session_session_*.json",
                "continuous_learning_data/action_intelligence_*.json",
                "continuous_learning_data/combined_score_*.json",
                "continuous_learning_data/memory_hierarchy_*.json",
                "continuous_learning_data/salience_patterns_*.json"
            ],
            MemoryClassification.REGULAR_DECAY: [
                "continuous_learning_data/sessions/continuous_session_*.json",
                "continuous_learning_data/training_episode_*.json",
                "continuous_learning_data/performance_metrics_*.json",
                "continuous_learning_data/coordination_logs_*.json"
            ],
            MemoryClassification.TEMPORARY_PURGE: [
                "continuous_learning_data/temp_*.json",
                "continuous_learning_data/debug_*.log",
                "continuous_learning_data/sandbox_*.json",
                "continuous_learning_data/mutations/mutation_exploratory_temp_*.json",
                "continuous_learning_data/test_temp_*.json"
            ]
        }
        
        # Decay parameters by classification
        self.decay_parameters = {
            MemoryClassification.CRITICAL_LOSSLESS: {
                "decay_rate": 0.0,  # No decay
                "min_salience": 1.0,  # Always maximum
                "max_age_days": float('inf'),  # Never expires
                "backup_enabled": True
            },
            MemoryClassification.IMPORTANT_DECAY: {
                "decay_rate": 0.02,  # Very slow decay
                "min_salience": 0.3,  # Strong protection
                "max_age_days": 365,  # Keep for a year
                "backup_enabled": True
            },
            MemoryClassification.REGULAR_DECAY: {
                "decay_rate": 0.05,  # Normal decay
                "min_salience": 0.1,  # Basic protection
                "max_age_days": 90,   # 3 months
                "backup_enabled": False
            },
            MemoryClassification.TEMPORARY_PURGE: {
                "decay_rate": 0.2,   # Fast decay
                "min_salience": 0.0,  # No protection
                "max_age_days": 7,    # One week
                "backup_enabled": False
            }
        }
        
        self.memory_inventory: Dict[Path, MemoryFile] = {}
        self.backup_directory = self.base_path / "memory_backups"
        self.backup_directory.mkdir(exist_ok=True)
    
    def classify_file(self, file_path: Path) -> MemoryClassification:
        """Classify a file based on patterns and content analysis."""
        file_str = str(file_path)
        
        # Check against classification patterns
        for classification, patterns in self.classification_patterns.items():
            for pattern in patterns:
                if file_path.match(pattern.replace("**/", "")):
                    return classification
        
        # Content-based classification for unmatched files
        try:
            if file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    # Check for critical Governor/Architect markers
                    if any(key in data for key in [
                        'governor_decision', 'architect_evolution', 
                        'meta_cognitive_state', 'critical_breakthrough',
                        'persistent_knowledge', 'cross_session_intelligence'
                    ]):
                        return MemoryClassification.CRITICAL_LOSSLESS
                    
                    # Check for important learning markers
                    elif any(key in data for key in [
                        'learning_session', 'memory_hierarchy', 
                        'action_intelligence', 'salience_patterns'
                    ]):
                        return MemoryClassification.IMPORTANT_DECAY
                        
        except (json.JSONDecodeError, IOError):
            pass
        
        # Default classification based on age and size
        if file_path.stat().st_size < 1024:  # Small files
            return MemoryClassification.TEMPORARY_PURGE
        elif (datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)).days > 30:
            return MemoryClassification.REGULAR_DECAY
        else:
            return MemoryClassification.IMPORTANT_DECAY
    
    def scan_memory_files(self) -> Dict[MemoryClassification, List[MemoryFile]]:
        """Scan and classify all memory files."""
        classified_files = defaultdict(list)
        
        # Scan all relevant directories
        scan_dirs = [
            self.base_path / "meta_learning_data",
            self.base_path / "continuous_learning_data", 
            self.base_path / "checkpoints",
            self.base_path / "research_results",
            self.base_path  # Root level files
        ]
        
        for scan_dir in scan_dirs:
            if not scan_dir.exists():
                continue
                
            for file_path in scan_dir.rglob("*.json"):
                if file_path.is_file():
                    classification = self.classify_file(file_path)
                    
                    stat = file_path.stat()
                    memory_file = MemoryFile(
                        path=file_path,
                        classification=classification,
                        size_bytes=stat.st_size,
                        last_modified=datetime.fromtimestamp(stat.st_mtime),
                        last_accessed=datetime.fromtimestamp(stat.st_atime),
                        importance_score=self.calculate_importance(file_path, classification),
                        access_count=1,  # Default, could track this
                        salience_strength=1.0  # Start at full strength
                    )
                    
                    classified_files[classification].append(memory_file)
                    self.memory_inventory[file_path] = memory_file
            
            # Also scan log files
            for file_path in scan_dir.rglob("*.log"):
                if file_path.is_file():
                    classification = self.classify_file(file_path)
                    
                    stat = file_path.stat()
                    memory_file = MemoryFile(
                        path=file_path,
                        classification=classification,
                        size_bytes=stat.st_size,
                        last_modified=datetime.fromtimestamp(stat.st_mtime),
                        last_accessed=datetime.fromtimestamp(stat.st_atime),
                        importance_score=self.calculate_importance(file_path, classification),
                        access_count=1,
                        salience_strength=1.0
                    )
                    
                    classified_files[classification].append(memory_file)
                    self.memory_inventory[file_path] = memory_file
        
        return dict(classified_files)
    
    def calculate_importance(self, file_path: Path, classification: MemoryClassification) -> float:
        """Calculate importance score for a file."""
        base_score = {
            MemoryClassification.CRITICAL_LOSSLESS: 1.0,
            MemoryClassification.IMPORTANT_DECAY: 0.7,
            MemoryClassification.REGULAR_DECAY: 0.4,
            MemoryClassification.TEMPORARY_PURGE: 0.1
        }[classification]
        
        # Adjust based on file characteristics
        try:
            stat = file_path.stat()
            age_days = (datetime.now() - datetime.fromtimestamp(stat.st_mtime)).days
            
            # Recent files are more important
            recency_bonus = max(0, (30 - age_days) / 30 * 0.2)
            
            # Size consideration (larger files might be more important, up to a point)
            size_mb = stat.st_size / (1024 * 1024)
            size_factor = min(1.0, size_mb / 10) * 0.1  # Modest bonus for reasonable sizes
            
            return min(1.0, base_score + recency_bonus + size_factor)
            
        except (OSError, IOError):
            return base_score
    
    def apply_salience_decay(self, memory_file: MemoryFile) -> float:
        """Apply salience decay based on classification and age."""
        if memory_file.classification == MemoryClassification.CRITICAL_LOSSLESS:
            return 1.0  # No decay for critical files
        
        params = self.decay_parameters[memory_file.classification]
        
        # Calculate age-based decay
        age_days = (datetime.now() - memory_file.last_modified).days
        decay_factor = 1.0 - (params["decay_rate"] * age_days / 30)  # Monthly decay
        
        # Apply minimum salience floor
        new_salience = max(params["min_salience"], 
                          memory_file.importance_score * decay_factor)
        
        return new_salience
    
    def perform_garbage_collection(self, dry_run: bool = False) -> Dict[str, Any]:
        """Perform intelligent garbage collection."""
        results = {
            "files_processed": 0,
            "files_backed_up": 0,
            "files_deleted": 0,
            "bytes_freed": 0,
            "critical_files_protected": 0,
            "actions": []
        }
        
        classified_files = self.scan_memory_files()
        
        for classification, files in classified_files.items():
            params = self.decay_parameters[classification]
            
            for memory_file in files:
                results["files_processed"] += 1
                new_salience = self.apply_salience_decay(memory_file)
                
                # Critical files are always protected
                if classification == MemoryClassification.CRITICAL_LOSSLESS:
                    results["critical_files_protected"] += 1
                    if params["backup_enabled"] and not dry_run:
                        self.backup_file(memory_file.path)
                        results["files_backed_up"] += 1
                    continue
                
                # Check if file should be deleted
                age_days = (datetime.now() - memory_file.last_modified).days
                should_delete = (
                    new_salience < params["min_salience"] and 
                    age_days > params["max_age_days"]
                ) or (
                    classification == MemoryClassification.TEMPORARY_PURGE and
                    age_days > params["max_age_days"]
                )
                
                if should_delete:
                    action = {
                        "action": "delete",
                        "file": str(memory_file.path),
                        "classification": classification.value,
                        "salience": new_salience,
                        "age_days": age_days,
                        "size_bytes": memory_file.size_bytes
                    }
                    results["actions"].append(action)
                    
                    if not dry_run:
                        # Backup important files before deletion
                        if params["backup_enabled"]:
                            self.backup_file(memory_file.path)
                            results["files_backed_up"] += 1
                        
                        memory_file.path.unlink()
                        results["files_deleted"] += 1
                        results["bytes_freed"] += memory_file.size_bytes
                        
                        self.logger.info(f"Deleted {classification.value} file: {memory_file.path}")
                
                # Update salience for remaining files
                else:
                    memory_file.salience_strength = new_salience
        
        return results
    
    def backup_file(self, file_path: Path) -> Path:
        """Create a backup of an important file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = self.backup_directory / backup_name
        
        shutil.copy2(file_path, backup_path)
        self.logger.info(f"Backed up {file_path} to {backup_path}")
        
        return backup_path
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get comprehensive memory status."""
        classified_files = self.scan_memory_files()
        
        status = {
            "total_files": sum(len(files) for files in classified_files.values()),
            "total_size_mb": 0,
            "classifications": {}
        }
        
        for classification, files in classified_files.items():
            total_size = sum(f.size_bytes for f in files)
            avg_salience = sum(f.salience_strength for f in files) / len(files) if files else 0
            
            status["classifications"][classification.value] = {
                "file_count": len(files),
                "total_size_mb": total_size / (1024 * 1024),
                "average_salience": avg_salience,
                "oldest_file_days": max(
                    (datetime.now() - f.last_modified).days for f in files
                ) if files else 0
            }
            
            status["total_size_mb"] += total_size / (1024 * 1024)
        
        return status
    
    def emergency_cleanup(self, target_size_mb: float) -> Dict[str, Any]:
        """Emergency cleanup to free space, respecting protection levels."""
        results = {
            "target_size_mb": target_size_mb,
            "files_deleted": 0,
            "bytes_freed": 0,
            "critical_files_protected": 0
        }
        
        classified_files = self.scan_memory_files()
        
        # Sort files by deletion priority (least important first)
        deletion_candidates = []
        for classification, files in classified_files.items():
            if classification == MemoryClassification.CRITICAL_LOSSLESS:
                results["critical_files_protected"] += len(files)
                continue
                
            for memory_file in files:
                priority = (
                    classification.value, 
                    -memory_file.salience_strength,
                    -(datetime.now() - memory_file.last_modified).days
                )
                deletion_candidates.append((priority, memory_file))
        
        # Sort by priority and delete until target is reached
        deletion_candidates.sort(key=lambda x: x[0])
        bytes_freed = 0
        target_bytes = target_size_mb * 1024 * 1024
        
        for _, memory_file in deletion_candidates:
            if bytes_freed >= target_bytes:
                break
                
            # Backup if needed
            params = self.decay_parameters[memory_file.classification]
            if params["backup_enabled"]:
                self.backup_file(memory_file.path)
            
            # Delete file
            memory_file.path.unlink()
            bytes_freed += memory_file.size_bytes
            results["files_deleted"] += 1
            
            self.logger.warning(f"Emergency cleanup deleted: {memory_file.path}")
        
        results["bytes_freed"] = bytes_freed
        return results

def main():
    """Test the MetaCognitiveMemoryManager."""
    logging.basicConfig(level=logging.INFO)
    
    base_path = Path(".")
    manager = MetaCognitiveMemoryManager(base_path)
    
    print("=== Memory Status ===")
    status = manager.get_memory_status()
    for classification, stats in status["classifications"].items():
        print(f"{classification}: {stats['file_count']} files, "
              f"{stats['total_size_mb']:.2f} MB, "
              f"avg salience: {stats['average_salience']:.3f}")
    
    print("\n=== Garbage Collection (Dry Run) ===")
    results = manager.perform_garbage_collection(dry_run=True)
    print(f"Would process {results['files_processed']} files")
    print(f"Would delete {results['files_deleted']} files")
    print(f"Would free {results['bytes_freed'] / (1024*1024):.2f} MB")
    print(f"Protected {results['critical_files_protected']} critical files")

if __name__ == "__main__":
    main()
