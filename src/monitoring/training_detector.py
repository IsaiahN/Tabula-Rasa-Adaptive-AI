"""
TRAINING PROCESS DETECTOR & MONITOR
Comprehensive system to detect, monitor, and manage all training processes
"""

import psutil
import subprocess
import time
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

@dataclass
class TrainingProcess:
    pid: int
    name: str
    cmdline: str
    create_time: float
    status: str
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    training_type: str = "unknown"
    session_id: Optional[str] = None

class TrainingDetector:
    """
    Comprehensive training process detection and monitoring system.
    Always finds running training sessions regardless of memory state.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.training_keywords = [
            'master_arc_trainer.py',
            'run_9hour',
            'training',
            'arc_trainer',
            'continuous_learning',
            'tabula_rasa'
        ]
        
        self.training_types = {
            'master_arc_trainer.py': 'core_training',
            'train.py': 'orchestrator',
            'parallel.py': 'scaled_training'
        }
    
    def find_all_training_processes(self) -> List[TrainingProcess]:
        """Find all training-related processes currently running."""
        training_processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'cpu_percent', 'memory_info']):
                try:
                    if not proc.info['cmdline']:
                        continue
                        
                    cmdline = ' '.join(proc.info['cmdline'])
                    
                    # Check if this is a training process
                    is_training = any(keyword in cmdline.lower() for keyword in self.training_keywords)
                    
                    if is_training:
                        # Determine training type
                        training_type = "unknown"
                        for pattern, t_type in self.training_types.items():
                            if pattern in cmdline:
                                training_type = t_type
                                break
                        
                        # Extract session ID if possible
                        session_id = self._extract_session_id(cmdline)
                        
                        training_processes.append(TrainingProcess(
                            pid=proc.info['pid'],
                            name=proc.info['name'],
                            cmdline=cmdline,
                            create_time=proc.info['create_time'],
                            status='running',
                            cpu_percent=proc.cpu_percent(),
                            memory_mb=proc.info['memory_info'].rss / 1024 / 1024 if proc.info['memory_info'] else 0.0,
                            training_type=training_type,
                            session_id=session_id
                        ))
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error scanning processes: {e}")
            
        return training_processes
    
    def _extract_session_id(self, cmdline: str) -> Optional[str]:
        """Extract session ID from command line if present."""
        import re
        
        # Look for session ID patterns
        patterns = [
            r'--session-id\s+(\S+)',
            r'session[_-]?id[=:]\s*(\S+)',
            r'director_session_\d+',
            r'training_session_\d+'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, cmdline, re.IGNORECASE)
            if match:
                return match.group(1) if match.groups() else match.group(0)
        
        return None
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all training processes."""
        processes = self.find_all_training_processes()
        
        summary = {
            'total_processes': len(processes),
            'processes': [],
            'by_type': {},
            'total_cpu': 0.0,
            'total_memory_mb': 0.0,
            'oldest_process': None,
            'newest_process': None,
            'timestamp': datetime.now().isoformat()
        }
        
        if not processes:
            return summary
        
        # Process each training process
        for proc in processes:
            proc_dict = {
                'pid': proc.pid,
                'name': proc.name,
                'cmdline': proc.cmdline,
                'training_type': proc.training_type,
                'session_id': proc.session_id,
                'create_time': datetime.fromtimestamp(proc.create_time).isoformat(),
                'cpu_percent': proc.cpu_percent,
                'memory_mb': proc.memory_mb,
                'status': proc.status
            }
            summary['processes'].append(proc_dict)
            
            # Group by type
            if proc.training_type not in summary['by_type']:
                summary['by_type'][proc.training_type] = 0
            summary['by_type'][proc.training_type] += 1
            
            # Accumulate totals
            summary['total_cpu'] += proc.cpu_percent
            summary['total_memory_mb'] += proc.memory_mb
        
        # Find oldest and newest
        sorted_by_time = sorted(processes, key=lambda x: x.create_time)
        summary['oldest_process'] = {
            'pid': sorted_by_time[0].pid,
            'create_time': datetime.fromtimestamp(sorted_by_time[0].create_time).isoformat(),
            'training_type': sorted_by_time[0].training_type
        }
        summary['newest_process'] = {
            'pid': sorted_by_time[-1].pid,
            'create_time': datetime.fromtimestamp(sorted_by_time[-1].create_time).isoformat(),
            'training_type': sorted_by_time[-1].training_type
        }
        
        return summary
    
    def monitor_training_continuously(self, interval_seconds: int = 30) -> None:
        """Continuously monitor training processes."""
        print(f" Starting continuous training monitoring (interval: {interval_seconds}s)")
        
        try:
            while True:
                summary = self.get_training_summary()
                
                print(f"\n TRAINING MONITOR - {summary['timestamp']}")
                print(f"   Total processes: {summary['total_processes']}")
                print(f"   Total CPU: {summary['total_cpu']:.1f}%")
                print(f"   Total Memory: {summary['total_memory_mb']:.1f} MB")
                
                if summary['processes']:
                    print(f"   Process types: {summary['by_type']}")
                    for proc in summary['processes']:
                        print(f"   - PID {proc['pid']}: {proc['training_type']} (CPU: {proc['cpu_percent']:.1f}%, Mem: {proc['memory_mb']:.1f}MB)")
                else:
                    print("   No training processes detected")
                
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n Training monitoring stopped")
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")

# Quick access functions
def find_training_processes() -> List[TrainingProcess]:
    """Quick function to find all training processes."""
    detector = TrainingDetector()
    return detector.find_all_training_processes()

def get_training_summary() -> Dict[str, Any]:
    """Quick function to get training summary."""
    detector = TrainingDetector()
    return detector.get_training_summary()

def start_monitoring(interval_seconds: int = 30) -> None:
    """Quick function to start continuous monitoring."""
    detector = TrainingDetector()
    detector.monitor_training_continuously(interval_seconds)

if __name__ == "__main__":
    # Quick test
    detector = TrainingDetector()
    summary = detector.get_training_summary()
    print(json.dumps(summary, indent=2))
