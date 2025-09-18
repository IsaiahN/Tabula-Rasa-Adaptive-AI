"""
DIRECTOR TRAINING MONITOR INTEGRATION
Integrates training process detection with Director's autonomous loop
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys
import os

# Add monitoring to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'monitoring'))
from training_detector import TrainingDetector, TrainingProcess

class DirectorTrainingMonitor:
    """
    Director's training monitoring system.
    Provides real-time awareness of all training processes.
    """
    
    def __init__(self):
        self.detector = TrainingDetector()
        self.logger = logging.getLogger(__name__)
    
    async def get_training_status(self) -> Dict[str, Any]:
        """Get comprehensive training status for Director analysis."""
        processes = self.detector.find_all_training_processes()
        
        # Categorize processes
        core_training = [p for p in processes if p.training_type == 'core_training']
        orchestrators = [p for p in processes if p.training_type == 'orchestrator']
        monitors = [p for p in processes if p.training_type == 'monitor']
        
        status = {
            'total_processes': len(processes),
            'core_training_count': len(core_training),
            'orchestrator_count': len(orchestrators),
            'monitor_count': len(monitors),
            'is_training_active': len(core_training) > 0,
            'is_orchestrator_active': len(orchestrators) > 0,
            'processes': [
                {
                    'pid': p.pid,
                    'type': p.training_type,
                    'session_id': p.session_id,
                    'cpu_percent': p.cpu_percent,
                    'memory_mb': p.memory_mb,
                    'uptime_hours': (time.time() - p.create_time) / 3600,
                    'status': p.status
                }
                for p in processes
            ],
            'summary': {
                'total_cpu': sum(p.cpu_percent for p in processes),
                'total_memory_mb': sum(p.memory_mb for p in processes),
                'oldest_uptime_hours': min((time.time() - p.create_time) / 3600 for p in processes) if processes else 0,
                'newest_uptime_hours': max((time.time() - p.create_time) / 3600 for p in processes) if processes else 0
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return status
    
    async def detect_training_issues(self) -> List[Dict[str, Any]]:
        """Detect potential issues with training processes."""
        issues = []
        processes = self.detector.find_all_training_processes()
        
        # Check for no training processes
        if not processes:
            issues.append({
                'type': 'NO_TRAINING',
                'severity': 'CRITICAL',
                'message': 'No training processes detected',
                'recommendation': 'Start training session immediately'
            })
        
        # Check for high CPU usage
        high_cpu_processes = [p for p in processes if p.cpu_percent > 80]
        if high_cpu_processes:
            issues.append({
                'type': 'HIGH_CPU',
                'severity': 'WARNING',
                'message': f'{len(high_cpu_processes)} processes using >80% CPU',
                'processes': [p.pid for p in high_cpu_processes],
                'recommendation': 'Monitor system performance'
            })
        
        # Check for high memory usage
        high_memory_processes = [p for p in processes if p.memory_mb > 1000]
        if high_memory_processes:
            issues.append({
                'type': 'HIGH_MEMORY',
                'severity': 'WARNING',
                'message': f'{len(high_memory_processes)} processes using >1GB memory',
                'processes': [p.pid for p in high_memory_processes],
                'recommendation': 'Check for memory leaks'
            })
        
        # Check for stuck processes (running too long without progress)
        long_running = [p for p in processes if (time.time() - p.create_time) > 3600]  # 1 hour
        if long_running:
            issues.append({
                'type': 'LONG_RUNNING',
                'severity': 'INFO',
                'message': f'{len(long_running)} processes running >1 hour',
                'processes': [p.pid for p in long_running],
                'recommendation': 'Monitor for progress'
            })
        
        return issues
    
    async def get_training_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations based on training status."""
        recommendations = []
        status = await self.get_training_status()
        issues = await self.detect_training_issues()
        
        # No training processes
        if not status['is_training_active']:
            recommendations.append({
                'type': 'START_TRAINING',
                'priority': 'CRITICAL',
                'message': 'No active training detected - start training immediately',
                'action': 'Execute training script'
            })
        
        # Multiple orchestrators
        if status['orchestrator_count'] > 1:
            recommendations.append({
                'type': 'MULTIPLE_ORCHESTRATORS',
                'priority': 'MEDIUM',
                'message': f'{status["orchestrator_count"]} orchestrators running - may cause conflicts',
                'action': 'Review and consolidate orchestrators'
            })
        
        # High resource usage
        if status['summary']['total_cpu'] > 150:
            recommendations.append({
                'type': 'HIGH_RESOURCE_USAGE',
                'priority': 'MEDIUM',
                'message': f'Total CPU usage {status["summary"]["total_cpu"]:.1f}% - consider optimization',
                'action': 'Monitor and optimize training parameters'
            })
        
        return recommendations

# Global instance
_training_monitor = None

def get_training_monitor() -> DirectorTrainingMonitor:
    """Get global training monitor instance."""
    global _training_monitor
    if _training_monitor is None:
        _training_monitor = DirectorTrainingMonitor()
    return _training_monitor

# Quick access functions
async def get_training_status() -> Dict[str, Any]:
    """Quick access to training status."""
    monitor = get_training_monitor()
    return await monitor.get_training_status()

async def detect_training_issues() -> List[Dict[str, Any]]:
    """Quick access to training issues."""
    monitor = get_training_monitor()
    return await monitor.detect_training_issues()

async def get_training_recommendations() -> List[Dict[str, Any]]:
    """Quick access to training recommendations."""
    monitor = get_training_monitor()
    return await monitor.get_training_recommendations()
