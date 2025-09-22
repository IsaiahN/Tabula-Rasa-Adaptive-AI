"""
Gameplay Automation Package

Comprehensive automation for gameplay error detection, correction, and monitoring.
"""

from .error_automation import (
    GameplayErrorAutomation,
    GameplayErrorType,
    GameplayError,
    gameplay_automation,
    process_gameplay_errors,
    get_gameplay_health
)

from .action_corrector import (
    ActionCorrector,
    CorrectionType,
    ActionCorrection,
    action_corrector,
    correct_action,
    get_correction_stats
)

from .realtime_monitor import (
    RealTimeGameplayMonitor,
    GameplayEvent,
    realtime_monitor,
    start_gameplay_monitoring,
    stop_gameplay_monitoring,
    get_gameplay_events,
    get_monitoring_status,
    enable_auto_fix,
    disable_auto_fix
)

__all__ = [
    # Error automation
    'GameplayErrorAutomation',
    'GameplayErrorType', 
    'GameplayError',
    'gameplay_automation',
    'process_gameplay_errors',
    'get_gameplay_health',
    
    # Action correction
    'ActionCorrector',
    'CorrectionType',
    'ActionCorrection', 
    'action_corrector',
    'correct_action',
    'get_correction_stats',
    
    # Real-time monitoring
    'RealTimeGameplayMonitor',
    'GameplayEvent',
    'realtime_monitor',
    'start_gameplay_monitoring',
    'stop_gameplay_monitoring',
    'get_gameplay_events',
    'get_monitoring_status',
    'enable_auto_fix',
    'disable_auto_fix'
]
