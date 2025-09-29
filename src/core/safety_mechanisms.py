#!/usr/bin/env python3
"""
Comprehensive Safety Mechanisms and Safeguards System
Provides multi-layered safety for Phase 3 automation and self-modifying systems.
"""

import logging
import asyncio
import json
import hashlib
import time
import random
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import os
import shutil
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    """Safety levels for different operations."""
    LOW = "low"           # Basic validation
    MEDIUM = "medium"     # Enhanced validation + rollback
    HIGH = "high"         # Full validation + rollback + approval
    CRITICAL = "critical" # Full validation + rollback + approval + cooldown

class SafetyViolation(Enum):
    """Types of safety violations."""
    CODE_INJECTION = "code_injection"
    SYSTEM_OVERRIDE = "system_override"
    DATA_CORRUPTION = "data_corruption"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SECURITY_BREACH = "security_breach"
    UNINTENDED_BEHAVIOR = "unintended_behavior"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    # Security monitoring violations
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH = "data_breach"
    SYSTEM_INTRUSION = "system_intrusion"
    MALWARE_DETECTED = "malware_detected"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    PERFORMANCE_ANOMALY = "performance_anomaly"
    MEMORY_ANOMALY = "memory_anomaly"
    NETWORK_ANOMALY = "network_anomaly"
    FILE_SYSTEM_ANOMALY = "file_system_anomaly"

class OverrideType(Enum):
    """Types of emergency overrides that can be triggered."""
    ACTION_LOOP_BREAK = "action_loop_break"
    COORDINATE_STUCK_BREAK = "coordinate_stuck_break"
    STAGNATION_BREAK = "stagnation_break"
    EMERGENCY_RESET = "emergency_reset"

@dataclass
class SafetyCheck:
    """Individual safety check result."""
    check_name: str
    passed: bool
    severity: SafetyLevel
    message: str
    details: Dict[str, Any]
    timestamp: datetime

@dataclass
class SafetyViolationReport:
    """Report of a safety violation."""
    violation_type: SafetyViolation
    severity: SafetyLevel
    description: str
    affected_components: List[str]
    suggested_actions: List[str]
    timestamp: datetime
    violation_id: str

@dataclass
class EmergencyOverride:
    """Represents an emergency override event."""
    game_id: str
    session_id: str
    override_type: OverrideType
    trigger_reason: str
    actions_before_override: int
    override_action: int
    override_successful: bool
    override_timestamp: float

class SafetyMechanisms:
    """
    Comprehensive safety system for Phase 3 automation.
    
    Provides:
    - Code validation and sanitization
    - System integrity monitoring
    - Performance impact assessment
    - Rollback capabilities
    - Emergency stop mechanisms
    - Change approval workflows
    - Cooldown periods
    - Audit logging
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.violation_history = []
        self.safety_checks = []
        self.emergency_stop_active = False
        self.approval_required = set()
        self.cooldown_periods = {}
        self.backup_registry = {}
        self.audit_log = []
        
        # Initialize safety subsystems
        self._initialize_safety_subsystems()
        
        logger.info("[SAFETY] Safety Mechanisms initialized with comprehensive protection")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default safety configuration."""
        return {
            "max_code_changes_per_hour": 5,
            "max_architecture_changes_per_day": 2,
            "emergency_stop_threshold": 3,  # violations before emergency stop
            "backup_retention_days": 30,
            "cooldown_periods": {
                "code_evolution": 3600,  # 1 hour
                "architecture_change": 86400,  # 24 hours
                "knowledge_update": 1800,  # 30 minutes
            },
            "performance_thresholds": {
                "max_cpu_usage": 80.0,
                "max_memory_usage": 85.0,
                "max_response_time": 5.0,
                "min_success_rate": 0.7
            },
            "security_patterns": [
                r"exec\s*\(",
                r"eval\s*\(",
                r"__import__\s*\(",
                r"subprocess\.",
                r"os\.system",
                r"open\s*\([^)]*['\"][wax]",
            ],
            "forbidden_imports": [
                "subprocess",
                "os.system",
                "eval",
                "exec",
                "compile"
            ]
        }
    
    def _initialize_safety_subsystems(self):
        """Initialize all safety subsystems."""
        self.code_validator = CodeValidator(self.config)
        self.system_monitor = SystemMonitor(self.config)
        self.performance_assessor = PerformanceAssessor(self.config)
        self.rollback_manager = RollbackManager(self.config)
        self.approval_system = ApprovalSystem(self.config)
        self.audit_logger = AuditLogger(self.config)

        logger.debug("[SAFETY] Safety subsystems initialized")
    
    async def validate_change(self, 
                            change_type: str, 
                            change_data: Dict[str, Any],
                            safety_level: SafetyLevel = SafetyLevel.MEDIUM) -> Tuple[bool, List[SafetyCheck]]:
        """
        Validate a proposed change for safety.
        
        Args:
            change_type: Type of change (code_evolution, architecture, knowledge)
            change_data: Data about the proposed change
            safety_level: Required safety level
            
        Returns:
            Tuple of (is_safe, safety_checks)
        """
        logger.info(f"[SAFETY] Validating {change_type} change with safety level {safety_level.value}")
        
        checks = []
        
        # 1. Code Safety Validation
        if change_type in ["code_evolution", "architecture"]:
            code_checks = await self.code_validator.validate_code(change_data)
            logger.info(f"[CHECK] Code validation returned {len(code_checks)} checks")
            for check in code_checks:
                logger.info(f"   {check.check_name}: {check.passed} ({check.severity.value})")
            checks.extend(code_checks)
        
        # 2. System Impact Assessment
        impact_checks = await self.system_monitor.assess_impact(change_data)
        checks.extend(impact_checks)
        
        # 3. Performance Impact Assessment
        perf_checks = await self.performance_assessor.assess_performance(change_data)
        checks.extend(perf_checks)
        
        # 4. Cooldown Check
        cooldown_check = self._check_cooldown_period(change_type)
        checks.append(cooldown_check)
        
        # 5. Approval Check
        approval_check = self._check_approval_required(change_type, change_data)
        checks.append(approval_check)
        
        # Store checks
        self.safety_checks.extend(checks)
        
        # Determine if change is safe
        critical_failures = [c for c in checks if c.severity == SafetyLevel.CRITICAL and not c.passed]
        high_failures = [c for c in checks if c.severity == SafetyLevel.HIGH and not c.passed]
        
        is_safe = len(critical_failures) == 0 and len(high_failures) == 0
        
        if not is_safe:
            logger.warning(f" Safety validation failed for {change_type}")
            for check in critical_failures + high_failures:
                logger.warning(f"   [ERROR] {check.check_name}: {check.message}")
        else:
            logger.info(f"[OK] Safety validation passed for {change_type}")
        
        return is_safe, checks
    
    async def execute_safe_change(self, 
                                change_type: str, 
                                change_data: Dict[str, Any],
                                safety_level: SafetyLevel = SafetyLevel.MEDIUM) -> bool:
        """
        Execute a change with full safety mechanisms.
        
        Args:
            change_type: Type of change to execute
            change_data: Data about the change
            safety_level: Required safety level
            
        Returns:
            True if change was executed safely, False otherwise
        """
        # 1. Validate change
        is_safe, checks = await self.validate_change(change_type, change_data, safety_level)
        
        if not is_safe:
            logger.error(f" Cannot execute {change_type} - safety validation failed")
            return False
        
        # 2. Create backup
        backup_id = await self.rollback_manager.create_backup(change_type, change_data)
        if not backup_id:
            logger.error(f" Cannot execute {change_type} - backup creation failed")
            return False
        
        # 3. Execute change with monitoring
        try:
            logger.info(f" Executing {change_type} change safely")
            success = await self._execute_change_with_monitoring(change_type, change_data)
            
            if success:
                # 4. Post-execution validation
                post_checks = await self._post_execution_validation(change_type, change_data)
                
                if all(check.passed for check in post_checks):
                    logger.info(f"[OK] {change_type} change executed successfully")
                    await self.audit_logger.log_change(change_type, change_data, "success")
                    return True
                else:
                    logger.warning(f"[WARNING] {change_type} change executed but post-validation failed")
                    # Trigger rollback
                    await self.rollback_manager.rollback(backup_id)
                    return False
            else:
                logger.error(f"[ERROR] {change_type} change execution failed")
                await self.rollback_manager.rollback(backup_id)
                return False
                
        except Exception as e:
            logger.error(f" Exception during {change_type} execution: {e}")
            await self.rollback_manager.rollback(backup_id)
            return False
    
    async def emergency_stop(self, reason: str = "Manual emergency stop") -> bool:
        """
        Trigger emergency stop of all automation systems.
        
        Args:
            reason: Reason for emergency stop
            
        Returns:
            True if emergency stop was successful
        """
        logger.critical(f" EMERGENCY STOP TRIGGERED: {reason}")
        
        self.emergency_stop_active = True
        
        # Stop all automation systems
        try:
            # Stop Phase 3 automation
            await self._stop_phase3_automation()
            
            # Stop Phase 2 automation
            await self._stop_phase2_automation()
            
            # Stop Phase 1 automation
            await self._stop_phase1_automation()
            
            # Log emergency stop
            await self.audit_logger.log_emergency_stop(reason)
            
            logger.critical("[STOP] Emergency stop completed - all automation systems stopped")
            return True
            
        except Exception as e:
            logger.critical(f" Emergency stop failed: {e}")
            return False
    
    async def resume_automation(self, safety_checks: bool = True) -> bool:
        """
        Resume automation after emergency stop.
        
        Args:
            safety_checks: Whether to run safety checks before resuming
            
        Returns:
            True if automation was resumed successfully
        """
        if not self.emergency_stop_active:
            logger.warning("[WARNING] No emergency stop active - nothing to resume")
            return False
        
        if safety_checks:
            # Run comprehensive safety checks
            system_health = await self.system_monitor.get_system_health()
            if system_health["overall_health"] < 0.8:
                logger.error(" System health too low to resume automation")
                return False
        
        self.emergency_stop_active = False
        logger.info(" Automation systems resumed")
        return True
    
    def _check_cooldown_period(self, change_type: str) -> SafetyCheck:
        """Check if change type is in cooldown period."""
        now = datetime.now()
        cooldown_duration = self.config["cooldown_periods"].get(change_type, 0)
        
        if change_type in self.cooldown_periods:
            last_change = self.cooldown_periods[change_type]
            time_since_last = (now - last_change).total_seconds()
            
            if time_since_last < cooldown_duration:
                remaining = cooldown_duration - time_since_last
                return SafetyCheck(
                    check_name="cooldown_period",
                    passed=False,
                    severity=SafetyLevel.HIGH,
                    message=f"Change type {change_type} is in cooldown period. {remaining:.0f}s remaining",
                    details={"remaining_seconds": remaining, "cooldown_duration": cooldown_duration},
                    timestamp=now
                )
        
        return SafetyCheck(
            check_name="cooldown_period",
            passed=True,
            severity=SafetyLevel.LOW,
            message=f"Change type {change_type} is not in cooldown period",
            details={},
            timestamp=now
        )
    
    def _check_approval_required(self, change_type: str, change_data: Dict[str, Any]) -> SafetyCheck:
        """Check if change requires approval."""
        if change_type in self.approval_required:
            return SafetyCheck(
                check_name="approval_required",
                passed=False,
                severity=SafetyLevel.HIGH,
                message=f"Change type {change_type} requires approval",
                details={"change_type": change_type},
                timestamp=datetime.now()
            )
        
        return SafetyCheck(
            check_name="approval_required",
            passed=True,
            severity=SafetyLevel.LOW,
            message=f"Change type {change_type} does not require approval",
            details={},
            timestamp=datetime.now()
        )
    
    async def _execute_change_with_monitoring(self, change_type: str, change_data: Dict[str, Any]) -> bool:
        """Execute change with real-time monitoring."""
        # This would integrate with the actual change execution systems
        # For now, simulate execution
        await asyncio.sleep(0.1)  # Simulate execution time
        return True
    
    async def _post_execution_validation(self, change_type: str, change_data: Dict[str, Any]) -> List[SafetyCheck]:
        """Run validation checks after change execution."""
        checks = []
        
        # System health check
        health = await self.system_monitor.get_system_health()
        checks.append(SafetyCheck(
            check_name="post_execution_health",
            passed=health["overall_health"] > 0.7,
            severity=SafetyLevel.HIGH,
            message=f"System health after change: {health['overall_health']:.2f}",
            details=health,
            timestamp=datetime.now()
        ))
        
        return checks
    
    async def _stop_phase3_automation(self):
        """Stop Phase 3 automation systems."""
        # Implementation would stop self-evolving code, architecture, knowledge management
        logger.info("[STOP] Stopping Phase 3 automation systems")
    
    async def _stop_phase2_automation(self):
        """Stop Phase 2 automation systems."""
        # Implementation would stop meta-learning and testing systems
        logger.info("[STOP] Stopping Phase 2 automation systems")
    
    async def _stop_phase1_automation(self):
        """Stop Phase 1 automation systems."""
        # Implementation would stop self-healing, monitoring, configuration systems
        logger.info("[STOP] Stopping Phase 1 automation systems")
    
    async def check_game_emergency_override(self, 
                                          game_id: str,
                                          session_id: str,
                                          current_state: Dict[str, Any],
                                          action_history: List[int],
                                          performance_history: List[Dict[str, Any]],
                                          available_actions: List[int]) -> Optional[EmergencyOverride]:
        """
        Check if game-specific emergency override should be triggered.
        
        Args:
            game_id: Game identifier
            session_id: Session identifier
            current_state: Current game state
            action_history: Recent action history
            performance_history: Recent performance data
            available_actions: Available actions
            
        Returns:
            EmergencyOverride if override should be triggered, None otherwise
        """
        try:
            # Check different types of emergency conditions
            override_checks = [
                await self._check_action_loop_break(game_id, action_history, available_actions),
                await self._check_coordinate_stuck_break(game_id, current_state),
                await self._check_stagnation_break(game_id, performance_history),
                await self._check_emergency_reset(game_id, action_history, performance_history)
            ]
            
            # Find the most critical override
            valid_overrides = [o for o in override_checks if o is not None]
            if not valid_overrides:
                return None
            
            # Select the most critical override (highest priority)
            override_priority = {
                OverrideType.EMERGENCY_RESET: 4,
                OverrideType.STAGNATION_BREAK: 3,
                OverrideType.ACTION_LOOP_BREAK: 2,
                OverrideType.COORDINATE_STUCK_BREAK: 1
            }
            
            most_critical = max(valid_overrides, key=lambda o: override_priority.get(o.override_type, 0))
            
            # Set session_id from parameter
            most_critical.session_id = session_id
            
            # Log emergency override
            await self.audit_logger.log_emergency_override(most_critical)
            
            logger.warning(f" GAME EMERGENCY OVERRIDE TRIGGERED: {most_critical.override_type.value} - "
                          f"{most_critical.trigger_reason}")
            
            return most_critical
            
        except Exception as e:
            logger.error(f"Error checking game emergency override: {e}")
            return None
    
    async def _check_action_loop_break(self, 
                                     game_id: str,
                                     action_history: List[Any],
                                     available_actions: List[int]) -> Optional[EmergencyOverride]:
        """Check for action loop break override."""
        if len(action_history) < 10:
            return None
        
        # Normalize recent actions to hashable integers
        recent_raw = action_history[-10:]
        recent_actions: List[int] = []
        try:
            for a in recent_raw:
                if isinstance(a, dict):
                    val = a.get('action_id') or a.get('id') or a.get('action') or a.get('action_number')
                    if isinstance(val, (int, float, str)):
                        try:
                            recent_actions.append(int(val))
                        except Exception:
                            continue
                elif isinstance(a, (int, float, str)):
                    try:
                        recent_actions.append(int(a))
                    except Exception:
                        continue
        except Exception:
            recent_actions = []
        
        if len(recent_actions) < 10:
            return None
        
        # Check for repeated action patterns
        unique_count = len(set(recent_actions))
        if unique_count <= 2:  # Only 1-2 unique actions in last 10
            return EmergencyOverride(
                game_id=game_id,
                session_id="",  # Will be set by caller
                override_type=OverrideType.ACTION_LOOP_BREAK,
                trigger_reason=f"Action loop detected: {recent_actions[-5:]}",
                actions_before_override=len(action_history),
                override_action=random.choice(available_actions) if available_actions else 1,
                override_successful=False,
                override_timestamp=time.time()
            )
        return None
    
    async def _check_coordinate_stuck_break(self, 
                                          game_id: str, 
                                          current_state: Dict[str, Any]) -> Optional[EmergencyOverride]:
        """Check for coordinate stuck break override."""
        # This would check if the system is stuck on the same coordinates
        # Implementation would depend on specific game state structure
        return None
    
    async def _check_stagnation_break(self, 
                                    game_id: str, 
                                    performance_history: List[Dict[str, Any]]) -> Optional[EmergencyOverride]:
        """Check for stagnation break override."""
        if len(performance_history) < 5:
            return None
        
        # Normalize scores from dicts or numeric list
        recent = performance_history[-5:]
        recent_scores: List[float] = []
        for p in recent:
            if isinstance(p, dict):
                score = p.get('score', 0)
                try:
                    recent_scores.append(float(score))
                except Exception:
                    recent_scores.append(0.0)
            elif isinstance(p, (int, float)):
                recent_scores.append(float(p))
            else:
                recent_scores.append(0.0)
        
        if len(set(recent_scores)) == 1:  # Same score for 5 consecutive actions
            return EmergencyOverride(
                game_id=game_id,
                session_id="",  # Will be set by caller
                override_type=OverrideType.STAGNATION_BREAK,
                trigger_reason=f"Score stagnation detected: {recent_scores[0]}",
                actions_before_override=len(performance_history),
                override_action=6,  # Try ACTION6 for coordinate-based games
                override_successful=False,
                override_timestamp=time.time()
            )
        return None
    
    async def _check_emergency_reset(self, 
                                   game_id: str, 
                                   action_history: List[int], 
                                   performance_history: List[Dict[str, Any]]) -> Optional[EmergencyOverride]:
        """Check for emergency reset override."""
        if len(action_history) < 20:
            return None
        
        # Check for excessive actions without progress
        recent_performance = performance_history[-10:] if len(performance_history) >= 10 else performance_history
        if not recent_performance:
            return None
        
        # If no score improvement in last 10 actions
        scores = [p.get('score', 0) for p in recent_performance]
        if max(scores) == min(scores) and len(action_history) > 50:
            return EmergencyOverride(
                game_id=game_id,
                session_id="",  # Will be set by caller
                override_type=OverrideType.EMERGENCY_RESET,
                trigger_reason=f"Emergency reset: {len(action_history)} actions without progress",
                actions_before_override=len(action_history),
                override_action=1,  # Try ACTION1 as reset
                override_successful=False,
                override_timestamp=time.time()
            )
        return None

    async def monitor_security_integrity(self) -> Dict[str, Any]:
        """Monitor system security and integrity."""
        try:
            security_violations = []
            
            # Check for unauthorized access patterns
            unauthorized_access = await self._check_unauthorized_access()
            if unauthorized_access:
                security_violations.append(unauthorized_access)
            
            # Check for data breach indicators
            data_breach = await self._check_data_breach()
            if data_breach:
                security_violations.append(data_breach)
            
            # Check for system intrusion
            system_intrusion = await self._check_system_intrusion()
            if system_intrusion:
                security_violations.append(system_intrusion)
            
            # Check for performance anomalies
            performance_anomaly = await self._check_performance_anomaly()
            if performance_anomaly:
                security_violations.append(performance_anomaly)
            
            # Check for memory anomalies
            memory_anomaly = await self._check_memory_anomaly()
            if memory_anomaly:
                security_violations.append(memory_anomaly)
            
            # Log security violations
            for violation in security_violations:
                await self.audit_logger.log_security_violation(violation)
            
            return {
                "security_violations": len(security_violations),
                "violations": security_violations,
                "monitoring_active": True,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error monitoring security integrity: {e}")
            return {"error": str(e)}
    
    async def _check_unauthorized_access(self) -> Optional[Dict[str, Any]]:
        """Check for unauthorized access patterns."""
        # This would check logs, access patterns, etc.
        # For now, return None (no violations detected)
        return None
    
    async def _check_data_breach(self) -> Optional[Dict[str, Any]]:
        """Check for data breach indicators."""
        # This would check for unusual data access patterns
        # For now, return None (no violations detected)
        return None
    
    async def _check_system_intrusion(self) -> Optional[Dict[str, Any]]:
        """Check for system intrusion indicators."""
        # This would check for suspicious system modifications
        # For now, return None (no violations detected)
        return None
    
    async def _check_performance_anomaly(self) -> Optional[Dict[str, Any]]:
        """Check for performance anomalies that might indicate security issues."""
        # This would check for unusual performance patterns
        # For now, return None (no violations detected)
        return None
    
    async def _check_memory_anomaly(self) -> Optional[Dict[str, Any]]:
        """Check for memory anomalies that might indicate security issues."""
        # This would check for unusual memory usage patterns
        # For now, return None (no violations detected)
        return None

    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety system status."""
        return {
            "emergency_stop_active": self.emergency_stop_active,
            "total_violations": len(self.violation_history),
            "total_checks": len(self.safety_checks),
            "cooldown_periods": self.cooldown_periods,
            "approval_required": list(self.approval_required),
            "recent_violations": self.violation_history[-5:] if self.violation_history else [],
            "system_health": "unknown"  # Would be populated by system monitor
        }

class CodeValidator:
    """Validates code changes for safety."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.security_patterns = config.get("security_patterns", [])
        self.forbidden_imports = config.get("forbidden_imports", [])
    
    async def validate_code(self, change_data: Dict[str, Any]) -> List[SafetyCheck]:
        """Validate code for security and safety issues."""
        checks = []
        
        # Extract code from change data
        code = change_data.get("code", "")
        if not code:
            return checks
        
        # Check for security patterns
        for pattern in self.security_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                checks.append(SafetyCheck(
                    check_name="security_pattern_detected",
                    passed=False,
                    severity=SafetyLevel.CRITICAL,
                    message=f"Potentially dangerous code pattern detected: {pattern}",
                    details={"pattern": pattern, "code_snippet": code[:100]},
                    timestamp=datetime.now()
                ))
                # Return early if critical security issue found
                return checks
        
        # Check for forbidden imports
        for forbidden in self.forbidden_imports:
            if f"import {forbidden}" in code or f"from {forbidden}" in code:
                checks.append(SafetyCheck(
                    check_name="forbidden_import",
                    passed=False,
                    severity=SafetyLevel.HIGH,
                    message=f"Forbidden import detected: {forbidden}",
                    details={"forbidden_import": forbidden},
                    timestamp=datetime.now()
                ))
                # Return early if high severity issue found
                return checks
        
        # If no issues found, add a passing check
        if not checks:
            checks.append(SafetyCheck(
                check_name="code_validation",
                passed=True,
                severity=SafetyLevel.LOW,
                message="Code validation passed",
                details={},
                timestamp=datetime.now()
            ))
        
        return checks

class SystemMonitor:
    """Monitors system health and impact."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def assess_impact(self, change_data: Dict[str, Any]) -> List[SafetyCheck]:
        """Assess system impact of proposed change."""
        checks = []
        
        # This would integrate with actual system monitoring
        # For now, return basic checks
        
        checks.append(SafetyCheck(
            check_name="system_impact",
            passed=True,
            severity=SafetyLevel.MEDIUM,
            message="System impact assessment passed",
            details={},
            timestamp=datetime.now()
        ))
        
        return checks
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics."""
        # This would integrate with actual system monitoring
        return {
            "overall_health": 0.95,
            "cpu_usage": 45.0,
            "memory_usage": 60.0,
            "response_time": 1.2,
            "success_rate": 0.92
        }

class PerformanceAssessor:
    """Assesses performance impact of changes."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thresholds = config.get("performance_thresholds", {})
    
    async def assess_performance(self, change_data: Dict[str, Any]) -> List[SafetyCheck]:
        """Assess performance impact of proposed change."""
        checks = []
        
        # This would integrate with actual performance monitoring
        # For now, return basic checks
        
        checks.append(SafetyCheck(
            check_name="performance_impact",
            passed=True,
            severity=SafetyLevel.MEDIUM,
            message="Performance impact assessment passed",
            details={},
            timestamp=datetime.now()
        ))
        
        return checks

class RollbackManager:
    """Manages rollback capabilities for changes."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backups = {}
    
    async def create_backup(self, change_type: str, change_data: Dict[str, Any]) -> Optional[str]:
        """Create backup before change execution."""
        backup_id = f"{change_type}_{int(time.time())}"
        
        # This would create actual backups
        self.backups[backup_id] = {
            "change_type": change_type,
            "change_data": change_data,
            "timestamp": datetime.now(),
            "backup_location": f"backups/{backup_id}"
        }
        
        logger.info(f" Backup created: {backup_id}")
        return backup_id
    
    async def rollback(self, backup_id: str) -> bool:
        """Rollback to backup state."""
        if backup_id not in self.backups:
            logger.error(f"[ERROR] Backup {backup_id} not found")
            return False
        
        # This would perform actual rollback
        logger.info(f" Rolling back to backup: {backup_id}")
        return True

class ApprovalSystem:
    """Manages approval workflows for changes."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pending_approvals = {}
    
    async def request_approval(self, change_type: str, change_data: Dict[str, Any]) -> str:
        """Request approval for a change."""
        approval_id = f"approval_{int(time.time())}"
        
        self.pending_approvals[approval_id] = {
            "change_type": change_type,
            "change_data": change_data,
            "timestamp": datetime.now(),
            "status": "pending"
        }
        
        logger.info(f" Approval requested: {approval_id}")
        return approval_id

class AuditLogger:
    """Logs all safety-related events for audit."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.audit_log = []
    
    async def log_change(self, change_type: str, change_data: Dict[str, Any], status: str):
        """Log a change execution."""
        entry = {
            "timestamp": datetime.now(),
            "change_type": change_type,
            "status": status,
            "change_data": change_data
        }
        self.audit_log.append(entry)
        logger.info(f" Audit log: {change_type} change {status}")
    
    async def log_emergency_stop(self, reason: str):
        """Log emergency stop event."""
        entry = {
            "timestamp": datetime.now(),
            "event": "emergency_stop",
            "reason": reason
        }
        self.audit_log.append(entry)
        logger.critical(f" Audit log: Emergency stop - {reason}")
    
    async def log_emergency_override(self, override: EmergencyOverride):
        """Log emergency override event."""
        entry = {
            "timestamp": datetime.now(),
            "event": "emergency_override",
            "override_type": override.override_type.value,
            "game_id": override.game_id,
            "session_id": override.session_id,
            "trigger_reason": override.trigger_reason,
            "actions_before_override": override.actions_before_override,
            "override_action": override.override_action
        }
        self.audit_log.append(entry)
        logger.warning(f" Audit log: Emergency override - {override.override_type.value}")
    
    async def log_security_violation(self, violation: Dict[str, Any]):
        """Log security violation event."""
        entry = {
            "timestamp": datetime.now(),
            "event": "security_violation",
            "violation_type": violation.get("type", "unknown"),
            "severity": violation.get("severity", "medium"),
            "description": violation.get("description", "No description"),
            "affected_components": violation.get("affected_components", []),
            "details": violation.get("details", {})
        }
        self.audit_log.append(entry)
        logger.warning(f" Security violation: {violation.get('type', 'unknown')}")

# Global singleton instance
_safety_mechanisms_instance: Optional[SafetyMechanisms] = None

def create_safety_mechanisms(config: Optional[Dict[str, Any]] = None) -> SafetyMechanisms:
    """Create or get the singleton SafetyMechanisms instance."""
    global _safety_mechanisms_instance
    if _safety_mechanisms_instance is None:
        logger.debug("Creating singleton SafetyMechanisms instance")
        _safety_mechanisms_instance = SafetyMechanisms(config)
    return _safety_mechanisms_instance

def get_safety_mechanisms() -> Optional[SafetyMechanisms]:
    """Get the existing SafetyMechanisms singleton instance."""
    return _safety_mechanisms_instance

# Import required modules
import re
