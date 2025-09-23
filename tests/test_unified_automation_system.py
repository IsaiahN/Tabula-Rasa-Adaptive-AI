#!/usr/bin/env python3
"""
Comprehensive test suite for the Unified Automation System and Safety Mechanisms.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from src.core.safety_mechanisms import (
    SafetyMechanisms, SafetyLevel, SafetyViolation, 
    SafetyCheck, SafetyViolationReport
)
from src.core.unified_automation_system import (
    UnifiedAutomationSystem, AutomationMode, SystemStatus
)

class TestSafetyMechanisms:
    """Test the Safety Mechanisms system."""
    
    @pytest.fixture
    def safety_mechanisms(self):
        """Create a SafetyMechanisms instance for testing."""
        config = {
            "max_code_changes_per_hour": 5,
            "emergency_stop_threshold": 3,
            "cooldown_periods": {
                "code_evolution": 3600,
                "architecture_change": 86400,
                "knowledge_update": 1800
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
        return SafetyMechanisms(config)
    
    @pytest.mark.asyncio
    async def test_validate_change_safe(self, safety_mechanisms):
        """Test validating a safe change."""
        change_data = {
            "code": "def safe_function():\n    return 'safe'",
            "change_type": "code_evolution"
        }
        
        is_safe, checks = await safety_mechanisms.validate_change(
            "code_evolution", change_data, SafetyLevel.MEDIUM
        )
        
        assert is_safe
        assert len(checks) > 0
        assert all(check.passed for check in checks if check.severity in [SafetyLevel.CRITICAL, SafetyLevel.HIGH])
    
    @pytest.mark.asyncio
    async def test_validate_change_unsafe_code(self, safety_mechanisms):
        """Test validating unsafe code."""
        change_data = {
            "code": "import subprocess\nsubprocess.call(['rm', '-rf', '/'])",
            "change_type": "code_evolution"
        }
        
        is_safe, checks = await safety_mechanisms.validate_change(
            "code_evolution", change_data, SafetyLevel.MEDIUM
        )
        
        assert not is_safe
        assert any(not check.passed for check in checks)
    
    @pytest.mark.asyncio
    async def test_cooldown_period_check(self, safety_mechanisms):
        """Test cooldown period enforcement."""
        # Set a cooldown period
        safety_mechanisms.cooldown_periods["code_evolution"] = datetime.now()
        
        change_data = {"code": "def test(): pass", "change_type": "code_evolution"}
        
        is_safe, checks = await safety_mechanisms.validate_change(
            "code_evolution", change_data, SafetyLevel.MEDIUM
        )
        
        # Should fail due to cooldown
        cooldown_check = next((c for c in checks if c.check_name == "cooldown_period"), None)
        assert cooldown_check is not None
        assert not cooldown_check.passed
    
    @pytest.mark.asyncio
    async def test_emergency_stop(self, safety_mechanisms):
        """Test emergency stop functionality."""
        result = await safety_mechanisms.emergency_stop("Test emergency stop")
        
        assert result
        assert safety_mechanisms.emergency_stop_active
    
    @pytest.mark.asyncio
    async def test_resume_automation(self, safety_mechanisms):
        """Test resuming automation after emergency stop."""
        # First trigger emergency stop
        await safety_mechanisms.emergency_stop("Test stop")
        
        # Then resume
        result = await safety_mechanisms.resume_automation()
        
        assert result
        assert not safety_mechanisms.emergency_stop_active
    
    def test_get_safety_status(self, safety_mechanisms):
        """Test getting safety status."""
        status = safety_mechanisms.get_safety_status()
        
        assert "emergency_stop_active" in status
        assert "total_violations" in status
        assert "total_checks" in status
        assert isinstance(status["emergency_stop_active"], bool)

class TestUnifiedAutomationSystem:
    """Test the Unified Automation System."""
    
    @pytest.fixture
    def unified_system(self):
        """Create a UnifiedAutomationSystem instance for testing."""
        config = {
            "safety": {
                "emergency_stop_threshold": 3,
                "max_changes_per_hour": 5
            }
        }
        return UnifiedAutomationSystem(config)
    
    @pytest.mark.asyncio
    async def test_start_automation(self, unified_system):
        """Test starting automation system."""
        result = await unified_system.start_automation(AutomationMode.PHASE1_ONLY)
        
        assert result
        assert unified_system.mode == AutomationMode.PHASE1_ONLY
        assert unified_system.status.phase1_active
    
    @pytest.mark.asyncio
    async def test_stop_automation(self, unified_system):
        """Test stopping automation system."""
        # Start first
        await unified_system.start_automation(AutomationMode.PHASE1_ONLY)
        
        # Then stop
        result = await unified_system.stop_automation()
        
        assert result
        assert unified_system.mode == AutomationMode.DISABLED
    
    @pytest.mark.asyncio
    async def test_emergency_stop(self, unified_system):
        """Test emergency stop functionality."""
        # Start automation first
        await unified_system.start_automation(AutomationMode.PHASE1_ONLY)
        
        # Trigger emergency stop
        result = await unified_system.emergency_stop("Test emergency stop")
        
        assert result
        assert unified_system.status.emergency_stop
        assert unified_system.mode == AutomationMode.DISABLED
    
    @pytest.mark.asyncio
    async def test_get_system_status(self, unified_system):
        """Test getting system status."""
        await unified_system.start_automation(AutomationMode.PHASE1_ONLY)
        
        status = await unified_system.get_system_status()
        
        assert isinstance(status, SystemStatus)
        assert status.mode == AutomationMode.PHASE1_ONLY
        assert status.phase1_active
        assert not status.emergency_stop
    
    @pytest.mark.asyncio
    async def test_mode_switching(self, unified_system):
        """Test switching between automation modes."""
        # Start in PHASE1_ONLY
        await unified_system.start_automation(AutomationMode.PHASE1_ONLY)
        assert unified_system.mode == AutomationMode.PHASE1_ONLY
        
        # Switch to FULL_AUTOMATION
        result = await unified_system.switch_mode(AutomationMode.FULL_AUTOMATION)
        
        assert result
        assert unified_system.mode == AutomationMode.FULL_AUTOMATION
    
    def test_automation_modes(self):
        """Test that all automation modes are properly defined."""
        modes = [
            AutomationMode.DISABLED,
            AutomationMode.PHASE1_ONLY,
            AutomationMode.PHASE2_ONLY,
            AutomationMode.PHASE3_ONLY,
            AutomationMode.FULL_AUTOMATION,
            AutomationMode.SAFETY_MODE
        ]
        
        for mode in modes:
            assert isinstance(mode.value, str)
            assert len(mode.value) > 0

class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_safety_mechanisms_integration(self):
        """Test safety mechanisms integration with unified system."""
        config = {
            "safety": {
                "emergency_stop_threshold": 2,
                "max_changes_per_hour": 3
            }
        }
        
        system = UnifiedAutomationSystem(config)
        
        # Test that safety mechanisms are initialized
        assert system.safety_mechanisms is not None
        assert isinstance(system.safety_mechanisms, SafetyMechanisms)
    
    @pytest.mark.asyncio
    async def test_full_automation_workflow(self):
        """Test complete automation workflow."""
        config = {
            "safety": {
                "emergency_stop_threshold": 5,
                "max_changes_per_hour": 10
            }
        }
        
        system = UnifiedAutomationSystem(config)
        
        # 1. Start automation
        result = await system.start_automation(AutomationMode.FULL_AUTOMATION)
        assert result
        
        # 2. Check status
        status = await system.get_system_status()
        assert status.mode == AutomationMode.FULL_AUTOMATION
        assert not status.emergency_stop
        
        # 3. Emergency stop
        result = await system.emergency_stop("Integration test")
        assert result
        assert system.status.emergency_stop
        
        # 4. Stop (emergency stop already stopped everything)
        result = await system.stop_automation()
        assert result
        assert system.mode == AutomationMode.DISABLED

class TestSafetyLevels:
    """Test safety level functionality."""
    
    def test_safety_levels(self):
        """Test that all safety levels are properly defined."""
        levels = [SafetyLevel.LOW, SafetyLevel.MEDIUM, SafetyLevel.HIGH, SafetyLevel.CRITICAL]
        
        for level in levels:
            assert isinstance(level.value, str)
            assert len(level.value) > 0
    
    def test_safety_violations(self):
        """Test that all safety violation types are properly defined."""
        violations = [
            SafetyViolation.CODE_INJECTION,
            SafetyViolation.SYSTEM_OVERRIDE,
            SafetyViolation.DATA_CORRUPTION,
            SafetyViolation.PERFORMANCE_DEGRADATION,
            SafetyViolation.SECURITY_BREACH,
            SafetyViolation.UNINTENDED_BEHAVIOR,
            SafetyViolation.RESOURCE_EXHAUSTION
        ]
        
        for violation in violations:
            assert isinstance(violation.value, str)
            assert len(violation.value) > 0

class TestSafetyCheck:
    """Test SafetyCheck dataclass."""
    
    def test_safety_check_creation(self):
        """Test creating a SafetyCheck instance."""
        check = SafetyCheck(
            check_name="test_check",
            passed=True,
            severity=SafetyLevel.MEDIUM,
            message="Test message",
            details={"key": "value"},
            timestamp=datetime.now()
        )
        
        assert check.check_name == "test_check"
        assert check.passed is True
        assert check.severity == SafetyLevel.MEDIUM
        assert check.message == "Test message"
        assert check.details == {"key": "value"}
        assert isinstance(check.timestamp, datetime)

class TestPerformance:
    """Performance tests for the automation system."""
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent operations on the automation system."""
        system = UnifiedAutomationSystem()
        
        # Start multiple concurrent operations
        tasks = [
            system.start_automation(AutomationMode.PHASE1_ONLY),
            system.get_system_status()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that operations completed without exceptions
        for result in results:
            assert not isinstance(result, Exception)
        
        # Test safety status separately (not async)
        safety_status = system.safety_mechanisms.get_safety_status()
        assert isinstance(safety_status, dict)
    
    @pytest.mark.asyncio
    async def test_rapid_mode_switching(self):
        """Test rapid mode switching."""
        system = UnifiedAutomationSystem()
        
        modes = [
            AutomationMode.PHASE1_ONLY,
            AutomationMode.PHASE2_ONLY,
            AutomationMode.PHASE3_ONLY,
            AutomationMode.FULL_AUTOMATION
        ]
        
        for mode in modes:
            result = await system.start_automation(mode)
            assert result
            assert system.mode == mode

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
