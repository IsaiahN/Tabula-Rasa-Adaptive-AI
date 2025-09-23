#!/usr/bin/env python3
"""
Unified Automation System
Integrates all phases of automation (Phase 1, 2, 3) with comprehensive coordination.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from .safety_mechanisms import SafetyMechanisms, SafetyLevel
from .unified_performance_monitor import get_performance_monitor

logger = logging.getLogger(__name__)

class AutomationMode(Enum):
    """Automation modes for the unified system."""
    DISABLED = "disabled"
    PHASE1_ONLY = "phase1_only"
    PHASE2_ONLY = "phase2_only"
    PHASE3_ONLY = "phase3_only"
    FULL_AUTOMATION = "full"
    SAFETY_MODE = "safety_mode"

@dataclass
class SystemStatus:
    """Current status of the unified automation system."""
    mode: AutomationMode
    phase1_active: bool
    phase2_active: bool
    phase3_active: bool
    safety_active: bool
    emergency_stop: bool
    overall_health: float
    last_update: datetime
    active_components: List[str]

class UnifiedAutomationSystem:
    """
    Unified Automation System that coordinates all phases of automation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.mode = AutomationMode.DISABLED
        self.status = SystemStatus(
            mode=AutomationMode.DISABLED,
            phase1_active=False,
            phase2_active=False,
            phase3_active=False,
            safety_active=True,
            emergency_stop=False,
            overall_health=1.0,
            last_update=datetime.now(),
            active_components=[]
        )
        
        # Initialize safety mechanisms
        from .safety_mechanisms import SafetyMechanisms
        self.safety_mechanisms = SafetyMechanisms(self.config.get("safety", {}))
        
        # Initialize performance monitoring
        self.performance_monitor = get_performance_monitor()
        
        logger.info("🤖 Unified Automation System initialized")
    
    async def start_automation(self, mode: AutomationMode = AutomationMode.FULL_AUTOMATION) -> bool:
        """Start the unified automation system."""
        logger.info(f"🚀 Starting unified automation system in {mode.value} mode")
        
        try:
            # Safety check
            if not await self._safety_check():
                logger.error("🚨 Safety check failed - cannot start automation")
                return False
            
            # Set mode and update status
            # Start performance monitoring
            self.performance_monitor.start_monitoring()
            
            self.mode = mode
            self.status.mode = mode
            await self._update_status()
            
            logger.info(f"✅ Unified automation system started in {mode.value} mode")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start automation system: {e}")
            return False
    
    async def stop_automation(self, graceful: bool = True) -> bool:
        """Stop the unified automation system."""
        logger.info(f"🛑 Stopping unified automation system (graceful={graceful})")
        
        try:
            # Stop performance monitoring
            self.performance_monitor.stop_monitoring()
            
            # Update status
            self.mode = AutomationMode.DISABLED
            await self._update_status()
            
            logger.info("✅ Unified automation system stopped")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to stop automation system: {e}")
            return False
    
    async def emergency_stop(self, reason: str = "Manual emergency stop") -> bool:
        """Trigger emergency stop of all automation."""
        logger.critical(f"🚨 EMERGENCY STOP: {reason}")
        
        try:
            # Stop all automation
            await self.stop_automation(graceful=False)
            
            # Activate safety mechanisms
            await self.safety_mechanisms.emergency_stop(reason)
            
            # Update status
            self.status.emergency_stop = True
            self.status.mode = AutomationMode.DISABLED
            await self._update_status()
            
            logger.critical("🛑 Emergency stop completed")
            return True
            
        except Exception as e:
            logger.critical(f"🚨 Emergency stop failed: {e}")
            return False
    
    async def switch_mode(self, new_mode: AutomationMode) -> bool:
        """Switch automation mode."""
        logger.info(f"🔄 Switching automation mode from {self.mode.value} to {new_mode.value}")
        
        try:
            # Stop current mode
            await self.stop_automation(graceful=True)
            
            # Start new mode
            success = await self.start_automation(new_mode)
            
            if success:
                logger.info(f"✅ Successfully switched to {new_mode.value} mode")
            else:
                logger.error(f"❌ Failed to switch to {new_mode.value} mode")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error switching mode: {e}")
            return False
    
    async def resume_automation(self, mode: Optional[AutomationMode] = None) -> bool:
        """Resume automation after emergency stop."""
        if not self.status.emergency_stop and not self.safety_mechanisms.emergency_stop_active:
            logger.warning("⚠️ No emergency stop active - nothing to resume")
            return False
        
        logger.info("🔄 Resuming automation after emergency stop")
        
        try:
            # Safety check
            if not await self._safety_check():
                logger.error("🚨 Safety check failed - cannot resume automation")
                return False
            
            # Resume safety mechanisms
            await self.safety_mechanisms.resume_automation()
            
            # Start automation in specified mode
            resume_mode = mode or self.mode
            success = await self.start_automation(resume_mode)
            
            if success:
                self.status.emergency_stop = False
                await self._update_status()
                logger.info("✅ Automation resumed successfully")
            else:
                logger.error("❌ Failed to resume automation")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error resuming automation: {e}")
            return False
    
    async def get_system_status(self) -> SystemStatus:
        """Get current system status."""
        await self._update_status()
        return self.status
    
    async def _safety_check(self) -> bool:
        """Run comprehensive safety check before starting automation."""
        try:
            # Check safety mechanisms
            safety_status = self.safety_mechanisms.get_safety_status()
            if safety_status["emergency_stop_active"]:
                logger.warning("⚠️ Emergency stop is active")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Safety check failed: {e}")
            return False
    
    async def _update_status(self):
        """Update system status."""
        self.status.last_update = datetime.now()
        
        # Update active components based on mode
        active_components = []
        if self.mode in [AutomationMode.PHASE1_ONLY, AutomationMode.FULL_AUTOMATION, AutomationMode.SAFETY_MODE]:
            active_components.append("phase1")
        if self.mode in [AutomationMode.PHASE2_ONLY, AutomationMode.FULL_AUTOMATION, AutomationMode.SAFETY_MODE]:
            active_components.append("phase2")
        if self.mode in [AutomationMode.PHASE3_ONLY, AutomationMode.FULL_AUTOMATION, AutomationMode.SAFETY_MODE]:
            active_components.append("phase3")
        
        self.status.active_components = active_components
        self.status.phase1_active = "phase1" in active_components
        self.status.phase2_active = "phase2" in active_components
        self.status.phase3_active = "phase3" in active_components