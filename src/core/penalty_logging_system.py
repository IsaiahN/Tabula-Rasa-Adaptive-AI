#!/usr/bin/env python3
"""
Penalty Logging System - Comprehensive logging for penalty application and decay events.

This system provides detailed logging for:
- Penalty application events
- Penalty decay events
- Coordinate avoidance decisions
- Failure learning insights
- System performance metrics
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

from src.database.system_integration import get_system_integration
from src.database.api import LogLevel, Component


class PenaltyEventType(Enum):
    """Types of penalty events."""
    PENALTY_APPLIED = "penalty_applied"
    PENALTY_DECAYED = "penalty_decayed"
    COORDINATE_AVOIDED = "coordinate_avoided"
    RECOVERY_ATTEMPTED = "recovery_attempted"
    FAILURE_LEARNED = "failure_learned"
    DIVERSITY_PROMOTED = "diversity_promoted"
    SYSTEM_STATUS_UPDATE = "system_status_update"


@dataclass
class PenaltyEvent:
    """Structured penalty event data."""
    event_type: PenaltyEventType
    game_id: str
    coordinate: Optional[tuple] = None
    penalty_score: Optional[float] = None
    penalty_reason: Optional[str] = None
    old_penalty: Optional[float] = None
    new_penalty: Optional[float] = None
    decay_factor: Optional[float] = None
    avoidance_score: Optional[float] = None
    diversity_factor: Optional[float] = None
    failure_type: Optional[str] = None
    learned_insights: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class PenaltyLoggingSystem:
    """Comprehensive logging system for penalty decay events."""
    
    def __init__(self):
        self.integration = get_system_integration()
        self.logger = logging.getLogger(__name__)
        
        # Event storage for analysis
        self.recent_events = []
        self.max_recent_events = 1000
        
        # Performance metrics
        self.metrics = {
            'total_events_logged': 0,
            'penalty_events': 0,
            'decay_events': 0,
            'avoidance_events': 0,
            'learning_events': 0
        }
    
    async def log_penalty_applied(
        self,
        game_id: str,
        coordinate: tuple,
        penalty_score: float,
        penalty_reason: str,
        context: Dict[str, Any] = None
    ):
        """Log penalty application event."""
        event = PenaltyEvent(
            event_type=PenaltyEventType.PENALTY_APPLIED,
            game_id=game_id,
            coordinate=coordinate,
            penalty_score=penalty_score,
            penalty_reason=penalty_reason,
            context=context or {}
        )
        
        await self._log_event(event)
        
        # Log to system integration
        await self.integration.log_system_event(
            level=LogLevel.INFO,
            component=Component.LEARNING_LOOP,
            message=f"Penalty applied to coordinate {coordinate}: {penalty_reason} (score: {penalty_score:.3f})",
            data={
                'game_id': game_id,
                'coordinate': coordinate,
                'penalty_score': penalty_score,
                'penalty_reason': penalty_reason,
                'context': context
            }
        )
        
        self.logger.info(f" PENALTY APPLIED: {coordinate} - {penalty_reason} (score: {penalty_score:.3f})")
    
    async def log_penalty_decayed(
        self,
        game_id: str,
        coordinate: tuple,
        old_penalty: float,
        new_penalty: float,
        decay_factor: float,
        context: Dict[str, Any] = None
    ):
        """Log penalty decay event."""
        event = PenaltyEvent(
            event_type=PenaltyEventType.PENALTY_DECAYED,
            game_id=game_id,
            coordinate=coordinate,
            old_penalty=old_penalty,
            new_penalty=new_penalty,
            decay_factor=decay_factor,
            context=context or {}
        )
        
        await self._log_event(event)
        
        # Log to system integration
        await self.integration.log_system_event(
            level=LogLevel.INFO,
            component=Component.LEARNING_LOOP,
            message=f"Penalty decayed for coordinate {coordinate}: {old_penalty:.3f} -> {new_penalty:.3f}",
            data={
                'game_id': game_id,
                'coordinate': coordinate,
                'old_penalty': old_penalty,
                'new_penalty': new_penalty,
                'decay_factor': decay_factor,
                'context': context
            }
        )
        
        self.logger.info(f"⏰ PENALTY DECAYED: {coordinate} - {old_penalty:.3f} -> {new_penalty:.3f} (decay: {decay_factor:.3f})")
    
    async def log_coordinate_avoided(
        self,
        game_id: str,
        coordinate: tuple,
        avoidance_score: float,
        reason: str,
        context: Dict[str, Any] = None
    ):
        """Log coordinate avoidance event."""
        event = PenaltyEvent(
            event_type=PenaltyEventType.COORDINATE_AVOIDED,
            game_id=game_id,
            coordinate=coordinate,
            avoidance_score=avoidance_score,
            context=context or {'reason': reason}
        )
        
        await self._log_event(event)
        
        self.logger.info(f" COORDINATE AVOIDED: {coordinate} - {reason} (score: {avoidance_score:.3f})")
    
    async def log_recovery_attempted(
        self,
        game_id: str,
        coordinate: tuple,
        recovery_type: str,
        success: bool,
        context: Dict[str, Any] = None
    ):
        """Log recovery attempt event."""
        event = PenaltyEvent(
            event_type=PenaltyEventType.RECOVERY_ATTEMPTED,
            game_id=game_id,
            coordinate=coordinate,
            context=context or {'recovery_type': recovery_type, 'success': success}
        )
        
        await self._log_event(event)
        
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f" RECOVERY {status}: {coordinate} - {recovery_type}")
    
    async def log_failure_learned(
        self,
        game_id: str,
        coordinate: tuple,
        failure_type: str,
        learned_insights: Dict[str, Any],
        context: Dict[str, Any] = None
    ):
        """Log failure learning event."""
        event = PenaltyEvent(
            event_type=PenaltyEventType.FAILURE_LEARNED,
            game_id=game_id,
            coordinate=coordinate,
            failure_type=failure_type,
            learned_insights=learned_insights,
            context=context or {}
        )
        
        await self._log_event(event)
        
        self.logger.info(f" FAILURE LEARNED: {coordinate} - {failure_type}")
        for insight in learned_insights.get('recommendations', []):
            self.logger.info(f"   {insight}")
    
    async def log_diversity_promoted(
        self,
        game_id: str,
        coordinate: tuple,
        diversity_factor: float,
        reason: str,
        context: Dict[str, Any] = None
    ):
        """Log diversity promotion event."""
        event = PenaltyEvent(
            event_type=PenaltyEventType.DIVERSITY_PROMOTED,
            game_id=game_id,
            coordinate=coordinate,
            diversity_factor=diversity_factor,
            context=context or {'reason': reason}
        )
        
        await self._log_event(event)
        
        self.logger.info(f" DIVERSITY PROMOTED: {coordinate} - {reason} (factor: {diversity_factor:.3f})")
    
    async def log_system_status_update(
        self,
        game_id: str,
        status_data: Dict[str, Any],
        context: Dict[str, Any] = None
    ):
        """Log system status update."""
        event = PenaltyEvent(
            event_type=PenaltyEventType.SYSTEM_STATUS_UPDATE,
            game_id=game_id,
            context=context or status_data
        )
        
        await self._log_event(event)
        
        self.logger.info(f" SYSTEM STATUS UPDATE: {game_id}")
        for key, value in status_data.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"  {key}: {value}")
            elif isinstance(value, dict):
                self.logger.info(f"  {key}: {len(value)} items")
    
    async def _log_event(self, event: PenaltyEvent):
        """Internal method to log event and update metrics."""
        # Add to recent events
        self.recent_events.append(event)
        if len(self.recent_events) > self.max_recent_events:
            self.recent_events = self.recent_events[-self.max_recent_events:]
        
        # Update metrics
        self.metrics['total_events_logged'] += 1
        self.metrics[f'{event.event_type.value}_events'] = self.metrics.get(f'{event.event_type.value}_events', 0) + 1
        
        # Store in database
        await self._store_event_in_database(event)
    
    async def _store_event_in_database(self, event: PenaltyEvent):
        """Store event in database for persistence."""
        try:
            await self.integration.db.execute(
                """
                INSERT INTO system_logs 
                (log_level, component, message, data, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "INFO",  # log_level
                    "PENALTY_SYSTEM",  # component
                    f"Penalty event: {event.event_type.value}",
                    json.dumps(asdict(event), default=str),
                    event.timestamp
                )
            )
        except Exception as e:
            self.logger.error(f"Failed to store event in database: {e}")
    
    async def get_recent_events(
        self, 
        event_type: Optional[PenaltyEventType] = None,
        limit: int = 50
    ) -> List[PenaltyEvent]:
        """Get recent events, optionally filtered by type."""
        events = self.recent_events
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-limit:]
    
    async def get_penalty_summary(self, game_id: str) -> Dict[str, Any]:
        """Get penalty summary for a game."""
        try:
            # Get penalty statistics from database
            penalty_stats = await self.integration.db.fetch_one(
                """
                SELECT 
                    COUNT(*) as total_penalties,
                    AVG(penalty_score) as avg_penalty,
                    MAX(penalty_score) as max_penalty,
                    MIN(penalty_score) as min_penalty,
                    COUNT(CASE WHEN is_stuck_coordinate = 1 THEN 1 END) as stuck_coordinates
                FROM coordinate_penalties
                WHERE game_id = ?
                """,
                (game_id,)
            )
            
            # Get recent events
            recent_events = await self.get_recent_events(limit=20)
            
            # Get failure learning statistics
            failure_stats = await self.integration.db.fetch_all(
                """
                SELECT 
                    failure_type,
                    COUNT(*) as count,
                    AVG(failure_count) as avg_failures
                FROM failure_learning
                WHERE game_id = ?
                GROUP BY failure_type
                """,
                (game_id,)
            )
            
            return {
                'penalty_statistics': penalty_stats[0] if penalty_stats else {},
                'recent_events': [asdict(e) for e in recent_events],
                'failure_statistics': {row['failure_type']: row for row in failure_stats},
                'logging_metrics': self.metrics
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get penalty summary: {e}")
            return {'error': str(e)}
    
    async def generate_penalty_report(self, game_id: str) -> str:
        """Generate a comprehensive penalty report."""
        try:
            summary = await self.get_penalty_summary(game_id)
            
            report = []
            report.append("=" * 60)
            report.append(f"PENALTY DECAY SYSTEM REPORT - {game_id}")
            report.append("=" * 60)
            report.append("")
            
            # Penalty statistics
            penalty_stats = summary.get('penalty_statistics', {})
            report.append(" PENALTY STATISTICS:")
            report.append(f"  Total penalties: {penalty_stats.get('total_penalties', 0)}")
            report.append(f"  Average penalty: {penalty_stats.get('avg_penalty', 0):.3f}")
            report.append(f"  Max penalty: {penalty_stats.get('max_penalty', 0):.3f}")
            report.append(f"  Min penalty: {penalty_stats.get('min_penalty', 0):.3f}")
            report.append(f"  Stuck coordinates: {penalty_stats.get('stuck_coordinates', 0)}")
            report.append("")
            
            # Failure statistics
            failure_stats = summary.get('failure_statistics', {})
            if failure_stats:
                report.append(" FAILURE STATISTICS:")
                for failure_type, stats in failure_stats.items():
                    report.append(f"  {failure_type}: {stats['count']} occurrences (avg: {stats['avg_failures']:.1f})")
                report.append("")
            
            # Recent events
            recent_events = summary.get('recent_events', [])
            if recent_events:
                report.append(" RECENT EVENTS:")
                for event in recent_events[-10:]:  # Last 10 events
                    event_type = event['event_type']
                    coordinate = event.get('coordinate')
                    timestamp = event['timestamp']
                    
                    if coordinate:
                        report.append(f"  {timestamp} - {event_type}: {coordinate}")
                    else:
                        report.append(f"  {timestamp} - {event_type}")
                report.append("")
            
            # Logging metrics
            metrics = summary.get('logging_metrics', {})
            report.append(" LOGGING METRICS:")
            for key, value in metrics.items():
                report.append(f"  {key}: {value}")
            
            report.append("=" * 60)
            
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"Failed to generate penalty report: {e}")
            return f"Error generating report: {e}"


# Global instance
_penalty_logging_system = None

def get_penalty_logging_system() -> PenaltyLoggingSystem:
    """Get the global penalty logging system instance."""
    global _penalty_logging_system
    if _penalty_logging_system is None:
        _penalty_logging_system = PenaltyLoggingSystem()
    return _penalty_logging_system
