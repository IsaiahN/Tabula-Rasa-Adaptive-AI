"""
DIRECTOR ANALYTICS WRAPPER FUNCTIONS
High-level analytics functions that execute the comprehensive queries from director_analytics_queries.md
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from .api import get_database
from .system_integration import get_system_integration

# ============================================================================
# DIRECTOR ANALYTICS WRAPPER FUNCTIONS
# ============================================================================

class DirectorAnalytics:
    """
    High-level analytics wrapper that executes comprehensive queries from director_analytics_queries.md
    Provides easy-to-use methods for deep gameplay insights and strategic analysis.
    """
    
    def __init__(self):
        self.db = get_database()
        self.integration = get_system_integration()
        self.logger = logging.getLogger(__name__)
    
    # ============================================================================
    # GAMEPLAY STATE & PROGRESSION ANALYTICS
    # ============================================================================
    
    async def get_current_session_overview(self) -> Dict[str, Any]:
        """Get real-time session status and performance."""
        query = """
        SELECT 
            ts.session_id,
            ts.mode,
            ts.status,
            ts.total_actions,
            ts.total_wins,
            ts.total_games,
            ROUND(ts.win_rate * 100, 2) as win_rate_percent,
            ROUND(ts.avg_score, 2) as avg_score,
            ts.energy_level,
            ts.memory_operations,
            ts.sleep_cycles,
            datetime(ts.start_time) as session_start,
            CASE 
                WHEN ts.end_time IS NOT NULL 
                THEN ROUND((julianday(ts.end_time) - julianday(ts.start_time)) * 24 * 60, 2)
                ELSE ROUND((julianday('now') - julianday(ts.start_time)) * 24 * 60, 2)
            END as duration_minutes
        FROM training_sessions ts
        WHERE ts.status = 'running'
        ORDER BY ts.start_time DESC
        LIMIT 10
        """
        
        results = await self.db.execute_query(query)
        
        return {
            "active_sessions": results,
            "session_count": len(results),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def get_game_specific_progress(self, game_types: List[str] = None) -> Dict[str, Any]:
        """Analyze progress for specific games."""
        if game_types is None:
            game_types = ['sp80', 'vc33', 'test']
        
        game_filter = " OR ".join([f"gr.game_id LIKE '%{gt}%'" for gt in game_types])
        
        query = f"""
        SELECT 
            gr.game_id,
            gr.session_id,
            gr.status,
            gr.final_score,
            gr.total_actions,
            gr.win_detected,
            gr.level_completions,
            gr.coordinate_attempts,
            gr.coordinate_successes,
            ROUND(CAST(gr.coordinate_successes AS FLOAT) / NULLIF(gr.coordinate_attempts, 0) * 100, 2) as coordinate_success_rate,
            gr.frame_changes,
            datetime(gr.start_time) as game_start,
            CASE 
                WHEN gr.end_time IS NOT NULL 
                THEN ROUND((julianday(gr.end_time) - julianday(gr.start_time)) * 24 * 60, 2)
                ELSE ROUND((julianday('now') - julianday(gr.start_time)) * 24 * 60, 2)
            END as duration_minutes
        FROM game_results gr
        WHERE {game_filter}
        ORDER BY gr.start_time DESC
        LIMIT 20
        """
        
        results = await self.db.execute_query(query)
        
        return {
            "game_progress": results,
            "game_count": len(results),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def get_performance_trends(self, days: int = 7) -> Dict[str, Any]:
        """Track performance trends across sessions."""
        query = f"""
        SELECT 
            DATE(ts.start_time) as session_date,
            COUNT(*) as sessions_count,
            ROUND(AVG(ts.win_rate) * 100, 2) as avg_win_rate,
            ROUND(AVG(ts.avg_score), 2) as avg_score,
            ROUND(AVG(ts.total_actions), 0) as avg_actions,
            ROUND(AVG(ts.energy_level), 1) as avg_energy,
            SUM(ts.total_wins) as total_wins,
            SUM(ts.total_games) as total_games
        FROM training_sessions ts
        WHERE ts.start_time >= datetime('now', '-{days} days')
        GROUP BY DATE(ts.start_time)
        ORDER BY session_date DESC
        """
        
        results = await self.db.execute_query(query)
        
        return {
            "performance_trends": results,
            "trend_days": days,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    # ============================================================================
    # ACTION INTELLIGENCE & STRATEGY ANALYTICS
    # ============================================================================
    
    async def get_action_effectiveness_by_game_type(self) -> Dict[str, Any]:
        """Find the most effective actions for different game types."""
        query = """
        SELECT 
            SUBSTR(ae.game_id, 1, 4) as game_type,
            ae.action_number,
            COUNT(*) as games_played,
            SUM(ae.attempts) as total_attempts,
            SUM(ae.successes) as total_successes,
            ROUND(AVG(ae.success_rate) * 100, 2) as avg_success_rate,
            ROUND(AVG(ae.avg_score_impact), 2) as avg_score_impact,
            MAX(ae.last_used) as last_used
        FROM action_effectiveness ae
        WHERE ae.attempts > 0
        GROUP BY SUBSTR(ae.game_id, 1, 4), ae.action_number
        HAVING total_attempts >= 5
        ORDER BY game_type, avg_success_rate DESC
        """
        
        results = await self.db.execute_query(query)
        
        return {
            "action_effectiveness": results,
            "action_count": len(results),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def get_coordinate_intelligence_analysis(self) -> Dict[str, Any]:
        """Analyze coordinate effectiveness patterns."""
        query = """
        SELECT 
            SUBSTR(ci.game_id, 1, 4) as game_type,
            ci.x,
            ci.y,
            COUNT(*) as games_played,
            SUM(ci.attempts) as total_attempts,
            SUM(ci.successes) as total_successes,
            ROUND(AVG(ci.success_rate) * 100, 2) as avg_success_rate,
            ROUND(AVG(ci.frame_changes), 1) as avg_frame_changes,
            MAX(ci.last_used) as last_used
        FROM coordinate_intelligence ci
        WHERE ci.attempts > 0
        GROUP BY SUBSTR(ci.game_id, 1, 4), ci.x, ci.y
        HAVING total_attempts >= 3
        ORDER BY game_type, avg_success_rate DESC, total_attempts DESC
        LIMIT 50
        """
        
        results = await self.db.execute_query(query)
        
        return {
            "coordinate_intelligence": results,
            "coordinate_count": len(results),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def get_winning_sequence_analysis(self) -> Dict[str, Any]:
        """Find the most successful action sequences."""
        query = """
        SELECT 
            ws.game_id,
            ws.sequence,
            ws.frequency,
            ROUND(ws.avg_score, 2) as avg_score,
            ROUND(ws.success_rate * 100, 2) as success_rate_percent,
            ws.last_used,
            LENGTH(ws.sequence) - LENGTH(REPLACE(ws.sequence, ',', '')) + 1 as sequence_length
        FROM winning_sequences ws
        WHERE ws.frequency > 1
        ORDER BY ws.success_rate DESC, ws.frequency DESC, ws.avg_score DESC
        LIMIT 20
        """
        
        results = await self.db.execute_query(query)
        
        return {
            "winning_sequences": results,
            "sequence_count": len(results),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    # ============================================================================
    # ERROR DETECTION & DEBUGGING ANALYTICS
    # ============================================================================
    
    async def get_critical_error_analysis(self) -> Dict[str, Any]:
        """Find the most frequent and critical errors."""
        query = """
        SELECT 
            el.error_type,
            el.error_message,
            el.occurrence_count,
            el.first_seen,
            el.last_seen,
            el.resolved,
            ROUND((julianday('now') - julianday(el.first_seen)) * 24, 1) as hours_since_first,
            ROUND((julianday('now') - julianday(el.last_seen)) * 24, 1) as hours_since_last
        FROM error_logs el
        WHERE el.resolved = FALSE
        ORDER BY el.occurrence_count DESC, el.last_seen DESC
        LIMIT 20
        """
        
        results = await self.db.execute_query(query)
        
        return {
            "critical_errors": results,
            "error_count": len(results),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def get_system_health_monitoring(self) -> Dict[str, Any]:
        """Monitor system component health."""
        query = """
        SELECT 
            sl.component,
            sl.log_level,
            COUNT(*) as log_count,
            MAX(sl.timestamp) as last_log,
            ROUND((julianday('now') - julianday(MAX(sl.timestamp))) * 24 * 60, 1) as minutes_since_last_log
        FROM system_logs sl
        WHERE sl.timestamp >= datetime('now', '-1 hour')
        GROUP BY sl.component, sl.log_level
        ORDER BY sl.component, 
            CASE sl.log_level 
                WHEN 'CRITICAL' THEN 1 
                WHEN 'ERROR' THEN 2 
                WHEN 'WARNING' THEN 3 
                WHEN 'INFO' THEN 4 
                WHEN 'DEBUG' THEN 5 
            END
        """
        
        results = await self.db.execute_query(query)
        
        return {
            "system_health": results,
            "component_count": len(set(r.get('component', '') for r in results)),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def get_performance_degradation_detection(self) -> Dict[str, Any]:
        """Detect performance degradation patterns."""
        query = """
        SELECT 
            gr.game_id,
            gr.session_id,
            gr.final_score,
            gr.total_actions,
            gr.win_detected,
            ROUND(CAST(gr.coordinate_successes AS FLOAT) / NULLIF(gr.coordinate_attempts, 0) * 100, 2) as coordinate_success_rate,
            gr.frame_changes,
            datetime(gr.start_time) as game_start,
            CASE 
                WHEN gr.final_score = 0 AND gr.total_actions > 50 THEN 'High Actions, No Score'
                WHEN gr.coordinate_attempts > 0 AND gr.coordinate_successes = 0 THEN 'Coordinate Failures'
                WHEN gr.frame_changes = 0 AND gr.total_actions > 10 THEN 'No Frame Changes'
                ELSE 'Normal'
            END as issue_type
        FROM game_results gr
        WHERE gr.start_time >= datetime('now', '-24 hours')
            AND (gr.final_score = 0 AND gr.total_actions > 50)
            OR (gr.coordinate_attempts > 0 AND gr.coordinate_successes = 0)
            OR (gr.frame_changes = 0 AND gr.total_actions > 10)
        ORDER BY gr.start_time DESC
        """
        
        results = await self.db.execute_query(query)
        
        return {
            "performance_issues": results,
            "issue_count": len(results),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    # ============================================================================
    # LEARNING & PATTERN ANALYSIS
    # ============================================================================
    
    async def get_learned_pattern_effectiveness(self) -> Dict[str, Any]:
        """Analyze the effectiveness of learned patterns."""
        query = """
        SELECT 
            lp.pattern_type,
            COUNT(*) as pattern_count,
            ROUND(AVG(lp.confidence), 3) as avg_confidence,
            ROUND(AVG(lp.success_rate), 3) as avg_success_rate,
            SUM(lp.frequency) as total_usage,
            MAX(lp.updated_at) as last_updated
        FROM learned_patterns lp
        GROUP BY lp.pattern_type
        ORDER BY avg_success_rate DESC, total_usage DESC
        """
        
        results = await self.db.execute_query(query)
        
        return {
            "pattern_effectiveness": results,
            "pattern_type_count": len(results),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def get_learning_progress_tracking(self, days: int = 7) -> Dict[str, Any]:
        """Track learning progress over time."""
        query = f"""
        SELECT 
            DATE(ph.timestamp) as date,
            COUNT(DISTINCT ph.session_id) as sessions,
            ROUND(AVG(ph.win_rate) * 100, 2) as avg_win_rate,
            ROUND(AVG(ph.learning_efficiency), 3) as avg_learning_efficiency,
            ROUND(AVG(ph.score), 2) as avg_score,
            COUNT(*) as data_points
        FROM performance_history ph
        WHERE ph.timestamp >= datetime('now', '-{days} days')
        GROUP BY DATE(ph.timestamp)
        ORDER BY date DESC
        """
        
        results = await self.db.execute_query(query)
        
        return {
            "learning_progress": results,
            "tracking_days": days,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def get_stagnation_detection(self) -> Dict[str, Any]:
        """Detect games that might be stuck or stagnating."""
        query = """
        SELECT 
            ft.game_id,
            ft.stagnation_detected,
            COUNT(*) as stagnation_events,
            MAX(ft.timestamp) as last_stagnation,
            ROUND((julianday('now') - julianday(MAX(ft.timestamp))) * 24 * 60, 1) as minutes_since_stagnation
        FROM frame_tracking ft
        WHERE ft.timestamp >= datetime('now', '-1 hour')
            AND ft.stagnation_detected = TRUE
        GROUP BY ft.game_id
        ORDER BY stagnation_events DESC, last_stagnation DESC
        """
        
        results = await self.db.execute_query(query)
        
        return {
            "stagnation_detection": results,
            "stagnated_games": len(results),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    # ============================================================================
    # GAME-SPECIFIC INSIGHTS
    # ============================================================================
    
    async def get_game_difficulty_analysis(self, days: int = 7) -> Dict[str, Any]:
        """Analyze game difficulty based on success rates."""
        query = f"""
        SELECT 
            SUBSTR(gr.game_id, 1, 4) as game_type,
            COUNT(*) as total_games,
            SUM(CASE WHEN gr.win_detected = TRUE THEN 1 ELSE 0 END) as wins,
            ROUND(CAST(SUM(CASE WHEN gr.win_detected = TRUE THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100, 2) as win_rate_percent,
            ROUND(AVG(gr.final_score), 2) as avg_score,
            ROUND(AVG(gr.total_actions), 1) as avg_actions,
            ROUND(AVG(gr.coordinate_successes), 1) as avg_coordinate_successes,
            ROUND(AVG(gr.frame_changes), 1) as avg_frame_changes
        FROM game_results gr
        WHERE gr.start_time >= datetime('now', '-{days} days')
        GROUP BY SUBSTR(gr.game_id, 1, 4)
        ORDER BY win_rate_percent ASC, avg_actions DESC
        """
        
        results = await self.db.execute_query(query)
        
        return {
            "game_difficulty": results,
            "game_type_count": len(results),
            "analysis_days": days,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def get_coordinate_hotspots(self, days: int = 7) -> Dict[str, Any]:
        """Find coordinate hotspots (frequently used coordinates)."""
        query = f"""
        SELECT 
            SUBSTR(ct.game_id, 1, 4) as game_type,
            ct.coordinate_x,
            ct.coordinate_y,
            COUNT(*) as usage_count,
            SUM(CASE WHEN ct.success = TRUE THEN 1 ELSE 0 END) as success_count,
            ROUND(CAST(SUM(CASE WHEN ct.success = TRUE THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100, 2) as success_rate_percent,
            MAX(ct.timestamp) as last_used
        FROM coordinate_tracking ct
        WHERE ct.timestamp >= datetime('now', '-{days} days')
        GROUP BY SUBSTR(ct.game_id, 1, 4), ct.coordinate_x, ct.coordinate_y
        HAVING usage_count >= 3
        ORDER BY game_type, success_rate_percent DESC, usage_count DESC
        LIMIT 50
        """
        
        results = await self.db.execute_query(query)
        
        return {
            "coordinate_hotspots": results,
            "hotspot_count": len(results),
            "analysis_days": days,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    # ============================================================================
    # STRATEGIC IMPROVEMENT ANALYTICS
    # ============================================================================
    
    async def get_underperforming_actions(self) -> Dict[str, Any]:
        """Find actions that are underperforming and need attention."""
        query = """
        SELECT 
            ae.action_number,
            SUBSTR(ae.game_id, 1, 4) as game_type,
            COUNT(*) as games_played,
            SUM(ae.attempts) as total_attempts,
            SUM(ae.successes) as total_successes,
            ROUND(AVG(ae.success_rate) * 100, 2) as success_rate_percent,
            ROUND(AVG(ae.avg_score_impact), 2) as avg_score_impact
        FROM action_effectiveness ae
        WHERE ae.attempts >= 10
        GROUP BY ae.action_number, SUBSTR(ae.game_id, 1, 4)
        HAVING success_rate_percent < 20
        ORDER BY success_rate_percent ASC, total_attempts DESC
        """
        
        results = await self.db.execute_query(query)
        
        return {
            "underperforming_actions": results,
            "action_count": len(results),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def get_high_potential_coordinates(self) -> Dict[str, Any]:
        """Find coordinates with high success rates but low usage (untapped potential)."""
        query = """
        SELECT 
            ci.x,
            ci.y,
            SUBSTR(ci.game_id, 1, 4) as game_type,
            COUNT(*) as games_played,
            SUM(ci.attempts) as total_attempts,
            SUM(ci.successes) as total_successes,
            ROUND(AVG(ci.success_rate) * 100, 2) as success_rate_percent,
            ROUND(AVG(ci.frame_changes), 1) as avg_frame_changes
        FROM coordinate_intelligence ci
        WHERE ci.attempts >= 3
        GROUP BY ci.x, ci.y, SUBSTR(ci.game_id, 1, 4)
        HAVING success_rate_percent > 70
            AND total_attempts < 20
        ORDER BY success_rate_percent DESC, total_attempts ASC
        LIMIT 30
        """
        
        results = await self.db.execute_query(query)
        
        return {
            "high_potential_coordinates": results,
            "coordinate_count": len(results),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    # ============================================================================
    # REAL-TIME MONITORING
    # ============================================================================
    
    async def get_live_session_dashboard(self) -> Dict[str, Any]:
        """Real-time dashboard for active sessions."""
        query = """
        SELECT 
            ts.session_id,
            ts.mode,
            ts.total_actions,
            ts.total_wins,
            ts.total_games,
            ROUND(ts.win_rate * 100, 2) as win_rate_percent,
            ts.energy_level,
            ts.memory_operations,
            ROUND((julianday('now') - julianday(ts.start_time)) * 24 * 60, 1) as minutes_running,
            COUNT(gr.game_id) as active_games
        FROM training_sessions ts
        LEFT JOIN game_results gr ON ts.session_id = gr.session_id AND gr.end_time IS NULL
        WHERE ts.status = 'running'
        GROUP BY ts.session_id
        ORDER BY ts.start_time DESC
        """
        
        results = await self.db.execute_query(query)
        
        return {
            "live_dashboard": results,
            "active_sessions": len(results),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def get_recent_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Quick performance summary for the last hour."""
        query = f"""
        SELECT 
            'Last {hours} Hour(s)' as period,
            COUNT(DISTINCT ts.session_id) as active_sessions,
            SUM(ts.total_actions) as total_actions,
            SUM(ts.total_wins) as total_wins,
            SUM(ts.total_games) as total_games,
            ROUND(CAST(SUM(ts.total_wins) AS FLOAT) / NULLIF(SUM(ts.total_games), 0) * 100, 2) as overall_win_rate,
            ROUND(AVG(ts.avg_score), 2) as avg_score,
            ROUND(AVG(ts.energy_level), 1) as avg_energy
        FROM training_sessions ts
        WHERE ts.start_time >= datetime('now', '-{hours} hour')
            AND ts.status = 'running'
        """
        
        results = await self.db.execute_query(query)
        
        return {
            "performance_summary": results[0] if results else {},
            "summary_hours": hours,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    # ============================================================================
    # COMPREHENSIVE ANALYSIS FUNCTIONS
    # ============================================================================
    
    async def get_comprehensive_gameplay_analysis(self) -> Dict[str, Any]:
        """Get comprehensive gameplay analysis combining multiple analytics."""
        try:
            # Run all analytics in parallel for efficiency
            tasks = [
                self.get_current_session_overview(),
                self.get_performance_trends(7),
                self.get_action_effectiveness_by_game_type(),
                self.get_coordinate_intelligence_analysis(),
                self.get_critical_error_analysis(),
                self.get_system_health_monitoring(),
                self.get_learned_pattern_effectiveness(),
                self.get_game_difficulty_analysis(7),
                self.get_underperforming_actions(),
                self.get_high_potential_coordinates(),
                self.get_live_session_dashboard()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            analysis = {
                "session_overview": results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])},
                "performance_trends": results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])},
                "action_effectiveness": results[2] if not isinstance(results[2], Exception) else {"error": str(results[2])},
                "coordinate_intelligence": results[3] if not isinstance(results[3], Exception) else {"error": str(results[3])},
                "critical_errors": results[4] if not isinstance(results[4], Exception) else {"error": str(results[4])},
                "system_health": results[5] if not isinstance(results[5], Exception) else {"error": str(results[5])},
                "pattern_effectiveness": results[6] if not isinstance(results[6], Exception) else {"error": str(results[6])},
                "game_difficulty": results[7] if not isinstance(results[7], Exception) else {"error": str(results[7])},
                "underperforming_actions": results[8] if not isinstance(results[8], Exception) else {"error": str(results[8])},
                "high_potential_coordinates": results[9] if not isinstance(results[9], Exception) else {"error": str(results[9])},
                "live_dashboard": results[10] if not isinstance(results[10], Exception) else {"error": str(results[10])},
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_status": "COMPREHENSIVE"
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {e}")
            return {
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_status": "FAILED"
            }
    
    async def get_strategic_insights(self) -> Dict[str, Any]:
        """Get strategic insights for Director decision-making."""
        try:
            # Get key strategic data
            underperforming = await self.get_underperforming_actions()
            high_potential = await self.get_high_potential_coordinates()
            game_difficulty = await self.get_game_difficulty_analysis(7)
            errors = await self.get_critical_error_analysis()
            
            # Generate strategic recommendations
            recommendations = []
            
            # Action recommendations
            if underperforming.get("underperforming_actions"):
                recommendations.append({
                    "type": "ACTION_IMPROVEMENT",
                    "priority": "HIGH",
                    "message": f"Found {len(underperforming['underperforming_actions'])} underperforming actions that need attention",
                    "actions": [f"Action {a['action_number']} in {a['game_type']} (success rate: {a['success_rate_percent']}%)" 
                              for a in underperforming['underperforming_actions'][:5]]
                })
            
            # Coordinate recommendations
            if high_potential.get("high_potential_coordinates"):
                recommendations.append({
                    "type": "COORDINATE_EXPLORATION",
                    "priority": "MEDIUM",
                    "message": f"Found {len(high_potential['high_potential_coordinates'])} high-potential coordinates",
                    "coordinates": [f"({c['x']}, {c['y']}) in {c['game_type']} (success rate: {c['success_rate_percent']}%)" 
                                  for c in high_potential['high_potential_coordinates'][:5]]
                })
            
            # Game difficulty recommendations
            if game_difficulty.get("game_difficulty"):
                difficult_games = [g for g in game_difficulty['game_difficulty'] if g['win_rate_percent'] < 20]
                if difficult_games:
                    recommendations.append({
                        "type": "GAME_DIFFICULTY",
                        "priority": "HIGH",
                        "message": f"Found {len(difficult_games)} very difficult game types",
                        "games": [f"{g['game_type']} (win rate: {g['win_rate_percent']}%)" for g in difficult_games]
                    })
            
            # Error recommendations
            if errors.get("critical_errors"):
                recommendations.append({
                    "type": "ERROR_RESOLUTION",
                    "priority": "CRITICAL",
                    "message": f"Found {len(errors['critical_errors'])} unresolved critical errors",
                    "errors": [f"{e['error_type']}: {e['error_message'][:100]}..." for e in errors['critical_errors'][:3]]
                })
            
            return {
                "strategic_insights": {
                    "recommendations": recommendations,
                    "total_recommendations": len(recommendations),
                    "priority_breakdown": {
                        "CRITICAL": len([r for r in recommendations if r['priority'] == 'CRITICAL']),
                        "HIGH": len([r for r in recommendations if r['priority'] == 'HIGH']),
                        "MEDIUM": len([r for r in recommendations if r['priority'] == 'MEDIUM']),
                        "LOW": len([r for r in recommendations if r['priority'] == 'LOW'])
                    }
                },
                "data_sources": {
                    "underperforming_actions": underperforming,
                    "high_potential_coordinates": high_potential,
                    "game_difficulty": game_difficulty,
                    "critical_errors": errors
                },
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Strategic insights failed: {e}")
            return {
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat(),
                "strategic_insights": {"recommendations": [], "total_recommendations": 0}
            }

# ============================================================================
# DIRECTOR SELF-MODEL PERSISTENCE FUNCTIONS
# ============================================================================

class DirectorSelfModel:
    """
    Director self-model persistence and management.
    Handles storing and retrieving Director thoughts, insights, and learning.
    """
    
    def __init__(self):
        self.integration = get_system_integration()
        self.logger = logging.getLogger(__name__)
    
    async def store_director_thought(self, thought: str, thought_type: str = "reflection", 
                                   importance: int = 3, session_id: str = None, 
                                   metadata: Dict[str, Any] = None) -> bool:
        """
        Store a Director thought or insight.
        
        Args:
            thought: The thought content to store
            thought_type: Type of thought ('reflection', 'memory', 'trait', 'insight')
            importance: Importance level (1-5)
            session_id: Associated session ID
            metadata: Additional metadata
            
        Returns:
            True if successfully stored
        """
        try:
            if metadata is None:
                metadata = {}
            
            metadata.update({
                "thought_type": thought_type,
                "stored_at": datetime.now().isoformat(),
                "director_version": "1.0"
            })
            
            success = await self.integration.add_self_model_entry(
                type=thought_type,
                content=thought,
                session_id=session_id or "director_thoughts",
                importance=importance,
                metadata=metadata
            )
            
            if success:
                self.logger.info(f"Stored Director thought: {thought_type} (importance: {importance})")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to store Director thought: {e}")
            return False
    
    async def store_learning_insight(self, insight: str, learning_type: str = "pattern", 
                                   confidence: float = 0.8, session_id: str = None) -> bool:
        """
        Store a learning insight.
        
        Args:
            insight: The learning insight
            learning_type: Type of learning ('pattern', 'strategy', 'coordinate', 'action')
            confidence: Confidence in the insight (0.0-1.0)
            session_id: Associated session ID
            
        Returns:
            True if successfully stored
        """
        metadata = {
            "learning_type": learning_type,
            "confidence": confidence,
            "insight_category": "learning"
        }
        
        return await self.store_director_thought(
            thought=insight,
            thought_type="reflection",
            importance=4 if confidence > 0.8 else 3,
            session_id=session_id,
            metadata=metadata
        )
    
    async def store_strategic_decision(self, decision: str, reasoning: str, 
                                     expected_outcome: str = None, session_id: str = None) -> bool:
        """
        Store a strategic decision and reasoning.
        
        Args:
            decision: The decision made
            reasoning: Reasoning behind the decision
            expected_outcome: Expected outcome (optional)
            session_id: Associated session ID
            
        Returns:
            True if successfully stored
        """
        metadata = {
            "decision_type": "strategic",
            "reasoning": reasoning,
            "expected_outcome": expected_outcome,
            "decision_category": "strategy"
        }
        
        return await self.store_director_thought(
            thought=f"DECISION: {decision} | REASONING: {reasoning}",
            thought_type="reflection",
            importance=5,
            session_id=session_id,
            metadata=metadata
        )
    
    async def store_performance_analysis(self, analysis: str, performance_type: str = "session", 
                                       session_id: str = None) -> bool:
        """
        Store performance analysis insights.
        
        Args:
            analysis: The performance analysis
            performance_type: Type of performance ('session', 'game', 'action', 'coordinate')
            session_id: Associated session ID
            
        Returns:
            True if successfully stored
        """
        metadata = {
            "performance_type": performance_type,
            "analysis_category": "performance"
        }
        
        return await self.store_director_thought(
            thought=analysis,
            thought_type="reflection",
            importance=3,
            session_id=session_id,
            metadata=metadata
        )
    
    async def get_recent_thoughts(self, limit: int = 20, thought_type: str = None) -> List[Dict[str, Any]]:
        """
        Get recent Director thoughts.
        
        Args:
            limit: Maximum number of thoughts to retrieve
            thought_type: Filter by thought type (optional)
            
        Returns:
            List of recent thoughts
        """
        try:
            thoughts = await self.integration.get_self_model_entries(
                limit=limit,
                type=thought_type
            )
            
            return thoughts
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve recent thoughts: {e}")
            return []
    
    async def get_learning_insights(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent learning insights.
        
        Args:
            limit: Maximum number of insights to retrieve
            
        Returns:
            List of learning insights
        """
        try:
            insights = await self.integration.get_self_model_entries(
                limit=limit,
                type="reflection"
            )
            
            # Filter for learning-related insights
            learning_insights = []
            for insight in insights:
                metadata = insight.get('metadata', {})
                if (metadata.get('insight_category') == 'learning' or 
                    'learning' in insight.get('content', '').lower() or
                    'pattern' in insight.get('content', '').lower() or
                    'strategy' in insight.get('content', '').lower()):
                    learning_insights.append(insight)
            
            return learning_insights[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve learning insights: {e}")
            return []
    
    async def get_strategic_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent strategic decisions.
        
        Args:
            limit: Maximum number of decisions to retrieve
            
        Returns:
            List of strategic decisions
        """
        try:
            decisions = await self.integration.get_self_model_entries(
                limit=limit,
                type="reflection"
            )
            
            # Filter for strategic decisions
            strategic_decisions = []
            for decision in decisions:
                metadata = decision.get('metadata', {})
                if (metadata.get('decision_category') == 'strategy' or
                    'DECISION:' in decision.get('content', '')):
                    strategic_decisions.append(decision)
            
            return strategic_decisions[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve strategic decisions: {e}")
            return []
    
    async def get_director_summary(self) -> Dict[str, Any]:
        """
        Get a summary of Director's recent activity and insights.
        
        Returns:
            Director activity summary
        """
        try:
            # Get recent thoughts
            recent_thoughts = await self.get_recent_thoughts(limit=50)
            learning_insights = await self.get_learning_insights(limit=10)
            strategic_decisions = await self.get_strategic_decisions(limit=10)
            
            # Analyze thought patterns
            thought_types = {}
            importance_levels = {}
            
            for thought in recent_thoughts:
                thought_type = thought.get('type', 'unknown')
                importance = thought.get('importance', 1)
                
                thought_types[thought_type] = thought_types.get(thought_type, 0) + 1
                importance_levels[importance] = importance_levels.get(importance, 0) + 1
            
            return {
                "total_thoughts": len(recent_thoughts),
                "thought_type_breakdown": thought_types,
                "importance_breakdown": importance_levels,
                "recent_learning_insights": len(learning_insights),
                "recent_strategic_decisions": len(strategic_decisions),
                "most_common_thought_type": max(thought_types.items(), key=lambda x: x[1])[0] if thought_types else "none",
                "average_importance": sum(importance * count for importance, count in importance_levels.items()) / sum(importance_levels.values()) if importance_levels else 0,
                "summary_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate Director summary: {e}")
            return {
                "error": str(e),
                "summary_timestamp": datetime.now().isoformat()
            }

# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

# Global instances for easy access
_director_analytics = None
_director_self_model = None

def get_director_analytics() -> DirectorAnalytics:
    """Get global Director analytics instance."""
    global _director_analytics
    if _director_analytics is None:
        _director_analytics = DirectorAnalytics()
    return _director_analytics

def get_director_self_model() -> DirectorSelfModel:
    """Get global Director self-model instance."""
    global _director_self_model
    if _director_self_model is None:
        _director_self_model = DirectorSelfModel()
    return _director_self_model

# ============================================================================
# QUICK ACCESS FUNCTIONS
# ============================================================================

async def get_comprehensive_analysis() -> Dict[str, Any]:
    """Quick access to comprehensive gameplay analysis."""
    analytics = get_director_analytics()
    return await analytics.get_comprehensive_gameplay_analysis()

async def get_strategic_insights() -> Dict[str, Any]:
    """Quick access to strategic insights."""
    analytics = get_director_analytics()
    return await analytics.get_strategic_insights()

async def store_director_thought(thought: str, thought_type: str = "reflection", 
                               importance: int = 3, session_id: str = None) -> bool:
    """Quick access to store Director thought."""
    self_model = get_director_self_model()
    return await self_model.store_director_thought(thought, thought_type, importance, session_id)

async def get_director_summary() -> Dict[str, Any]:
    """Quick access to Director summary."""
    self_model = get_director_self_model()
    return await self_model.get_director_summary()
