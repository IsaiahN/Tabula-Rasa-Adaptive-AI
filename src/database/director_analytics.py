"""
UNIFIED DIRECTOR ANALYTICS API
Comprehensive analytics system combining simple and advanced analysis capabilities
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
# UNIFIED DIRECTOR ANALYTICS CLASS
# ============================================================================

class DirectorAnalytics:
    """
    Unified Director analytics system combining simple and comprehensive analysis.
    Uses working database API calls with advanced analytics capabilities.
    """
    
    def __init__(self):
        self.db = get_database()
        self.integration = get_system_integration()
        self.logger = logging.getLogger(__name__)
    
    # ============================================================================
    # BASIC ANALYTICS (from simple_director_analysis.py)
    # ============================================================================
    
    async def get_current_session_status(self) -> Dict[str, Any]:
        """Get current session status and performance."""
        try:
            # Get active sessions
            active_sessions = await self.db.get_active_sessions()
            
            # Get recent game results
            recent_games = await self.db.get_game_results()
            
            # Calculate summary statistics
            total_sessions = len(active_sessions)
            total_wins = sum(session.total_wins for session in active_sessions)
            total_games = sum(session.total_games for session in active_sessions)
            total_actions = sum(session.total_actions for session in active_sessions)
            
            win_rate = (total_wins / total_games * 100) if total_games > 0 else 0
            avg_score = sum(session.avg_score for session in active_sessions) / max(total_sessions, 1)
            
            return {
                "active_sessions": total_sessions,
                "total_wins": total_wins,
                "total_games": total_games,
                "total_actions": total_actions,
                "win_rate_percent": round(win_rate, 2),
                "avg_score": round(avg_score, 2),
                "session_details": [
                    {
                        "session_id": session.session_id,
                        "mode": session.mode,
                        "status": session.status,
                        "wins": session.total_wins,
                        "games": session.total_games,
                        "actions": session.total_actions,
                        "win_rate": round(session.win_rate * 100, 2),
                        "avg_score": round(session.avg_score, 2),
                        "energy": session.energy_level
                    }
                    for session in active_sessions
                ],
                "recent_games": len(recent_games),
                "analysis_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to get session status: {e}")
            return {
                "error": str(e),
                "active_sessions": 0,
                "total_wins": 0,
                "total_games": 0,
                "win_rate_percent": 0,
                "analysis_timestamp": datetime.now().isoformat()
            }
    
    async def get_recent_game_results(self, limit: int = 20) -> Dict[str, Any]:
        """Get recent game results and performance."""
        try:
            # Get recent game results
            game_results = await self.db.get_game_results()
            
            # Sort by start_time (most recent first)
            game_results.sort(key=lambda x: x.start_time, reverse=True)
            
            # Take the most recent games
            recent_games = game_results[:limit]
            
            # Analyze results
            wins = sum(1 for game in recent_games if game.win_detected)
            total_games = len(recent_games)
            win_rate = (wins / total_games * 100) if total_games > 0 else 0
            
            # Group by game type
            game_types = {}
            for game in recent_games:
                game_type = game.game_id[:4] if len(game.game_id) >= 4 else "unknown"
                if game_type not in game_types:
                    game_types[game_type] = {"total": 0, "wins": 0, "avg_score": 0}
                game_types[game_type]["total"] += 1
                if game.win_detected:
                    game_types[game_type]["wins"] += 1
                game_types[game_type]["avg_score"] += game.final_score
            
            # Calculate averages
            for game_type in game_types:
                total = game_types[game_type]["total"]
                game_types[game_type]["win_rate"] = round((game_types[game_type]["wins"] / total * 100), 2)
                game_types[game_type]["avg_score"] = round(game_types[game_type]["avg_score"] / total, 2)
            
            return {
                "recent_games": [
                    {
                        "game_id": game.game_id,
                        "session_id": game.session_id,
                        "status": game.status,
                        "final_score": game.final_score,
                        "total_actions": game.total_actions,
                        "win_detected": game.win_detected,
                        "level_completions": game.level_completions,
                        "coordinate_attempts": game.coordinate_attempts,
                        "coordinate_successes": game.coordinate_successes,
                        "frame_changes": game.frame_changes,
                        "start_time": game.start_time.isoformat()
                    }
                    for game in recent_games
                ],
                "summary": {
                    "total_games": total_games,
                    "wins": wins,
                    "win_rate_percent": round(win_rate, 2),
                    "game_types": game_types
                },
                "analysis_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to get game results: {e}")
            return {
                "error": str(e),
                "recent_games": [],
                "summary": {"total_games": 0, "wins": 0, "win_rate_percent": 0},
                "analysis_timestamp": datetime.now().isoformat()
            }
    
    async def get_action_effectiveness_analysis(self) -> Dict[str, Any]:
        """Get action effectiveness analysis."""
        try:
            # Get action effectiveness data
            action_data = await self.db.get_action_effectiveness()
            
            if not action_data:
                return {
                    "action_effectiveness": [],
                    "summary": {"total_actions": 0, "avg_success_rate": 0},
                    "analysis_timestamp": datetime.now().isoformat()
                }
            
            # Analyze action effectiveness
            total_attempts = sum(action.attempts for action in action_data)
            total_successes = sum(action.successes for action in action_data)
            avg_success_rate = (total_successes / total_attempts * 100) if total_attempts > 0 else 0
            
            # Group by action number
            action_summary = {}
            for action in action_data:
                action_num = action.action_number
                if action_num not in action_summary:
                    action_summary[action_num] = {
                        "total_attempts": 0,
                        "total_successes": 0,
                        "games_played": 0
                    }
                action_summary[action_num]["total_attempts"] += action.attempts
                action_summary[action_num]["total_successes"] += action.successes
                action_summary[action_num]["games_played"] += 1
            
            # Calculate success rates
            for action_num in action_summary:
                summary = action_summary[action_num]
                summary["success_rate_percent"] = round(
                    (summary["total_successes"] / summary["total_attempts"] * 100) 
                    if summary["total_attempts"] > 0 else 0, 2
                )
            
            return {
                "action_effectiveness": [
                    {
                        "action_number": action.action_number,
                        "game_id": action.game_id,
                        "attempts": action.attempts,
                        "successes": action.successes,
                        "success_rate": round(action.success_rate * 100, 2),
                        "avg_score_impact": action.avg_score_impact,
                        "last_used": action.last_used.isoformat() if hasattr(action.last_used, 'isoformat') else str(action.last_used) if action.last_used else None
                    }
                    for action in action_data
                ],
                "action_summary": action_summary,
                "summary": {
                    "total_actions": len(action_data),
                    "total_attempts": total_attempts,
                    "total_successes": total_successes,
                    "avg_success_rate": round(avg_success_rate, 2)
                },
                "analysis_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to get action effectiveness: {e}")
            return {
                "error": str(e),
                "action_effectiveness": [],
                "summary": {"total_actions": 0, "avg_success_rate": 0},
                "analysis_timestamp": datetime.now().isoformat()
            }
    
    async def get_coordinate_intelligence_analysis(self) -> Dict[str, Any]:
        """Get coordinate intelligence analysis."""
        try:
            # Get coordinate intelligence data
            coord_data = await self.db.get_coordinate_intelligence()
            
            if not coord_data:
                return {
                    "coordinate_intelligence": [],
                    "summary": {"total_coordinates": 0, "avg_success_rate": 0},
                    "analysis_timestamp": datetime.now().isoformat()
                }
            
            # Analyze coordinate effectiveness
            total_attempts = sum(coord.attempts for coord in coord_data)
            total_successes = sum(coord.successes for coord in coord_data)
            avg_success_rate = (total_successes / total_attempts * 100) if total_attempts > 0 else 0
            
            # Find high-success coordinates
            high_success_coords = [
                coord for coord in coord_data 
                if coord.success_rate > 0.5 and coord.attempts >= 3
            ]
            
            # Find low-success coordinates
            low_success_coords = [
                coord for coord in coord_data 
                if coord.success_rate < 0.2 and coord.attempts >= 5
            ]
            
            return {
                "coordinate_intelligence": [
                    {
                        "x": coord.x,
                        "y": coord.y,
                        "game_id": coord.game_id,
                        "attempts": coord.attempts,
                        "successes": coord.successes,
                        "success_rate": round(coord.success_rate * 100, 2),
                        "frame_changes": coord.frame_changes,
                        "last_used": coord.last_used.isoformat() if hasattr(coord.last_used, 'isoformat') else str(coord.last_used) if coord.last_used else None
                    }
                    for coord in coord_data
                ],
                "high_success_coordinates": [
                    {
                        "x": coord.x,
                        "y": coord.y,
                        "success_rate": round(coord.success_rate * 100, 2),
                        "attempts": coord.attempts
                    }
                    for coord in high_success_coords
                ],
                "low_success_coordinates": [
                    {
                        "x": coord.x,
                        "y": coord.y,
                        "success_rate": round(coord.success_rate * 100, 2),
                        "attempts": coord.attempts
                    }
                    for coord in low_success_coords
                ],
                "summary": {
                    "total_coordinates": len(coord_data),
                    "total_attempts": total_attempts,
                    "total_successes": total_successes,
                    "avg_success_rate": round(avg_success_rate, 2),
                    "high_success_count": len(high_success_coords),
                    "low_success_count": len(low_success_coords)
                },
                "analysis_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to get coordinate intelligence: {e}")
            return {
                "error": str(e),
                "coordinate_intelligence": [],
                "summary": {"total_coordinates": 0, "avg_success_rate": 0},
                "analysis_timestamp": datetime.now().isoformat()
            }
    
    async def get_system_health_analysis(self) -> Dict[str, Any]:
        """Get system health analysis."""
        try:
            # Get system status
            system_status = await self.db.get_system_status()
            
            # Get recent sessions
            recent_sessions = await self.db.get_training_sessions(hours=24)
            
            # Analyze health indicators
            active_sessions_list = system_status.get("active_sessions", [])
            active_sessions = len(active_sessions_list) if isinstance(active_sessions_list, list) else 0
            recent_performance = recent_sessions.get("metrics", {})
            
            # Calculate health score
            health_score = 0.0
            if active_sessions > 0:
                health_score += 0.3
            if recent_performance.get("win_rate", 0) > 0.1:
                health_score += 0.4
            if recent_performance.get("total_games", 0) > 0:
                health_score += 0.3
            
            # Determine health status
            if health_score >= 0.8:
                health_status = "EXCELLENT"
            elif health_score >= 0.6:
                health_status = "GOOD"
            elif health_score >= 0.4:
                health_status = "FAIR"
            elif health_score >= 0.2:
                health_status = "POOR"
            else:
                health_status = "CRITICAL"
            
            return {
                "health_score": round(health_score, 2),
                "health_status": health_status,
                "active_sessions": active_sessions,
                "recent_performance": recent_performance,
                "system_status": system_status,
                "analysis_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to get system health: {e}")
            return {
                "error": str(e),
                "health_score": 0.0,
                "health_status": "CRITICAL",
                "analysis_timestamp": datetime.now().isoformat()
            }
    
    # ============================================================================
    # ADVANCED ANALYTICS (from director_analytics.py - fixed database calls)
    # ============================================================================
    
    async def get_performance_trends(self, days: int = 7) -> Dict[str, Any]:
        """Track performance trends across sessions."""
        try:
            # Get training sessions data
            sessions_data = await self.db.get_training_sessions(hours=days * 24)
            
            if not sessions_data.get("session_data"):
                return {
                    "performance_trends": [],
                    "trend_days": days,
                    "analysis_timestamp": datetime.now().isoformat()
                }
            
            # Process session data
            sessions = sessions_data["session_data"]
            if not isinstance(sessions, list):
                sessions = [sessions] if sessions else []
            
            # Group by date
            daily_stats = {}
            for session in sessions:
                if isinstance(session, dict):
                    start_time = session.get("start_time")
                    if start_time:
                        date = start_time.split("T")[0] if "T" in str(start_time) else str(start_time)[:10]
                        if date not in daily_stats:
                            daily_stats[date] = {
                                "sessions_count": 0,
                                "total_wins": 0,
                                "total_games": 0,
                                "total_actions": 0,
                                "total_energy": 0
                            }
                        
                        daily_stats[date]["sessions_count"] += 1
                        daily_stats[date]["total_wins"] += session.get("total_wins", 0)
                        daily_stats[date]["total_games"] += session.get("total_games", 0)
                        daily_stats[date]["total_actions"] += session.get("total_actions", 0)
                        daily_stats[date]["total_energy"] += session.get("energy_level", 0)
            
            # Calculate trends
            trends = []
            for date, stats in sorted(daily_stats.items(), reverse=True):
                avg_win_rate = (stats["total_wins"] / stats["total_games"] * 100) if stats["total_games"] > 0 else 0
                avg_energy = stats["total_energy"] / stats["sessions_count"] if stats["sessions_count"] > 0 else 0
                
                trends.append({
                    "session_date": date,
                    "sessions_count": stats["sessions_count"],
                    "avg_win_rate": round(avg_win_rate, 2),
                    "avg_score": 0.0,  # Not available in basic data
                    "avg_actions": round(stats["total_actions"] / stats["sessions_count"], 0),
                    "avg_energy": round(avg_energy, 1),
                    "total_wins": stats["total_wins"],
                    "total_games": stats["total_games"]
                })
            
            return {
                "performance_trends": trends,
                "trend_days": days,
                "analysis_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to get performance trends: {e}")
            return {
                "error": str(e),
                "performance_trends": [],
                "trend_days": days,
                "analysis_timestamp": datetime.now().isoformat()
            }
    
    async def get_game_difficulty_analysis(self, days: int = 7) -> Dict[str, Any]:
        """Analyze game difficulty based on success rates."""
        try:
            # Get game results
            game_results = await self.db.get_game_results()
            
            # Filter by date (last N days)
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_games = [
                game for game in game_results 
                if game.start_time >= cutoff_date
            ]
            
            if not recent_games:
                return {
                    "game_difficulty": [],
                    "game_type_count": 0,
                    "analysis_days": days,
                    "analysis_timestamp": datetime.now().isoformat()
                }
            
            # Group by game type
            game_types = {}
            for game in recent_games:
                game_type = game.game_id[:4] if len(game.game_id) >= 4 else "unknown"
                if game_type not in game_types:
                    game_types[game_type] = {
                        "total_games": 0,
                        "wins": 0,
                        "total_score": 0,
                        "total_actions": 0,
                        "total_coordinate_successes": 0,
                        "total_frame_changes": 0
                    }
                
                game_types[game_type]["total_games"] += 1
                if game.win_detected:
                    game_types[game_type]["wins"] += 1
                game_types[game_type]["total_score"] += game.final_score
                game_types[game_type]["total_actions"] += game.total_actions
                game_types[game_type]["total_coordinate_successes"] += game.coordinate_successes
                game_types[game_type]["total_frame_changes"] += game.frame_changes
            
            # Calculate difficulty metrics
            difficulty_analysis = []
            for game_type, stats in game_types.items():
                win_rate = (stats["wins"] / stats["total_games"] * 100) if stats["total_games"] > 0 else 0
                avg_score = stats["total_score"] / stats["total_games"] if stats["total_games"] > 0 else 0
                avg_actions = stats["total_actions"] / stats["total_games"] if stats["total_games"] > 0 else 0
                avg_coordinate_successes = stats["total_coordinate_successes"] / stats["total_games"] if stats["total_games"] > 0 else 0
                avg_frame_changes = stats["total_frame_changes"] / stats["total_games"] if stats["total_games"] > 0 else 0
                
                difficulty_analysis.append({
                    "game_type": game_type,
                    "total_games": stats["total_games"],
                    "wins": stats["wins"],
                    "win_rate_percent": round(win_rate, 2),
                    "avg_score": round(avg_score, 2),
                    "avg_actions": round(avg_actions, 1),
                    "avg_coordinate_successes": round(avg_coordinate_successes, 1),
                    "avg_frame_changes": round(avg_frame_changes, 1)
                })
            
            # Sort by difficulty (lowest win rate first)
            difficulty_analysis.sort(key=lambda x: x["win_rate_percent"])
            
            return {
                "game_difficulty": difficulty_analysis,
                "game_type_count": len(difficulty_analysis),
                "analysis_days": days,
                "analysis_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to get game difficulty analysis: {e}")
            return {
                "error": str(e),
                "game_difficulty": [],
                "game_type_count": 0,
                "analysis_days": days,
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
                self.get_current_session_status(),
                self.get_recent_game_results(),
                self.get_action_effectiveness_analysis(),
                self.get_coordinate_intelligence_analysis(),
                self.get_system_health_analysis(),
                self.get_performance_trends(7),
                self.get_game_difficulty_analysis(7)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            analysis = {
                "session_status": results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])},
                "game_results": results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])},
                "action_analysis": results[2] if not isinstance(results[2], Exception) else {"error": str(results[2])},
                "coordinate_analysis": results[3] if not isinstance(results[3], Exception) else {"error": str(results[3])},
                "system_health": results[4] if not isinstance(results[4], Exception) else {"error": str(results[4])},
                "performance_trends": results[5] if not isinstance(results[5], Exception) else {"error": str(results[5])},
                "game_difficulty": results[6] if not isinstance(results[6], Exception) else {"error": str(results[6])},
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
            action_analysis = await self.get_action_effectiveness_analysis()
            coord_analysis = await self.get_coordinate_intelligence_analysis()
            game_difficulty = await self.get_game_difficulty_analysis(7)
            
            # Generate strategic recommendations
            recommendations = []
            
            # Action recommendations
            if action_analysis.get("action_summary"):
                underperforming_actions = [
                    action for action in action_analysis["action_summary"].values()
                    if action.get("success_rate_percent", 0) < 20 and action.get("total_attempts", 0) >= 10
                ]
                if underperforming_actions:
                    recommendations.append({
                        "type": "ACTION_IMPROVEMENT",
                        "priority": "HIGH",
                        "message": f"Found {len(underperforming_actions)} underperforming actions that need attention",
                        "actions": [f"Action {action_num} (success rate: {action['success_rate_percent']}%)" 
                                  for action_num, action in action_analysis["action_summary"].items()
                                  if action.get("success_rate_percent", 0) < 20 and action.get("total_attempts", 0) >= 10][:5]
                    })
            
            # Coordinate recommendations
            if coord_analysis.get("high_success_coordinates"):
                recommendations.append({
                    "type": "COORDINATE_EXPLORATION",
                    "priority": "MEDIUM",
                    "message": f"Found {len(coord_analysis['high_success_coordinates'])} high-potential coordinates",
                    "coordinates": [f"({c['x']}, {c['y']}) (success rate: {c['success_rate']}%)" 
                                  for c in coord_analysis["high_success_coordinates"][:5]]
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
            
            return {
                "strategic_insights": {
                    "recommendations": recommendations,
                    "total_recommendations": len(recommendations),
                    "priority_breakdown": {
                        "HIGH": len([r for r in recommendations if r['priority'] == 'HIGH']),
                        "MEDIUM": len([r for r in recommendations if r['priority'] == 'MEDIUM']),
                        "LOW": len([r for r in recommendations if r['priority'] == 'LOW'])
                    }
                },
                "data_sources": {
                    "action_analysis": action_analysis,
                    "coordinate_analysis": coord_analysis,
                    "game_difficulty": game_difficulty
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
        """Store a Director thought or insight."""
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
        """Store a learning insight."""
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
        """Store a strategic decision and reasoning."""
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
    
    async def get_recent_thoughts(self, limit: int = 20, thought_type: str = None) -> List[Dict[str, Any]]:
        """Get recent Director thoughts."""
        try:
            thoughts = await self.integration.get_self_model_entries(
                limit=limit,
                type=thought_type
            )
            return thoughts
        except Exception as e:
            self.logger.error(f"Failed to retrieve recent thoughts: {e}")
            return []
    
    async def get_director_summary(self) -> Dict[str, Any]:
        """Get a summary of Director's recent activity and insights."""
        try:
            # Get recent thoughts
            recent_thoughts = await self.get_recent_thoughts(limit=50)
            
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
