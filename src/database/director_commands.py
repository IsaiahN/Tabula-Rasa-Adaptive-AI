"""
DIRECTOR COMMAND INTERFACE
Easy-to-use commands for Director/LLM system analysis and control
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from .api import get_database, log_director_decision, get_director_status, get_director_insights
from .director_training_monitor import get_training_status, detect_training_issues, get_training_recommendations
# GAN system imports are optional and will be loaded lazily when needed
PatternAwareGAN = None
GANTrainingConfig = None

def _get_gan_classes():
    """Lazy loader for GAN classes to avoid torch import at startup."""
    global PatternAwareGAN, GANTrainingConfig
    if PatternAwareGAN is None:
        try:
            from ..core.gan_system import PatternAwareGAN as _PatternAwareGAN, GANTrainingConfig as _GANTrainingConfig
            PatternAwareGAN = _PatternAwareGAN
            GANTrainingConfig = _GANTrainingConfig
        except ImportError:
            try:
                from src.core.gan_system import PatternAwareGAN as _PatternAwareGAN, GANTrainingConfig as _GANTrainingConfig
                PatternAwareGAN = _PatternAwareGAN
                GANTrainingConfig = _GANTrainingConfig
            except ImportError:
                # Define dummy classes if GAN system is not available
                class PatternAwareGAN:
                    def __init__(self, *args, **kwargs):
                        pass
                
                class GANTrainingConfig:
                    def __init__(self, *args, **kwargs):
                        pass
                
                PatternAwareGAN = PatternAwareGAN
                GANTrainingConfig = GANTrainingConfig
    return PatternAwareGAN, GANTrainingConfig

# ============================================================================
# DIRECTOR COMMAND INTERFACE
# ============================================================================

class DirectorCommands:
    """
    High-level command interface for Director/LLM system control and analysis.
    Provides easy-to-use methods for system monitoring, analysis, and control.
    """
    
    def __init__(self):
        self.db = get_database()
        self.logger = logging.getLogger(__name__)
    
    # ============================================================================
    # SYSTEM STATUS COMMANDS
    # ============================================================================
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """
        Get comprehensive system overview.
        
        Returns:
            Dict containing system status, performance, and key metrics
        """
        status = await get_director_status()
        
        # Add calculated insights
        active_sessions = status.get("active_sessions", [])
        recent_performance = status.get("recent_performance", [])
        
        overview = {
            "system_status": {
                "active_sessions": len(active_sessions),
                "total_actions": sum(s.get("total_actions", 0) for s in active_sessions),
                "total_wins": sum(s.get("total_wins", 0) for s in active_sessions),
                "total_games": sum(s.get("total_games", 0) for s in active_sessions),
                "avg_win_rate": sum(s.get("win_rate", 0) for s in active_sessions) / max(len(active_sessions), 1),
                "avg_score": sum(s.get("avg_score", 0) for s in active_sessions) / max(len(active_sessions), 1),
            },
            "recent_trends": recent_performance,
            "action_effectiveness": status.get("action_effectiveness", []),
            "global_counters": status.get("global_counters", {}),
            "timestamp": datetime.now().isoformat()
        }
        
        await log_director_decision("system_overview", "Retrieved system overview", 1.0)
        return overview
    
    async def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get performance summary for specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Performance summary with key metrics
        """
        metrics = await self.db.get_training_sessions(hours=hours)
        
        summary = {
            "time_period_hours": hours,
            "total_sessions": metrics["metrics"]["total_sessions"],
            "total_games": metrics["metrics"]["total_games"],
            "total_wins": metrics["metrics"]["total_wins"],
            "win_rate": metrics["metrics"]["win_rate"],
            "recent_sessions": metrics["session_data"][:5] if isinstance(metrics["session_data"], list) else [metrics["session_data"]],
            "recent_games": metrics["game_results"][:10],
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        await log_director_decision("performance_summary", f"Analyzed {hours}h performance", 1.0)
        return summary
    
    async def get_learning_analysis(self, game_id: str = None) -> Dict[str, Any]:
        """
        Get learning analysis and insights.
        
        Args:
            game_id: Specific game to analyze (optional)
            
        Returns:
            Learning insights and patterns
        """
        insights = await get_director_insights(game_id)
        
        analysis = {
            "coordinate_insights": insights.get("coordinate_insights", []),
            "winning_sequences": insights.get("winning_sequences", []),
            "recent_patterns": insights.get("recent_patterns", []),
            "learning_effectiveness": self._calculate_learning_effectiveness(insights),
            "recommendations": self._generate_learning_recommendations(insights),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        await log_director_decision("learning_analysis", f"Analyzed learning for {game_id or 'all games'}", 0.9)
        return analysis
    
    # ============================================================================
    # ACTION INTELLIGENCE COMMANDS
    # ============================================================================
    
    async def get_action_effectiveness(self, game_id: str = None, action_number: int = None) -> Dict[str, Any]:
        """
        Get action effectiveness analysis.
        
        Args:
            game_id: Specific game to analyze
            action_number: Specific action to analyze
            
        Returns:
            Action effectiveness data and insights
        """
        effectiveness = await self.db.get_action_effectiveness(game_id, action_number)
        
        analysis = {
            "effectiveness_data": [asdict(e) for e in effectiveness],
            "summary": self._summarize_action_effectiveness(effectiveness),
            "recommendations": self._generate_action_recommendations(effectiveness),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        await log_director_decision("action_effectiveness", f"Analyzed actions for {game_id or 'all games'}", 0.8)
        return analysis
    
    async def get_coordinate_intelligence(self, game_id: str = None, min_success_rate: float = 0.1) -> Dict[str, Any]:
        """
        Get coordinate intelligence analysis.
        
        Args:
            game_id: Specific game to analyze
            min_success_rate: Minimum success rate to include
            
        Returns:
            Coordinate intelligence data and insights
        """
        intelligence = await self.db.get_coordinate_intelligence(game_id, min_success_rate)
        
        analysis = {
            "coordinate_data": [asdict(c) for c in intelligence],
            "summary": self._summarize_coordinate_intelligence(intelligence),
            "hotspots": self._identify_coordinate_hotspots(intelligence),
            "recommendations": self._generate_coordinate_recommendations(intelligence),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        await log_director_decision("coordinate_intelligence", f"Analyzed coordinates for {game_id or 'all games'}", 0.8)
        return analysis
    
    # ============================================================================
    # SYSTEM CONTROL COMMANDS
    # ============================================================================
    
    async def create_training_session(self, mode: str = "maximum-intelligence", 
                                    session_id: str = None) -> Dict[str, Any]:
        """
        Create a new training session.
        
        Args:
            mode: Training mode
            session_id: Custom session ID (optional)
            
        Returns:
            Session creation result
        """
        if not session_id:
            session_id = f"director_session_{int(datetime.now().timestamp())}"
        
        from .api import TrainingSession
        session = TrainingSession(
            session_id=session_id,
            start_time=datetime.now(),
            mode=mode,
            status="running"
        )
        
        success = await self.db.create_session(session)
        
        result = {
            "success": success,
            "session_id": session_id,
            "mode": mode,
            "created_at": datetime.now().isoformat()
        }
        
        await log_director_decision("create_session", f"Created {mode} session {session_id}", 1.0 if success else 0.0)
        return result
    
    async def update_session_status(self, session_id: str, status: str, 
                                   updates: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Update training session status.
        
        Args:
            session_id: Session to update
            status: New status
            updates: Additional updates
            
        Returns:
            Update result
        """
        update_data = {"status": status}
        if updates:
            update_data.update(updates)
        
        success = await self.db.update_session(session_id, update_data)
        
        result = {
            "success": success,
            "session_id": session_id,
            "status": status,
            "updated_at": datetime.now().isoformat()
        }
        
        await log_director_decision("update_session", f"Updated session {session_id} to {status}", 1.0 if success else 0.0)
        return result
    
    async def get_active_sessions(self) -> Dict[str, Any]:
        """
        Get all active training sessions.
        
        Returns:
            List of active sessions
        """
        sessions = await self.db.get_active_sessions()
        
        result = {
            "active_sessions": [asdict(s) for s in sessions],
            "count": len(sessions),
            "timestamp": datetime.now().isoformat()
        }
        
        await log_director_decision("get_active_sessions", f"Retrieved {len(sessions)} active sessions", 1.0)
        return result
    
    # ============================================================================
    # ANALYSIS AND INSIGHTS
    # ============================================================================
    
    async def analyze_system_health(self) -> Dict[str, Any]:
        """
        Analyze overall system health.
        
        Returns:
            System health analysis
        """
        overview = await self.get_system_overview()
        performance = await self.get_performance_summary(24)
        
        health_score = self._calculate_health_score(overview, performance)
        
        analysis = {
            "health_score": health_score,
            "status": self._get_health_status(health_score),
            "issues": self._identify_health_issues(overview, performance),
            "recommendations": self._generate_health_recommendations(overview, performance),
            "overview": overview,
            "performance": performance,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        await log_director_decision("system_health", f"Analyzed system health: {health_score:.2f}", 0.9)
        return analysis
    
    async def get_learning_progress(self, game_id: str = None) -> Dict[str, Any]:
        """
        Get learning progress analysis.
        
        Args:
            game_id: Specific game to analyze
            
        Returns:
            Learning progress data
        """
        insights = await get_director_insights(game_id)
        effectiveness = await self.get_action_effectiveness(game_id)
        coordinates = await self.get_coordinate_intelligence(game_id)
        
        progress = {
            "learning_metrics": self._calculate_learning_metrics(insights, effectiveness, coordinates),
            "progress_trends": self._analyze_progress_trends(insights),
            "improvement_areas": self._identify_improvement_areas(insights, effectiveness, coordinates),
            "next_steps": self._recommend_next_steps(insights, effectiveness, coordinates),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        await log_director_decision("learning_progress", f"Analyzed learning progress for {game_id or 'all games'}", 0.8)
        return progress
    
    # ============================================================================
    # TRAINING MONITORING COMMANDS
    # ============================================================================
    
    async def get_training_status(self) -> Dict[str, Any]:
        """
        Get comprehensive training process status.
        
        Returns:
            Training process status and monitoring data
        """
        status = await get_training_status()
        
        await log_director_decision("training_status", f"Retrieved training status: {status['total_processes']} processes", 1.0)
        return status
    
    async def detect_training_issues(self) -> Dict[str, Any]:
        """
        Detect issues with training processes.
        
        Returns:
            Training issues and recommendations
        """
        issues = await detect_training_issues()
        recommendations = await get_training_recommendations()
        
        analysis = {
            "issues": issues,
            "recommendations": recommendations,
            "total_issues": len(issues),
            "total_recommendations": len(recommendations),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        await log_director_decision("training_issues", f"Detected {len(issues)} issues, {len(recommendations)} recommendations", 0.9)
        return analysis
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    def _calculate_learning_effectiveness(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate learning effectiveness metrics."""
        coordinate_insights = insights.get("coordinate_insights", [])
        winning_sequences = insights.get("winning_sequences", [])
        recent_patterns = insights.get("recent_patterns", [])
        
        return {
            "coordinate_learning": len(coordinate_insights),
            "sequence_learning": len(winning_sequences),
            "pattern_learning": len(recent_patterns),
            "overall_effectiveness": (len(coordinate_insights) + len(winning_sequences) + len(recent_patterns)) / 3
        }
    
    def _generate_learning_recommendations(self, insights: Dict[str, Any]) -> List[str]:
        """Generate learning recommendations."""
        recommendations = []
        
        coordinate_insights = insights.get("coordinate_insights", [])
        if len(coordinate_insights) < 5:
            recommendations.append("Increase coordinate exploration to improve Action 6 effectiveness")
        
        winning_sequences = insights.get("winning_sequences", [])
        if len(winning_sequences) < 3:
            recommendations.append("Focus on identifying and learning winning action sequences")
        
        recent_patterns = insights.get("recent_patterns", [])
        if len(recent_patterns) < 10:
            recommendations.append("Increase pattern recognition and learning activities")
        
        return recommendations
    
    def _summarize_action_effectiveness(self, effectiveness: List) -> Dict[str, Any]:
        """Summarize action effectiveness data."""
        if not effectiveness:
            return {"total_actions": 0, "avg_success_rate": 0.0, "best_action": None}
        
        total_attempts = sum(e.attempts for e in effectiveness)
        total_successes = sum(e.successes for e in effectiveness)
        avg_success_rate = total_successes / total_attempts if total_attempts > 0 else 0.0
        
        best_action = max(effectiveness, key=lambda x: x.success_rate) if effectiveness else None
        
        return {
            "total_actions": len(effectiveness),
            "total_attempts": total_attempts,
            "total_successes": total_successes,
            "avg_success_rate": avg_success_rate,
            "best_action": best_action.action_number if best_action else None,
            "best_success_rate": best_action.success_rate if best_action else 0.0
        }
    
    def _generate_action_recommendations(self, effectiveness: List) -> List[str]:
        """Generate action recommendations."""
        recommendations = []
        
        if not effectiveness:
            recommendations.append("No action data available - start training to collect data")
            return recommendations
        
        # Find actions with low success rates
        low_success_actions = [e for e in effectiveness if e.success_rate < 0.1]
        if low_success_actions:
            recommendations.append(f"Actions {[e.action_number for e in low_success_actions]} have low success rates - investigate and improve")
        
        # Find best performing actions
        best_actions = [e for e in effectiveness if e.success_rate > 0.5]
        if best_actions:
            recommendations.append(f"Actions {[e.action_number for e in best_actions]} are performing well - increase usage")
        
        return recommendations
    
    def _summarize_coordinate_intelligence(self, intelligence: List) -> Dict[str, Any]:
        """Summarize coordinate intelligence data."""
        if not intelligence:
            return {"total_coordinates": 0, "avg_success_rate": 0.0, "best_coordinate": None}
        
        total_attempts = sum(c.attempts for c in intelligence)
        total_successes = sum(c.successes for c in intelligence)
        avg_success_rate = total_successes / total_attempts if total_attempts > 0 else 0.0
        
        best_coordinate = max(intelligence, key=lambda x: x.success_rate) if intelligence else None
        
        return {
            "total_coordinates": len(intelligence),
            "total_attempts": total_attempts,
            "total_successes": total_successes,
            "avg_success_rate": avg_success_rate,
            "best_coordinate": (best_coordinate.x, best_coordinate.y) if best_coordinate else None,
            "best_success_rate": best_coordinate.success_rate if best_coordinate else 0.0
        }
    
    def _identify_coordinate_hotspots(self, intelligence: List) -> List[Dict[str, Any]]:
        """Identify coordinate hotspots."""
        hotspots = []
        for coord in intelligence:
            if coord.success_rate > 0.3 and coord.attempts > 5:
                hotspots.append({
                    "coordinate": (coord.x, coord.y),
                    "success_rate": coord.success_rate,
                    "attempts": coord.attempts,
                    "game_id": coord.game_id
                })
        
        return sorted(hotspots, key=lambda x: x["success_rate"], reverse=True)
    
    def _generate_coordinate_recommendations(self, intelligence: List) -> List[str]:
        """Generate coordinate recommendations."""
        recommendations = []
        
        if not intelligence:
            recommendations.append("No coordinate data available - start Action 6 training to collect data")
            return recommendations
        
        # Find high-success coordinates
        high_success = [c for c in intelligence if c.success_rate > 0.5]
        if high_success:
            recommendations.append(f"Found {len(high_success)} high-success coordinates - prioritize these areas")
        
        # Find low-success coordinates
        low_success = [c for c in intelligence if c.success_rate < 0.1 and c.attempts > 10]
        if low_success:
            recommendations.append(f"Found {len(low_success)} low-success coordinates - avoid these areas")
        
        return recommendations
    
    def _calculate_health_score(self, overview: Dict[str, Any], performance: Dict[str, Any]) -> float:
        """Calculate system health score (0-1)."""
        score = 0.0
        
        # Active sessions score
        active_sessions = overview.get("system_status", {}).get("active_sessions", 0)
        if active_sessions > 0:
            score += 0.3
        
        # Win rate score
        win_rate = overview.get("system_status", {}).get("avg_win_rate", 0)
        score += min(win_rate * 0.4, 0.4)
        
        # Action effectiveness score
        action_effectiveness = overview.get("action_effectiveness", [])
        if action_effectiveness:
            avg_effectiveness = sum(a.get("avg_success_rate", 0) for a in action_effectiveness) / len(action_effectiveness)
            score += min(avg_effectiveness * 0.3, 0.3)
        
        return min(score, 1.0)
    
    def _get_health_status(self, health_score: float) -> str:
        """Get health status based on score."""
        if health_score >= 0.8:
            return "EXCELLENT"
        elif health_score >= 0.6:
            return "GOOD"
        elif health_score >= 0.4:
            return "FAIR"
        elif health_score >= 0.2:
            return "POOR"
        else:
            return "CRITICAL"
    
    def _identify_health_issues(self, overview: Dict[str, Any], performance: Dict[str, Any]) -> List[str]:
        """Identify system health issues."""
        issues = []
        
        # Check for no active sessions
        active_sessions = overview.get("system_status", {}).get("active_sessions", 0)
        if active_sessions == 0:
            issues.append("No active training sessions")
        
        # Check for low win rate
        win_rate = overview.get("system_status", {}).get("avg_win_rate", 0)
        if win_rate < 0.1:
            issues.append("Very low win rate - system may be struggling")
        
        # Check for low action effectiveness
        action_effectiveness = overview.get("action_effectiveness", [])
        if action_effectiveness:
            avg_effectiveness = sum(a.get("avg_success_rate", 0) for a in action_effectiveness) / len(action_effectiveness)
            if avg_effectiveness < 0.1:
                issues.append("Low action effectiveness - actions are not working well")
        
        return issues
    
    def _generate_health_recommendations(self, overview: Dict[str, Any], performance: Dict[str, Any]) -> List[str]:
        """Generate health recommendations."""
        recommendations = []
        
        # Check for no active sessions
        active_sessions = overview.get("system_status", {}).get("active_sessions", 0)
        if active_sessions == 0:
            recommendations.append("Start a new training session to begin learning")
        
        # Check for low win rate
        win_rate = overview.get("system_status", {}).get("avg_win_rate", 0)
        if win_rate < 0.1:
            recommendations.append("Investigate why win rate is low - check action effectiveness and coordinate selection")
        
        # Check for low action effectiveness
        action_effectiveness = overview.get("action_effectiveness", [])
        if action_effectiveness:
            avg_effectiveness = sum(a.get("avg_success_rate", 0) for a in action_effectiveness) / len(action_effectiveness)
            if avg_effectiveness < 0.1:
                recommendations.append("Improve action effectiveness - focus on better coordinate selection and action sequencing")
        
        return recommendations
    
    def _calculate_learning_metrics(self, insights: Dict[str, Any], effectiveness: Dict[str, Any], coordinates: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate learning metrics."""
        return {
            "coordinate_learning": len(insights.get("coordinate_insights", [])),
            "sequence_learning": len(insights.get("winning_sequences", [])),
            "pattern_learning": len(insights.get("recent_patterns", [])),
            "action_learning": len(effectiveness.get("effectiveness_data", [])),
            "overall_progress": "GOOD" if len(insights.get("coordinate_insights", [])) > 5 else "NEEDS_IMPROVEMENT"
        }
    
    def _analyze_progress_trends(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze learning progress trends."""
        return {
            "coordinate_trend": "IMPROVING" if len(insights.get("coordinate_insights", [])) > 5 else "STABLE",
            "sequence_trend": "IMPROVING" if len(insights.get("winning_sequences", [])) > 3 else "STABLE",
            "pattern_trend": "IMPROVING" if len(insights.get("recent_patterns", [])) > 10 else "STABLE"
        }
    
    def _identify_improvement_areas(self, insights: Dict[str, Any], effectiveness: Dict[str, Any], coordinates: Dict[str, Any]) -> List[str]:
        """Identify areas for improvement."""
        areas = []
        
        if len(insights.get("coordinate_insights", [])) < 5:
            areas.append("Coordinate learning - need more successful coordinate patterns")
        
        if len(insights.get("winning_sequences", [])) < 3:
            areas.append("Sequence learning - need to identify winning action sequences")
        
        if len(insights.get("recent_patterns", [])) < 10:
            areas.append("Pattern learning - need more pattern recognition")
        
        return areas
    
    def _recommend_next_steps(self, insights: Dict[str, Any], effectiveness: Dict[str, Any], coordinates: Dict[str, Any]) -> List[str]:
        """Recommend next steps for improvement."""
        steps = []
        
        if len(insights.get("coordinate_insights", [])) < 5:
            steps.append("Focus on Action 6 coordinate exploration to build coordinate intelligence")
        
        if len(insights.get("winning_sequences", [])) < 3:
            steps.append("Analyze successful games to identify winning action sequences")
        
        if len(insights.get("recent_patterns", [])) < 10:
            steps.append("Increase pattern recognition activities and learning sessions")
        
        return steps
    
    # ============================================================================
    # GAN SYSTEM COMMANDS
    # ============================================================================
    
    async def start_gan_training(self, session_name: str = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Start GAN training session for synthetic data generation.
        
        Args:
            session_name: Optional name for the training session
            config: Optional GAN configuration parameters
            
        Returns:
            Dict containing session information and status
        """
        try:
            # Create GAN configuration
            gan_config = GANTrainingConfig()
            if config:
                for key, value in config.items():
                    if hasattr(gan_config, key):
                        setattr(gan_config, key, value)
            
            # Initialize GAN system
            gan_system = PatternAwareGAN(config=gan_config)
            
            # Start training session
            session_id = await gan_system.start_training_session(session_name)
            
            # Store GAN system reference (in production, this would be managed better)
            if not hasattr(self, '_gan_systems'):
                self._gan_systems = {}
            self._gan_systems[session_id] = gan_system
            
            await log_director_decision("gan_training_started", f"GAN training session {session_id} started", 1.0)
            
            return {
                "status": "success",
                "session_id": session_id,
                "message": f"GAN training session {session_id} started successfully",
                "config": asdict(gan_config)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to start GAN training: {e}")
            return {
                "status": "error",
                "message": f"Failed to start GAN training: {str(e)}"
            }
    
    async def generate_synthetic_states(self, session_id: str, count: int = 10, 
                                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate synthetic game states using GAN.
        
        Args:
            session_id: GAN training session ID
            count: Number of synthetic states to generate
            context: Optional context for generation
            
        Returns:
            Dict containing generated states and metadata
        """
        try:
            # Get GAN system
            if not hasattr(self, '_gan_systems') or session_id not in self._gan_systems:
                return {
                    "status": "error",
                    "message": f"GAN session {session_id} not found"
                }
            
            gan_system = self._gan_systems[session_id]
            
            # Generate synthetic states
            synthetic_states = await gan_system.generate_synthetic_states(count, context)
            
            # Convert to serializable format
            states_data = [state.to_dict() for state in synthetic_states]
            
            await log_director_decision("synthetic_states_generated", 
                                      f"Generated {len(synthetic_states)} synthetic states", 1.0)
            
            return {
                "status": "success",
                "count": len(synthetic_states),
                "states": states_data,
                "session_id": session_id
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate synthetic states: {e}")
            return {
                "status": "error",
                "message": f"Failed to generate synthetic states: {str(e)}"
            }
    
    async def train_gan_epoch(self, session_id: str, real_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train GAN for one epoch using real game states.
        
        Args:
            session_id: GAN training session ID
            real_states: List of real game states for training
            
        Returns:
            Dict containing training metrics and status
        """
        try:
            # Get GAN system
            if not hasattr(self, '_gan_systems') or session_id not in self._gan_systems:
                return {
                    "status": "error",
                    "message": f"GAN session {session_id} not found"
                }
            
            gan_system = self._gan_systems[session_id]
            
            # Convert real states to GameState objects
            from ..core.gan_system import GameState
            game_states = [GameState.from_dict(state) for state in real_states]
            
            # Train epoch
            metrics = await gan_system.train_epoch(game_states)
            
            await log_director_decision("gan_epoch_trained", 
                                      f"GAN epoch trained with {len(real_states)} real states", 1.0)
            
            return {
                "status": "success",
                "metrics": metrics,
                "session_id": session_id
            }
            
        except Exception as e:
            self.logger.error(f"Failed to train GAN epoch: {e}")
            return {
                "status": "error",
                "message": f"Failed to train GAN epoch: {str(e)}"
            }
    
    async def get_gan_training_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get GAN training status and metrics.
        
        Args:
            session_id: GAN training session ID
            
        Returns:
            Dict containing training status and metrics
        """
        try:
            # Get GAN system
            if not hasattr(self, '_gan_systems') or session_id not in self._gan_systems:
                return {
                    "status": "error",
                    "message": f"GAN session {session_id} not found"
                }
            
            gan_system = self._gan_systems[session_id]
            
            # Get training status
            status = await gan_system.get_training_status()
            
            return {
                "status": "success",
                "training_status": status,
                "session_id": session_id
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get GAN training status: {e}")
            return {
                "status": "error",
                "message": f"Failed to get GAN training status: {str(e)}"
            }
    
    async def reverse_engineer_game_mechanics(self, session_id: str, game_id: str) -> Dict[str, Any]:
        """
        Use GAN to reverse engineer game mechanics.
        
        Args:
            session_id: GAN training session ID
            game_id: Game ID to analyze
            
        Returns:
            Dict containing discovered game mechanics and rules
        """
        try:
            # Get GAN system
            if not hasattr(self, '_gan_systems') or session_id not in self._gan_systems:
                return {
                    "status": "error",
                    "message": f"GAN session {session_id} not found"
                }
            
            gan_system = self._gan_systems[session_id]
            
            # Reverse engineer mechanics
            discovered_rules = await gan_system.reverse_engineer_game_mechanics(game_id)
            
            await log_director_decision("game_mechanics_discovered", 
                                      f"Discovered mechanics for game {game_id}", 1.0)
            
            return {
                "status": "success",
                "game_id": game_id,
                "discovered_rules": discovered_rules,
                "session_id": session_id
            }
            
        except Exception as e:
            self.logger.error(f"Failed to reverse engineer game mechanics: {e}")
            return {
                "status": "error",
                "message": f"Failed to reverse engineer game mechanics: {str(e)}"
            }
    
    async def get_gan_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get GAN analytics and performance metrics.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dict containing GAN analytics and insights
        """
        try:
            # Get GAN sessions from database
            sessions = await self.db.fetch_all("""
                SELECT * FROM gan_training_sessions 
                WHERE start_time >= datetime('now', '-{} hours')
                ORDER BY start_time DESC
            """.format(hours))
            
            # Get performance metrics
            metrics = await self.db.fetch_all("""
                SELECT * FROM gan_performance_metrics 
                WHERE timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp DESC
            """.format(hours))
            
            # Get generated states count
            generated_states = await self.db.fetch_all("""
                SELECT session_id, COUNT(*) as count, AVG(quality_score) as avg_quality
                FROM gan_generated_states 
                WHERE created_at >= datetime('now', '-{} hours')
                GROUP BY session_id
            """.format(hours))
            
            # Calculate analytics
            total_sessions = len(sessions)
            total_generated = sum(gs['count'] for gs in generated_states)
            avg_quality = sum(gs['avg_quality'] for gs in generated_states) / max(len(generated_states), 1)
            
            # Get recent patterns
            recent_patterns = await self.db.fetch_all("""
                SELECT * FROM gan_pattern_learning 
                WHERE last_updated >= datetime('now', '-{} hours')
                ORDER BY last_updated DESC
                LIMIT 10
            """.format(hours))
            
            return {
                "status": "success",
                "analytics": {
                    "total_sessions": total_sessions,
                    "total_generated_states": total_generated,
                    "average_quality": avg_quality,
                    "recent_sessions": [dict(s) for s in sessions[:5]],
                    "recent_metrics": [dict(m) for m in metrics[:20]],
                    "recent_patterns": [dict(p) for p in recent_patterns]
                },
                "time_period_hours": hours
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get GAN analytics: {e}")
            return {
                "status": "error",
                "message": f"Failed to get GAN analytics: {str(e)}"
            }

# ============================================================================
# GLOBAL DIRECTOR COMMANDS INSTANCE
# ============================================================================

# Global instance for easy access
_director_commands = None

def get_director_commands() -> DirectorCommands:
    """Get global Director commands instance."""
    global _director_commands
    if _director_commands is None:
        _director_commands = DirectorCommands()
    return _director_commands

# ============================================================================
# QUICK ACCESS FUNCTIONS
# ============================================================================

async def get_system_status() -> Dict[str, Any]:
    """Quick access to system status."""
    commands = get_director_commands()
    return await commands.get_system_overview()

async def get_learning_analysis(game_id: str = None) -> Dict[str, Any]:
    """Quick access to learning analysis."""
    commands = get_director_commands()
    return await commands.get_learning_analysis(game_id)

async def get_system_health() -> Dict[str, Any]:
    """Quick access to system health analysis."""
    commands = get_director_commands()
    return await commands.analyze_system_health()
