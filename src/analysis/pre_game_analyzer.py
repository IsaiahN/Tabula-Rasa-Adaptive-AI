"""Pre-game analysis system for reviewing past game data before starting new games."""

import logging
import sqlite3
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import json
from dataclasses import dataclass
import numpy as np

from src.learning.game_type_classifier import GameTypeClassifier
from src.analysis.game_lifecycle_analyzer import GameLifecycleAnalyzer
from src.analysis.frame_analysis import analyze_frame_sequence

logger = logging.getLogger(__name__)

class FailureMode(Enum):
    """Failure modes that can be detected."""
    UNKNOWN = "unknown"
    STAGNATION = "stagnation"
    OSCILLATION = "oscillation"
    LOW_EFFICIENCY = "low_efficiency"
    INVALID_ACTION = "invalid_action"

@dataclass
class GameAnalysis:
    """Analysis results for a game type."""
    game_type: str
    common_failures: List[Dict[str, Any]]
    successful_patterns: List[Dict[str, Any]]
    recommended_actions: List[Dict[str, Any]]
    risk_factors: Dict[str, float]
    improvement_areas: List[str]

class PreGameAnalyzer:
    """Analyzes past game data to provide insights before starting new games."""

    def __init__(self, 
                 db_path: str,
                 game_type_classifier: GameTypeClassifier,
                 lifecycle_analyzer: Optional[GameLifecycleAnalyzer] = None):
        """Initialize pre-game analyzer.
        
        Args:
            db_path: Path to the SQLite database
            game_type_classifier: GameTypeClassifier instance for game type analysis
            lifecycle_analyzer: Optional GameLifecycleAnalyzer for failure pattern analysis
        """
        self.db_path = db_path
        self.game_type_classifier = game_type_classifier
        self.lifecycle_analyzer = lifecycle_analyzer
        self._conn: Optional[sqlite3.Connection] = None
        self._cursor: Optional[sqlite3.Cursor] = None

    def connect(self) -> None:
        """Establish database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._cursor = self._conn.cursor()

    @property
    def cursor(self) -> sqlite3.Cursor:
        """Get the database cursor, connecting if needed."""
        if self._cursor is None:
            self.connect()
        assert self._cursor is not None
        return self._cursor

    async def analyze_game_type(self, game_id: str) -> GameAnalysis:
        """Analyze past data for a specific game type before starting.
        
        Args:
            game_id: ID of the game about to be played
            
        Returns:
            GameAnalysis containing insights and recommendations
        """
        self.connect()
        game_type = self.game_type_classifier.extract_game_type(game_id)
        
        # Get game type profile
        profile = self.game_type_classifier.get_game_type_profile(game_type)
        
        # Analyze common failure patterns
        common_failures = await self._analyze_failure_patterns(game_type)
        
        # Get successful patterns
        successful_patterns = await self._analyze_successful_patterns(game_type)
        
        # Generate recommended actions
        recommended_actions = await self._generate_recommendations(
            game_type, common_failures, successful_patterns
        )
        
        # Calculate risk factors
        risk_factors = await self._calculate_risk_factors(
            game_type, common_failures, profile
        )
        
        # Identify areas for improvement
        improvement_areas = await self._identify_improvement_areas(
            game_type, risk_factors, profile
        )

        return GameAnalysis(
            game_type=game_type,
            common_failures=common_failures,
            successful_patterns=successful_patterns,
            recommended_actions=recommended_actions,
            risk_factors=risk_factors,
            improvement_areas=improvement_areas
        )

    async def _analyze_failure_patterns(self, game_type: str) -> List[Dict[str, Any]]:
        """Analyze common failure patterns for the game type."""
        failures = []
        
        try:
            # Query recent failures
            self.cursor.execute("""
                SELECT failure_mode, action_sequence, score_trajectory, 
                       frequency, avg_actions_to_failure
                FROM game_lifecycle_patterns
                WHERE game_type = ? 
                  AND success_rate < 0.3
                ORDER BY frequency DESC, last_updated DESC
                LIMIT 10
            """, (game_type,))
            
            for row in self.cursor.fetchall():
                pattern = {
                    'failure_mode': FailureMode(row['failure_mode']),
                    'action_sequence': json.loads(row['action_sequence']),
                    'score_trajectory': json.loads(row['score_trajectory']),
                    'frequency': row['frequency'],
                    'avg_actions_to_failure': row['avg_actions_to_failure']
                }
                failures.append(pattern)
                
            # Add lifecycle analyzer patterns if available
            if self.lifecycle_analyzer and game_type in self.lifecycle_analyzer.game_over_patterns:
                pattern = self.lifecycle_analyzer.game_over_patterns[game_type]
                failures.append({
                    'failure_mode': pattern.failure_mode,
                    'action_sequence': pattern.failure_action_window,
                    'score_trajectory': pattern.score_trajectory,
                    'frequency': 1,
                    'avg_actions_to_failure': pattern.avg_actions_to_failure
                })
                
        except Exception as e:
            logger.error(f"Error analyzing failure patterns: {e}")
            
        return failures

    async def _analyze_successful_patterns(self, game_type: str) -> List[Dict[str, Any]]:
        """Analyze successful patterns and strategies."""
        patterns = []
        
        try:
            # Get winning sequences
            self.cursor.execute("""
                SELECT pattern_data, success_rate, frequency
                FROM learned_patterns
                WHERE pattern_type = 'winning_sequence' 
                  AND game_context = ?
                  AND success_rate >= 0.5
                ORDER BY success_rate DESC, frequency DESC
                LIMIT 10
            """, (game_type,))
            
            for row in self.cursor.fetchall():
                pattern = {
                    'sequence': json.loads(row['pattern_data']),
                    'success_rate': row['success_rate'],
                    'frequency': row['frequency']
                }
                patterns.append(pattern)
                
            # Get successful coordinates
            self.cursor.execute("""
                SELECT x, y, success_rate, attempts, successes
                FROM coordinate_intelligence
                WHERE game_id LIKE ? 
                  AND success_rate >= 0.5
                ORDER BY success_rate DESC, successes DESC
                LIMIT 10
            """, (f"{game_type}%",))
            
            for row in self.cursor.fetchall():
                pattern = {
                    'coordinate': (row['x'], row['y']),
                    'success_rate': row['success_rate'],
                    'attempts': row['attempts'],
                    'successes': row['successes']
                }
                patterns.append(pattern)
                
        except Exception as e:
            logger.error(f"Error analyzing successful patterns: {e}")
            
        return patterns

    async def _generate_recommendations(self,
                                     game_type: str,
                                     failures: List[Dict[str, Any]],
                                     successes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        try:
            # Recommend avoiding problematic action sequences
            for failure in failures:
                if 'action_sequence' in failure:
                    recommendations.append({
                        'type': 'avoid_sequence',
                        'sequence': failure['action_sequence'],
                        'reason': f"Led to {failure['failure_mode']} failure",
                        'confidence': failure['frequency'] / 10.0  # Scale frequency to confidence
                    })
                    
            # Recommend successful sequences
            for success in successes:
                if 'sequence' in success:
                    recommendations.append({
                        'type': 'try_sequence',
                        'sequence': success['sequence'],
                        'reason': "Previously successful pattern",
                        'confidence': success['success_rate']
                    })
                elif 'coordinate' in success:
                    recommendations.append({
                        'type': 'explore_coordinate',
                        'coordinate': success['coordinate'],
                        'reason': "High success rate location",
                        'confidence': success['success_rate']
                    })
                    
            # Sort by confidence
            recommendations.sort(key=lambda x: x['confidence'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            
        return recommendations

    async def _calculate_risk_factors(self,
                                    game_type: str,
                                    failures: List[Dict[str, Any]],
                                    profile: Any) -> Dict[str, float]:
        """Calculate risk factors for various aspects of gameplay."""
        risks = {
            'overall_failure_risk': 0.0,
            'stagnation_risk': 0.0,
            'oscillation_risk': 0.0,
            'efficiency_risk': 0.0
        }
        
        try:
            # Calculate overall failure risk
            if profile.total_games > 0:
                failure_rate = 1 - (profile.successful_games / profile.total_games)
                risks['overall_failure_risk'] = failure_rate
                
            # Get all failure patterns that include frame data
            frame_patterns = [f for f in failures if f.get('frame_sequence')]
            
            for pattern in frame_patterns:
                # Analyze frame sequences for stagnation and oscillation
                result = analyze_frame_sequence(
                    frames=pattern['frame_sequence'],
                    actions=pattern['action_sequence']
                )
                
                # Update risk metrics based on frame analysis
                if result.is_stagnating:
                    risks['stagnation_risk'] += 0.2
                    
                if result.is_oscillating:
                    risks['oscillation_risk'] += 0.2
                    
                # Penalize low action diversity more heavily
                if result.action_diversity < 0.3:
                    risks['stagnation_risk'] += 0.3
                    
                # Check for unchanged regions indicating "safe zone" behavior
                if len(result.repeated_regions) > 3:  # Multiple unchanged regions
                    risks['stagnation_risk'] += 0.1 * len(result.repeated_regions)
                    
                # Check for action cycles
                if result.cycles_detected:
                    risks['oscillation_risk'] += 0.15 * len(result.cycles_detected)
            
            # Normalize risk values to 0-1 range
            risks['stagnation_risk'] = min(1.0, risks['stagnation_risk'])
            risks['oscillation_risk'] = min(1.0, risks['oscillation_risk'])
                
            # Calculate efficiency risk based on average score
            if profile.avg_score < 10:
                risks['efficiency_risk'] = 0.8
            elif profile.avg_score < 50:
                risks['efficiency_risk'] = 0.5
            else:
                risks['efficiency_risk'] = 0.2
                
        except Exception as e:
            logger.error(f"Error calculating risk factors: {e}")
            
        return risks

    async def _identify_improvement_areas(self,
                                        game_type: str,
                                        risks: Dict[str, float],
                                        profile: Any) -> List[str]:
        """Identify areas that need improvement based on analysis."""
        areas = []
        
        try:
            # Check overall success rate
            if profile.total_games > 0:
                success_rate = profile.successful_games / profile.total_games
                if success_rate < 0.3:
                    areas.append("Overall success rate is very low")
                elif success_rate < 0.6:
                    areas.append("Room for improvement in success rate")
                    
            # Check risk factors
            if risks['stagnation_risk'] > 0.4:
                areas.append("High risk of score stagnation")
            if risks['oscillation_risk'] > 0.3:
                areas.append("Score oscillation patterns detected")
            if risks['efficiency_risk'] > 0.6:
                areas.append("Low score efficiency")
                
            # Check action patterns
            if profile.action6_centric_count > 0:
                areas.append("Heavy reliance on action 6 (coordinate clicks)")
                
        except Exception as e:
            logger.error(f"Error identifying improvement areas: {e}")
            
        return areas