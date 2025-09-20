#!/usr/bin/env python3
"""
Position Tracker - Tracks agent position and movement patterns.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import json
from datetime import datetime

class PositionTracker:
    """Tracks agent position and movement patterns in ARC-AGI-3 frames."""
    
    def __init__(self, base_path: str = "."):
        self.previous_frame = None
        self.agent_position = None  # (x, y) coordinates
        self.frame_history = []  # Store recent frames for pattern detection
        self.max_history = 10  # Increased for better tracking
        self.grid_size = 64  # ARC-AGI-3 uses 64x64 grids
        
        # Position tracking data
        self.position_history = deque(maxlen=50)
        self.movement_vectors = deque(maxlen=20)
        self.stability_scores = deque(maxlen=30)
        
        # Coordinate tracking for learning
        self.coordinate_attempts = {}  # Track coordinate effectiveness
        self.avoidance_scores = {}  # Track coordinates to avoid
        self.success_patterns = {}  # Track successful coordinate patterns
        
        # Meta-learning integration
        self.meta_learner = None
        self._initialize_meta_learner()
    
    def _initialize_meta_learner(self):
        """Initialize meta-learner for position tracking."""
        try:
            from ...core.reward_cap_meta_learner import RewardCapMetaLearner
            self.meta_learner = RewardCapMetaLearner()
        except ImportError:
            try:
                from ...core.reward_cap_meta_learner import RewardCapMetaLearner
                self.meta_learner = RewardCapMetaLearner()
            except ImportError:
                # Fallback for when meta-learner is not available
                class RewardCapMetaLearner:
                    def __init__(self, *args, **kwargs):
                        pass
                    def update_performance(self, *args, **kwargs):
                        pass
                    def get_current_caps(self):
                        return type('Caps', (), {
                            'productivity_multiplier': 25.0,
                            'productivity_max': 100.0,
                            'recent_gains_multiplier': 15.0,
                            'recent_gains_max': 75.0,
                            'recent_losses_multiplier': 10.0,
                            'recent_losses_max': 50.0,
                            'exploration_bonus': 15.0,
                            'movement_bonus': 20.0
                        })()
                self.meta_learner = RewardCapMetaLearner()
    
    def reset_for_new_game(self, game_id: str = None):
        """Reset tracking data for a new game."""
        self.previous_frame = None
        self.agent_position = None
        self.frame_history.clear()
        self.position_history.clear()
        self.movement_vectors.clear()
        self.stability_scores.clear()
        
        # Reset coordinate tracking
        self.coordinate_attempts.clear()
        self.avoidance_scores.clear()
        self.success_patterns.clear()
        
        if game_id:
            print(f"Position tracking reset for game {game_id}")
    
    def update_position(self, x: int, y: int, confidence: float = 1.0):
        """Update the current agent position."""
        if self.agent_position is not None:
            # Calculate movement vector
            old_x, old_y = self.agent_position
            movement_x = x - old_x
            movement_y = y - old_y
            movement_magnitude = np.sqrt(movement_x**2 + movement_y**2)
            
            # Store movement data
            self.movement_vectors.append({
                'dx': movement_x,
                'dy': movement_y,
                'magnitude': movement_magnitude,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            })
            
            # Calculate stability score
            stability = self._calculate_stability_score(x, y)
            self.stability_scores.append(stability)
        
        # Update position
        self.agent_position = (x, y)
        self.position_history.append({
            'x': x,
            'y': y,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })
    
    def _calculate_stability_score(self, x: int, y: int) -> float:
        """Calculate how stable the agent position is."""
        if len(self.position_history) < 3:
            return 1.0
        
        # Get recent positions
        recent_positions = list(self.position_history)[-3:]
        positions = [(p['x'], p['y']) for p in recent_positions]
        
        # Calculate variance in position
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        
        x_variance = np.var(x_coords)
        y_variance = np.var(y_coords)
        
        # Lower variance = higher stability
        stability = 1.0 / (1.0 + x_variance + y_variance)
        return min(1.0, stability)
    
    def get_position(self) -> Optional[Tuple[int, int]]:
        """Get the current agent position."""
        return self.agent_position
    
    def get_movement_analysis(self) -> Dict[str, Any]:
        """Get analysis of recent movement patterns."""
        if not self.movement_vectors:
            return {
                'total_movements': 0,
                'average_magnitude': 0.0,
                'movement_direction': 'unknown',
                'stability_score': 0.0
            }
        
        # Calculate movement statistics
        magnitudes = [mv['magnitude'] for mv in self.movement_vectors]
        avg_magnitude = np.mean(magnitudes)
        
        # Determine primary movement direction
        if len(self.movement_vectors) >= 3:
            recent_movements = list(self.movement_vectors)[-3:]
            avg_dx = np.mean([mv['dx'] for mv in recent_movements])
            avg_dy = np.mean([mv['dy'] for mv in recent_movements])
            
            if abs(avg_dx) > abs(avg_dy):
                direction = 'horizontal'
            elif abs(avg_dy) > abs(avg_dx):
                direction = 'vertical'
            else:
                direction = 'diagonal'
        else:
            direction = 'unknown'
        
        # Calculate stability
        stability = np.mean(list(self.stability_scores)) if self.stability_scores else 0.0
        
        return {
            'total_movements': len(self.movement_vectors),
            'average_magnitude': float(avg_magnitude),
            'movement_direction': direction,
            'stability_score': float(stability),
            'recent_movements': list(self.movement_vectors)[-5:]  # Last 5 movements
        }
    
    def record_coordinate_attempt(self, x: int, y: int, was_successful: bool, score_change: float = 0):
        """Record a coordinate attempt for learning."""
        coord_key = f"{x},{y}"
        
        if coord_key not in self.coordinate_attempts:
            self.coordinate_attempts[coord_key] = {
                'attempts': 0,
                'successes': 0,
                'total_score_change': 0.0,
                'last_attempt': None
            }
        
        record = self.coordinate_attempts[coord_key]
        record['attempts'] += 1
        record['total_score_change'] += score_change
        record['last_attempt'] = datetime.now().isoformat()
        
        if was_successful:
            record['successes'] += 1
        
        # Update avoidance scores
        if not was_successful and score_change < 0:
            self.avoidance_scores[coord_key] = self.avoidance_scores.get(coord_key, 0) + abs(score_change)
        elif was_successful and score_change > 0:
            # Reduce avoidance score for successful coordinates
            self.avoidance_scores[coord_key] = max(0, self.avoidance_scores.get(coord_key, 0) - score_change)
    
    def get_coordinate_effectiveness(self, x: int, y: int) -> Dict[str, Any]:
        """Get effectiveness data for a specific coordinate."""
        coord_key = f"{x},{y}"
        
        if coord_key not in self.coordinate_attempts:
            return {
                'attempts': 0,
                'success_rate': 0.0,
                'average_score_change': 0.0,
                'avoidance_score': 0.0,
                'recommendation': 'unknown'
            }
        
        record = self.coordinate_attempts[coord_key]
        success_rate = record['successes'] / record['attempts'] if record['attempts'] > 0 else 0.0
        avg_score_change = record['total_score_change'] / record['attempts'] if record['attempts'] > 0 else 0.0
        avoidance_score = self.avoidance_scores.get(coord_key, 0.0)
        
        # Generate recommendation
        if success_rate > 0.7 and avg_score_change > 0:
            recommendation = 'highly_recommended'
        elif success_rate > 0.5 and avg_score_change > 0:
            recommendation = 'recommended'
        elif avoidance_score > 10:
            recommendation = 'avoid'
        elif success_rate < 0.3:
            recommendation = 'not_recommended'
        else:
            recommendation = 'neutral'
        
        return {
            'attempts': record['attempts'],
            'success_rate': success_rate,
            'average_score_change': avg_score_change,
            'avoidance_score': avoidance_score,
            'recommendation': recommendation
        }
    
    def reset_coordinate_tracking(self):
        """Reset coordinate tracking data."""
        self.coordinate_attempts.clear()
        self.avoidance_scores.clear()
        self.success_patterns.clear()
    
    def get_meta_learning_summary(self) -> Dict[str, Any]:
        """Get summary of meta-learning data."""
        total_coordinates = len(self.coordinate_attempts)
        successful_coordinates = sum(1 for coord in self.coordinate_attempts.values() if coord['successes'] > 0)
        
        return {
            'total_coordinates_tracked': total_coordinates,
            'successful_coordinates': successful_coordinates,
            'coordinate_success_rate': successful_coordinates / total_coordinates if total_coordinates > 0 else 0.0,
            'avoidance_coordinates': len(self.avoidance_scores),
            'position_stability': np.mean(list(self.stability_scores)) if self.stability_scores else 0.0,
            'recent_movement_count': len(self.movement_vectors)
        }