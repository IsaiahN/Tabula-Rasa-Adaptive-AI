"""
Training Monitor

Monitors training-specific metrics and progress.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class TrainingMetric(Enum):
    """Training metrics to monitor."""
    GAMES_COMPLETED = "games_completed"
    WIN_RATE = "win_rate"
    AVERAGE_SCORE = "average_score"
    LEARNING_RATE = "learning_rate"
    MEMORY_USAGE = "memory_usage"
    ACTION_COUNT = "action_count"
    SESSION_DURATION = "session_duration"


@dataclass
class TrainingProgress:
    """Training progress status."""
    session_id: str
    games_completed: int
    win_rate: float
    average_score: float
    current_game: Optional[str]
    session_duration: float
    memory_usage: float
    action_count: int
    timestamp: datetime
    status: str  # 'running', 'paused', 'completed', 'error'


class TrainingMonitor:
    """
    Monitors training progress and metrics.
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._running = False
        self._thread = None
        self._callbacks: List[Callable[[TrainingProgress], None]] = []
        self._metrics_history: List[Dict[str, Any]] = []
        self._max_history = 1000
        
        # Current metrics
        self.games_completed = 0
        self.wins = 0
        self.total_score = 0.0
        self.current_game = None
        self.session_start_time = time.time()
        self.action_count = 0
        self.memory_usage = 0.0
        self.status = 'running'
    
    def start_monitoring(self) -> None:
        """Start training monitoring."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop training monitoring."""
        self._running = False
        if self._thread:
            self._thread.join()
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                progress = self._get_current_progress()
                self._metrics_history.append({
                    'timestamp': progress.timestamp,
                    'games_completed': progress.games_completed,
                    'win_rate': progress.win_rate,
                    'average_score': progress.average_score,
                    'action_count': progress.action_count,
                    'memory_usage': progress.memory_usage,
                    'status': progress.status
                })
                
                # Keep only recent history
                if len(self._metrics_history) > self._max_history:
                    self._metrics_history = self._metrics_history[-self._max_history:]
                
                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(progress)
                    except Exception as e:
                        print(f"Error in training monitor callback: {e}")
                
                time.sleep(1)  # Update every second
            except Exception as e:
                print(f"Error in training monitoring: {e}")
                time.sleep(1)
    
    def _get_current_progress(self) -> TrainingProgress:
        """Get current training progress."""
        session_duration = time.time() - self.session_start_time
        win_rate = (self.wins / self.games_completed) if self.games_completed > 0 else 0.0
        average_score = (self.total_score / self.games_completed) if self.games_completed > 0 else 0.0
        
        return TrainingProgress(
            session_id=self.session_id,
            games_completed=self.games_completed,
            win_rate=win_rate,
            average_score=average_score,
            current_game=self.current_game,
            session_duration=session_duration,
            memory_usage=self.memory_usage,
            action_count=self.action_count,
            timestamp=datetime.now(),
            status=self.status
        )
    
    def add_callback(self, callback: Callable[[TrainingProgress], None]) -> None:
        """Add a callback for progress updates."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[TrainingProgress], None]) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def record_game_start(self, game_id: str) -> None:
        """Record the start of a new game."""
        self.current_game = game_id
    
    def record_game_end(self, won: bool, score: float) -> None:
        """Record the end of a game."""
        self.games_completed += 1
        if won:
            self.wins += 1
        self.total_score += score
        self.current_game = None
    
    def record_action(self) -> None:
        """Record an action taken."""
        self.action_count += 1
    
    def update_memory_usage(self, usage: float) -> None:
        """Update memory usage."""
        self.memory_usage = usage
    
    def set_status(self, status: str) -> None:
        """Set training status."""
        self.status = status
    
    def get_current_progress(self) -> TrainingProgress:
        """Get current training progress."""
        return self._get_current_progress()
    
    def get_metrics_history(self, duration_minutes: int = 60) -> List[Dict[str, Any]]:
        """Get metrics history for specified duration."""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        return [m for m in self._metrics_history if m['timestamp'] >= cutoff_time]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        progress = self._get_current_progress()
        return {
            'session_id': self.session_id,
            'games_completed': progress.games_completed,
            'win_rate': progress.win_rate,
            'average_score': progress.average_score,
            'total_actions': progress.action_count,
            'session_duration': progress.session_duration,
            'actions_per_minute': progress.action_count / max(1, progress.session_duration / 60),
            'games_per_hour': progress.games_completed / max(1, progress.session_duration / 3600),
            'status': progress.status
        }
    
    def get_learning_curve(self) -> List[Dict[str, Any]]:
        """Get learning curve data."""
        return [
            {
                'timestamp': m['timestamp'],
                'win_rate': m['win_rate'],
                'average_score': m['average_score'],
                'games_completed': m['games_completed']
            }
            for m in self._metrics_history
        ]
    
    def reset_session(self) -> None:
        """Reset session metrics."""
        self.games_completed = 0
        self.wins = 0
        self.total_score = 0.0
        self.current_game = None
        self.session_start_time = time.time()
        self.action_count = 0
        self.memory_usage = 0.0
        self.status = 'running'
        self._metrics_history.clear()
