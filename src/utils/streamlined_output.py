#!/usr/bin/env python3
"""
Streamlined Output System for Tabula Rasa

This module provides clean, focused terminal output that shows only essential information:
- Current score and progress
- Errors and exceptions
- Fallbacks and warnings
- System status updates

All verbose debug information is suppressed unless explicitly requested.
"""

import sys
import time
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

class OutputLevel(Enum):
    """Output verbosity levels."""
    MINIMAL = 1      # Only critical info (scores, errors)
    NORMAL = 2       # Normal operation info
    VERBOSE = 3      # Detailed debugging info
    DEBUG = 4        # Full debug output

class StreamlinedOutput:
    """
    Clean, focused output system for Tabula Rasa training.
    """
    
    def __init__(self, level: OutputLevel = OutputLevel.NORMAL):
        self.level = level
        self.start_time = time.time()
        self.current_score = 0
        self.current_game = None
        self.session_id = None
        self.errors_count = 0
        self.warnings_count = 0
        self.last_update = time.time()
        
        # Color support
        self._init_colors()
        
        # Status tracking
        self.status = {
            'initialized': False,
            'training': False,
            'paused': False,
            'error': False
        }
    
    def _init_colors(self):
        """Initialize color support."""
        try:
            import colorama
            from colorama import Fore, Style
            colorama.init()
            self.Fore = Fore
            self.Style = Style
            self.colors_available = True
        except ImportError:
            self.colors_available = False
            # Fallback color constants
            class Fore:
                GREEN = YELLOW = RED = BLUE = CYAN = MAGENTA = ""
            class Style:
                RESET_ALL = BOLD = ""
            self.Fore = Fore()
            self.Style = Style()
    
    def _format_time(self) -> str:
        """Format elapsed time."""
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    
    def _should_show(self, level: OutputLevel) -> bool:
        """Check if output should be shown at current verbosity level."""
        return level.value <= self.level.value
    
    def _print(self, message: str, level: OutputLevel = OutputLevel.NORMAL, 
               color: str = "", prefix: str = ""):
        """Internal print method with level checking."""
        if not self._should_show(level):
            return
        
        timestamp = self._format_time()
        if prefix:
            message = f"{prefix} {message}"
        
        if color and self.colors_available:
            print(f"{color}[{timestamp}] {message}{self.Style.RESET_ALL}", flush=True)
        else:
            print(f"[{timestamp}] {message}", flush=True)
    
    def init_session(self, session_id: str, mode: str = "training"):
        """Initialize training session."""
        self.session_id = session_id
        self.status['initialized'] = True
        self.status['training'] = True
        
        if self._should_show(OutputLevel.MINIMAL):
            self._print(f" Starting {mode.upper()} session: {session_id[:8]}...", 
                       OutputLevel.MINIMAL, self.Fore.GREEN)
    
    def update_score(self, score: int, game_id: str = None, actions: int = 0):
        """Update current score display."""
        self.current_score = score
        if game_id:
            self.current_game = game_id
        
        if self._should_show(OutputLevel.MINIMAL):
            game_info = f" (Game: {game_id[:8]})" if game_id else ""
            action_info = f" [Actions: {actions}]" if actions > 0 else ""
            self._print(f" Score: {score}{game_info}{action_info}", 
                       OutputLevel.MINIMAL, self.Fore.CYAN)
    
    def game_start(self, game_id: str, level: int = 1):
        """Game started."""
        if self._should_show(OutputLevel.NORMAL):
            self._print(f" Starting game {game_id[:8]} (Level {level})", 
                       OutputLevel.NORMAL, self.Fore.BLUE)
    
    def game_end(self, game_id: str, final_score: int, win: bool = False, 
                 actions_taken: int = 0, duration: float = 0):
        """Game ended."""
        if self._should_show(OutputLevel.MINIMAL):
            status = " WIN" if win else " LOSS"
            duration_str = f" ({duration:.1f}s)" if duration > 0 else ""
            self._print(f" Game {game_id[:8]}: {final_score} points {status} [{actions_taken} actions]{duration_str}", 
                       OutputLevel.MINIMAL, self.Fore.GREEN if win else self.Fore.RED)
    
    def error(self, message: str, exception: Exception = None, component: str = ""):
        """Log error with fallback information."""
        self.errors_count += 1
        self.status['error'] = True
        
        if self._should_show(OutputLevel.MINIMAL):
            comp_info = f" [{component}]" if component else ""
            exc_info = f" - {type(exception).__name__}: {str(exception)}" if exception else ""
            self._print(f" ERROR{comp_info}: {message}{exc_info}", 
                       OutputLevel.MINIMAL, self.Fore.RED)
    
    def warning(self, message: str, component: str = ""):
        """Log warning."""
        self.warnings_count += 1
        
        if self._should_show(OutputLevel.NORMAL):
            comp_info = f" [{component}]" if component else ""
            self._print(f" WARNING{comp_info}: {message}", 
                       OutputLevel.NORMAL, self.Fore.YELLOW)
    
    def fallback(self, original: str, fallback: str, component: str = ""):
        """Log fallback usage."""
        if self._should_show(OutputLevel.NORMAL):
            comp_info = f" [{component}]" if component else ""
            self._print(f" FALLBACK{comp_info}: {original} -> {fallback}",
                       OutputLevel.NORMAL, self.Fore.MAGENTA)
    
    def system_status(self, status: str, details: str = ""):
        """Log system status changes."""
        if self._should_show(OutputLevel.NORMAL):
            details_str = f" - {details}" if details else ""
            self._print(f" {status.upper()}{details_str}", 
                       OutputLevel.NORMAL, self.Fore.BLUE)
    
    def progress(self, current: int, total: int, item: str = "items"):
        """Log progress."""
        if self._should_show(OutputLevel.NORMAL):
            percentage = (current / total * 100) if total > 0 else 0
            self._print(f" Progress: {current}/{total} {item} ({percentage:.1f}%)", 
                       OutputLevel.NORMAL, self.Fore.CYAN)
    
    def debug(self, message: str, component: str = ""):
        """Debug message (only shown in debug mode)."""
        comp_info = f" [{component}]" if component else ""
        self._print(f" DEBUG{comp_info}: {message}", 
                   OutputLevel.DEBUG, self.Fore.MAGENTA)
    
    def verbose(self, message: str, component: str = ""):
        """Verbose message (only shown in verbose mode)."""
        comp_info = f" [{component}]" if component else ""
        self._print(f" {component.upper() if component else 'INFO'}: {message}", 
                   OutputLevel.VERBOSE, "")
    
    def session_summary(self):
        """Print session summary."""
        if not self._should_show(OutputLevel.MINIMAL):
            return
        
        duration = self._format_time()
        print(f"\n{'='*60}")
        print(f" SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Duration: {duration}")
        print(f"Final Score: {self.current_score}")
        print(f"Errors: {self.errors_count}")
        print(f"Warnings: {self.warnings_count}")
        print(f"Status: {' SUCCESS' if not self.status['error'] else ' ERRORS DETECTED'}")
        print(f"{'='*60}")
    
    def set_level(self, level: OutputLevel):
        """Change output verbosity level."""
        self.level = level
        if self._should_show(OutputLevel.NORMAL):
            self._print(f"Output level set to: {level.name}", 
                       OutputLevel.NORMAL, self.Fore.BLUE)

# Global streamlined output instance
_streamlined_output = None

def get_streamlined_output() -> StreamlinedOutput:
    """Get global streamlined output instance."""
    global _streamlined_output
    if _streamlined_output is None:
        _streamlined_output = StreamlinedOutput()
    return _streamlined_output

def init_streamlined_output(level: OutputLevel = OutputLevel.NORMAL) -> StreamlinedOutput:
    """Initialize global streamlined output with specified level."""
    global _streamlined_output
    _streamlined_output = StreamlinedOutput(level)
    return _streamlined_output

# Convenience functions for easy use
def log_score(score: int, game_id: str = None, actions: int = 0):
    """Log score update."""
    get_streamlined_output().update_score(score, game_id, actions)

def log_error(message: str, exception: Exception = None, component: str = ""):
    """Log error."""
    get_streamlined_output().error(message, exception, component)

def log_warning(message: str, component: str = ""):
    """Log warning."""
    get_streamlined_output().warning(message, component)

def log_fallback(original: str, fallback: str, component: str = ""):
    """Log fallback usage."""
    get_streamlined_output().fallback(original, fallback, component)

def log_system_status(status: str, details: str = ""):
    """Log system status."""
    get_streamlined_output().system_status(status, details)

def log_progress(current: int, total: int, item: str = "items"):
    """Log progress."""
    get_streamlined_output().progress(current, total, item)

def log_debug(message: str, component: str = ""):
    """Log debug message."""
    get_streamlined_output().debug(message, component)

def log_verbose(message: str, component: str = ""):
    """Log verbose message."""
    get_streamlined_output().verbose(message, component)

def show_session_summary():
    """Show session summary."""
    get_streamlined_output().session_summary()
