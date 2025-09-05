#!/usr/bin/env python3
"""
Meta-Cognitive Visualization Dashboard

Real-time monitoring interface for Governor decisions, Architect evolutions,
and system performance metrics during training sessions. Provides developers
with insights into meta-cognitive decision-making processes.
"""

import json
import os
import time
import logging
import threading
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
from collections import deque, defaultdict
import statistics
from datetime import datetime, timedelta

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.animation import FuncAnimation
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("matplotlib not available - using text-based dashboard")

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext
    import tkinter.font as font
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("tkinter not available - dashboard will be console-based")

class DashboardMode(Enum):
    """Different modes for the dashboard display."""
    CONSOLE = "console"
    GUI = "gui"
    WEB = "web"
    HEADLESS = "headless"

class MetricType(Enum):
    """Types of metrics tracked by the dashboard."""
    PERFORMANCE = "performance"
    DECISION = "decision"
    EVOLUTION = "evolution"
    LEARNING = "learning"
    SYSTEM = "system"

@dataclass
class DashboardMetric:
    """A single metric data point."""
    timestamp: float
    metric_type: MetricType
    source: str  # "governor", "architect", "system"
    name: str
    value: float
    metadata: Dict[str, Any]

@dataclass  
class DashboardEvent:
    """A significant event in the meta-cognitive system."""
    timestamp: float
    event_type: str
    source: str
    title: str
    description: str
    importance: float  # 0.0 to 1.0
    data: Dict[str, Any]

class MetaCognitiveDashboard:
    """Real-time visualization dashboard for meta-cognitive systems."""
    
    def __init__(self, mode: DashboardMode = DashboardMode.CONSOLE,
                 update_interval: float = 2.0,
                 data_retention_hours: int = 24,
                 logger: Optional[logging.Logger] = None):
        
        self.mode = mode
        self.update_interval = update_interval
        self.data_retention_hours = data_retention_hours
        self.logger = logger or logging.getLogger(f"{__name__}.Dashboard")
        
        # Data storage
        self.metrics = deque(maxlen=10000)
        self.events = deque(maxlen=1000)
        self.performance_history = defaultdict(lambda: deque(maxlen=500))
        self.decision_history = deque(maxlen=200)
        
        # Current state tracking
        self.current_session_id = None
        self.session_start_time = None
        self.last_update = time.time()
        
        # Dashboard state
        self.is_running = False
        self.update_thread = None
        self.gui_root = None
        self.gui_components = {}
        
        # Event subscribers
        self.event_subscribers = []
        
        # Initialize display
        self._initialize_display()
        
        self.logger.info(f"Meta-cognitive dashboard initialized in {mode.value} mode")
    
    def start(self, session_id: str = None):
        """Start the dashboard monitoring."""
        self.current_session_id = session_id or f"session_{int(time.time())}"
        self.session_start_time = time.time()
        self.is_running = True
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        # Log session start event
        self.log_event("SESSION_START", "system", "Session Started", 
                      f"Meta-cognitive monitoring session {self.current_session_id} started",
                      importance=0.8)
        
        self.logger.info(f"Dashboard monitoring started for session {self.current_session_id}")
    
    def stop(self):
        """Stop the dashboard monitoring."""
        self.is_running = False
        
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5.0)
        
        # Log session end event
        if self.session_start_time:
            duration = time.time() - self.session_start_time
            self.log_event("SESSION_END", "system", "Session Ended",
                          f"Session lasted {duration:.1f} seconds", importance=0.8)
        
        self.logger.info("Dashboard monitoring stopped")
    
    def log_governor_decision(self, recommendation: Dict[str, Any], 
                            confidence: float, context: Dict[str, Any]):
        """Log a Governor decision."""
        timestamp = time.time()
        
        # Add metrics
        self.add_metric(MetricType.DECISION, "governor", "confidence", confidence,
                       {"recommendation_type": recommendation.get("type", "unknown")})
        
        self.add_metric(MetricType.DECISION, "governor", "decision_count", 1.0, {})
        
        # Add event
        rec_type = recommendation.get("type", "unknown")
        title = f"Governor Recommendation: {rec_type}"
        description = f"Confidence: {confidence:.2f}, Changes: {recommendation.get('configuration_changes', {})}"
        
        self.log_event("GOVERNOR_DECISION", "governor", title, description, 
                      importance=confidence, data=recommendation)
        
        # Store in decision history
        self.decision_history.append({
            'timestamp': timestamp,
            'source': 'governor',
            'type': rec_type,
            'confidence': confidence,
            'context': context
        })
    
    def log_architect_evolution(self, mutation: Dict[str, Any], 
                               test_result: Dict[str, Any]):
        """Log an Architect evolution event."""
        timestamp = time.time()
        
        # Add metrics
        success_score = test_result.get("success_score", 0.0)
        self.add_metric(MetricType.EVOLUTION, "architect", "mutation_success", success_score, 
                       {"mutation_type": mutation.get("type", "unknown")})
        
        self.add_metric(MetricType.EVOLUTION, "architect", "evolution_count", 1.0, {})
        
        # Add event
        mutation_type = mutation.get("type", "unknown")
        title = f"Architect Evolution: {mutation_type}"
        description = f"Success: {success_score:.2f}, Changes: {mutation.get('changes', {})}"
        
        self.log_event("ARCHITECT_EVOLUTION", "architect", title, description,
                      importance=success_score, data={"mutation": mutation, "result": test_result})
    
    def log_performance_update(self, metrics: Dict[str, float], source: str = "system"):
        """Log performance metrics update."""
        timestamp = time.time()
        
        for metric_name, value in metrics.items():
            # Ensure value is a float to prevent type errors
            try:
                float_value = float(value) if value is not None else 0.0
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid metric value for {metric_name}: {value}, using 0.0")
                float_value = 0.0
                
            self.add_metric(MetricType.PERFORMANCE, source, metric_name, float_value, {})
            
            # Store in performance history
            self.performance_history[metric_name].append({
                'timestamp': timestamp,
                'value': float_value,
                'source': source
            })
    
    def log_learning_update(self, patterns_learned: int, success_rate: float, 
                          insights: List[str], source: str = "learning_manager"):
        """Log cross-session learning updates."""
        self.add_metric(MetricType.LEARNING, source, "patterns_learned", patterns_learned, {})
        self.add_metric(MetricType.LEARNING, source, "success_rate", success_rate, {})
        
        if insights:
            title = "Learning Insights"
            description = f"Success rate: {success_rate:.1%}, New insights: {len(insights)}"
            self.log_event("LEARNING_UPDATE", source, title, description,
                          importance=success_rate, data={"insights": insights})
    
    def add_metric(self, metric_type: MetricType, source: str, name: str, 
                  value: float, metadata: Dict[str, Any] = None):
        """Add a metric data point."""
        # Ensure value is a float to prevent arithmetic errors
        try:
            float_value = float(value) if value is not None else 0.0
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid metric value for {name}: {value}, using 0.0")
            float_value = 0.0
            
        metric = DashboardMetric(
            timestamp=time.time(),
            metric_type=metric_type,
            source=source,
            name=name,
            value=float_value,
            metadata=metadata or {}
        )
        
        self.metrics.append(metric)
        self._notify_metric_update(metric)
    
    def log_event(self, event_type: str, source: str, title: str, 
                 description: str, importance: float = 0.5, 
                 data: Dict[str, Any] = None):
        """Log a significant event."""
        event = DashboardEvent(
            timestamp=time.time(),
            event_type=event_type,
            source=source,
            title=title,
            description=description,
            importance=importance,
            data=data or {}
        )
        
        self.events.append(event)
        self._notify_event_update(event)
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for the specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        
        summary = {
            'timeframe': f"Last {hours} hour(s)",
            'metrics': {},
            'events': {
                'total': 0,
                'by_type': defaultdict(int),
                'by_source': defaultdict(int)
            },
            'decisions': {
                'governor': 0,
                'architect': 0,
                'average_confidence': 0.0
            }
        }
        
        # Analyze metrics
        recent_metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
        
        for metric in recent_metrics:
            key = f"{metric.source}_{metric.name}"
            if key not in summary['metrics']:
                summary['metrics'][key] = {
                    'count': 0,
                    'average': 0.0,
                    'min': float('inf'),
                    'max': float('-inf'),
                    'latest': None
                }
            
            stats = summary['metrics'][key]
            stats['count'] += 1
            
            # Safe averaging calculation with type checking
            try:
                metric_value = float(metric.value) if metric.value is not None else 0.0
                current_average = float(stats['average']) if stats['average'] is not None else 0.0
                stats['average'] = (current_average * (stats['count'] - 1) + metric_value) / stats['count']
                stats['min'] = min(stats['min'], metric_value)
                stats['max'] = max(stats['max'], metric_value)
                stats['latest'] = metric_value
            except (ValueError, TypeError, ZeroDivisionError) as e:
                self.logger.warning(f"Error calculating metric stats for {key}: {e}")
                stats['latest'] = 0.0
        
        # Analyze events
        recent_events = [e for e in self.events if e.timestamp > cutoff_time]
        summary['events']['total'] = len(recent_events)
        
        for event in recent_events:
            summary['events']['by_type'][event.event_type] += 1
            summary['events']['by_source'][event.source] += 1
        
        # Analyze decisions
        recent_decisions = [d for d in self.decision_history if d['timestamp'] > cutoff_time]
        
        gov_decisions = [d for d in recent_decisions if d['source'] == 'governor']
        arch_decisions = [d for d in recent_decisions if d['source'] == 'architect']
        
        summary['decisions']['governor'] = len(gov_decisions)
        summary['decisions']['architect'] = len(arch_decisions)
        
        if recent_decisions:
            try:
                # Safe confidence averaging with type checking
                confidence_values = []
                for d in recent_decisions:
                    try:
                        conf = float(d.get('confidence', 0.0)) if d.get('confidence') is not None else 0.0
                        confidence_values.append(conf)
                    except (ValueError, TypeError):
                        confidence_values.append(0.0)
                
                if confidence_values:
                    avg_confidence = statistics.mean(confidence_values)
                    summary['decisions']['average_confidence'] = avg_confidence
                else:
                    summary['decisions']['average_confidence'] = 0.0
            except Exception as e:
                self.logger.warning(f"Error calculating average confidence: {e}")
                summary['decisions']['average_confidence'] = 0.0
        
        return summary
    
    def subscribe_to_events(self, callback: Callable[[DashboardEvent], None]):
        """Subscribe to dashboard events."""
        self.event_subscribers.append(callback)
    
    def _initialize_display(self):
        """Initialize the appropriate display based on mode."""
        if self.mode == DashboardMode.GUI and TKINTER_AVAILABLE:
            self._initialize_gui()
        elif self.mode == DashboardMode.CONSOLE:
            self._initialize_console()
        else:
            self.logger.warning(f"Display mode {self.mode.value} not available, using console")
            self.mode = DashboardMode.CONSOLE
            self._initialize_console()
    
    def _initialize_console(self):
        """Initialize console-based display."""
        print("\n" + "="*80)
        print("🧠 META-COGNITIVE DASHBOARD - CONSOLE MODE")
        print("="*80)
    
    def _initialize_gui(self):
        """Initialize GUI-based display."""
        self.gui_root = tk.Tk()
        self.gui_root.title("Meta-Cognitive Dashboard")
        self.gui_root.geometry("1200x800")
        
        # Create notebook for tabs
        notebook = ttk.Notebook(self.gui_root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Overview tab
        overview_frame = ttk.Frame(notebook)
        notebook.add(overview_frame, text="Overview")
        self._create_overview_tab(overview_frame)
        
        # Decisions tab  
        decisions_frame = ttk.Frame(notebook)
        notebook.add(decisions_frame, text="Decisions")
        self._create_decisions_tab(decisions_frame)
        
        # Performance tab
        performance_frame = ttk.Frame(notebook)
        notebook.add(performance_frame, text="Performance")
        self._create_performance_tab(performance_frame)
        
        # Events tab
        events_frame = ttk.Frame(notebook)
        notebook.add(events_frame, text="Events")
        self._create_events_tab(events_frame)
        
        # Start GUI update loop
        self.gui_root.after(1000, self._update_gui)
    
    def _create_overview_tab(self, parent):
        """Create overview tab content."""
        # Session info frame
        session_frame = ttk.LabelFrame(parent, text="Current Session", padding=10)
        session_frame.pack(fill='x', padx=5, pady=5)
        
        self.gui_components['session_id'] = ttk.Label(session_frame, text="Session: Not Started")
        self.gui_components['session_id'].pack(anchor='w')
        
        self.gui_components['session_duration'] = ttk.Label(session_frame, text="Duration: 0s")
        self.gui_components['session_duration'].pack(anchor='w')
        
        # Quick stats frame
        stats_frame = ttk.LabelFrame(parent, text="Quick Statistics", padding=10)
        stats_frame.pack(fill='x', padx=5, pady=5)
        
        stats_container = ttk.Frame(stats_frame)
        stats_container.pack(fill='x')
        
        # Create columns for stats
        for i, stat_name in enumerate(['Governor Decisions', 'Architect Evolutions', 'Performance Updates', 'Learning Insights']):
            col_frame = ttk.Frame(stats_container)
            col_frame.grid(row=0, column=i, sticky='ew', padx=5)
            stats_container.columnconfigure(i, weight=1)
            
            label = ttk.Label(col_frame, text=stat_name, font=('Arial', 10, 'bold'))
            label.pack()
            
            value_label = ttk.Label(col_frame, text="0", font=('Arial', 16))
            value_label.pack()
            
            self.gui_components[f'stat_{i}'] = value_label
        
        # Status log
        log_frame = ttk.LabelFrame(parent, text="Recent Activity", padding=10)
        log_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.gui_components['activity_log'] = scrolledtext.ScrolledText(
            log_frame, height=15, width=80, font=('Courier', 9)
        )
        self.gui_components['activity_log'].pack(fill='both', expand=True)
    
    def _create_decisions_tab(self, parent):
        """Create decisions tab content."""
        # Decision summary
        summary_frame = ttk.LabelFrame(parent, text="Decision Summary", padding=10)
        summary_frame.pack(fill='x', padx=5, pady=5)
        
        self.gui_components['decision_summary'] = ttk.Label(
            summary_frame, text="No decisions recorded yet", font=('Arial', 10)
        )
        self.gui_components['decision_summary'].pack(anchor='w')
        
        # Decision history
        history_frame = ttk.LabelFrame(parent, text="Decision History", padding=10)
        history_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create treeview for decisions
        columns = ('Time', 'Source', 'Type', 'Confidence', 'Context')
        decision_tree = ttk.Treeview(history_frame, columns=columns, show='headings', height=20)
        
        for col in columns:
            decision_tree.heading(col, text=col)
            decision_tree.column(col, width=150)
        
        scrollbar = ttk.Scrollbar(history_frame, orient='vertical', command=decision_tree.yview)
        decision_tree.configure(yscrollcommand=scrollbar.set)
        
        decision_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        self.gui_components['decision_tree'] = decision_tree
    
    def _create_performance_tab(self, parent):
        """Create performance tab content."""
        perf_label = ttk.Label(parent, text="Performance metrics will be displayed here", 
                              font=('Arial', 12))
        perf_label.pack(expand=True)
        
        # TODO: Add matplotlib integration for performance graphs
        if MATPLOTLIB_AVAILABLE:
            # Placeholder for performance graphs
            pass
    
    def _create_events_tab(self, parent):
        """Create events tab content."""
        events_label = ttk.Label(parent, text="System events will be displayed here", 
                                font=('Arial', 12))
        events_label.pack(expand=True)
    
    def _update_loop(self):
        """Main update loop for the dashboard."""
        while self.is_running:
            try:
                self._update_display()
                time.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"Error in dashboard update loop: {e}")
                time.sleep(1.0)
    
    def _update_display(self):
        """Update the display based on current mode."""
        if self.mode == DashboardMode.CONSOLE:
            self._update_console()
        # GUI updates are handled by _update_gui method
        
        self.last_update = time.time()
    
    def _update_console(self):
        """Update console display."""
        # Clear screen (works on Windows and Unix)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n" + "="*80)
        print("🧠 META-COGNITIVE DASHBOARD")
        print("="*80)
        
        # Session info
        if self.current_session_id:
            duration = time.time() - self.session_start_time if self.session_start_time else 0
            print(f"📊 Session: {self.current_session_id}")
            print(f"⏱️  Duration: {duration:.1f}s")
            print()
        
        # Quick statistics
        summary = self.get_performance_summary(hours=1)
        
        print("📈 QUICK STATISTICS (Last Hour)")
        print("-" * 40)
        print(f"Governor Decisions: {summary['decisions']['governor']}")
        print(f"Architect Evolutions: {summary['decisions']['architect']}")
        print(f"Average Confidence: {summary['decisions']['average_confidence']:.2f}")
        print(f"Total Events: {summary['events']['total']}")
        print()
        
        # Recent events
        print("📋 RECENT EVENTS")
        print("-" * 40)
        recent_events = list(self.events)[-5:]  # Last 5 events
        
        if recent_events:
            for event in reversed(recent_events):
                timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
                importance_indicator = "🔥" if event.importance > 0.7 else "⚡" if event.importance > 0.4 else "📌"
                print(f"{timestamp} {importance_indicator} {event.source}: {event.title}")
                if len(event.description) < 60:
                    print(f"         {event.description}")
                else:
                    print(f"         {event.description[:57]}...")
                print()
        else:
            print("No events recorded yet")
        
        print("-" * 80)
        print(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    
    def _update_gui(self):
        """Update GUI components."""
        if not self.gui_root:
            return
        
        try:
            # Update session info
            if self.current_session_id:
                self.gui_components['session_id'].config(text=f"Session: {self.current_session_id}")
                
                if self.session_start_time:
                    duration = time.time() - self.session_start_time
                    self.gui_components['session_duration'].config(text=f"Duration: {duration:.1f}s")
            
            # Update quick stats
            summary = self.get_performance_summary(hours=1)
            stats = [
                summary['decisions']['governor'],
                summary['decisions']['architect'], 
                len([m for m in self.metrics if m.metric_type == MetricType.PERFORMANCE]),
                len([e for e in self.events if 'LEARNING' in e.event_type])
            ]
            
            for i, stat in enumerate(stats):
                if f'stat_{i}' in self.gui_components:
                    self.gui_components[f'stat_{i}'].config(text=str(stat))
            
            # Update activity log
            if 'activity_log' in self.gui_components:
                log_widget = self.gui_components['activity_log']
                
                # Add recent events
                recent_events = list(self.events)[-3:]  # Last 3 events
                current_content = log_widget.get('1.0', tk.END)
                
                for event in recent_events:
                    timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
                    log_line = f"[{timestamp}] {event.source}: {event.title}\n"
                    
                    if log_line not in current_content:
                        log_widget.insert(tk.END, log_line)
                        log_widget.see(tk.END)
                
                # Keep log manageable
                lines = log_widget.get('1.0', tk.END).split('\n')
                if len(lines) > 100:
                    log_widget.delete('1.0', f'{len(lines)-50}.0')
            
            # Update decision tree
            if 'decision_tree' in self.gui_components:
                tree = self.gui_components['decision_tree']
                
                # Clear and repopulate with recent decisions
                for item in tree.get_children():
                    tree.delete(item)
                
                recent_decisions = list(self.decision_history)[-20:]  # Last 20 decisions
                for decision in reversed(recent_decisions):
                    timestamp = datetime.fromtimestamp(decision['timestamp']).strftime("%H:%M:%S")
                    tree.insert('', 0, values=(
                        timestamp,
                        decision['source'],
                        decision['type'],
                        f"{decision['confidence']:.2f}",
                        str(decision.get('context', {}))[:30] + "..."
                    ))
            
            # Schedule next update
            self.gui_root.after(2000, self._update_gui)
            
        except Exception as e:
            self.logger.error(f"Error updating GUI: {e}")
    
    def _notify_metric_update(self, metric: DashboardMetric):
        """Notify about metric updates."""
        # Could be extended to trigger specific updates
        pass
    
    def _notify_event_update(self, event: DashboardEvent):
        """Notify about event updates."""
        # Notify event subscribers
        for subscriber in self.event_subscribers:
            try:
                subscriber(event)
            except Exception as e:
                self.logger.error(f"Error notifying event subscriber: {e}")
    
    def run_gui(self):
        """Run the GUI main loop (blocking)."""
        if self.mode == DashboardMode.GUI and self.gui_root:
            self.gui_root.mainloop()
        else:
            self.logger.warning("GUI mode not available or not initialized")
    
    def shutdown(self):
        """Gracefully shutdown the dashboard."""
        self.stop()
        
        if self.gui_root:
            self.gui_root.quit()
    
    def export_session_data(self, filepath: Path) -> bool:
        """Export current session data to a file."""
        try:
            session_data = {
                'session_id': self.current_session_id,
                'session_start': self.session_start_time,
                'session_end': time.time(),
                'metrics': [asdict(m) for m in self.metrics],
                'events': [asdict(e) for e in self.events],
                'decisions': list(self.decision_history),
                'performance_summary': self.get_performance_summary(hours=24)
            }
            
            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            
            self.logger.info(f"Session data exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export session data: {e}")
            return False
