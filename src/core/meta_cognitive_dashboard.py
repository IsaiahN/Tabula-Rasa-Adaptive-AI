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
        # Basic configuration
        self.mode = mode
        self.update_interval = float(update_interval)
        self.data_retention_hours = int(data_retention_hours)
        self.logger = logger or logging.getLogger(f"{__name__}.Dashboard")

        # Data storage
        self.metrics = deque(maxlen=10000)
        self.events = deque(maxlen=1000)
        self.performance_history = defaultdict(lambda: deque(maxlen=500))
        self.decision_history = deque(maxlen=200)

        # Current state tracking
        self.current_session_id: Optional[str] = None
        self.session_start_time: Optional[float] = None
        self.last_update = time.time()
        # Track last scores per source for delta computation
        self._last_scores: Dict[str, float] = {}

        # Dashboard runtime state
        self.is_running = False
        self.update_thread: Optional[threading.Thread] = None
        self.gui_root = None
        self.gui_components: Dict[str, Any] = {}

        # Event subscribers
        self.event_subscribers: List[Callable[[DashboardEvent], None]] = []

        # Initialize display (GUI or console)
        self._initialize_display()

        self.logger.info(f"Meta-cognitive dashboard initialized in {self.mode.value} mode")
    
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
            # Enhanced value processing to handle different data types
            try:
                # Handle string status values specially
                if isinstance(value, str):
                    if value.lower() in ['starting', 'started', 'begin']:
                        float_value = 1.0  # Starting state
                    elif value.lower() in ['completed', 'finished', 'done', 'success']:
                        float_value = 100.0  # Completed state
                    elif value.lower() in ['failed', 'error', 'failed']:
                        float_value = -1.0  # Error state
                    elif value.lower() in ['running', 'active', 'in_progress']:
                        float_value = 50.0  # Active state
                    else:
                        # Try to extract numeric value from string
                        import re
                        numbers = re.findall(r'-?\d+\.?\d*', str(value))
                        float_value = float(numbers[0]) if numbers else 0.0
                elif isinstance(value, bool):
                    float_value = 1.0 if value else 0.0
                else:
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
        print(" META-COGNITIVE DASHBOARD - CONSOLE MODE")
        print("="*80)
    
    def _initialize_gui(self):
        """Initialize GUI-based display with enhanced error handling."""
        try:
            self.gui_root = tk.Tk()
            self.gui_root.title("Meta-Cognitive Dashboard")
            self.gui_root.geometry("1200x800")
            
            # Set up proper encoding for Windows
            if os.name == 'nt':  # Windows
                try:
                    self.gui_root.tk.call('encoding', 'system', 'utf-8')
                except:
                    pass  # Ignore if encoding setting fails
            
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

            # Frame Analyzer tab (GUI only) - shows frames, requested actions, scores
            frame_analyzer_frame = ttk.Frame(notebook)
            notebook.add(frame_analyzer_frame, text="Frame Analyzer")
            self._create_frame_analyzer_tab(frame_analyzer_frame)

            # File Ops tab - show deletions, ranking and progress on learning files
            file_ops_frame = ttk.Frame(notebook)
            notebook.add(file_ops_frame, text="File Ops")
            self._create_file_ops_tab(file_ops_frame)
            
            # Start GUI update loop
            self.gui_root.after(1000, self._update_gui)
            
            self.logger.info("GUI dashboard initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GUI: {e}")
            # Fallback to console mode
            self.mode = DashboardMode.CONSOLE
            self._initialize_console()
            raise
    
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
        perf_frame = ttk.Frame(parent)
        perf_frame.pack(fill='both', expand=True, padx=6, pady=6)

        stats_frame = ttk.LabelFrame(perf_frame, text='Live Performance', padding=6)
        stats_frame.pack(fill='x', padx=4, pady=4)
        self.gui_components['perf_latest'] = ttk.Label(stats_frame, text='Latest: N/A')
        self.gui_components['perf_latest'].pack(anchor='w')

        # If matplotlib is available, embed a small time-series plot
        if MATPLOTLIB_AVAILABLE:
            try:
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                fig = plt.Figure(figsize=(6, 3), dpi=80)
                ax = fig.add_subplot(111)
                ax.set_title('Score over time')
                ax.set_ylabel('Score')
                ax.set_xlabel('Time')
                canvas = FigureCanvasTkAgg(fig, master=perf_frame)
                canvas.get_tk_widget().pack(fill='both', expand=True)
                self.gui_components['perf_fig'] = fig
                self.gui_components['perf_ax'] = ax
                self.gui_components['perf_canvas'] = canvas
            except Exception:
                # Graceful fallback if embedding fails
                self.gui_components['perf_fig'] = None
                self.gui_components['perf_ax'] = None
                self.gui_components['perf_canvas'] = None
        else:
            note = ttk.Label(perf_frame, text='Matplotlib not available - install to see live graphs')
            note.pack()
    
    def _create_events_tab(self, parent):
        """Create events tab content with a searchable tree/list."""
        # Tree for events: Time, Type, Source, Title, Importance
        columns = ('Time', 'Type', 'Source', 'Title', 'Importance')
        events_tree = ttk.Treeview(parent, columns=columns, show='headings', height=20)
        for col in columns:
            events_tree.heading(col, text=col)
            events_tree.column(col, width=150)
        events_tree.pack(side='left', fill='both', expand=True, padx=6, pady=6)
        scrollbar = ttk.Scrollbar(parent, orient='vertical', command=events_tree.yview)
        events_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self.gui_components['events_tree'] = events_tree
        # search/filter entry
        search_frame = ttk.Frame(parent)
        search_frame.pack(fill='x', padx=6)
        ttk.Label(search_frame, text='Filter:').pack(side='left')
        filter_entry = ttk.Entry(search_frame)
        filter_entry.pack(side='left', fill='x', expand=True, padx=4)
        self.gui_components['events_filter'] = filter_entry

    def _create_frame_analyzer_tab(self, parent):
        """Create the Frame Analyzer GUI components."""
        # Left: frame preview (simulated with Canvas)
        left = ttk.Frame(parent)
        left.pack(side='left', fill='both', expand=True, padx=5, pady=5)

        preview_frame = ttk.LabelFrame(left, text="Frame Preview", padding=6)
        preview_frame.pack(fill='both', expand=True)

        # Canvas to draw simple rectangles/text to simulate frame updates
        canvas = tk.Canvas(preview_frame, bg='black', width=480, height=360)
        canvas.pack(fill='both', expand=True)
        self.gui_components['frame_canvas'] = canvas

        # Right: action list and score info
        right = ttk.Frame(parent)
        right.pack(side='right', fill='y', padx=5, pady=5)

        actions_frame = ttk.LabelFrame(right, text='Requested Actions', padding=6)
        actions_frame.pack(fill='both', expand=False)
        actions_list = tk.Listbox(actions_frame, height=10, width=40)
        actions_list.pack(fill='both', expand=True)
        self.gui_components['actions_list'] = actions_list

        score_frame = ttk.LabelFrame(right, text='Score', padding=6)
        score_frame.pack(fill='x', pady=6)
        self.gui_components['current_score'] = ttk.Label(score_frame, text='Score: 0')
        self.gui_components['current_score'].pack(anchor='w')
        self.gui_components['win_score'] = ttk.Label(score_frame, text='Win Score: 0')
        self.gui_components['win_score'].pack(anchor='w')
        self.gui_components['score_delta'] = ttk.Label(score_frame, text='Δ: 0')
        self.gui_components['score_delta'].pack(anchor='w')

        # Session/scorecard full names
        id_frame = ttk.LabelFrame(right, text='Session / Scorecard', padding=6)
        id_frame.pack(fill='x', pady=6)
        self.gui_components['full_scorecard_name'] = ttk.Label(id_frame, text='Scorecard: N/A')
        self.gui_components['full_scorecard_name'].pack(anchor='w')
        self.gui_components['full_session_name'] = ttk.Label(id_frame, text='Session: N/A')
        self.gui_components['full_session_name'].pack(anchor='w')

    def _create_file_ops_tab(self, parent):
        """Create File Operations tab for deletion and ranking visualization."""
        top = ttk.Frame(parent)
        top.pack(fill='both', expand=True, padx=5, pady=5)

        ops_frame = ttk.LabelFrame(top, text='File Operations', padding=6)
        ops_frame.pack(fill='both', expand=True)

        # Tree for files with ranking/priority
        columns = ('File', 'Rank', 'Status')
        file_tree = ttk.Treeview(ops_frame, columns=columns, show='headings', height=20)
        for col in columns:
            file_tree.heading(col, text=col)
            file_tree.column(col, width=200)
        file_tree.pack(side='left', fill='both', expand=True)
        scrollbar = ttk.Scrollbar(ops_frame, orient='vertical', command=file_tree.yview)
        file_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self.gui_components['file_tree'] = file_tree

        # Controls to simulate rank-up / delete
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill='x', padx=5, pady=5)
        ttk.Button(control_frame, text='Simulate Rank Up', command=self._simulate_rank_up).pack(side='left', padx=4)
        ttk.Button(control_frame, text='Simulate Delete', command=self._simulate_delete).pack(side='left', padx=4)

        # Status label
        self.gui_components['file_ops_status'] = ttk.Label(parent, text='No operations yet')
        self.gui_components['file_ops_status'].pack(anchor='w', padx=8, pady=4)

    # --- File ops simulation helpers ---
    def _simulate_rank_up(self):
        """Simulate ranking up the top candidate file."""
        if not hasattr(self, '_sim_files') or not self._sim_files:
            return
        # Pick highest-priority 'consider' file
        candidate = None
        for f in self._sim_files:
            if f['status'] == 'consider':
                candidate = f
                break
        if not candidate:
            candidate = sorted(self._sim_files, key=lambda x: x['rank'])[0]
        candidate['rank'] += 1
        candidate['status'] = 'ranked'
        self.log_event('FILE_RANK_UP', 'file_ops', f"Ranked up {candidate['file']}", f"New rank: {candidate['rank']}", importance=0.6, data=candidate)

    def _simulate_delete(self):
        """Simulate deleting the lowest-ranked file."""
        if not hasattr(self, '_sim_files') or not self._sim_files:
            return
        # Pick lowest rank
        lowest = sorted(self._sim_files, key=lambda x: x['rank'])[0]
        lowest['status'] = 'deleted'
        self.log_event('FILE_DELETED', 'file_ops', f"Deleted {lowest['file']}", f"Rank was: {lowest['rank']}", importance=0.8, data=lowest)
    
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
        """Update console display with enhanced Windows compatibility."""
        try:
            # Clear screen (works on Windows and Unix)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Use safe printing for Windows compatibility
            def safe_console_print(text):
                try:
                    print(text, flush=False)
                except (UnicodeEncodeError, OSError):
                    # Fallback for Windows encoding issues
                    safe_text = text.encode('ascii', errors='replace').decode('ascii')
                    print(safe_text, flush=False)
            
            safe_console_print("\n" + "="*80)
            safe_console_print(" META-COGNITIVE DASHBOARD")
            safe_console_print("="*80)
            
            # Session info
            if self.current_session_id:
                duration = time.time() - self.session_start_time if self.session_start_time else 0
                safe_console_print(f" Session: {self.current_session_id}")
                safe_console_print(f"⏱  Duration: {duration:.1f}s")
                safe_console_print("")
            
            # Quick statistics
            summary = self.get_performance_summary(hours=1)
            
            safe_console_print(" QUICK STATISTICS (Last Hour)")
            safe_console_print("-" * 40)
            safe_console_print(f"Governor Decisions: {summary['decisions']['governor']}")
            safe_console_print(f"Architect Evolutions: {summary['decisions']['architect']}")
            safe_console_print(f"Average Confidence: {summary['decisions']['average_confidence']:.2f}")
            safe_console_print(f"Total Events: {summary['events']['total']}")
            safe_console_print("")
            
            # Recent events
            safe_console_print(" RECENT EVENTS")
            safe_console_print("-" * 40)
            recent_events = list(self.events)[-5:]  # Last 5 events
            
            if recent_events:
                for event in reversed(recent_events):
                    timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
                    importance_indicator = "" if event.importance > 0.7 else "" if event.importance > 0.4 else ""
                    
                    # Safe title and description handling
                    safe_title = str(event.title).encode('ascii', errors='replace').decode('ascii')
                    safe_desc = str(event.description).encode('ascii', errors='replace').decode('ascii')
                    
                    safe_console_print(f"{timestamp} {importance_indicator} {event.source}: {safe_title}")
                    if len(safe_desc) < 60:
                        safe_console_print(f"         {safe_desc}")
                    else:
                        safe_console_print(f"         {safe_desc[:57]}...")
                    safe_console_print("")
            else:
                safe_console_print("No events recorded yet")
            
            safe_console_print("-" * 80)
            safe_console_print(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
            
        except Exception as e:
            # Ultimate fallback for any console update issues
            try:
                print(f"Dashboard update error: {e}")
                print("Meta-Cognitive Dashboard is running...")
                print(f"Session: {self.current_session_id or 'None'}")
                print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
            except:
                pass  # Silent fallback if even basic printing fails
    
    def _update_gui(self):
        """Update GUI components with enhanced error handling."""
        if not self.gui_root:
            return
        
        try:
            # Update session info
            if self.current_session_id:
                safe_session_id = str(self.current_session_id).encode('ascii', errors='replace').decode('ascii')
                self.gui_components['session_id'].config(text=f"Session: {safe_session_id}")
                
                if self.session_start_time:
                    duration = time.time() - self.session_start_time
                    self.gui_components['session_duration'].config(text=f"Duration: {duration:.1f}s")
            
            # Update quick stats with safe value handling
            try:
                summary = self.get_performance_summary(hours=1)
                stats = [
                    summary['decisions']['governor'],
                    summary['decisions']['architect'], 
                    len([m for m in self.metrics if m.metric_type == MetricType.PERFORMANCE]),
                    len([e for e in self.events if 'LEARNING' in e.event_type])
                ]
                
                for i, stat in enumerate(stats):
                    if f'stat_{i}' in self.gui_components:
                        # Ensure stat is safely displayable
                        safe_stat = str(stat) if stat is not None else "0"
                        self.gui_components[f'stat_{i}'].config(text=safe_stat)
            except Exception as stats_error:
                self.logger.warning(f"Failed to update stats: {stats_error}")
            
            # Update activity log with safe text handling
            if 'activity_log' in self.gui_components:
                try:
                    log_widget = self.gui_components['activity_log']
                    
                    # Add recent events safely
                    recent_events = list(self.events)[-3:]  # Last 3 events
                    current_content = log_widget.get('1.0', tk.END)
                    
                    for event in recent_events:
                        timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
                        
                        # Safe text handling for GUI
                        safe_source = str(event.source).encode('ascii', errors='replace').decode('ascii')
                        safe_title = str(event.title).encode('ascii', errors='replace').decode('ascii')
                        log_line = f"[{timestamp}] {safe_source}: {safe_title}\n"
                        
                        if log_line not in current_content:
                            log_widget.insert(tk.END, log_line)
                            log_widget.see(tk.END)
                    
                    # Keep log manageable
                    lines = log_widget.get('1.0', tk.END).split('\n')
                    if len(lines) > 100:
                        log_widget.delete('1.0', f'{len(lines)-50}.0')
                except Exception as log_error:
                    self.logger.warning(f"Failed to update activity log: {log_error}")
            
            # Update decision tree with safe data handling
            if 'decision_tree' in self.gui_components:
                try:
                    tree = self.gui_components['decision_tree']
                    
                    # Clear and repopulate with recent decisions
                    for item in tree.get_children():
                        tree.delete(item)
                    
                    recent_decisions = list(self.decision_history)[-20:]  # Last 20 decisions
                    for decision in reversed(recent_decisions):
                        timestamp = datetime.fromtimestamp(decision['timestamp']).strftime("%H:%M:%S")
                        
                        # Safe value extraction and display
                        safe_source = str(decision.get('source', 'Unknown')).encode('ascii', errors='replace').decode('ascii')
                        safe_type = str(decision.get('type', 'Unknown')).encode('ascii', errors='replace').decode('ascii')
                        safe_confidence = f"{decision.get('confidence', 0.0):.2f}"
                        safe_context = str(decision.get('context', {}))[:30] + "..."
                        
                        tree.insert('', 0, values=(
                            timestamp,
                            safe_source,
                            safe_type,
                            safe_confidence,
                            safe_context
                        ))
                except Exception as tree_error:
                    self.logger.warning(f"Failed to update decision tree: {tree_error}")

            # Update Frame Analyzer GUI components
            try:
                # Simulated frame drawing: draw moving rectangles to indicate motion
                if 'frame_canvas' in self.gui_components:
                    canvas = self.gui_components['frame_canvas']
                    canvas.delete('all')
                    # Use last few performance metrics to generate simple shapes
                    recent_perf = list(self.performance_history.values())
                    # Simple moving box based on time
                    t = int(time.time() * 10) % 400
                    x = 20 + (t % 440)
                    y = 20 + ((t * 3) % 300)
                    canvas.create_rectangle(x, y, x+60, y+40, fill='lime')
                    canvas.create_text(10, 10, anchor='nw', text=f"Updated: {datetime.now().strftime('%H:%M:%S')}", fill='white')

                # Actions list - show last performance metrics as 'requested actions'
                if 'actions_list' in self.gui_components:
                    actions_list = self.gui_components['actions_list']
                    actions_list.delete(0, tk.END)
                    # Convert recent metrics into human-friendly actions and show score deltas
                    recent_metrics = list(self.metrics)[-12:]
                    for m in reversed(recent_metrics):
                        try:
                            name = str(m.name)
                            src = str(m.source)
                            val = float(m.value) if m.value is not None else 0.0
                        except Exception:
                            name = str(m.name)
                            src = str(m.source)
                            val = 0.0

                        delta_text = ''
                        # If this metric looks like a score, compute delta versus last seen
                        if 'score' in name.lower():
                            prev = self._last_scores.get(src, 0.0)
                            delta = val - prev
                            symbol = '+' if delta > 0 else '-' if delta < 0 else ' '
                            delta_text = f" ({symbol}{abs(delta):.1f})"
                            self._last_scores[src] = val

                        label = f"{src}:{name}={val:.1f}{delta_text}"
                        actions_list.insert(tk.END, label)

                # Score display - show current and win score with delta
                if 'current_score' in self.gui_components:
                    cur = 0.0
                    win = 0.0
                    # Look for 'score' and 'win_score' metrics
                    for m in list(self.metrics)[-50:]:
                        if m.name.lower() in ('score', 'average_score', 'game_score'):
                            try:
                                cur = float(m.value)
                            except:
                                cur = cur
                        if m.name.lower() in ('win_score', 'target_score', 'target'):
                            try:
                                win = float(m.value)
                            except:
                                win = win

                    delta = cur - getattr(self, '_last_displayed_score', 0.0)
                    self._last_displayed_score = cur
                    self.gui_components['current_score'].config(text=f"Score: {cur:.1f}")
                    self.gui_components['win_score'].config(text=f"Win Score: {win:.1f}")
                    self.gui_components['score_delta'].config(text=f"Δ: {delta:+.1f}")

                    # Full names
                    full_scorecard = getattr(self, '_scorecard_name', 'Scorecard: N/A')
                    full_session = str(self.current_session_id or 'Session: N/A')
                    self.gui_components['full_scorecard_name'].config(text=f"Scorecard: {full_scorecard}")
                    self.gui_components['full_session_name'].config(text=f"Session: {full_session}")

            except Exception as fa_err:
                self.logger.warning(f"Failed to update Frame Analyzer UI: {fa_err}")

            # Update File Ops view
            try:
                if 'file_tree' in self.gui_components:
                    tree = self.gui_components['file_tree']
                    # Keep an internal list of simulated learning files
                    if not hasattr(self, '_sim_files'):
                        # Initialize with some entries
                        self._sim_files = [
                            {'file': 'learned_patterns.pkl', 'rank': 5, 'status': 'kept'},
                            {'file': 'policy_weights_v1.pt', 'rank': 3, 'status': 'kept'},
                            {'file': 'candidate_rule_set.json', 'rank': 1, 'status': 'consider'},
                        ]

                    # Repopulate tree
                    for iid in tree.get_children():
                        tree.delete(iid)
                    for f in self._sim_files:
                        tree.insert('', 'end', values=(f['file'], f['rank'], f['status']))
                    # Update status label
                    self.gui_components['file_ops_status'].config(text=f"Files: {len(self._sim_files)} | Last update: {datetime.now().strftime('%H:%M:%S')}")
            except Exception as fo_err:
                self.logger.warning(f"Failed to update File Ops UI: {fo_err}")
            # Update Events tree
            try:
                if 'events_tree' in self.gui_components:
                    tree = self.gui_components['events_tree']
                    filt = None
                    if 'events_filter' in self.gui_components:
                        filt = self.gui_components['events_filter'].get().strip()

                    # Keep last 50 events
                    recent_events = list(self.events)[-50:]
                    # Clear tree
                    for iid in tree.get_children():
                        tree.delete(iid)

                    for ev in reversed(recent_events):
                        ts = datetime.fromtimestamp(ev.timestamp).strftime('%H:%M:%S')
                        if filt and filt.lower() not in (ev.event_type.lower() + ev.source.lower() + ev.title.lower()):
                            continue
                        tree.insert('', 'end', values=(ts, ev.event_type, ev.source, str(ev.title)[:60], f"{ev.importance:.2f}"))
            except Exception as ev_err:
                self.logger.warning(f"Failed to update events tree: {ev_err}")

            # Update Performance plot if available
            try:
                if self.gui_components.get('perf_ax') is not None:
                    ax = self.gui_components['perf_ax']
                    fig = self.gui_components['perf_fig']
                    canvas = self.gui_components['perf_canvas']
                    ax.clear()
                    ax.set_title('Score over time')
                    ax.set_ylabel('Score')
                    ax.set_xlabel('Time')
                    # Aggregate recent score-like metrics
                    times = []
                    values = []
                    # look for any performance history keys that include 'score'
                    for k, hist in self.performance_history.items():
                        if 'score' in k:
                            for entry in list(hist)[-200:]:
                                times.append(datetime.fromtimestamp(entry['timestamp']))
                                values.append(entry['value'])
                    if times and values:
                        ax.plot(times, values, '-o', markersize=4)
                        fig.autofmt_xdate()
                    canvas.draw_idle()
                    # update latest label
                    if values:
                        self.gui_components['perf_latest'].config(text=f"Latest score: {values[-1]:.1f}")
            except Exception as perf_err:
                self.logger.warning(f"Failed to update performance plot: {perf_err}")
            
            # Schedule next update
            if self.gui_root:
                self.gui_root.after(2000, self._update_gui)
            
        except Exception as e:
            self.logger.error(f"Error updating GUI: {e}")
            # Try to continue with basic functionality
            try:
                if self.gui_root:
                    self.gui_root.after(5000, self._update_gui)  # Retry in 5 seconds
            except:
                pass  # Silent failure for scheduling
    
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
        """Run the GUI main loop (blocking) with enhanced error handling."""
        if self.mode == DashboardMode.GUI and self.gui_root:
            try:
                self.logger.info("Starting GUI main loop...")
                self.gui_root.mainloop()
            except Exception as e:
                self.logger.error(f"GUI main loop error: {e}")
                # Try to recover or fallback to console mode
                try:
                    self.mode = DashboardMode.CONSOLE
                    self._initialize_console()
                    self.logger.info("Fell back to console mode after GUI error")
                except Exception as fallback_error:
                    self.logger.error(f"Fallback to console failed: {fallback_error}")
        else:
            self.logger.warning("GUI mode not available or not initialized - use console mode instead")
            if self.mode != DashboardMode.CONSOLE:
                self.mode = DashboardMode.CONSOLE
                self._initialize_console()
    
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
