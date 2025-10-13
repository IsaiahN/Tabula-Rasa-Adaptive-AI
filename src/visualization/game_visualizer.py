#!/usr/bin/env python3
"""
ARC Game Visualizer - Windows Widget for Real-time and Replay Visualization
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import sqlite3
import threading
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import queue
import colorsys

class GameVisualizer:
    """Windows widget for visualizing ARC game sessions."""

    def __init__(self, db_path: str = None):
        """Initialize the visualizer."""
        self.db_path = db_path or "tabula_rasa.db"
        self.root = tk.Tk()
        self.root.title("ARC Game Visualizer")
        self.root.geometry("1200x800")

        # Visualization state
        self.current_session = None
        self.current_frame_data = []
        self.current_frame_index = 0
        self.is_playing = False
        self.play_speed = 1.0  # Frames per second

        # Hypothesis data for current session
        self.session_hypothesis_data = {}

        # Thread-safe communication
        self.frame_queue = queue.Queue()
        self.live_mode = False

        # Colors for different cell values (ARC uses 0-9)
        self.colors = self._generate_arc_colors()

        self._setup_ui()
        self._start_update_loop()

    def _generate_arc_colors(self) -> Dict[int, str]:
        """Generate colors for ARC game cells (0-9)."""
        colors = {
            0: "#000000",  # Black for empty
            1: "#0074D9",  # Blue
            2: "#FF4136",  # Red
            3: "#2ECC40",  # Green
            4: "#FFDC00",  # Yellow
            5: "#AAAAAA",  # Gray
            6: "#F012BE",  # Fuchsia
            7: "#FF851B",  # Orange
            8: "#7FDBFF",  # Aqua
            9: "#870C25",  # Maroon
        }
        return colors

    def _setup_ui(self):
        """Setup the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Mode selection
        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="replay")
        ttk.Radiobutton(mode_frame, text="Live", variable=self.mode_var,
                       value="live", command=self._mode_changed).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Radiobutton(mode_frame, text="Replay", variable=self.mode_var,
                       value="replay", command=self._mode_changed).pack(side=tk.LEFT, padx=(5, 0))

        # Session selection
        session_frame = ttk.Frame(control_frame)
        session_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(session_frame, text="Session:").pack(side=tk.LEFT)
        self.session_combo = ttk.Combobox(session_frame, width=40)
        self.session_combo.pack(side=tk.LEFT, padx=(10, 0))
        self.session_combo.bind('<<ComboboxSelected>>', self._session_selected)

        ttk.Button(session_frame, text="Refresh",
                  command=self._refresh_sessions).pack(side=tk.LEFT, padx=(5, 0))

        # Playback controls
        playback_frame = ttk.Frame(control_frame)
        playback_frame.pack(fill=tk.X, pady=(5, 0))

        self.play_button = ttk.Button(playback_frame, text="Play", command=self._toggle_play)
        self.play_button.pack(side=tk.LEFT)

        ttk.Button(playback_frame, text="Prev", command=self._prev_frame).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(playback_frame, text="Next", command=self._next_frame).pack(side=tk.LEFT, padx=(5, 0))

        # Frame slider
        self.frame_var = tk.IntVar()
        self.frame_slider = ttk.Scale(playback_frame, from_=0, to=100,
                                     variable=self.frame_var, orient=tk.HORIZONTAL,
                                     command=self._frame_changed)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))

        # Speed control
        ttk.Label(playback_frame, text="Speed:").pack(side=tk.LEFT, padx=(10, 0))
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(playback_frame, from_=0.1, to=5.0,
                               variable=self.speed_var, orient=tk.HORIZONTAL, length=100)
        speed_scale.pack(side=tk.RIGHT)

        # Main content area
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Game visualization
        viz_frame = ttk.LabelFrame(content_frame, text="Game Frame", padding=10)
        viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(viz_frame, bg="white", width=400, height=400)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Information panel
        info_frame = ttk.LabelFrame(content_frame, text="Information", padding=10)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        self.info_text = tk.Text(info_frame, width=30, height=20, wrap=tk.WORD)
        info_scroll = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scroll.set)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Initialize
        self._refresh_sessions()

    def _mode_changed(self):
        """Handle mode change between live and replay."""
        mode = self.mode_var.get()
        self.live_mode = (mode == "live")

        if self.live_mode:
            self._start_live_mode()
        else:
            self._stop_live_mode()

    def _start_live_mode(self):
        """Start live visualization mode."""
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, "Live mode active.\nWaiting for game data...\n\n")
        self._start_live_monitoring()

    def _stop_live_mode(self):
        """Stop live visualization mode."""
        self.live_mode = False

    def _start_live_monitoring(self):
        """Start monitoring for live game data."""
        def monitor():
            while self.live_mode:
                try:
                    # Check for latest live session
                    latest_session = self._get_latest_live_session()
                    if latest_session and latest_session != self.current_session:
                        self._load_session_frames(latest_session)
                        self.current_session = latest_session

                        # Auto-advance to latest frame
                        if self.current_frame_data:
                            self.current_frame_index = len(self.current_frame_data) - 1
                            self.root.after(0, self._display_current_frame)

                    time.sleep(1.0)  # Check every second
                except Exception as e:
                    print(f"Live monitoring error: {e}")
                    time.sleep(5.0)

        if self.live_mode:
            threading.Thread(target=monitor, daemon=True).start()

    def _get_latest_live_session(self) -> Optional[str]:
        """Get the most recent live session ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT session_id FROM visualization_sessions
                    WHERE end_time IS NULL OR end_time > datetime('now', '-1 minute')
                    ORDER BY start_time DESC LIMIT 1
                """)
                result = cursor.fetchone()
                return result[0] if result else None
        except Exception:
            return None

    def _refresh_sessions(self):
        """Refresh the list of available sessions."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT s.session_id, s.game_id, s.start_time, s.total_actions, s.final_score, s.game_won
                    FROM visualization_sessions s
                    ORDER BY s.start_time DESC
                    LIMIT 50
                """)

                sessions = cursor.fetchall()
                session_options = []

                for session_id, game_id, start_time, total_actions, final_score, game_won in sessions:
                    win_status = "WON" if game_won else "LOST"
                    display_text = f"{start_time} | {game_id[:12]}... | {total_actions} actions | Score: {final_score} | {win_status}"
                    session_options.append((display_text, session_id))

                self.session_combo['values'] = [opt[0] for opt in session_options]
                self.session_data = {opt[0]: opt[1] for opt in session_options}

                if session_options:
                    self.session_combo.set(session_options[0][0])

        except Exception as e:
            messagebox.showerror("Database Error", f"Could not load sessions: {e}")

    def _session_selected(self, event=None):
        """Handle session selection."""
        selected_display = self.session_combo.get()
        if selected_display in self.session_data:
            session_id = self.session_data[selected_display]
            self._load_session_frames(session_id)

    def _load_session_frames(self, session_id: str):
        """Load frame data for a session."""
        try:
            print(f"[VIZ DEBUG] Loading frames for session: {session_id}")
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT action_number, frame_data, frame_width, frame_height,
                           action_taken, action_x, action_y, score_before, score_after, available_actions
                    FROM game_visualization
                    WHERE session_id = ?
                    ORDER BY action_number
                """, (session_id,))

                frames = cursor.fetchall()
                print(f"[VIZ DEBUG] Found {len(frames)} frames")
                
                # Load hypothesis data for this session
                hypothesis_data = {}
                try:
                    cursor.execute("""
                        SELECT h.hypothesis_id, h.hypothesis_type, h.description, h.confidence,
                               h.reasoning, h.success_rate, h.test_count, h.created_at,
                               htr.outcome, htr.total_score_change, htr.actions_count,
                               htr.learning_insights, htr.recommendations
                        FROM game_hypotheses h
                        LEFT JOIN hypothesis_test_results htr ON h.hypothesis_id = htr.hypothesis_id
                        WHERE htr.session_id = ? OR h.hypothesis_id IN (
                            SELECT hypothesis_id FROM hypothesis_test_results WHERE session_id = ?
                        )
                        ORDER BY h.created_at
                    """, (session_id, session_id))
                    
                    hypothesis_rows = cursor.fetchall()
                    for row in hypothesis_rows:
                        hyp_id, hyp_type, desc, conf, reasoning, success_rate, test_count, created_at, outcome, score_change, actions_count, insights, recommendations = row
                        if hyp_id not in hypothesis_data:
                            hypothesis_data[hyp_id] = {
                                'hypothesis_id': hyp_id,
                                'hypothesis_type': hyp_type,
                                'description': desc,
                                'confidence': conf,
                                'reasoning': reasoning,
                                'success_rate': success_rate,
                                'test_count': test_count,
                                'created_at': created_at,
                                'test_results': []
                            }
                        
                        if outcome:  # Test result exists
                            hypothesis_data[hyp_id]['test_results'].append({
                                'outcome': outcome,
                                'score_change': score_change,
                                'actions_count': actions_count,
                                'insights': insights,
                                'recommendations': recommendations
                            })
                    
                    print(f"[VIZ DEBUG] Found {len(hypothesis_data)} hypotheses for session")
                except Exception as e:
                    print(f"[VIZ DEBUG] Error loading hypothesis data: {e}")
                    hypothesis_data = {}

                self.current_frame_data = []
                self.session_hypothesis_data = hypothesis_data  # Store hypothesis data

                for i, frame_data in enumerate(frames):
                    action_num, frame_json, width, height, action, ax, ay, score_before, score_after, actions_json = frame_data

                    print(f"[VIZ DEBUG] Processing frame {i+1}: action {action_num}")
                    print(f"[VIZ DEBUG] Raw frame_json length: {len(frame_json) if frame_json else 0}")
                    
                    try:
                        frame_array = json.loads(frame_json)
                        print(f"[VIZ DEBUG] Parsed frame array type: {type(frame_array)}")
                        print(f"[VIZ DEBUG] Frame array length: {len(frame_array) if frame_array else 0}")
                        
                        if frame_array and len(frame_array) > 0:
                            first_row = frame_array[0]
                            print(f"[VIZ DEBUG] First row type: {type(first_row)}, length: {len(first_row) if hasattr(first_row, '__len__') else 'N/A'}")
                            
                            if hasattr(first_row, '__len__') and len(first_row) > 0:
                                first_cell = first_row[0]
                                print(f"[VIZ DEBUG] First cell type: {type(first_cell)}, value: {first_cell}")
                    except Exception as e:
                        print(f"[VIZ DEBUG] JSON parse error: {e}")
                        frame_array = []

                    available_actions = json.loads(actions_json) if actions_json else []

                    self.current_frame_data.append({
                        'action_number': action_num,
                        'frame': frame_array,
                        'width': width,
                        'height': height,
                        'action': action,
                        'action_x': ax,
                        'action_y': ay,
                        'score_before': score_before,
                        'score_after': score_after,
                        'available_actions': available_actions
                    })

                self.current_session = session_id
                self.current_frame_index = 0

                # Update slider
                max_frames = len(self.current_frame_data) - 1 if self.current_frame_data else 0
                self.frame_slider.configure(to=max_frames)
                self.frame_var.set(0)

                print(f"[VIZ DEBUG] Loaded {len(self.current_frame_data)} frames, displaying first frame")
                self._display_current_frame()

        except Exception as e:
            print(f"[VIZ DEBUG] Load error: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Load Error", f"Could not load session frames: {e}")

    def _get_cell_value(self, cell) -> int:
        """Extract numeric value from cell (handles different formats)."""
        try:
            if isinstance(cell, (list, tuple)) and len(cell) > 0:
                return int(cell[0]) if isinstance(cell[0], (int, float)) else 0
            elif isinstance(cell, (int, float)):
                return int(cell)
            else:
                return 0
        except:
            return 0

    def _debug_frame_data(self, frame_data):
        """Debug frame data structure."""
        try:
            frame = frame_data['frame']
            print(f"[VIZ DEBUG] Frame type: {type(frame)}")
            print(f"[VIZ DEBUG] Frame length: {len(frame) if frame else 0}")
            
            if frame and len(frame) > 0:
                first_row = frame[0]
                print(f"[VIZ DEBUG] First row type: {type(first_row)}, length: {len(first_row) if hasattr(first_row, '__len__') else 'N/A'}")
                
                if hasattr(first_row, '__len__') and len(first_row) > 0:
                    first_cell = first_row[0]
                    print(f"[VIZ DEBUG] First cell type: {type(first_cell)}, value: {first_cell}")
                    
        except Exception as e:
            print(f"[VIZ DEBUG] Debug error: {e}")

    def _display_current_frame(self):
        """Display the current frame."""
        if not self.current_frame_data or self.current_frame_index >= len(self.current_frame_data):
            print("[VIZ DEBUG] No frame data or invalid index")
            return

        frame_data = self.current_frame_data[self.current_frame_index]
        frame = frame_data['frame']

        # Debug frame data
        self._debug_frame_data(frame_data)

        # Clear canvas
        self.canvas.delete("all")

        if not frame:
            print("[VIZ DEBUG] Frame is empty or None")
            # Draw "No Data" message
            canvas_width = self.canvas.winfo_width() or 400
            canvas_height = self.canvas.winfo_height() or 400
            self.canvas.create_text(canvas_width/2, canvas_height/2, 
                                  text="No Frame Data", 
                                  font=("Arial", 16), fill="red")
            return

        # Calculate cell size
        canvas_width = self.canvas.winfo_width() or 400
        canvas_height = self.canvas.winfo_height() or 400

        rows = len(frame)
        cols = len(frame[0]) if frame and len(frame) > 0 else 1

        if rows == 0 or cols == 0:
            print(f"[VIZ DEBUG] Invalid frame dimensions: {rows}x{cols}")
            self.canvas.create_text(canvas_width/2, canvas_height/2, 
                                  text=f"Invalid Frame: {rows}x{cols}", 
                                  font=("Arial", 16), fill="red")
            return

        # Adjust cell size for different grid sizes
        available_width = canvas_width - 40  # Leave margins
        available_height = canvas_height - 40
        
        cell_width = available_width / cols
        cell_height = available_height / rows
        cell_size = min(cell_width, cell_height)

        # For large grids (like 64x64), ensure minimum visibility
        if rows >= 32 or cols >= 32:
            # For large grids, use smaller cells but ensure they're at least 1 pixel
            cell_size = max(cell_size, 1.0)
            print(f"[VIZ DEBUG] Large grid detected ({rows}x{cols}), using cell_size: {cell_size}")
        else:
            # For smaller grids, ensure cells are at least 10 pixels for visibility
            cell_size = max(cell_size, 10.0)
            print(f"[VIZ DEBUG] Small grid detected ({rows}x{cols}), using cell_size: {cell_size}")

        # Center the grid
        grid_width = cols * cell_size
        grid_height = rows * cell_size
        start_x = (canvas_width - grid_width) / 2
        start_y = (canvas_height - grid_height) / 2

        print(f"[VIZ DEBUG] Drawing {rows}x{cols} grid, cell_size: {cell_size}")

        # Draw grid
        cells_drawn = 0
        for y, row in enumerate(frame):
            if not hasattr(row, '__len__'):
                print(f"[VIZ DEBUG] Row {y} is not iterable: {type(row)}")
                continue
                
            for x, cell in enumerate(row):
                if x >= cols:  # Safety check
                    break
                    
                x1 = start_x + x * cell_size
                y1 = start_y + y * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                # Extract numeric value from cell (handles nested structures)
                cell_value = self._get_cell_value(cell)

                # Get color for cell value
                color = self.colors.get(cell_value, "#FFFFFF")

                # Draw cell (skip if too small to be visible)
                if cell_size >= 0.5:
                    self.canvas.create_rectangle(x1, y1, x2, y2,
                                               fill=color, outline="#CCCCCC" if cell_size >= 2 else "", 
                                               width=1 if cell_size >= 2 else 0)
                    cells_drawn += 1

                    # Show cell value if non-zero and cell is large enough
                    if cell_value != 0 and cell_size >= 8:
                        text_color = "#FFFFFF" if cell_value in [1, 6, 9] else "#000000"
                        self.canvas.create_text(x1 + cell_size/2, y1 + cell_size/2,
                                              text=str(cell_value), fill=text_color, 
                                              font=("Arial", max(6, int(cell_size/3))))

        print(f"[VIZ DEBUG] Drew {cells_drawn} cells")

        # Highlight action coordinates if applicable
        if frame_data['action'] == "ACTION6" and frame_data['action_x'] is not None and frame_data['action_y'] is not None:
            ax, ay = frame_data['action_x'], frame_data['action_y']
            if 0 <= ay < rows and 0 <= ax < cols:
                x1 = start_x + ax * cell_size
                y1 = start_y + ay * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                # Draw action highlight (make it visible even for small cells)
                highlight_width = max(2, cell_size / 4)
                self.canvas.create_rectangle(x1-highlight_width, y1-highlight_width, 
                                           x2+highlight_width, y2+highlight_width,
                                           outline="#FF0000", width=int(highlight_width))
                print(f"[VIZ DEBUG] Drew action highlight at ({ax}, {ay})")

        # Add grid info overlay for large grids
        if rows >= 32 or cols >= 32:
            info_text = f"{rows}x{cols} Grid\nCell size: {cell_size:.1f}px"
            self.canvas.create_text(20, 20, text=info_text, anchor="nw", 
                                  fill="black", font=("Arial", 10))

        # Update info panel
        self._update_info_panel(frame_data)
        print(f"[VIZ DEBUG] Frame display completed for action {frame_data.get('action_number', 'unknown')}")

    def _update_info_panel(self, frame_data: Dict[str, Any]):
        """Update the information panel."""
        self.info_text.delete(1.0, tk.END)

        info = []
        info.append(f"Action Number: {frame_data['action_number']}")
        info.append(f"Frame Size: {frame_data['width']}x{frame_data['height']}")
        info.append(f"Action Taken: {frame_data['action']}")

        if frame_data['action'] == "ACTION6":
            info.append(f"Coordinates: ({frame_data['action_x']}, {frame_data['action_y']})")

        info.append(f"Score Before: {frame_data['score_before']}")
        info.append(f"Score After: {frame_data['score_after']}")
        info.append(f"Available Actions: {frame_data['available_actions']}")

        # Add hypothesis information if available
        if hasattr(self, 'session_hypothesis_data') and self.session_hypothesis_data:
            info.append("\n" + "="*30)
            info.append("HYPOTHESIS SYSTEM")
            info.append("="*30)
            info.append(f"Hypotheses for this session: {len(self.session_hypothesis_data)}")
            
            # Show top 3 hypotheses with highest confidence
            sorted_hypotheses = sorted(
                self.session_hypothesis_data.values(), 
                key=lambda h: h.get('confidence', 0), 
                reverse=True
            )
            
            for i, hyp in enumerate(sorted_hypotheses[:3]):
                info.append(f"\nHypothesis {i+1}:")
                info.append(f"  Type: {hyp.get('hypothesis_type', 'Unknown')}")
                info.append(f"  Confidence: {hyp.get('confidence', 0):.2f}")
                info.append(f"  Description: {hyp.get('description', 'No description')[:80]}...")
                
                if hyp.get('test_results'):
                    test_result = hyp['test_results'][0]  # Show first test result
                    info.append(f"  Test Outcome: {test_result.get('outcome', 'Unknown')}")
                    info.append(f"  Score Change: {test_result.get('score_change', 0):.2f}")
                
                if hyp.get('reasoning'):
                    reasoning = hyp['reasoning'][:100]  # Truncate long reasoning
                    info.append(f"  Reasoning: {reasoning}...")

        info.append("\n" + "="*30)
        info.append("Frame Analysis:")

        # Analyze frame
        frame = frame_data['frame']
        if frame and len(frame) > 0:
            try:
                total_cells = sum(len(row) for row in frame if hasattr(row, '__len__'))
                non_zero_cells = 0
                value_counts = {}

                for row in frame:
                    if hasattr(row, '__len__'):
                        for cell in row:
                            cell_value = self._get_cell_value(cell)
                            
                            if cell_value != 0:
                                non_zero_cells += 1
                            
                            value_counts[cell_value] = value_counts.get(cell_value, 0) + 1

                density = non_zero_cells / total_cells if total_cells > 0 else 0

                info.append(f"Total Cells: {total_cells}")
                info.append(f"Non-zero Cells: {non_zero_cells}")
                info.append(f"Density: {density:.2%}")

                info.append("\nCell Value Counts:")
                for value, count in sorted(value_counts.items()):
                    info.append(f"  {value}: {count}")

            except Exception as e:
                info.append(f"Frame analysis error: {e}")
                info.append(f"Raw frame type: {type(frame)}")
                if frame and len(frame) > 0:
                    info.append(f"First row type: {type(frame[0])}")
                    if hasattr(frame[0], '__len__') and len(frame[0]) > 0:
                        info.append(f"First cell type: {type(frame[0][0])}")
                        info.append(f"First cell value: {frame[0][0]}")
        else:
            info.append("No frame data available")

        self.info_text.insert(tk.END, "\n".join(info))

    def _toggle_play(self):
        """Toggle playback."""
        self.is_playing = not self.is_playing
        self.play_button.configure(text="Pause" if self.is_playing else "Play")

        if self.is_playing:
            self._start_playback()

    def _start_playback(self):
        """Start automatic playback."""
        def play():
            while self.is_playing and self.current_frame_data:
                if self.current_frame_index < len(self.current_frame_data) - 1:
                    self.current_frame_index += 1
                    self.frame_var.set(self.current_frame_index)
                    self.root.after(0, self._display_current_frame)

                    speed = self.speed_var.get()
                    time.sleep(1.0 / speed)
                else:
                    # End of frames
                    self.is_playing = False
                    self.root.after(0, lambda: self.play_button.configure(text="Play"))
                    break

        if self.is_playing:
            threading.Thread(target=play, daemon=True).start()

    def _prev_frame(self):
        """Go to previous frame."""
        if self.current_frame_data and self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.frame_var.set(self.current_frame_index)
            self._display_current_frame()

    def _next_frame(self):
        """Go to next frame."""
        if self.current_frame_data and self.current_frame_index < len(self.current_frame_data) - 1:
            self.current_frame_index += 1
            self.frame_var.set(self.current_frame_index)
            self._display_current_frame()

    def _frame_changed(self, value):
        """Handle frame slider change."""
        try:
            new_index = int(float(value))
            if 0 <= new_index < len(self.current_frame_data):
                self.current_frame_index = new_index
                self._display_current_frame()
        except (ValueError, IndexError):
            pass

    def _start_update_loop(self):
        """Start the UI update loop."""
        def update():
            # Process any queued updates
            try:
                while True:
                    self.frame_queue.get_nowait()
                    # Handle real-time updates
            except queue.Empty:
                pass

            self.root.after(100, update)  # Update every 100ms

        self.root.after(100, update)

    def run(self):
        """Run the visualizer."""
        print("Starting ARC Game Visualizer...")
        print(f"Database: {self.db_path}")
        self.root.mainloop()

    def close(self):
        """Close the visualizer."""
        self.live_mode = False
        self.is_playing = False
        self.root.destroy()


def main():
    """Main function for standalone usage."""
    import sys

    db_path = "tabula_rasa.db"
    if len(sys.argv) > 1:
        db_path = sys.argv[1]

    visualizer = GameVisualizer(db_path)

    try:
        visualizer.run()
    except KeyboardInterrupt:
        print("\nShutting down visualizer...")
        visualizer.close()


if __name__ == "__main__":
    main()