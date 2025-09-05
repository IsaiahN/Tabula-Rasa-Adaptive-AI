#!/usr/bin/env python3
"""
🧠 SIMPLE CONTINUOUS TRAINING - WINDOWS COMPATIBLE VERSION
===========================================================

This version fixes Unicode/emoji issues on Windows while maintaining
full meta-cognitive functionality.
"""

import asyncio
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
import signal
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# Configure Windows-compatible logging
def setup_windows_logging():
    """Set up logging that works properly on Windows."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create handlers with UTF-8 encoding
    handlers = []
    
    # Console handler with UTF-8
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)
    
    # File handler with UTF-8 encoding
    try:
        file_handler = logging.FileHandler('meta_cognitive_training.log', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    except Exception:
        pass  # Skip file logging if it fails
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )
    
    # Disable some noisy loggers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('git').setLevel(logging.WARNING)

setup_windows_logging()

try:
    # Import the working systems
    from src.core.meta_cognitive_dashboard import MetaCognitiveDashboard, DashboardMode
    from real_arc_training_with_metacognition import MetaCognitiveARCTraining
    
    # Color support (fallback if not available)
    try:
        import colorama
        from colorama import Fore, Style
        colorama.init(autoreset=True)
    except ImportError:
        class Fore:
            GREEN = CYAN = YELLOW = RED = BLUE = ""
        class Style:
            RESET_ALL = ""
            
except ImportError as e:
    print(f"ERROR: Import failed: {e}")
    print("Make sure all dependencies are installed")
    sys.exit(1)

class ContinuousMetaCognitiveRunner:
    """Windows-compatible continuous runner for meta-cognitive ARC training."""
    
    def __init__(self, dashboard_mode='console'):
        self.dashboard_mode = dashboard_mode
        self.running = True
        self.session_count = 0
        self.dashboard = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set up signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n{Fore.YELLOW}SHUTDOWN: Gracefully stopping...{Style.RESET_ALL}")
        self.running = False
        
    async def start_continuous_training(self):
        """Start continuous training with meta-cognitive monitoring."""
        
        print(f"{Fore.CYAN}CONTINUOUS META-COGNITIVE ARC TRAINING")
        print(f"=" * 60)
        print(f"Dashboard Mode: {self.dashboard_mode}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"=" * 60 + Style.RESET_ALL)
        
        # Initialize dashboard
        await self._setup_dashboard()
        
        # Run continuous training loop
        await self._run_continuous_loop()
        
    async def _setup_dashboard(self):
        """Set up the meta-cognitive dashboard."""
        print(f"\n{Fore.BLUE}DASHBOARD: Initializing ({self.dashboard_mode} mode)...{Style.RESET_ALL}")
        
        # Map dashboard modes
        mode_map = {
            'gui': DashboardMode.GUI,
            'console': DashboardMode.CONSOLE, 
            'headless': DashboardMode.HEADLESS,
            'web': DashboardMode.WEB
        }
        
        dashboard_mode = mode_map.get(self.dashboard_mode, DashboardMode.CONSOLE)
        
        try:
            self.dashboard = MetaCognitiveDashboard(
                mode=dashboard_mode,
                update_interval=3.0  # Update every 3 seconds for continuous mode
            )
            
            # Start monitoring
            session_id = f"continuous_training_{int(time.time())}"
            self.dashboard.start(session_id)
            
            print(f"{Fore.GREEN}SUCCESS: Dashboard initialized and running{Style.RESET_ALL}")
            self.logger.info(f"Dashboard started in {self.dashboard_mode} mode")
            
        except Exception as e:
            print(f"{Fore.YELLOW}WARNING: Dashboard failed to initialize: {e}")
            print("Continuing without dashboard...{Style.RESET_ALL}")
            self.dashboard = None
            self.logger.warning(f"Dashboard initialization failed: {e}")
            
    async def _run_continuous_loop(self):
        """Run the continuous training loop."""
        print(f"\n{Fore.YELLOW}TRAINING: Starting Continuous Loop...")
        print(f"Press Ctrl+C to stop gracefully{Style.RESET_ALL}\n")
        
        total_start = time.time()
        
        while self.running:
            self.session_count += 1
            session_start = time.time()
            
            print(f"\n{Fore.CYAN}{'='*60}")
            print(f"TRAINING SESSION #{self.session_count}")
            print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*60}{Style.RESET_ALL}")
            
            try:
                # Create and run training session
                training_session = MetaCognitiveARCTraining()
                
                # If we have a dashboard, log session start
                if self.dashboard:
                    self.dashboard.log_performance_update({
                        'session_number': self.session_count,
                        'status': 'starting'
                    }, 'continuous_runner')
                
                # Run the actual training
                print(f"{Fore.BLUE}META-COGNITIVE: Running ARC training...{Style.RESET_ALL}")
                success = await training_session.run_training_with_monitoring()
                
                session_time = time.time() - session_start
                
                # Log results
                if success:
                    print(f"\n{Fore.GREEN}SUCCESS: Session #{self.session_count} completed!")
                    print(f"   Duration: {session_time:.1f} seconds{Style.RESET_ALL}")
                    
                    if self.dashboard:
                        self.dashboard.log_performance_update({
                            'session_number': self.session_count,
                            'duration': session_time,
                            'status': 'completed',
                            'success': True
                        }, 'continuous_runner')
                        
                else:
                    print(f"\n{Fore.YELLOW}WARNING: Session #{self.session_count} completed with issues")
                    print(f"   Duration: {session_time:.1f} seconds{Style.RESET_ALL}")
                    
                    if self.dashboard:
                        self.dashboard.log_performance_update({
                            'session_number': self.session_count,
                            'duration': session_time,
                            'status': 'completed',
                            'success': False
                        }, 'continuous_runner')
                
                self.logger.info(f"Session {self.session_count} completed: success={success}, duration={session_time:.1f}s")
                
                # Show progress every 3 sessions
                if self.session_count % 3 == 0:
                    await self._show_progress()
                
                # Wait between sessions (adaptive timing)
                wait_time = self._calculate_wait_time(session_time, success)
                if wait_time > 0 and self.running:
                    print(f"\n{Fore.BLUE}WAIT: {wait_time:.0f}s before next session...{Style.RESET_ALL}")
                    
                    # Wait in chunks so we can respond to shutdown signals
                    for _ in range(int(wait_time)):
                        if not self.running:
                            break
                        await asyncio.sleep(1)
                
            except Exception as e:
                print(f"\n{Fore.RED}ERROR: Session #{self.session_count} failed: {e}{Style.RESET_ALL}")
                self.logger.error(f"Session {self.session_count} failed: {e}")
                
                # Log error to dashboard
                if self.dashboard:
                    self.dashboard.log_performance_update({
                        'session_number': self.session_count,
                        'status': 'failed',
                        'error': str(e)
                    }, 'continuous_runner')
                
                # Wait longer after errors
                if self.running:
                    print(f"{Fore.BLUE}WAIT: 120s after error...{Style.RESET_ALL}")
                    for _ in range(120):
                        if not self.running:
                            break
                        await asyncio.sleep(1)
        
        # Shutdown
        await self._shutdown_gracefully(total_start)
        
    def _calculate_wait_time(self, session_duration, success):
        """Calculate wait time between sessions."""
        base_wait = 60  # Base 1 minute wait
        
        # Adjust based on session duration
        if session_duration < 30:  # Very short session
            return base_wait * 2
        elif session_duration > 300:  # Long session (5+ minutes)
            return base_wait * 0.5
        
        # Adjust based on success
        if not success:
            return base_wait * 1.5
            
        return base_wait
        
    async def _show_progress(self):
        """Show progress summary."""
        total_runtime = time.time() - self.start_time if hasattr(self, 'start_time') else 0
        
        print(f"\n{Fore.CYAN}PROGRESS SUMMARY")
        print(f"{'='*40}")
        print(f"Sessions Completed: {self.session_count}")
        print(f"Total Runtime: {total_runtime/60:.1f} minutes")
        
        if self.dashboard:
            try:
                summary = self.dashboard.get_performance_summary(hours=1)
                print(f"Governor Decisions: {summary.get('decisions', {}).get('governor', 0)}")
                print(f"Architect Evolutions: {summary.get('decisions', {}).get('architect', 0)}")
            except Exception as e:
                print("Dashboard summary unavailable")
                self.logger.warning(f"Dashboard summary failed: {e}")
                
        print(f"{'='*40}{Style.RESET_ALL}")
        
    async def _shutdown_gracefully(self, total_start):
        """Graceful shutdown procedure."""
        total_runtime = time.time() - total_start
        
        print(f"\n{Fore.YELLOW}SHUTDOWN: Graceful shutdown in progress")
        print(f"{'='*50}")
        
        # Stop dashboard
        if self.dashboard:
            print(f"DASHBOARD: Stopping...")
            self.dashboard.stop()
            
            # Export session data
            export_path = Path(f"continuous_session_{int(time.time())}.json")
            try:
                if self.dashboard.export_session_data(export_path):
                    print(f"EXPORT: Session data saved to {export_path}")
                    self.logger.info(f"Session data exported to {export_path}")
            except Exception as e:
                print(f"EXPORT: Failed to save session data: {e}")
                self.logger.error(f"Export failed: {e}")
        
        # Final summary
        print(f"\n{Fore.GREEN}COMPLETE: CONTINUOUS TRAINING FINISHED")
        print(f"Sessions: {self.session_count}")
        print(f"Runtime: {total_runtime/60:.1f} minutes")
        print(f"Avg per session: {total_runtime/max(self.session_count,1):.1f}s")
        print(f"{'='*50}{Style.RESET_ALL}")
        
        self.logger.info(f"Training completed: {self.session_count} sessions, {total_runtime/60:.1f} minutes")
        print("Thank you for using the Meta-Cognitive ARC Training System!")

async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Windows-Compatible Continuous Meta-Cognitive ARC Training")
    parser.add_argument('--dashboard', choices=['console', 'gui', 'headless'], 
                       default='console', help='Dashboard mode')
    
    args = parser.parse_args()
    
    print(f"{Fore.CYAN}TABULA RASA - CONTINUOUS META-COGNITIVE TRAINING")
    print(f"Windows-compatible version with enhanced error handling{Style.RESET_ALL}")
    
    runner = ContinuousMetaCognitiveRunner(dashboard_mode=args.dashboard)
    runner.start_time = time.time()
    
    try:
        await runner.start_continuous_training()
        return 0
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}INTERRUPTED: Stopped by user{Style.RESET_ALL}")
        return 0
    except Exception as e:
        print(f"\n{Fore.RED}SYSTEM ERROR: {e}{Style.RESET_ALL}")
        logging.error(f"System error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
