"""
Performance Monitoring System

Provides comprehensive performance monitoring and optimization tools.
"""

import time
import psutil
import os
import gc
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
from collections import defaultdict, deque
import threading
import json
from pathlib import Path


class PerformanceMetrics:
    """Stores performance metrics for analysis."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.lock = threading.Lock()
    
    def add_metric(self, name: str, value: float, timestamp: Optional[float] = None):
        """Add a metric value."""
        if timestamp is None:
            timestamp = time.time()
        
        with self.lock:
            self.metrics[name].append({
                'value': value,
                'timestamp': timestamp
            })
    
    def get_metric(self, name: str) -> List[Dict[str, Any]]:
        """Get metric history."""
        with self.lock:
            return list(self.metrics[name])
    
    def get_latest(self, name: str) -> Optional[Dict[str, Any]]:
        """Get latest metric value."""
        with self.lock:
            if self.metrics[name]:
                return self.metrics[name][-1]
            return None
    
    def get_average(self, name: str, last_n: Optional[int] = None) -> Optional[float]:
        """Get average metric value."""
        with self.lock:
            history = list(self.metrics[name])
            if not history:
                return None
            
            if last_n is not None:
                history = history[-last_n:]
            
            return sum(item['value'] for item in history) / len(history)
    
    def clear(self, name: Optional[str] = None):
        """Clear metrics."""
        with self.lock:
            if name is None:
                self.metrics.clear()
            else:
                self.metrics[name].clear()


class PerformanceMonitor:
    """Main performance monitoring class."""
    
    def __init__(self, enable_auto_monitoring: bool = True):
        self.process = psutil.Process(os.getpid())
        self.metrics = PerformanceMetrics()
        self.enable_auto_monitoring = enable_auto_monitoring
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        if enable_auto_monitoring:
            self.start_auto_monitoring()
    
    def start_auto_monitoring(self):
        """Start automatic performance monitoring."""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.stop_monitoring.clear()
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
    
    def stop_auto_monitoring(self):
        """Stop automatic performance monitoring."""
        self.stop_monitoring.set()
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                # Memory usage
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                self.metrics.add_metric('memory_usage', memory_mb)
                
                # CPU usage
                cpu_percent = self.process.cpu_percent()
                self.metrics.add_metric('cpu_usage', cpu_percent)
                
                # Thread count
                thread_count = self.process.num_threads()
                self.metrics.add_metric('thread_count', thread_count)
                
                # File descriptor count
                try:
                    fd_count = self.process.num_fds()
                    self.metrics.add_metric('file_descriptors', fd_count)
                except (psutil.AccessDenied, AttributeError):
                    pass
                
                time.sleep(1.0)  # Monitor every second
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(5.0)
    
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager for monitoring operations."""
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024
        
        print(f"Starting {operation_name}...")
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Record metrics
            self.metrics.add_metric(f'{operation_name}_duration', duration)
            self.metrics.add_metric(f'{operation_name}_memory_delta', memory_delta)
            
            print(f"Finished {operation_name}: {duration:.3f}s, {memory_delta:+.1f}MB")
    
    def measure_function(self, func: Callable, *args, **kwargs) -> tuple[Any, float]:
        """Measure function execution time and memory usage."""
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024
        
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        
        # Record metrics
        self.metrics.add_metric(f'{func.__name__}_duration', duration)
        self.metrics.add_metric(f'{func.__name__}_memory_delta', memory_delta)
        
        return result, duration
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            'memory_usage_mb': self.get_memory_usage(),
            'cpu_usage_percent': self.get_cpu_usage(),
            'thread_count': self.process.num_threads(),
            'pid': self.process.pid,
            'create_time': self.process.create_time(),
            'status': self.process.status()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'current_memory_mb': self.get_memory_usage(),
            'current_cpu_percent': self.get_cpu_usage(),
            'average_memory_mb': self.metrics.get_average('memory_usage', last_n=60),
            'average_cpu_percent': self.metrics.get_average('cpu_usage', last_n=60),
            'peak_memory_mb': max(item['value'] for item in self.metrics.get_metric('memory_usage')) if self.metrics.get_metric('memory_usage') else 0,
            'total_operations': sum(len(history) for history in self.metrics.metrics.values())
        }
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        data = {
            'timestamp': time.time(),
            'system_info': self.get_system_info(),
            'metrics': {name: list(history) for name, history in self.metrics.metrics.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop_auto_monitoring()
        self.metrics.clear()


class MemoryProfiler:
    """Memory profiling utilities."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024
    
    def get_memory_delta(self) -> float:
        """Get memory delta from baseline."""
        current_memory = self.process.memory_info().rss / 1024 / 1024
        return current_memory - self.baseline_memory
    
    def force_gc(self):
        """Force garbage collection."""
        collected = gc.collect()
        return collected
    
    def get_memory_breakdown(self) -> Dict[str, float]:
        """Get memory breakdown by type."""
        memory_info = self.process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'shared_mb': memory_info.shared / 1024 / 1024,
            'text_mb': memory_info.text / 1024 / 1024,
            'data_mb': memory_info.data / 1024 / 1024,
            'lib_mb': memory_info.lib / 1024 / 1024,
            'dirty_mb': memory_info.dirty / 1024 / 1024
        }


# Global performance monitor instance
performance_monitor = PerformanceMonitor()

# Convenience functions
def monitor_operation(operation_name: str):
    """Convenience function for monitoring operations."""
    return performance_monitor.monitor_operation(operation_name)

def measure_function(func: Callable, *args, **kwargs):
    """Convenience function for measuring functions."""
    return performance_monitor.measure_function(func, *args, **kwargs)

def get_memory_usage() -> float:
    """Convenience function for getting memory usage."""
    return performance_monitor.get_memory_usage()

def get_performance_summary() -> Dict[str, Any]:
    """Convenience function for getting performance summary."""
    return performance_monitor.get_performance_summary()
