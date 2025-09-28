"""
Automated Test Runner

Comprehensive test suite for the modularized ARC-AGI-3 system.
"""

import unittest
import sys
import os
import time
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

class TestResult:
    """Test result container."""
    
    def __init__(self, test_name: str, success: bool, duration: float, 
                 memory_usage: float, error: Optional[str] = None):
        self.test_name = test_name
        self.success = success
        self.duration = duration
        self.memory_usage = memory_usage
        self.error = error
        self.timestamp = time.time()

class PerformanceTestSuite(unittest.TestCase):
    """Performance and functionality test suite."""
    
    def setUp(self):
        """Set up test environment."""
        self.process = psutil.Process(os.getpid())
        self.start_memory = self.process.memory_info().rss / 1024 / 1024
        self.start_time = time.time()
    
    def tearDown(self):
        """Clean up after test."""
        self.end_time = time.time()
        self.end_memory = self.process.memory_info().rss / 1024 / 1024
        self.duration = self.end_time - self.start_time
        self.memory_delta = self.end_memory - self.start_memory
    
    def test_package_imports(self):
        """Test that all packages can be imported."""
        packages = [
            'src.training',
            'src.vision', 
            'src.adapters',
            'src.analysis',
            'src.learning',
            'src.monitoring',
            'src.core'
        ]
        
        for package in packages:
            with self.subTest(package=package):
                try:
                    __import__(package)
                except Exception as e:
                    self.fail(f"Failed to import {package}: {e}")
    
    def test_core_classes_instantiation(self):
        """Test that core classes can be instantiated."""
        try:
            from src.training import ContinuousLearningLoop, MasterARCTrainer, MasterTrainingConfig
            from src.vision import FrameAnalyzer
            from src.core import SystemGenome
            
            # Test instantiation
            config = MasterTrainingConfig()
            trainer = MasterARCTrainer(config)
            loop = ContinuousLearningLoop()
            analyzer = FrameAnalyzer()
            genome = SystemGenome()
            
            # Verify they're the right types
            self.assertIsInstance(config, MasterTrainingConfig)
            self.assertIsInstance(trainer, MasterARCTrainer)
            self.assertIsInstance(loop, ContinuousLearningLoop)
            self.assertIsInstance(analyzer, FrameAnalyzer)
            self.assertIsInstance(genome, SystemGenome)
            
        except Exception as e:
            self.fail(f"Failed to instantiate core classes: {e}")
    
    def test_vision_system_functionality(self):
        """Test vision system functionality."""
        try:
            from src.vision import FrameAnalyzer
            import numpy as np
            
            analyzer = FrameAnalyzer()
            
            # Test with a simple frame
            test_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            result = analyzer.analyze_frame_for_action6_targets(test_frame.tolist())
            
            # Verify result structure
            self.assertIsInstance(result, dict)
            self.assertIn('targets', result)
            self.assertIn('movement_data', result)
            self.assertIn('color_anomalies', result)
            self.assertIn('geometric_shapes', result)
            
        except Exception as e:
            self.fail(f"Vision system test failed: {e}")
    
    def test_lazy_loading_system(self):
        """Test lazy loading system."""
        try:
            from src.training.utils.lazy_loading import vision_system, training_system, learning_system
            
            # Test vision system
            analyzer = vision_system.get('analyzer')
            self.assertIsNotNone(analyzer)
            
            # Test training system
            loop = training_system.get('loop')
            self.assertIsNotNone(loop)
            
            # Test learning system
            meta_learning = learning_system.get('meta_learning')
            self.assertIsNotNone(meta_learning)
            
        except Exception as e:
            self.fail(f"Lazy loading test failed: {e}")
    
    def test_performance_monitoring(self):
        """Test performance monitoring system."""
        try:
            from src.monitoring import performance_monitor, get_memory_usage, get_performance_summary
            
            # Test memory usage
            memory = get_memory_usage()
            self.assertIsInstance(memory, float)
            self.assertGreater(memory, 0)
            
            # Test performance summary
            summary = get_performance_summary()
            self.assertIsInstance(summary, dict)
            self.assertIn('current_memory_mb', summary)
            self.assertIn('current_cpu_percent', summary)
            
        except Exception as e:
            self.fail(f"Performance monitoring test failed: {e}")
    
    def test_backward_compatibility(self):
        """Test backward compatibility wrappers."""
        import importlib
        
        wrappers = [
            ('src.arc_integration.continuous_learning_loop', 'ContinuousLearningLoop'),
            ('src.arc_integration.opencv_feature_extractor', 'FeatureExtractor'),
            ('src.arc_integration.arc_agent_adapter', 'AdaptiveLearningARCAgent'),
            ('src.arc_integration.action_trace_analyzer', 'PatternAnalyzer'),
            ('src.arc_integration.arc_meta_learning', 'ARCMetaLearningSystem')
        ]
        
        for module_path, class_name in wrappers:
            with self.subTest(wrapper=module_path):
                try:
                    module = importlib.import_module(module_path)
                    self.assertTrue(hasattr(module, class_name))
                    cls = getattr(module, class_name)
                    self.assertIsNotNone(cls)
                except Exception as e:
                    self.fail(f"Backward compatibility test failed for {module_path}: {e}")
    
    def test_memory_efficiency(self):
        """Test memory efficiency."""
        # This test runs in setUp/tearDown
        # Check that memory usage is reasonable
        self.assertLess(self.memory_delta, 100, "Memory usage increased by more than 100MB")
    
    def test_import_performance(self):
        """Test import performance."""
        # This test runs in setUp/tearDown
        # Check that imports are reasonably fast
        self.assertLess(self.duration, 10, "Import took more than 10 seconds")


class TestRunner:
    """Main test runner class."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.process = psutil.Process(os.getpid())
    
    def run_tests(self) -> Dict[str, Any]:
        """Run all tests and return results."""
        print(" RUNNING AUTOMATED TEST SUITE")
        print("=" * 40)
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(PerformanceTestSuite)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Collect results
        test_results = {
            'total_tests': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100,
            'failures': [{'test': str(f[0]), 'error': str(f[1])} for f in result.failures],
            'errors': [{'test': str(e[0]), 'error': str(e[1])} for e in result.errors]
        }
        
        return test_results
    
    def run_performance_benchmark(self) -> Dict[str, Any]:
        """Run performance benchmark."""
        print("\\n RUNNING PERFORMANCE BENCHMARK")
        print("=" * 40)
        
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Test package imports
        packages = ['src.training', 'src.vision', 'src.adapters', 'src.analysis', 'src.learning', 'src.monitoring', 'src.core']
        for package in packages:
            __import__(package)
        
        # Test core instantiation
        from src.training import ContinuousLearningLoop, MasterARCTrainer, MasterTrainingConfig
        from src.vision import FrameAnalyzer
        from src.core import SystemGenome
        
        config = MasterTrainingConfig()
        trainer = MasterARCTrainer(config)
        loop = ContinuousLearningLoop()
        analyzer = FrameAnalyzer()
        genome = SystemGenome()
        
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024
        
        return {
            'import_time': end_time - start_time,
            'memory_usage_mb': end_memory,
            'memory_delta_mb': end_memory - start_memory,
            'packages_loaded': len(packages),
            'objects_created': 5
        }
    
    def generate_report(self, test_results: Dict[str, Any], benchmark_results: Dict[str, Any]) -> str:
        """Generate test report."""
        report = []
        report.append(" AUTOMATED TEST REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Test results
        report.append(" TEST RESULTS:")
        report.append(f"  Total Tests: {test_results['total_tests']}")
        report.append(f"  Failures: {test_results['failures']}")
        report.append(f"  Errors: {test_results['errors']}")
        report.append(f"  Success Rate: {test_results['success_rate']:.1f}%")
        report.append("")
        
        # Performance results
        report.append(" PERFORMANCE BENCHMARK:")
        report.append(f"  Import Time: {benchmark_results['import_time']:.3f}s")
        report.append(f"  Memory Usage: {benchmark_results['memory_usage_mb']:.1f} MB")
        report.append(f"  Memory Delta: {benchmark_results['memory_delta_mb']:+.1f} MB")
        report.append(f"  Packages Loaded: {benchmark_results['packages_loaded']}")
        report.append(f"  Objects Created: {benchmark_results['objects_created']}")
        report.append("")
        
        # Overall status
        if test_results['success_rate'] >= 95:
            report.append(" STATUS: EXCELLENT - System is highly reliable")
        elif test_results['success_rate'] >= 90:
            report.append(" STATUS: GOOD - System is reliable with minor issues")
        elif test_results['success_rate'] >= 80:
            report.append(" STATUS: FAIR - System has some issues that need attention")
        else:
            report.append(" STATUS: POOR - System has significant issues")
        
        return "\\n".join(report)


def main():
    """Main test runner function."""
    runner = TestRunner()
    
    # Run tests
    test_results = runner.run_tests()
    
    # Run benchmark
    benchmark_results = runner.run_performance_benchmark()
    
    # Generate report
    report = runner.generate_report(test_results, benchmark_results)
    print(report)
    
    # Save report
    with open('test_report.json', 'w') as f:
        json.dump({
            'test_results': test_results,
            'benchmark_results': benchmark_results,
            'timestamp': time.time()
        }, f, indent=2)
    
    print("\\n Report saved to test_report.json")
    
    return test_results['success_rate'] >= 90


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
