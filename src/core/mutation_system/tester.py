#!/usr/bin/env python3
"""
Enhanced Sandbox Tester - Tests mutations with advanced analysis.
"""

import time
import logging
import asyncio
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from ..system_design.genome import SystemGenome, MutationType, MutationImpact
from .types import Mutation, TestResult

@dataclass
class TestConfiguration:
    """Configuration for mutation testing."""
    test_games: List[str]
    max_episodes: int = 5
    timeout_per_game: float = 30.0
    performance_metrics: List[str] = None
    baseline_comparison: bool = True
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = ['win_rate', 'average_score', 'learning_efficiency', 'sample_efficiency', 'robustness']

class TestAnalytics:
    """Analytics for test results and performance trends."""
    
    def __init__(self):
        self.performance_history = []
        self.mutation_effectiveness = {}
        self.trend_analysis = {}
    
    def analyze_test_result(self, test_result: TestResult, mutation: Mutation) -> Dict[str, Any]:
        """Analyze test result and extract insights."""
        analysis = {
            'mutation_id': mutation.id,
            'mutation_type': mutation.type.value,
            'success': test_result.success,
            'improvement': test_result.get_overall_improvement(),
            'confidence': mutation.confidence,
            'test_duration': test_result.test_duration,
            'performance_breakdown': self._analyze_performance_breakdown(test_result),
            'trend_indicators': self._calculate_trend_indicators(test_result),
            'risk_assessment': self._assess_risk(test_result, mutation)
        }
        
        self.performance_history.append(analysis)
        return analysis
    
    def _analyze_performance_breakdown(self, test_result: TestResult) -> Dict[str, Any]:
        """Analyze performance metrics breakdown."""
        metrics = test_result.performance_metrics
        improvements = test_result.improvement_over_baseline
        
        breakdown = {}
        for metric in ['win_rate', 'average_score', 'learning_efficiency', 'sample_efficiency', 'robustness']:
            breakdown[metric] = {
                'value': metrics.get(metric, 0.0),
                'improvement': improvements.get(metric, 0.0),
                'relative_improvement': improvements.get(metric, 0.0) / max(metrics.get(metric, 0.01), 0.01)
            }
        
        return breakdown
    
    def _calculate_trend_indicators(self, test_result: TestResult) -> Dict[str, Any]:
        """Calculate trend indicators from test result."""
        return {
            'consistency': self._calculate_consistency(test_result),
            'stability': self._calculate_stability(test_result),
            'scalability': self._calculate_scalability(test_result)
        }
    
    def _calculate_consistency(self, test_result: TestResult) -> float:
        """Calculate consistency score based on performance variance."""
        metrics = list(test_result.performance_metrics.values())
        if len(metrics) < 2:
            return 0.5
        
        variance = np.var(metrics)
        return max(0.0, 1.0 - variance)
    
    def _calculate_stability(self, test_result: TestResult) -> float:
        """Calculate stability score based on improvement consistency."""
        improvements = list(test_result.improvement_over_baseline.values())
        if not improvements:
            return 0.5
        
        # Stability is higher when improvements are consistent
        mean_improvement = np.mean(improvements)
        std_improvement = np.std(improvements)
        
        if std_improvement == 0:
            return 1.0
        
        stability = 1.0 - (std_improvement / abs(mean_improvement))
        return max(0.0, min(1.0, stability))
    
    def _calculate_scalability(self, test_result: TestResult) -> float:
        """Calculate scalability score based on performance vs complexity."""
        # Simple heuristic: higher performance with reasonable test duration indicates scalability
        performance = test_result.get_overall_improvement()
        duration = test_result.test_duration
        
        if duration == 0:
            return 0.5
        
        # Scalability score decreases with test duration but increases with performance
        scalability = performance / (1.0 + duration / 60.0)  # Normalize duration to minutes
        return max(0.0, min(1.0, scalability))
    
    def _assess_risk(self, test_result: TestResult, mutation: Mutation) -> Dict[str, Any]:
        """Assess risk factors for the mutation."""
        risk_factors = []
        risk_score = 0.0
        
        # High impact mutations are riskier
        if mutation.impact.value == 'significant':
            risk_factors.append('high_impact')
            risk_score += 0.3
        
        # Low confidence mutations are riskier
        if mutation.confidence < 0.5:
            risk_factors.append('low_confidence')
            risk_score += 0.2
        
        # Long test duration indicates complexity
        if mutation.test_duration_estimate > 20.0:
            risk_factors.append('complex_testing')
            risk_score += 0.1
        
        # Negative improvements are risky
        if test_result.get_overall_improvement() < 0:
            risk_factors.append('negative_improvement')
            risk_score += 0.4
        
        return {
            'risk_score': min(1.0, risk_score),
            'risk_factors': risk_factors,
            'recommendation': self._get_risk_recommendation(risk_score)
        }
    
    def _get_risk_recommendation(self, risk_score: float) -> str:
        """Get recommendation based on risk score."""
        if risk_score < 0.2:
            return 'safe_to_proceed'
        elif risk_score < 0.5:
            return 'proceed_with_caution'
        elif risk_score < 0.8:
            return 'requires_review'
        else:
            return 'high_risk_abort'

class SandboxTester:
    """Enhanced sandbox tester with advanced analysis capabilities."""
    
    def __init__(self, base_path: Path, logger: logging.Logger):
        self.base_path = base_path
        self.logger = logger
        # Database-only mode: No sandbox directory creation
        self.sandbox_dir = None  # Disabled for database-only mode
        
        # Test history and analysis
        self.test_history = []
        self.baseline_performance = {}
        self.performance_trends = {}
        self.test_analytics = TestAnalytics()
    
    async def test_mutation(self, mutation: Mutation, 
                          baseline_genome: SystemGenome,
                          test_games: List[str] = None,
                          test_config: Optional[TestConfiguration] = None) -> TestResult:
        """Test a mutation in a sandboxed environment with enhanced analysis."""
        start_time = time.time()
        sandbox_path = None
        
        # Create test configuration
        if test_config is None:
            test_config = TestConfiguration(
                test_games=test_games or ["test_game_1"],
                max_episodes=5,
                timeout_per_game=30.0
            )
        
        try:
            # Database-only mode: Skip sandbox testing
            if self.sandbox_dir is None:
                self.logger.warning("Sandbox testing disabled in database-only mode")
                return self._create_enhanced_mock_test_result(mutation, baseline_genome, start_time, test_config)
            
            # Create sandbox environment
            sandbox_path = self._create_sandbox(mutation)
            
            # Apply mutation to genome
            mutated_genome = mutation.apply_to_genome(baseline_genome)
            
            # Run tests in sandbox
            test_results = await self._run_sandbox_tests(sandbox_path, mutated_genome, test_games)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(test_results)
            improvement_over_baseline = self._calculate_improvement(test_results, baseline_genome)
            
            # Determine success
            success = self._evaluate_success(performance_metrics, improvement_over_baseline)
            
            # Create test result
            test_result = TestResult(
                mutation_id=mutation.id,
                genome_hash=mutated_genome.get_hash(),
                success=success,
                performance_metrics=performance_metrics,
                improvement_over_baseline=improvement_over_baseline,
                test_duration=time.time() - start_time,
                detailed_results=test_results
            )
            
            # Perform enhanced analysis
            analysis = self.test_analytics.analyze_test_result(test_result, mutation)
            test_result.detailed_results['analysis'] = analysis
            
            # Store in test history
            self.test_history.append({
                'mutation': mutation,
                'test_result': test_result,
                'analysis': analysis,
                'timestamp': time.time()
            })
            
            return test_result
            
        except Exception as e:
            self.logger.error(f"Sandbox test failed for mutation {mutation.id}: {e}")
            return TestResult(
                mutation_id=mutation.id,
                genome_hash=baseline_genome.get_hash(),
                success=False,
                performance_metrics={},
                improvement_over_baseline={},
                test_duration=time.time() - start_time,
                error_log=str(e)
            )
        finally:
            # Cleanup sandbox
            if sandbox_path and sandbox_path.exists():
                self._cleanup_sandbox(sandbox_path)
    
    def _create_enhanced_mock_test_result(self, mutation: Mutation, baseline_genome: SystemGenome, 
                                        start_time: float, test_config: TestConfiguration) -> TestResult:
        """Create an enhanced mock test result with realistic performance simulation."""
        # Simulate performance based on mutation characteristics
        base_performance = self._simulate_performance_from_mutation(mutation)
        
        # Add some realistic variation
        performance_metrics = {}
        for metric in test_config.performance_metrics:
            base_value = base_performance.get(metric, 0.5)
            variation = np.random.normal(0, 0.05)  # 5% standard deviation
            performance_metrics[metric] = max(0.0, min(1.0, base_value + variation))
        
        # Calculate improvement over baseline
        baseline_metrics = self._get_baseline_performance(baseline_genome)
        improvement_over_baseline = {}
        for metric in test_config.performance_metrics:
            baseline_value = baseline_metrics.get(metric, 0.5)
            current_value = performance_metrics[metric]
            improvement_over_baseline[metric] = current_value - baseline_value
        
        # Determine success based on overall improvement
        overall_improvement = sum(improvement_over_baseline.values()) / len(improvement_over_baseline)
        success = overall_improvement > 0.01  # 1% improvement threshold
        
        test_result = TestResult(
            mutation_id=mutation.id,
            genome_hash=baseline_genome.get_hash(),
            success=success,
            performance_metrics=performance_metrics,
            improvement_over_baseline=improvement_over_baseline,
            test_duration=time.time() - start_time,
            detailed_results={'mode': 'enhanced_mock', 'test_config': test_config.__dict__}
        )
        
        # Perform analysis on mock result
        analysis = self.test_analytics.analyze_test_result(test_result, mutation)
        test_result.detailed_results['analysis'] = analysis
        
        return test_result
    
    def _simulate_performance_from_mutation(self, mutation: Mutation) -> Dict[str, float]:
        """Simulate performance metrics based on mutation characteristics."""
        base_performance = {
            'win_rate': 0.6,
            'average_score': 70.0,
            'learning_efficiency': 0.55,
            'sample_efficiency': 0.45,
            'robustness': 0.65
        }
        
        # Adjust based on mutation type and impact
        if mutation.type == MutationType.PARAMETER_ADJUSTMENT:
            # Parameter adjustments typically improve efficiency
            base_performance['learning_efficiency'] += 0.05
            base_performance['sample_efficiency'] += 0.03
        elif mutation.type == MutationType.FEATURE_TOGGLE:
            # Feature toggles can improve robustness
            base_performance['robustness'] += 0.08
        elif mutation.type == MutationType.ARCHITECTURE_ENHANCEMENT:
            # Architecture changes can improve win rate
            base_performance['win_rate'] += 0.1
            base_performance['average_score'] += 10.0
        elif mutation.type == MutationType.MODE_MODIFICATION:
            # Mode changes can improve learning
            base_performance['learning_efficiency'] += 0.08
        
        # Adjust based on confidence
        confidence_factor = mutation.confidence
        for metric in base_performance:
            if metric in ['win_rate', 'learning_efficiency', 'sample_efficiency', 'robustness']:
                base_performance[metric] *= (0.8 + 0.4 * confidence_factor)
            else:
                base_performance[metric] *= (0.9 + 0.2 * confidence_factor)
        
        return base_performance
    
    def _get_baseline_performance(self, baseline_genome: SystemGenome) -> Dict[str, float]:
        """Get baseline performance metrics for comparison."""
        # Return default baseline performance
        return {
            'win_rate': 0.6,
            'average_score': 70.0,
            'learning_efficiency': 0.55,
            'sample_efficiency': 0.45,
            'robustness': 0.65
        }
    
    def _create_mock_test_result(self, mutation: Mutation, baseline_genome: SystemGenome, start_time: float) -> TestResult:
        """Create a mock test result for database-only mode."""
        # Simulate some performance metrics
        performance_metrics = {
            'win_rate': 0.6 + (hash(mutation.id) % 100) / 1000,  # Random between 0.6-0.7
            'average_score': 70.0 + (hash(mutation.id) % 200) / 10,  # Random between 70-90
            'learning_efficiency': 0.5 + (hash(mutation.id) % 100) / 1000,  # Random between 0.5-0.6
            'sample_efficiency': 0.4 + (hash(mutation.id) % 100) / 1000,  # Random between 0.4-0.5
            'robustness': 0.7 + (hash(mutation.id) % 100) / 1000  # Random between 0.7-0.8
        }
        
        # Simulate improvement over baseline
        improvement_over_baseline = {
            'win_rate': (hash(mutation.id) % 200 - 100) / 1000,  # Random between -0.1 and 0.1
            'average_score': (hash(mutation.id) % 200 - 100) / 10,  # Random between -10 and 10
            'learning_efficiency': (hash(mutation.id) % 200 - 100) / 1000,  # Random between -0.1 and 0.1
            'sample_efficiency': (hash(mutation.id) % 200 - 100) / 1000,  # Random between -0.1 and 0.1
            'robustness': (hash(mutation.id) % 200 - 100) / 1000  # Random between -0.1 and 0.1
        }
        
        # Determine success based on overall improvement
        overall_improvement = sum(improvement_over_baseline.values()) / len(improvement_over_baseline)
        success = overall_improvement > 0.01  # 1% improvement threshold
        
        return TestResult(
            mutation_id=mutation.id,
            genome_hash=baseline_genome.get_hash(),
            success=success,
            performance_metrics=performance_metrics,
            improvement_over_baseline=improvement_over_baseline,
            test_duration=time.time() - start_time,
            detailed_results={'mode': 'mock', 'reason': 'database_only'}
        )
    
    def _create_sandbox(self, mutation: Mutation) -> Path:
        """Create a sandbox environment for testing."""
        # This would create a temporary directory and copy the codebase
        # For now, return None since we're in database-only mode
        return None
    
    async def _run_sandbox_tests(self, sandbox_path: Path, genome: SystemGenome, test_games: List[str]) -> Dict[str, Any]:
        """Run tests in the sandbox environment."""
        # This would run the actual tests
        # For now, return mock data
        return {'mode': 'mock', 'tests_run': 0}
    
    def _calculate_performance_metrics(self, test_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance metrics from test results."""
        # This would calculate real metrics from test results
        # For now, return mock data
        return {
            'win_rate': 0.65,
            'average_score': 75.0,
            'learning_efficiency': 0.55,
            'sample_efficiency': 0.45,
            'robustness': 0.75
        }
    
    def _calculate_improvement(self, test_results: Dict[str, Any], baseline_genome: SystemGenome) -> Dict[str, float]:
        """Calculate improvement over baseline."""
        # This would calculate real improvement
        # For now, return mock data
        return {
            'win_rate': 0.05,
            'average_score': 5.0,
            'learning_efficiency': 0.05,
            'sample_efficiency': 0.05,
            'robustness': 0.05
        }
    
    def _evaluate_success(self, performance_metrics: Dict[str, float], improvement: Dict[str, float]) -> bool:
        """Evaluate if the mutation was successful."""
        # Simple success criteria: overall improvement > 0
        overall_improvement = sum(improvement.values()) / len(improvement)
        return overall_improvement > 0.01
    
    def _cleanup_sandbox(self, sandbox_path: Path):
        """Clean up sandbox environment."""
        # This would clean up the temporary directory
        pass
