#!/usr/bin/env python3
"""
Mutation System Types - Data classes for mutations and test results.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from ..system_design.genome import SystemGenome, MutationType, MutationImpact

@dataclass
class Mutation:
    """Represents a proposed change to the system genome."""
    id: str
    type: MutationType
    impact: MutationImpact
    changes: Dict[str, Any]
    rationale: str
    expected_improvement: float
    confidence: float
    test_duration_estimate: float  # in minutes
    
    def apply_to_genome(self, genome: SystemGenome) -> SystemGenome:
        """Apply this mutation to create a new genome."""
        new_genome_dict = genome.to_dict()
        
        # Apply changes
        for key, value in self.changes.items():
            if key in new_genome_dict:
                new_genome_dict[key] = value
        
        # Update metadata
        new_genome_dict['generation'] = genome.generation + 1
        new_genome_dict['parent_hash'] = genome.get_hash()
        new_genome_dict['mutation_history'] = genome.mutation_history + [self.id]
        
        return SystemGenome(**new_genome_dict)

@dataclass
class TestResult:
    """Results from testing a mutated genome in sandbox."""
    mutation_id: str
    genome_hash: str
    success: bool
    performance_metrics: Dict[str, float]
    improvement_over_baseline: Dict[str, float]
    test_duration: float
    error_log: Optional[str] = None
    detailed_results: Optional[Dict[str, Any]] = None
    
    def get_overall_improvement(self) -> float:
        """Calculate overall improvement score."""
        if not self.success:
            return -1.0
        
        # Weight different metrics
        weights = {
            'win_rate': 100.0,
            'average_score': 1.0,
            'learning_efficiency': 10.0,
            'sample_efficiency': 5.0,
            'robustness': 20.0
        }
        
        total_improvement = 0.0
        total_weight = 0.0
        
        for metric, improvement in self.improvement_over_baseline.items():
            if metric in weights:
                total_improvement += improvement * weights[metric]
                total_weight += weights[metric]
        
        return total_improvement / max(total_weight, 1.0)
