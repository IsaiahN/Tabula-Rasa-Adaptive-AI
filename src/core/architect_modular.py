#!/usr/bin/env python3
"""
Modular Architect - The "Zeroth Brain" using modular components.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import modular components
from .system_design.genome import SystemGenome
from .mutation_system import MutationEngine, SandboxTester, Mutation, TestResult
from .evolution_engine import EvolutionEngine
from .component_coordination import ComponentCoordinator, SystemIntegration

class Architect:
    """
    The "Zeroth Brain" - Self-Architecture Evolution System
    
    Performs safe, sandboxed experimentation on the AI's own architecture
    and hyperparameters using a general-intelligence fitness function.
    """
    
    def __init__(self, evolution_rate: float = 0.05, innovation_threshold: float = 0.8, 
                 memory_capacity: int = 500, base_path: str = ".", repo_path: str = ".", 
                 logger: Optional[logging.Logger] = None):
        self.base_path = Path(base_path)
        self.repo_path = Path(repo_path)
        self.logger = logger or logging.getLogger(f"{__name__}.Architect")
        
        # Initialize modular components
        self.current_genome = self._load_current_genome()
        self.mutation_engine = MutationEngine(self.current_genome, self.logger)
        self.sandbox_tester = SandboxTester(self.base_path, self.logger)
        self.evolution_engine = EvolutionEngine(self.logger)
        self.component_coordinator = ComponentCoordinator(self.logger)
        self.system_integration = SystemIntegration(self.logger)
        
        # Evolution state
        self.generation = 0
        self.mutation_history = []
        self.successful_mutations = []
        self.pending_requests = []
        
        # Game monitoring
        self.last_training_check = 0
        self.training_active = False
        self.game_activity_log = []
        
        # Safety measures
        self.max_concurrent_tests = 1
        self.human_approval_required = True
        self.auto_merge_threshold = 0.15
        
        self.logger.info("🔬 Modular Architect initialized - Zeroth Brain online")
    
    def _load_current_genome(self) -> SystemGenome:
        """Load current system genome from configuration."""
        # For now, return default genome
        # In real implementation, this would load from actual config files
        return SystemGenome()
    
    async def autonomous_evolution_cycle(self) -> Dict[str, Any]:
        """Run one cycle of autonomous evolution."""
        self.logger.info(f"🧬 Starting evolution cycle {self.generation}")
        
        try:
            # Use evolution engine for autonomous evolution
            evolution_result = self.evolution_engine.execute_autonomous_evolution()
            
            if evolution_result.get('success'):
                self.logger.info("🚀 Evolution Engine executed autonomous evolution")
                return evolution_result
            
            # Fallback to traditional mutation approach
            mutation = self.mutation_engine.generate_exploratory_mutation()
            
            # Test in sandbox
            test_result = await self.sandbox_tester.test_mutation(
                mutation, self.current_genome
            )
            
            # Record results
            self.mutation_history.append((mutation, test_result))
            
            improvement = test_result.get_overall_improvement()
            
            if test_result.success and improvement > 0.02:
                self.logger.info(f"✅ Beneficial mutation found: {improvement:.3f} improvement")
                
                return {
                    'success': True,
                    'generation': self.generation,
                    'improvement': improvement,
                    'mutation_id': mutation.id
                }
            else:
                self.logger.debug(f"📊 Mutation tested: {improvement:.3f} improvement (below threshold)")
                
                return {
                    'success': False,
                    'generation': self.generation,
                    'improvement': improvement,
                    'reason': 'insufficient_improvement'
                }
            
        except Exception as e:
            self.logger.error(f"❌ Error in evolution cycle: {e}")
            return {
                'success': False, 
                'error': str(e),
                'generation': self.generation
            }
        
        finally:
            self.generation += 1
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get current evolution status."""
        return {
            'generation': self.generation,
            'mutation_history_count': len(self.mutation_history),
            'successful_mutations_count': len(self.successful_mutations),
            'pending_requests_count': len(self.pending_requests),
            'current_genome_hash': self.current_genome.get_hash()
        }
    
    def evolve_strategy(self, available_actions: List[int], context: Dict[str, Any], 
                       performance_data: List[Dict[str, Any]], frame_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evolve strategy based on current performance and context.
        """
        try:
            game_id = context.get('game_id', 'unknown')
            
            # Generate evolved strategy using mutation engine
            mutation = self.mutation_engine.generate_mutation()
            
            # Create evolved strategy based on mutation
            evolved_strategy = {
                'actions': available_actions,
                'mutation_applied': mutation.id,
                'rationale': mutation.rationale,
                'confidence': mutation.confidence
            }
            
            # Generate reasoning
            reasoning = f"Strategy evolved using mutation {mutation.id}: {mutation.rationale}"
            
            return {
                'strategy': evolved_strategy,
                'reasoning': reasoning,
                'innovation_score': mutation.confidence,
                'mutation_id': mutation.id
            }
            
        except Exception as e:
            self.logger.error(f"Strategy evolution failed: {e}")
            return {
                'strategy': {'actions': available_actions, 'fallback': True},
                'reasoning': f"Fallback strategy due to error: {e}",
                'innovation_score': 0.0,
                'mutation_id': None
            }
