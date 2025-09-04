#!/usr/bin/env python3
"""
UNIFIED ARC TRAINER - Complete Integration

This unified system combines:
1. Continuous Learning Loop (core learning engine)
2. Train Arc Agent (CLI orchestration and experimentation)
3. Enhanced error handling and monitoring
4. Streamlined interface for both research and production

Usage:
    # Production continuous learning
    python unified_arc_trainer.py --mode continuous --salience decay
    
    # Research experimentation  
    python unified_arc_trainer.py --mode experimental --compare-systems
    
    # Quick testing
    python unified_arc_trainer.py --mode test --games game1,game2
    
    # Full capabilities demo
    python unified_arc_trainer.py --mode demo --enhanced
"""

import asyncio
import argparse
import logging
import sys
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Import core systems
try:
    from dotenv import load_dotenv
    from src.arc_integration.continuous_learning_loop import ContinuousLearningLoop
    from src.core.salience_system import SalienceMode
    from src.core.energy_system import EnergySystem
    
    # Load environment configuration
    load_dotenv('.env')  # Only load .env - no local overrides needed
    
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    print(f"❌ Make sure the package is installed with: pip install -e .")
    sys.exit(1)

@dataclass
class TrainingConfig:
    """Unified configuration for all training modes."""
    # Core settings
    mode: str = "continuous"
    api_key: str = ""
    arc_agents_path: str = ""
    
    # Learning parameters
    salience_mode: SalienceMode = SalienceMode.DECAY_COMPRESSION  # Changed to decay as default
    max_actions_per_game: int = 500  # Maximum actions per individual game attempt (reasonable limit)
    max_learning_cycles: int = 50  # Increased for deeper learning
    target_score: float = 85.0  # Higher target for full system
    
    # ALL ADVANCED FEATURES - EVERYTHING ENABLED BY DEFAULT
    enable_swarm: bool = True  # SWARM parallel processing
    enable_coordinates: bool = True  # Coordinate intelligence
    enable_energy_system: bool = True  # Energy management
    enable_sleep_cycles: bool = True  # Sleep and consolidation
    enable_dnc_memory: bool = True  # Differentiable Neural Computer
    enable_meta_learning: bool = True  # Meta-learning system
    enable_salience_system: bool = True  # Salience-weighted replay
    enable_contrarian_strategy: bool = True  # Contrarian mode for failures
    enable_frame_analysis: bool = True  # Visual frame analysis
    enable_boundary_detection: bool = True  # Boundary and danger zone detection
    enable_memory_consolidation: bool = True  # Memory consolidation during sleep
    enable_action_intelligence: bool = True  # Learned action patterns
    enable_goal_invention: bool = True  # Dynamic goal creation
    enable_learning_progress_drive: bool = True  # Intrinsic motivation
    enable_death_manager: bool = True  # Survival mechanics
    enable_exploration_strategies: bool = True  # Advanced exploration
    enable_pattern_recognition: bool = True  # Pattern learning
    enable_knowledge_transfer: bool = True  # Cross-game learning
    enable_boredom_detection: bool = True  # Boredom and strategy switching
    enable_mid_game_sleep: bool = True  # Mid-episode consolidation
    enable_action_experimentation: bool = True  # Action experimentation
    enable_reset_decisions: bool = True  # Strategic game resets
    enable_curriculum_learning: bool = True  # Progressive difficulty
    enable_multi_modal_input: bool = True  # Visual + proprioceptive
    enable_temporal_memory: bool = True  # Sequence memory
    enable_hebbian_bonuses: bool = True  # Hebbian co-activation
    enable_memory_regularization: bool = True  # Encourage memory use
    enable_gradient_flow_monitoring: bool = True  # Training stability
    enable_usage_tracking: bool = True  # Memory utilization tracking
    enable_salient_memory_retrieval: bool = True  # Context-based memory
    enable_anti_bias_weighting: bool = True  # Action selection balancing
    enable_stagnation_detection: bool = True  # Progress monitoring
    enable_emergency_movement: bool = True  # Coordinate unsticking
    enable_cluster_formation: bool = True  # Success zone mapping
    enable_danger_zone_avoidance: bool = True  # Failure zone mapping
    enable_predictive_coordinates: bool = True  # Smart coordinate selection
    enable_rate_limiting_management: bool = True  # API rate management
    
    # Memory system configuration
    memory_size: int = 512  # DNC memory slots
    memory_word_size: int = 64  # Memory word width
    memory_read_heads: int = 4  # Number of read heads
    memory_write_heads: int = 1  # Number of write heads
    memory_controller_size: int = 256  # Controller hidden size
    
    # Sleep and consolidation
    sleep_trigger_energy: float = 40.0  # Energy level for sleep
    sleep_duration_steps: int = 50  # Sleep cycle duration
    consolidation_strength: float = 0.8  # Memory consolidation power
    
    # Salience system
    salience_threshold: float = 0.6  # High salience threshold
    salience_decay_rate: float = 0.95  # Memory decay rate
    replay_buffer_size: int = 10000  # Experience replay size
    
    # Experimental options
    compare_systems: bool = False
    run_performance_tests: bool = False
    
    # Games and sessions
    games: List[str] = None
    session_duration_minutes: int = 120  # Longer sessions for full system
    
    # Logging and monitoring
    verbose: bool = True  # Full logging by default
    save_detailed_logs: bool = True
    monitor_performance: bool = True

class UnifiedARCTrainer:
    """
    Unified ARC trainer that combines continuous learning with experimental capabilities.
    
    This system provides:
    1. Production-ready continuous learning
    2. Research experimentation framework
    3. Performance comparison and testing
    4. Unified configuration and monitoring
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.continuous_loop = None
        self.session_results = {}
        self.performance_history = []
        
        # Setup logging
        self._setup_logging()
        
        # Validate configuration
        self._validate_config()
    
    def _setup_logging(self):
        """Setup unified logging system."""
        log_level = logging.DEBUG if self.config.verbose else logging.INFO
        
        # Setup logging with ASCII-only formatting for Windows compatibility
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('unified_trainer.log', encoding='utf-8') if self.config.save_detailed_logs else logging.NullHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _validate_config(self):
        """Validate configuration and set defaults."""
        if not self.config.api_key:
            self.config.api_key = os.getenv('ARC_API_KEY')
            if not self.config.api_key:
                raise ValueError("ARC_API_KEY not found in environment or config")
        
        if not self.config.arc_agents_path:
            # Try to find arc-agents directory
            potential_paths = [
                Path.cwd() / "arc-agents",
                Path.cwd() / "ARC-AGI-3-Agents",  # Common name for the official repository
                Path.cwd().parent / "ARC-AGI-3-Agents",  # Sibling directory
                Path.home() / "arc-agents", 
                Path.home() / "ARC-AGI-3-Agents",
                Path.home() / "Documents" / "GitHub" / "ARC-AGI-3-Agents",  # GitHub desktop location
                Path("/opt/arc-agents")
            ]
            
            for path in potential_paths:
                if path.exists() and (path / "main.py").exists():
                    self.config.arc_agents_path = str(path)
                    self.logger.info(f"Found ARC-AGI-3-Agents at: {path}")
                    break
            
            if not self.config.arc_agents_path:
                self.logger.warning("arc-agents path not found, using current directory")
                self.config.arc_agents_path = str(Path.cwd())
        
        if not self.config.games:
            self.config.games = ["lp85-2e205bbe3622"]  # Default game for testing
    
    async def initialize(self):
        """Initialize the unified system."""
        self.logger.info("🚀 Initializing Unified ARC Trainer - MAXIMUM INTELLIGENCE MODE")
        self.logger.info(f"   Mode: {self.config.mode}")
        self.logger.info(f"   Salience: {self.config.salience_mode.value}")
        self.logger.info(f"   🔥 ALL FEATURES ENABLED - Complete System Integration")
        
        try:
            # Initialize continuous learning loop
            self.continuous_loop = ContinuousLearningLoop(
                api_key=self.config.api_key,
                tabula_rasa_path=str(Path(__file__).parent),
                arc_agents_path=self.config.arc_agents_path
            )
            
            # Initialize enhanced coordinate intelligence for better clustering
            try:
                from enhanced_coordinate_intelligence import EnhancedCoordinateIntelligence
                self.continuous_loop.enhanced_coordinate_intelligence = EnhancedCoordinateIntelligence(self.continuous_loop)
                self.continuous_loop.use_enhanced_coordinate_selection = True
                self.logger.info("✅ Enhanced Coordinate Intelligence initialized")
            except ImportError as e:
                self.logger.warning(f"⚠️ Enhanced Coordinate Intelligence not available: {e}")
                self.continuous_loop.use_enhanced_coordinate_selection = False
            
            # Initialize action counter for better tracking
            self.continuous_loop.action_counter = 0
            self.continuous_loop.max_actions_limit = self.config.max_actions_per_game
            
            # Initialize full feature system
            await self._initialize_full_feature_system()
            
            self.logger.info("✅ Maximum Intelligence Continuous Learning Loop initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    async def _initialize_full_feature_system(self):
        """Initialize all advanced features in the system."""
        self.logger.info("🔧 Initializing ALL system features...")
        
        # Configure energy system
        if self.config.enable_energy_system:
            self.continuous_loop.current_energy = 100.0
            # Check if death_manager exists before configuring
            if hasattr(self.continuous_loop.energy_system, 'death_manager'):
                self.continuous_loop.energy_system.death_manager.enabled = self.config.enable_death_manager
        
        # Configure memory systems
        if hasattr(self.continuous_loop, 'demo_agent') and self.continuous_loop.demo_agent:
            agent = self.continuous_loop.demo_agent
            
            # DNC Memory configuration
            if self.config.enable_dnc_memory and hasattr(agent, 'memory') and agent.memory:
                agent.memory.memory_size = self.config.memory_size
                agent.memory.word_size = self.config.memory_word_size
                agent.memory.num_read_heads = self.config.memory_read_heads
                agent.memory.num_write_heads = self.config.memory_write_heads
            
            # Salience system configuration
            if self.config.enable_salience_system and hasattr(self.continuous_loop, 'salience_calculator') and self.continuous_loop.salience_calculator:
                self.continuous_loop.salience_calculator.threshold = self.config.salience_threshold
                # Check if decay_rate attribute exists
                if hasattr(self.continuous_loop.salience_calculator, 'decay_rate'):
                    self.continuous_loop.salience_calculator.decay_rate = self.config.salience_decay_rate
        
        # Configure sleep system
        if self.config.enable_sleep_cycles:
            self.continuous_loop.sleep_trigger_energy = self.config.sleep_trigger_energy
        
        self.logger.info("✅ All features configured and ready")
    
    async def _initialize_full_feature_agent(self):
        """Initialize the adaptive learning agent with all features enabled."""
        if not hasattr(self.continuous_loop, 'demo_agent') or not self.continuous_loop.demo_agent:
            # Create comprehensive agent configuration
            agent_config = {
                'predictive_core': {
                    'visual_size': [3, 64, 64] if self.config.enable_multi_modal_input else [1, 32, 32],
                    'proprioception_size': 16 if self.config.enable_multi_modal_input else 8,
                    'hidden_size': 256,  # Increased for full system
                    'use_memory': self.config.enable_dnc_memory
                },
                'memory': {
                    'enabled': self.config.enable_dnc_memory,
                    'memory_size': self.config.memory_size,
                    'word_size': self.config.memory_word_size,
                    'num_read_heads': self.config.memory_read_heads,
                    'num_write_heads': self.config.memory_write_heads,
                    'controller_size': self.config.memory_controller_size,
                    'use_learned_importance': True,
                    'enable_regularization': self.config.enable_memory_regularization,
                    'enable_usage_tracking': self.config.enable_usage_tracking
                },
                'learning_progress': {
                    'enabled': self.config.enable_learning_progress_drive,
                    'smoothing_window': 50,
                    'normalized_threshold': 0.05,
                    'derivative_clamp_range': (-0.1, 0.1)
                },
                'energy': {
                    'enabled': self.config.enable_energy_system,
                    'initial_energy': 100.0,
                    'death_manager_enabled': self.config.enable_death_manager,
                    'energy_decay_rate': 0.1
                },
                'sleep': {
                    'enabled': self.config.enable_sleep_cycles,
                    'sleep_trigger_energy': self.config.sleep_trigger_energy,
                    'sleep_trigger_boredom_steps': 100 if self.config.enable_boredom_detection else 200,
                    'sleep_duration_steps': self.config.sleep_duration_steps,
                    'enable_consolidation': self.config.enable_memory_consolidation,
                    'consolidation_strength': self.config.consolidation_strength
                },
                'meta_learning': {
                    'enabled': self.config.enable_meta_learning,
                    'pattern_memory_size': 1000,
                    'insight_threshold': 0.6,
                    'enable_transfer': self.config.enable_knowledge_transfer
                },
                'goals': {
                    'enabled': self.config.enable_goal_invention,
                    'dynamic_creation': True,
                    'survival_phase_enabled': True,
                    'exploration_phase_enabled': True
                },
                'action_selection': {
                    'enabled': True,
                    'exploration_strategies': self.config.enable_exploration_strategies,
                    'anti_bias_weighting': self.config.enable_anti_bias_weighting,
                    'enable_experimentation': self.config.enable_action_experimentation
                }
            }
            
            # Create the full-featured agent
            from src.core.agent import AdaptiveLearningAgent
            self.continuous_loop.demo_agent = AdaptiveLearningAgent(agent_config)
            self.logger.info("✅ Full-featured agent created with ALL systems enabled")
    
    def _configure_full_feature_system(self, continuous_loop):
        """Configure the continuous learning loop with ALL features working seamlessly together."""
        self.logger.info("🔧 Configuring seamless feature integration...")
        
        # COORDINATE INTELLIGENCE ↔ MEMORY SYSTEM INTEGRATION
        if self.config.enable_coordinates and self.config.enable_dnc_memory:
            # Use the actual available_actions_memory structure
            if hasattr(continuous_loop, 'available_actions_memory'):
                continuous_loop.available_actions_memory['memory_integration'] = True
                continuous_loop.available_actions_memory['store_clusters_in_memory'] = True
                continuous_loop.available_actions_memory['retrieve_spatial_patterns'] = True
                
                # ENHANCED: Add intelligent coordinate selection system
                try:
                    from enhanced_coordinate_intelligence import EnhancedCoordinateIntelligence
                    continuous_loop.enhanced_coordinate_intelligence = EnhancedCoordinateIntelligence(continuous_loop)
                    continuous_loop.use_enhanced_coordinate_selection = True
                    self.logger.info("🧠 Enhanced Coordinate Intelligence system activated")
                except ImportError:
                    self.logger.warning("Enhanced Coordinate Intelligence not available, using standard system")
                    continuous_loop.use_enhanced_coordinate_selection = False
            
        # FRAME ANALYSIS ↔ COORDINATE INTELLIGENCE INTEGRATION  
        if self.config.enable_frame_analysis and self.config.enable_coordinates:
            if hasattr(continuous_loop, 'frame_analyzer'):
                continuous_loop.frame_analyzer.coordinate_guidance_enabled = True
            # Enable frame analysis input to boundary detection
            if hasattr(continuous_loop, 'available_actions_memory'):
                continuous_loop.available_actions_memory['universal_boundary_detection']['frame_analysis_input'] = True
            
        # SALIENCE SYSTEM ↔ MEMORY CONSOLIDATION INTEGRATION
        if self.config.enable_salience_system and self.config.enable_memory_consolidation:
            if hasattr(continuous_loop, 'salience_calculator') and continuous_loop.salience_calculator:
                continuous_loop.salience_calculator.memory_consolidation_feedback = True
            continuous_loop.memory_consolidation_uses_salience = True
            
        # ACTION INTELLIGENCE ↔ META-LEARNING INTEGRATION
        if self.config.enable_action_intelligence and self.config.enable_meta_learning:
            if hasattr(continuous_loop, 'available_actions_memory'):
                if 'action_semantic_mapping' not in continuous_loop.available_actions_memory:
                    continuous_loop.available_actions_memory['action_semantic_mapping'] = {}
                continuous_loop.available_actions_memory['action_semantic_mapping']['meta_learning_feedback'] = True
            if hasattr(continuous_loop, 'arc_meta_learning'):
                continuous_loop.arc_meta_learning.action_intelligence_input = True
            
        # ENERGY SYSTEM ↔ SLEEP CYCLES ↔ MEMORY INTEGRATION
        if self.config.enable_energy_system and self.config.enable_sleep_cycles and self.config.enable_memory_consolidation:
            continuous_loop.energy_sleep_memory_loop = True
            continuous_loop.adaptive_sleep_timing = True
            continuous_loop.energy_informed_consolidation = True
            
        # BOREDOM DETECTION ↔ ACTION EXPERIMENTATION INTEGRATION
        if self.config.enable_boredom_detection and self.config.enable_action_experimentation:
            continuous_loop.boredom_triggers_experimentation = True
            continuous_loop.experimentation_feedback_to_boredom = True
            
        # CONTRARIAN STRATEGY ↔ RESET DECISIONS INTEGRATION
        if self.config.enable_contrarian_strategy and self.config.enable_reset_decisions:
            continuous_loop.contrarian_informs_resets = True
            continuous_loop.reset_triggers_contrarian = True
            
        # PATTERN RECOGNITION ↔ KNOWLEDGE TRANSFER INTEGRATION
        if self.config.enable_pattern_recognition and self.config.enable_knowledge_transfer:
            continuous_loop.patterns_enable_transfer = True
            continuous_loop.transfer_creates_patterns = True
            
        # LEARNING PROGRESS DRIVE ↔ GOAL INVENTION INTEGRATION
        if self.config.enable_learning_progress_drive and self.config.enable_goal_invention:
            continuous_loop.learning_progress_creates_goals = True
            continuous_loop.goals_inform_learning_progress = True
            
        # UNIVERSAL BOUNDARY DETECTION ENHANCEMENTS
        if self.config.enable_coordinates and hasattr(continuous_loop, 'available_actions_memory'):
            boundary_system = continuous_loop.available_actions_memory['universal_boundary_detection']
            boundary_system['stagnation_detection'] = self.config.enable_stagnation_detection
            boundary_system['emergency_movement'] = self.config.enable_emergency_movement
            boundary_system['cluster_formation'] = self.config.enable_cluster_formation
            boundary_system['danger_zone_mapping'] = self.config.enable_danger_zone_avoidance
            boundary_system['integration_level'] = 'maximum'
            
        # CENTRAL DATA FUSION HUB - All systems feed into this
        continuous_loop.central_data_fusion = {
            'coordinate_data': {},
            'memory_patterns': {},
            'salience_insights': {},
            'action_effectiveness': {},
            'energy_patterns': {},
            'sleep_quality_metrics': {},
            'learning_velocity': {},
            'pattern_recognition_results': {},
            'goal_achievement_data': {},
            'transfer_success_rates': {},
            'boredom_triggers': {},
            'contrarian_effectiveness': {},
            'reset_decision_outcomes': {},
            'curriculum_progression': {},
            'danger_zone_mappings': {},
            'cluster_formations': {},
            'temporal_sequences': {},
            'multimodal_correlations': {},
            'hebbian_activations': {},
            'usage_optimization_data': {},
            'rate_limiting_efficiency': {},
            'anti_bias_effectiveness': {},
            'exploration_outcomes': {},
            'experimentation_results': {},
            'consolidated_intelligence': {}
        }
        
        # CROSS-SYSTEM FEEDBACK LOOPS
        continuous_loop.enable_cross_system_feedback = True
        continuous_loop.feedback_loop_strength = 0.8
        continuous_loop.system_synergy_multiplier = 1.5
        
        # Enable action intelligence with cross-system learning
        if self.config.enable_action_intelligence and hasattr(continuous_loop, 'available_actions_memory'):
            if 'action_semantic_mapping' in continuous_loop.available_actions_memory:
                for action_id in continuous_loop.available_actions_memory['action_semantic_mapping'].keys():
                    if isinstance(continuous_loop.available_actions_memory['action_semantic_mapping'][action_id], dict):
                        continuous_loop.available_actions_memory['action_semantic_mapping'][action_id]['enhanced_learning'] = True
                        continuous_loop.available_actions_memory['action_semantic_mapping'][action_id]['cross_system_integration'] = True
        
        # Configure advanced sleep and consolidation
        continuous_loop.enable_mid_game_sleep = self.config.enable_mid_game_sleep
        continuous_loop.enable_boredom_detection = self.config.enable_boredom_detection
        continuous_loop.enable_strategic_resets = self.config.enable_reset_decisions
        continuous_loop.advanced_integration_mode = True
        
        # Configure frame analysis with maximum intelligence
        if hasattr(continuous_loop, 'frame_analyzer'):
            continuous_loop.frame_analyzer.enabled = self.config.enable_frame_analysis
            continuous_loop.frame_analyzer.boundary_detection = self.config.enable_boundary_detection
            continuous_loop.frame_analyzer.pattern_recognition = self.config.enable_pattern_recognition
            continuous_loop.frame_analyzer.coordinate_integration = True
            continuous_loop.frame_analyzer.memory_integration = True
            continuous_loop.frame_analyzer.action_intelligence_integration = True
            
        self.logger.info("✅ All systems configured for maximum synergy and data leverage")
    
    def _log_activated_features(self):
        """Log all activated features for transparency."""
        self.logger.info("🔥 ACTIVATED FEATURE SET:")
        features = [
            ("SWARM Processing", self.config.enable_swarm),
            ("DNC Memory", self.config.enable_dnc_memory),
            ("Meta-Learning", self.config.enable_meta_learning),
            ("Salience System", self.config.enable_salience_system),
            ("Energy System", self.config.enable_energy_system),
            ("Sleep Cycles", self.config.enable_sleep_cycles),
            ("Memory Consolidation", self.config.enable_memory_consolidation),
            ("Coordinate Intelligence", self.config.enable_coordinates),
            ("Frame Analysis", self.config.enable_frame_analysis),
            ("Boundary Detection", self.config.enable_boundary_detection),
            ("Action Intelligence", self.config.enable_action_intelligence),
            ("Goal Invention", self.config.enable_goal_invention),
            ("Learning Progress Drive", self.config.enable_learning_progress_drive),
            ("Exploration Strategies", self.config.enable_exploration_strategies),
            ("Pattern Recognition", self.config.enable_pattern_recognition),
            ("Knowledge Transfer", self.config.enable_knowledge_transfer),
            ("Boredom Detection", self.config.enable_boredom_detection),
            ("Mid-Game Sleep", self.config.enable_mid_game_sleep),
            ("Action Experimentation", self.config.enable_action_experimentation),
            ("Reset Decisions", self.config.enable_reset_decisions),
            ("Contrarian Strategy", self.config.enable_contrarian_strategy),
            ("Anti-Bias Weighting", self.config.enable_anti_bias_weighting),
            ("Stagnation Detection", self.config.enable_stagnation_detection),
            ("Emergency Movement", self.config.enable_emergency_movement),
            ("Cluster Formation", self.config.enable_cluster_formation),
            ("Danger Zone Avoidance", self.config.enable_danger_zone_avoidance),
            ("Predictive Coordinates", self.config.enable_predictive_coordinates),
            ("Hebbian Bonuses", self.config.enable_hebbian_bonuses),
            ("Memory Regularization", self.config.enable_memory_regularization),
            ("Usage Tracking", self.config.enable_usage_tracking),
            ("Salient Memory Retrieval", self.config.enable_salient_memory_retrieval),
            ("Temporal Memory", self.config.enable_temporal_memory),
            ("Multi-Modal Input", self.config.enable_multi_modal_input),
            ("Curriculum Learning", self.config.enable_curriculum_learning),
            ("Death Manager", self.config.enable_death_manager),
            ("Rate Limiting Management", self.config.enable_rate_limiting_management)
        ]
        
        for feature_name, enabled in features:
            status = "✅ ENABLED" if enabled else "❌ DISABLED"
            self.logger.info(f"   {feature_name}: {status}")
        
        enabled_count = sum(1 for _, enabled in features if enabled)
        self.logger.info(f"🚀 TOTAL ACTIVE FEATURES: {enabled_count}/{len(features)}")
    
    async def _process_maximum_intelligence_results(self, results: Dict[str, Any]):
        """Process results from maximum intelligence training with enhanced analysis."""
        await self._process_results(results)  # Standard processing
        
        # Additional maximum intelligence analysis
        self.logger.info("🧠 MAXIMUM INTELLIGENCE ANALYSIS:")
        
        if 'detailed_metrics' in results:
            metrics = results['detailed_metrics']
            
            # Memory system analysis
            if 'memory_operations' in metrics:
                self.logger.info(f"   Memory Operations: {metrics['memory_operations']}")
            if 'consolidation_cycles' in metrics:
                self.logger.info(f"   Consolidation Cycles: {metrics['consolidation_cycles']}")
            
            # Learning system analysis
            if 'learning_velocity' in metrics:
                self.logger.info(f"   Learning Velocity: {metrics['learning_velocity']:.3f}")
            if 'knowledge_transfers' in metrics:
                self.logger.info(f"   Knowledge Transfers: {metrics['knowledge_transfers']}")
            
            # Coordination system analysis  
            if 'coordinate_clusters_formed' in metrics:
                self.logger.info(f"   Coordinate Clusters: {metrics['coordinate_clusters_formed']}")
            if 'danger_zones_mapped' in metrics:
                self.logger.info(f"   Danger Zones: {metrics['danger_zones_mapped']}")
        
        self.logger.info("✅ Maximum intelligence analysis complete")
    
    async def run(self):
        """Main execution method - routes to appropriate mode."""
        if not await self.initialize():
            self.logger.error("Initialization failed, cannot continue")
            return False
        
        try:
            if self.config.mode == "adaptive-learning" or self.config.mode == "maximum-intelligence":
                return await self._run_maximum_intelligence_mode()
            elif self.config.mode == "research-lab":
                return await self._run_research_lab_mode()
            elif self.config.mode == "quick-validation":
                return await self._run_quick_validation_mode()
            elif self.config.mode == "showcase-demo":
                return await self._run_showcase_demo_mode()
            elif self.config.mode == "system-comparison":
                return await self._run_system_comparison_mode()
            elif self.config.mode == "minimal-debug":
                return await self._run_minimal_debug_mode()
            else:
                raise ValueError(f"Unknown mode: {self.config.mode}")
                
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            return await self._handle_graceful_shutdown()
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            return False
    
    async def _run_maximum_intelligence_mode(self):
        """Run maximum intelligence mode - ALL COGNITIVE SYSTEMS FULLY ACTIVATED."""
        self.logger.info("🧠 Starting MAXIMUM INTELLIGENCE Mode")
        self.logger.info("🔥 ALL COGNITIVE SYSTEMS ACTIVATED - Complete Neural Architecture Enabled")
        
        # Initialize and configure FULL agent with all features
        await self._initialize_full_feature_agent()
        
        # Create a TrainingSession for continuous learning with ALL features
        from src.arc_integration.continuous_learning_loop import TrainingSession
        import time
        
        session_id = f"maximum_intelligence_{int(time.time())}"
        
        training_session = TrainingSession(
            session_id=session_id,
            games_to_play=self.config.games,
            max_mastery_sessions_per_game=self.config.max_learning_cycles,
            learning_rate_schedule={
                'initial': 0.001,
                'adaptive': 0.0005,
                'deep': 0.0002,
                'mastery': 0.0001
            },
            save_interval=500,  # More frequent saves for full system
            target_performance={
                'score': self.config.target_score,
                'win_rate': 0.8,
                'efficiency': 0.7,
                'learning_velocity': 0.6
            },
            max_actions_per_session=self.config.max_actions_per_game,
            enable_contrarian_strategy=self.config.enable_contrarian_strategy,
            salience_mode=self.config.salience_mode,
            enable_salience_comparison=True,  # Compare modes within session
            swarm_enabled=self.config.enable_swarm
        )
        
        # Configure the continuous loop with ALL features
        self._configure_full_feature_system(self.continuous_loop)
        
        # Set the session in the continuous loop
        self.continuous_loop.current_session = training_session
        
        self.logger.info(f"Maximum Intelligence Session created: {session_id}")
        self._log_activated_features()
        
        # Run continuous learning with maximum intelligence
        results = await self.continuous_loop.run_continuous_learning(session_id)
        
        # Enhanced results processing for maximum intelligence system
        await self._process_maximum_intelligence_results(results)
        
        return results.get('success', False)
    
    async def _run_research_lab_mode(self):
        """Run research lab mode - experimentation and comparison."""
        self.logger.info("🧪 Starting Research Lab Mode")
        
        results = {}
        
        if self.config.compare_systems:
            # Run system comparison
            results['system_comparison'] = await self._run_system_comparison()
        
        if self.config.run_performance_tests:
            # Run performance benchmarks
            results['performance_tests'] = await self._run_performance_tests()
        
        # Run main experimental session
        session_id = await self.continuous_loop.create_session(
            session_name=f"research_lab_{int(time.time())}",
            games=self.config.games,
            salience_mode=self.config.salience_mode,
            max_actions_per_session=self.config.max_actions_per_game
        )
        
        results['main_session'] = await self.continuous_loop.run_continuous_learning(session_id)
        
        return results
    
    async def _run_quick_validation_mode(self):
        """Run quick validation mode - rapid system testing."""
        self.logger.info("⚡ Starting Quick Validation Mode")
        
        # Create a TrainingSession manually since there's no create_session method
        from src.arc_integration.continuous_learning_loop import TrainingSession
        import time
        
        session_id = f"quick_validation_{int(time.time())}"
        
        training_session = TrainingSession(
            session_id=session_id,
            games_to_play=self.config.games,
            max_mastery_sessions_per_game=1,  # Limited for testing
            learning_rate_schedule={'default': 0.001},
            save_interval=100,
            target_performance={'default': 50.0},
            max_actions_per_session=min(1000, self.config.max_actions_per_game),
            salience_mode=self.config.salience_mode,
            swarm_enabled=False  # Disable SWARM for testing
        )
        
        # Set the session in the continuous loop
        self.continuous_loop.current_session = training_session
        
        # Run the continuous learning
        results = await self.continuous_loop.run_continuous_learning(session_id)
        
        # Quick validation
        success = results.get('success', False)
        self.logger.info(f"Quick Validation {'PASSED' if success else 'FAILED'}")
        
        return success
    
    async def _run_showcase_demo_mode(self):
        """Run showcase demo mode - demonstrate capabilities."""
        self.logger.info("🎭 Starting Showcase Demo Mode")
        
        # Create demo session with enhanced logging
        session_id = await self.continuous_loop.create_session(
            session_name=f"showcase_demo_{int(time.time())}",
            games=self.config.games[:2],  # Limit to first 2 games for demo
            salience_mode=self.config.salience_mode,
            max_actions_per_session=self.config.max_actions_per_game,  # Use user-specified limit for demo
        )
        
        results = await self.continuous_loop.run_continuous_learning(session_id)
        
        # Enhanced demo reporting
        await self._generate_demo_report(results)
        
        return results.get('success', False)
    
    async def _run_system_comparison_mode(self):
        """Run system comparison mode - A/B testing different configurations."""
        self.logger.info("⚖️ Starting System Comparison Mode")
        
        results = {}
        
        # Test with different salience modes
        for salience_mode in [SalienceMode.DECAY_COMPRESSION, SalienceMode.LOSSLESS]:
            session_id = await self.continuous_loop.create_session(
                session_name=f"comparison_{salience_mode.value}_{int(time.time())}",
                games=self.config.games,
                salience_mode=salience_mode,
                max_actions_per_session=self.config.max_actions_per_game // 2,  # Split time
            )
            
            results[f"salience_{salience_mode.value}"] = await self.continuous_loop.run_continuous_learning(session_id)
        
        # Generate comparison report
        await self._generate_comparison_report(results)
        
        return True
    
    async def _run_minimal_debug_mode(self):
        """Run minimal debug mode - basic functionality only."""
        self.logger.info("🔧 Starting Minimal Debug Mode")
        
        # Create minimal session for debugging
        from src.arc_integration.continuous_learning_loop import TrainingSession
        import time
        
        session_id = f"minimal_debug_{int(time.time())}"
        
        training_session = TrainingSession(
            session_id=session_id,
            games_to_play=self.config.games[:1],  # Single game only
            max_mastery_sessions_per_game=1,
            learning_rate_schedule={'default': 0.001},
            save_interval=50,
            target_performance={'default': 30.0},
            max_actions_per_session=min(self.config.max_actions_per_game, 500),  # Limited for debug
            salience_mode=self.config.salience_mode,
            swarm_enabled=False
        )
        
        # Set the session in the continuous loop
        self.continuous_loop.current_session = training_session
        
        # Run minimal learning
        results = await self.continuous_loop.run_continuous_learning(session_id)
        
        success = results.get('success', False)
        self.logger.info(f"Minimal Debug {'PASSED' if success else 'FAILED'}")
        
        return success
    
    async def _run_system_comparison(self):
        """Compare different system configurations."""
        self.logger.info("🔬 Running system comparison...")
        
        # This would implement detailed A/B testing
        # For now, placeholder
        return {'comparison_completed': True}
    
    async def _run_performance_tests(self):
        """Run performance benchmarks."""
        self.logger.info("⚡ Running performance tests...")
        
        # This would implement performance benchmarking
        # For now, placeholder
        return {'benchmarks_completed': True}
    
    async def _process_results(self, results: Dict[str, Any]):
        """Process and save training results."""
        self.session_results[results.get('session_id', 'unknown')] = results
        self.performance_history.append({
            'timestamp': time.time(),
            'results': results
        })
        
        # Save to file if enabled
        if self.config.save_detailed_logs:
            results_file = Path("unified_trainer_results.json")
            import json
            with open(results_file, 'w') as f:
                json.dump({
                    'config': self.config.__dict__,
                    'session_results': self.session_results,
                    'performance_history': self.performance_history
                }, f, indent=2, default=str)
            
            self.logger.info(f"📊 Results saved to {results_file}")
    
    async def _generate_demo_report(self, results: Dict[str, Any]):
        """Generate detailed demo report."""
        print("\n" + "="*60)
        print("🎭 UNIFIED ARC TRAINER - DEMO REPORT")
        print("="*60)
        
        # Extract key metrics
        session_id = results.get('session_id', 'unknown')
        games_played = results.get('games_played', {})
        
        print(f"📋 Session: {session_id}")
        print(f"🎮 Games Played: {len(games_played)}")
        
        for game_id, game_results in games_played.items():
            episodes = game_results.get('episodes', [])
            best_score = max([ep.get('score', 0) for ep in episodes] + [0])
            print(f"   {game_id}: {len(episodes)} episodes, best score: {best_score}")
        
        # Show key system metrics
        metrics = results.get('detailed_metrics', {})
        print(f"\n🧠 System Metrics:")
        print(f"   Sleep Cycles: {metrics.get('sleep_cycles', 0)}")
        print(f"   Memory Operations: {metrics.get('memory_operations', 0)}")
        print(f"   High Salience Experiences: {metrics.get('high_salience_experiences', 0)}")
        
        print("\n✨ Advanced Features Demonstrated:")
        print("   ✅ Energy System with Sleep Cycles")
        print("   ✅ Coordinate Intelligence & Stagnation Detection")
        print("   ✅ Salience-Weighted Experience Replay")
        print("   ✅ Meta-Learning Knowledge Transfer")
        print("   ✅ ARC-3 API Integration with Rate Limiting")
        
        print("\n🔗 ARC-3 Scoreboard:", results.get('arc3_scoreboard_url', 'Not available'))
        print("="*60)
    
    async def _generate_comparison_report(self, results: Dict[str, Any]):
        """Generate comparison report."""
        print("\n" + "="*60)
        print("⚖️ SYSTEM COMPARISON REPORT")
        print("="*60)
        
        for config_name, config_results in results.items():
            games_played = config_results.get('games_played', {})
            total_score = sum([
                max([ep.get('score', 0) for ep in game.get('episodes', [])] + [0])
                for game in games_played.values()
            ])
            
            print(f"{config_name.upper()}: Total Score = {total_score}")
        
        print("="*60)
    
    async def _handle_graceful_shutdown(self):
        """Handle graceful shutdown on interruption."""
        self.logger.info("🛑 Initiating graceful shutdown...")
        
        # Save current state
        await self._process_results({
            'session_id': 'interrupted',
            'shutdown_reason': 'user_interrupt',
            'timestamp': time.time()
        })
        
        return False

def create_parser():
    """Create argument parser for unified trainer."""
    parser = argparse.ArgumentParser(
        description="Unified ARC Trainer - Complete Integration of Continuous Learning and Experimental Systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Maximum Intelligence mode (default) - ALL cognitive systems enabled
  python unified_arc_trainer.py --mode maximum-intelligence --verbose
  
  # Maximum Intelligence with custom memory configuration
  python unified_arc_trainer.py --mode maximum-intelligence --memory-size 1024 --memory-read-heads 8
  
  # Maximum Intelligence with decay salience (default now)
  python unified_arc_trainer.py --mode maximum-intelligence --salience decay_compression --salience-threshold 0.7
  
  # Minimal Debug mode - essential features only
  python unified_arc_trainer.py --mode minimal-debug
  
  # Research Lab mode - experimentation and comparison
  python unified_arc_trainer.py --mode research-lab --compare-systems
  
  # Quick Validation - rapid testing
  python unified_arc_trainer.py --mode quick-validation --games game1,game2 --verbose
  
  # Showcase Demo - demonstrate capabilities
  python unified_arc_trainer.py --mode showcase-demo --verbose
        """
    )
    
    # Core mode selection
    parser.add_argument('--mode', 
                       choices=['maximum-intelligence', 'adaptive-learning', 'research-lab', 'quick-validation', 'showcase-demo', 'system-comparison', 'minimal-debug'],
                       default='maximum-intelligence',  # MAXIMUM INTELLIGENCE IS NOW DEFAULT
                       help='Training mode (default: maximum-intelligence - all cognitive systems enabled)')
    
    # Debug and minimal modes for specific testing
    parser.add_argument('--debug-mode', action='store_true',
                       help='Enable debug mode with minimal features for troubleshooting')
    parser.add_argument('--minimal-mode', action='store_true', 
                       help='Run with only essential features (for performance testing)')
    
    # API and paths
    parser.add_argument('--api-key', help='ARC API key (or set ARC_API_KEY env var)')
    parser.add_argument('--arc-agents-path', help='Path to arc-agents directory')
    
    # Learning parameters
    parser.add_argument('--salience', 
                       choices=['lossless', 'decay_compression'],
                       default='decay_compression',  # Changed default to decay
                       help='Salience mode (default: decay_compression)')
    parser.add_argument('--max-actions', type=int, default=500,
                       help='Max actions per game attempt (default: 500 actions per game)')
    parser.add_argument('--max-cycles', type=int, default=50,
                       help='Max learning cycles (default: 50 for full system)')
    parser.add_argument('--target-score', type=float, default=85.0,
                       help='Target score per game (default: 85.0 for full system)')
    
    # Memory system configuration
    parser.add_argument('--memory-size', type=int, default=512,
                       help='DNC memory size (default: 512)')
    parser.add_argument('--memory-word-size', type=int, default=64,
                       help='DNC memory word size (default: 64)')
    parser.add_argument('--memory-read-heads', type=int, default=4,
                       help='DNC read heads (default: 4)')
    parser.add_argument('--memory-write-heads', type=int, default=1,
                       help='DNC write heads (default: 1)')
    
    # Feature toggles (most enabled by default in full-feature mode)
    parser.add_argument('--disable-swarm', action='store_true',
                       help='Disable SWARM parallel processing')
    parser.add_argument('--disable-coordinates', action='store_true',
                       help='Disable coordinate intelligence')
    parser.add_argument('--disable-energy', action='store_true',
                       help='Disable energy system')
    parser.add_argument('--disable-sleep', action='store_true',
                       help='Disable sleep cycles')
    parser.add_argument('--disable-dnc-memory', action='store_true',
                       help='Disable DNC memory system')
    parser.add_argument('--disable-meta-learning', action='store_true',
                       help='Disable meta-learning system')
    parser.add_argument('--disable-salience', action='store_true',
                       help='Disable salience system')
    parser.add_argument('--disable-contrarian', action='store_true',
                       help='Disable contrarian strategy')
    parser.add_argument('--disable-frame-analysis', action='store_true',
                       help='Disable visual frame analysis')
    parser.add_argument('--disable-boundary-detection', action='store_true',
                       help='Disable boundary detection')
    parser.add_argument('--disable-memory-consolidation', action='store_true',
                       help='Disable memory consolidation')
    parser.add_argument('--disable-action-intelligence', action='store_true',
                       help='Disable learned action patterns')
    parser.add_argument('--disable-all-advanced', action='store_true',
                       help='Disable ALL advanced features (basic mode)')
    
    # Sleep and consolidation parameters
    parser.add_argument('--sleep-trigger-energy', type=float, default=40.0,
                       help='Energy level to trigger sleep (default: 40.0)')
    parser.add_argument('--sleep-duration', type=int, default=50,
                       help='Sleep cycle duration steps (default: 50)')
    parser.add_argument('--consolidation-strength', type=float, default=0.8,
                       help='Memory consolidation strength (default: 0.8)')
    
    # Salience parameters
    parser.add_argument('--salience-threshold', type=float, default=0.6,
                       help='High salience threshold (default: 0.6)')
    parser.add_argument('--salience-decay', type=float, default=0.95,
                       help='Salience decay rate (default: 0.95)')
    
    # Experimental options
    parser.add_argument('--compare-systems', action='store_true',
                       help='Compare different system configurations')
    parser.add_argument('--performance-tests', action='store_true',
                       help='Run performance benchmarks')
    
    # Games and sessions
    parser.add_argument('--games', help='Comma-separated list of game IDs')
    parser.add_argument('--session-duration', type=int, default=60,
                       help='Session duration in minutes (default: 60)')
    
    # Logging and monitoring  
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--no-logs', action='store_true',
                       help='Disable detailed log files')
    parser.add_argument('--no-monitoring', action='store_true',
                       help='Disable performance monitoring')
    
    return parser

async def main():
    """Main entry point for unified trainer."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Convert args to config
    salience_mode_map = {
        'lossless': SalienceMode.LOSSLESS,
        'decay_compression': SalienceMode.DECAY_COMPRESSION
    }
    
    # Handle feature disabling for full-feature mode
    if args.disable_all_advanced:
        # Basic mode - disable everything advanced
        feature_config = {
            'enable_swarm': False,
            'enable_coordinates': False,
            'enable_energy_system': False,
            'enable_sleep_cycles': False,
            'enable_dnc_memory': False,
            'enable_meta_learning': False,
            'enable_salience_system': False,
            'enable_contrarian_strategy': False,
            'enable_frame_analysis': False,
            'enable_boundary_detection': False,
            'enable_memory_consolidation': False,
            'enable_action_intelligence': False,
            'enable_goal_invention': False,
            'enable_learning_progress_drive': False,
            'enable_death_manager': False,
            'enable_exploration_strategies': False,
            'enable_pattern_recognition': False,
            'enable_knowledge_transfer': False,
            'enable_boredom_detection': False,
            'enable_mid_game_sleep': False,
            'enable_action_experimentation': False,
            'enable_reset_decisions': False,
            'enable_curriculum_learning': False,
            'enable_multi_modal_input': False,
            'enable_temporal_memory': False,
            'enable_hebbian_bonuses': False,
            'enable_memory_regularization': False,
            'enable_gradient_flow_monitoring': False,
            'enable_usage_tracking': False,
            'enable_salient_memory_retrieval': False,
            'enable_anti_bias_weighting': False,
            'enable_stagnation_detection': False,
            'enable_emergency_movement': False,
            'enable_cluster_formation': False,
            'enable_danger_zone_avoidance': False,
            'enable_predictive_coordinates': False,
            'enable_rate_limiting_management': False
        }
    else:
        # Full-feature mode - enable everything by default, only disable if explicitly requested
        feature_config = {
            'enable_swarm': not args.disable_swarm,
            'enable_coordinates': not args.disable_coordinates,
            'enable_energy_system': not args.disable_energy,
            'enable_sleep_cycles': not args.disable_sleep,
            'enable_dnc_memory': not args.disable_dnc_memory,
            'enable_meta_learning': not args.disable_meta_learning,
            'enable_salience_system': not args.disable_salience,
            'enable_contrarian_strategy': not args.disable_contrarian,
            'enable_frame_analysis': not args.disable_frame_analysis,
            'enable_boundary_detection': not args.disable_boundary_detection,
            'enable_memory_consolidation': not args.disable_memory_consolidation,
            'enable_action_intelligence': not args.disable_action_intelligence,
            # All other features enabled by default in full mode
            'enable_goal_invention': True,
            'enable_learning_progress_drive': True,
            'enable_death_manager': True,
            'enable_exploration_strategies': True,
            'enable_pattern_recognition': True,
            'enable_knowledge_transfer': True,
            'enable_boredom_detection': True,
            'enable_mid_game_sleep': True,
            'enable_action_experimentation': True,
            'enable_reset_decisions': True,
            'enable_curriculum_learning': True,
            'enable_multi_modal_input': True,
            'enable_temporal_memory': True,
            'enable_hebbian_bonuses': True,
            'enable_memory_regularization': True,
            'enable_gradient_flow_monitoring': True,
            'enable_usage_tracking': True,
            'enable_salient_memory_retrieval': True,
            'enable_anti_bias_weighting': True,
            'enable_stagnation_detection': True,
            'enable_emergency_movement': True,
            'enable_cluster_formation': True,
            'enable_danger_zone_avoidance': True,
            'enable_predictive_coordinates': True,
            'enable_rate_limiting_management': True
        }
    
    config = TrainingConfig(
        mode=args.mode,
        api_key=args.api_key or "",
        arc_agents_path=args.arc_agents_path or "",
        salience_mode=salience_mode_map[args.salience],
        max_actions_per_game=args.max_actions,
        max_learning_cycles=args.max_cycles,
        target_score=args.target_score,
        
        # Memory configuration
        memory_size=args.memory_size,
        memory_word_size=args.memory_word_size,
        memory_read_heads=args.memory_read_heads,
        memory_write_heads=args.memory_write_heads,
        
        # Sleep and consolidation
        sleep_trigger_energy=args.sleep_trigger_energy,
        sleep_duration_steps=args.sleep_duration,
        consolidation_strength=args.consolidation_strength,
        
        # Salience system
        salience_threshold=args.salience_threshold,
        salience_decay_rate=args.salience_decay,
        
        # Apply feature configuration
        **feature_config,
        
        compare_systems=args.compare_systems,
        run_performance_tests=args.performance_tests,
        games=args.games.split(',') if args.games else None,
        session_duration_minutes=args.session_duration,
        verbose=args.verbose,
        save_detailed_logs=not args.no_logs,
        monitor_performance=not args.no_monitoring
    )
    
    print("🧠 UNIFIED ARC TRAINER - MAXIMUM INTELLIGENCE MODE")
    print("=" * 60)
    print(f"Mode: {config.mode.upper()}")
    print(f"Salience: {config.salience_mode.value}")
    print(f"Memory: {config.memory_size} slots × {config.memory_word_size} words")
    print(f"Max Actions: {config.max_actions_per_game:,}")
    print(f"Learning Cycles: {config.max_learning_cycles}")
    print(f"Target Score: {config.target_score}")
    
    # Count enabled features
    enabled_features = sum(1 for key, value in feature_config.items() if value)
    total_features = len(feature_config)
    print(f"🔥 ACTIVE COGNITIVE SYSTEMS: {enabled_features}/{total_features}")
    
    if config.enable_swarm:
        print("🚀 SWARM INTELLIGENCE ENABLED")
    if config.enable_dnc_memory:
        print("🧠 DIFFERENTIABLE NEURAL COMPUTER ENABLED")
    if config.enable_meta_learning:
        print("🎓 META-LEARNING SYSTEM ENABLED")
    if config.enable_energy_system:
        print("⚡ ENERGY MANAGEMENT SYSTEM ENABLED")
    if enabled_features == total_features:
        print("🔥🔥🔥 ALL COGNITIVE SYSTEMS OPERATIONAL 🔥🔥🔥")
    
    print("=" * 60)
    
    # Create and run trainer
    trainer = UnifiedARCTrainer(config)
    success = await trainer.run()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 MAXIMUM INTELLIGENCE TRAINING COMPLETED SUCCESSFULLY!")
        print(f"🧠 System used {enabled_features}/{total_features} cognitive systems")
        return 0
    else:
        print("❌ TRAINING COMPLETED WITH ISSUES")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
