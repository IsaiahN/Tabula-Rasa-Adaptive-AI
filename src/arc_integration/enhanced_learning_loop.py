"""
Enhanced Continuous Learning Loop with ARC API Integration

This module provides an enhanced version of the ContinuousLearningLoop that integrates
with the real ARC API for training and evaluation.
"""
import asyncio
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Import our new components
from .arc_api_client import ARCClient, ScorecardTracker, ARCScorecard
from .visualization import TrainingVisualizer

logger = logging.getLogger(__name__)

class EnhancedLearningLoop:
    """Enhanced learning loop with ARC API integration and visualization."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the enhanced learning loop.
        
        Args:
            config: Configuration dictionary with training parameters.
        """
        self.config = config or {}
        self.arc_client = None
        self.scorecard_tracker = ScorecardTracker()
        self.visualizer = TrainingVisualizer(
            output_dir=self.config.get("output_dir", "training_visualizations")
        )
        
        # Training state
        self.current_episode = 0
        self.best_score = float('-inf')
        self.start_time = time.time()
        
        # Initialize logger
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Configure logging for the training loop."""
        log_level = self.config.get("log_level", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('training.log')
            ]
        )
        
    async def initialize(self) -> None:
        """Initialize the learning loop and API client."""
        logger.info("Initializing EnhancedLearningLoop...")
        
        # Initialize ARC API client
        self.arc_client = await self._initialize_arc_client()
        
        # Load any existing state
        await self._load_state()
        
        logger.info("EnhancedLearningLoop initialized successfully")
        
    async def _initialize_arc_client(self) -> ARCClient:
        """Initialize and return an authenticated ARC client."""
        api_key = self.config.get("api_key") or os.getenv("ARC_API_KEY")
        if not api_key:
            logger.warning(
                "No API key provided. Some features may be limited. "
                "Set ARC_API_KEY environment variable or provide api_key in config."
            )
            
        return ARCClient(api_key=api_key)
        
    async def _load_state(self) -> None:
        """Load any saved state from disk."""
        state_file = Path(self.config.get("state_file", "training_state.json"))
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    
                self.current_episode = state.get('current_episode', 0)
                self.best_score = state.get('best_score', float('-inf'))
                
                # Load scorecards if available
                if 'scorecards' in state:
                    for sc in state['scorecards']:
                        self.scorecard_tracker.add_scorecard(sc)
                        
                logger.info(f"Loaded state from {state_file}")
                
            except Exception as e:
                logger.error(f"Error loading state: {e}")
                
    async def save_state(self) -> None:
        """Save the current state to disk."""
        if not self.arc_client:
            return
            
        state = {
            'current_episode': self.current_episode,
            'best_score': self.best_score,
            'timestamp': time.time(),
            'scorecards': [sc.to_dict() for sc in self.scorecard_tracker.scorecards]
        }
        
        state_file = Path(self.config.get("state_file", "training_state.json"))
        state_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug(f"Saved state to {state_file}")
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            
    async def train(
        self, 
        task_id: str, 
        num_episodes: int = 100,
        save_interval: int = 10
    ) -> None:
        """Run the training loop.
        
        Args:
            task_id: The ID of the task to train on.
            num_episodes: Number of episodes to run.
            save_interval: Save state and generate reports every N episodes.
        """
        if not self.arc_client:
            logger.error("ARC client not initialized. Call initialize() first.")
            return
            
        logger.info(f"Starting training on task {task_id} for {num_episodes} episodes")
        
        try:
            for episode in range(self.current_episode, self.current_episode + num_episodes):
                self.current_episode = episode
                
                # Run a training episode
                scorecard = await self._run_episode(task_id, episode)
                
                # Update best score
                if scorecard.score > self.best_score:
                    self.best_score = scorecard.score
                    logger.info(f"New best score: {self.best_score:.2f}")
                
                # Save state and generate reports at intervals
                if (episode + 1) % save_interval == 0:
                    await self._generate_reports()
                    await self.save_state()
                    
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
        finally:
            # Ensure we save state and generate final reports
            await self._generate_reports()
            await self.save_state()
            
    async def _run_episode(self, task_id: str, episode: int) -> ARCScorecard:
        """Run a single training episode.
        
        Args:
            task_id: The ID of the task to train on.
            episode: The current episode number.
            
        Returns:
            The scorecard from this episode.
        """
        logger.info(f"Starting episode {episode}")
        start_time = time.time()
        
        try:
            # Use the client in an async context
            async with self.arc_client as client:
                # Get the task details
                task_info = await client.get_task(task_id)
                
                # Here you would implement your agent's logic to generate a solution
                # For now, we'll use a placeholder that simulates a solution
                solution = self._generate_solution(task_info)
                
                # Submit the solution for evaluation
                scorecard = await client.submit_solution(
                    task_id=task_id,
                    solution=solution,
                    metadata={
                        'episode': episode,
                        'timestamp': time.time(),
                        'agent_version': '1.0.0'
                    }
                )
                
                # Track the scorecard
                self.scorecard_tracker.add_scorecard(scorecard)
                
                # Log the results
                duration = time.time() - start_time
                logger.info(
                    f"Episode {episode} completed in {duration:.1f}s | "
                    f"Score: {scorecard.score:.2f} | "
                    f"(Accuracy: {scorecard.accuracy:.2f}, "
                f"Efficiency: {scorecard.efficiency:.2f}, "
                f"Generalization: {scorecard.generalization:.2f})"
            )
            
            return scorecard
            
        except Exception as e:
            logger.error(f"Error in episode {episode}: {e}", exc_info=True)
            # Return a failed scorecard
            return ARCScorecard(
                task_id=task_id,
                score=0,
                accuracy=0,
                efficiency=0,
                generalization=0,
                timestamp=time.time(),
                metadata={
                    'error': str(e),
                    'episode': episode
                }
            )
            
    def _generate_solution(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a solution for the given task.
        
        This is a placeholder implementation. In a real scenario, this would
        use your agent's logic to generate a solution.
        
        Args:
            task_info: Information about the task.
            
        Returns:
            A solution in the format expected by the ARC API.
        """
        # This is a simplified example - adjust based on actual task requirements
        return {
            'algorithm': 'example_algorithm',
            'parameters': {
                'param1': 'value1',
                'param2': 42
            },
            'metadata': {
                'generated_at': time.time(),
                'version': '1.0.0'
            }
        }
        
    async def _generate_reports(self) -> None:
        """Generate training reports and visualizations."""
        if not self.scorecard_tracker.scorecards:
            logger.warning("No scorecards to generate reports from")
            return
            
        try:
            # Generate visualizations
            await asyncio.get_event_loop().run_in_executor(
                None,  # Use default executor
                self.visualizer.plot_scorecard_metrics,
                self.scorecard_tracker,
                f"Training Progress - Episode {self.current_episode}",
                False,  # Don't show the plot
                True    # Save the plot
            )
            
            # Generate HTML report
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.visualizer.generate_html_report,
                self.scorecard_tracker,
                f"Training Report - Episode {self.current_episode}",
                True    # Save the report
            )
            
            logger.info("Generated training reports and visualizations")
            
        except Exception as e:
            logger.error(f"Error generating reports: {e}", exc_info=True)
            
    async def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'arc_client') and self.arc_client:
            if hasattr(self.arc_client, 'session') and self.arc_client.session:
                await self.arc_client.session.close()
                
        logger.info("EnhancedLearningLoop closed")
