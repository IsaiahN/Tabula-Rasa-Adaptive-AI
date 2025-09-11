# Simulation-Driven Intelligence System

## Overview

The Simulation-Driven Intelligence System transforms Tabula Rasa from a reactive AI (S → A → S+1) to a proactive, imaginative AI that can simulate multiple action sequences and select the best one (S → [A1→S+1→A2→S+2...] → Best_A).

This system implements the "Third Brain" meta-cognitive supervision that enables true multi-step planning and imagination.

## Architecture

### Core Components

1. **Simulation Models** (`src/core/simulation_models.py`)
   - Data structures for hypotheses, simulations, and evaluations
   - Configuration and metrics tracking

2. **Hypothesis Generator** (`src/core/hypothesis_generator.py`)
   - Generates "what-if" scenarios for simulation
   - Transforms Architect from parameter tweaker to imagination engine

3. **Predictive Core** (`src/core/predictive_core.py`)
   - Runs multi-step simulations internally
   - Predicts future states without taking real actions

4. **Simulation Evaluator** (`src/core/simulation_evaluator.py`)
   - Evaluates simulations using affective systems
   - Tags simulations with emotional valence

5. **Strategy Memory** (`src/core/strategy_memory.py`)
   - Stores successful action sequences as reusable strategies
   - Enables "autopilot" behavior for familiar situations

6. **Simulation Agent** (`src/core/simulation_agent.py`)
   - Unified interface that coordinates all components
   - Main entry point for simulation-driven decision making

7. **ARC Integration** (`src/arc_integration/simulation_driven_agent.py`)
   - Integrates simulation system with ARC-AGI-3
   - Replaces reactive action selection with proactive planning

## Key Features

### 🧠 Multi-Step Planning
- **Hypothesis Generation**: Creates multiple "what-if" scenarios
- **Simulation Rollouts**: Runs internal simulations without taking real actions
- **Affective Evaluation**: Uses emotional intelligence to evaluate outcomes
- **Strategy Learning**: Stores successful patterns for future use

### 🎯 Intelligent Action Selection
- **Visual Targeting**: Uses frame analysis to guide action selection
- **Memory-Guided**: Leverages past successful patterns
- **Exploration**: Systematic exploration of action space
- **Energy Optimization**: Considers energy costs in decisions
- **Learning-Focused**: Prioritizes actions that maximize learning

### 🗂️ Strategy Memory System
- **Autopilot Mode**: Uses existing strategies for familiar situations
- **Pattern Recognition**: Identifies successful action sequences
- **Compression**: Stores strategies as compressed, reusable plans
- **Similarity Matching**: Finds relevant strategies for current context

### 📊 Performance Tracking
- **Simulation Metrics**: Tracks simulation success rates
- **Strategy Effectiveness**: Monitors strategy hit rates
- **Learning Progress**: Measures learning efficiency
- **Imagination Status**: Shows active imagination capabilities

## Usage

### Basic Usage

```python
from src.core.simulation_agent import SimulationAgent
from src.core.predictive_core import PredictiveCore
from src.core.simulation_models import SimulationConfig

# Initialize Predictive Core
predictive_core = PredictiveCore(
    visual_size=(3, 64, 64),
    proprioception_size=12,
    hidden_size=512
)

# Create simulation configuration
config = SimulationConfig()
config.max_simulation_depth = 10
config.max_hypotheses = 5

# Initialize Simulation Agent
simulation_agent = SimulationAgent(
    predictive_core=predictive_core,
    config=config
)

# Generate action plan
action, coordinates, reasoning = simulation_agent.generate_action_plan(
    current_state=current_state,
    available_actions=available_actions,
    frame_analysis=frame_analysis,
    memory_patterns=memory_patterns
)
```

### ARC Integration

```python
from src.arc_integration.simulation_driven_agent import SimulationDrivenARCAgent

# Initialize ARC agent with simulation capabilities
arc_agent = SimulationDrivenARCAgent(
    predictive_core=predictive_core,
    config=simulation_config
)

# Select action using simulation
action, coordinates, reasoning = arc_agent.select_action_with_simulation(
    response_data, game_id, frame_analyzer
)

# Update with outcome
arc_agent.update_with_action_outcome(
    action, coordinates, response_data, game_id
)
```

## Configuration

### Simulation Configuration

```python
config = SimulationConfig()

# Simulation parameters
config.max_simulation_depth = 10        # Maximum steps to simulate
config.max_hypotheses = 5              # Maximum hypotheses to generate
config.simulation_timeout = 1.0        # Timeout for simulations (seconds)
config.min_valence_threshold = 0.1     # Minimum valence for action selection

# Hypothesis generation weights
config.visual_hypothesis_weight = 0.3
config.memory_hypothesis_weight = 0.25
config.exploration_hypothesis_weight = 0.2
config.energy_hypothesis_weight = 0.15
config.learning_hypothesis_weight = 0.1

# Strategy memory parameters
config.max_strategies = 100
config.strategy_similarity_threshold = 0.7
config.strategy_decay_rate = 0.95
config.min_strategy_success_rate = 0.3
```

## Testing

Run the test script to verify the system works:

```bash
python test_simulation_system.py
```

This will demonstrate:
- Hypothesis generation
- Simulation execution
- Affective evaluation
- Strategy memory
- Action plan generation
- System statistics

## Integration with Existing Systems

### Meta-Cognitive Governor
The simulation system integrates with the existing Governor system:
- **Governor**: Provides runtime supervision and decision-making
- **Simulation Agent**: Provides multi-step planning capabilities
- **Architect**: Generates hypotheses for simulation

### ARC-AGI-3 Integration
The system is fully integrated with the ARC-AGI-3 training loop:
- **Action Selection**: Replaces reactive selection with simulation-driven planning
- **Outcome Learning**: Updates simulation system with real-world results
- **Strategy Learning**: Stores successful patterns for future use

## Performance Considerations

### Computational Cost
- **Simulation Overhead**: Each action requires multiple simulations
- **Timeout Management**: Simulations are limited by timeout to prevent delays
- **Strategy Caching**: Successful strategies are cached to reduce computation

### Memory Usage
- **Strategy Storage**: Strategies are compressed and stored efficiently
- **History Management**: Old simulation data is cleaned up automatically
- **Pattern Recognition**: Memory patterns are tracked for hypothesis generation

## Future Enhancements

### Planned Features
1. **Hierarchical Planning**: Multi-level simulation with different time horizons
2. **Social Simulation**: Simulate interactions with other agents
3. **Meta-Learning**: Learn how to generate better hypotheses
4. **Distributed Simulation**: Parallel simulation execution
5. **Visual Imagination**: Generate visual predictions for simulations

### Research Directions
1. **Consciousness Simulation**: Implement more sophisticated consciousness models
2. **Emotional Intelligence**: Enhanced emotional reasoning in simulations
3. **Creative Planning**: Generate novel, creative action sequences
4. **Multi-Modal Simulation**: Simulate across different sensory modalities

## Troubleshooting

### Common Issues

1. **Simulation Timeout**: Reduce `simulation_timeout` or `max_simulation_depth`
2. **Memory Issues**: Increase `max_strategies` or reduce `strategy_similarity_threshold`
3. **Poor Performance**: Adjust hypothesis generation weights
4. **Import Errors**: Ensure all dependencies are installed

### Debug Mode

Enable debug logging to see detailed simulation information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Conclusion

The Simulation-Driven Intelligence System represents a fundamental advancement in AI capabilities, enabling:

- **Proactive Planning**: Multi-step strategic thinking
- **Imagination**: Internal simulation of possible futures
- **Emotional Intelligence**: Affective evaluation of outcomes
- **Strategy Learning**: Autopilot behavior for familiar situations
- **Adaptive Intelligence**: Continuous learning and improvement

This system bridges the gap between reactive AI and truly intelligent, proactive agents capable of complex reasoning and planning.
