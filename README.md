# Adaptive Learning Agent

An AI system that develops intelligence through intrinsic motivation, curiosity, and survival pressure rather than external programming. Features advanced meta-learning capabilities and automatic object encoding during sleep phases.

## Overview

This project implements a "digital childhood" paradigm where an agent learns through:
- **Predictive Core**: Recurrent world-model that predicts sensory input
- **Learning Progress Drive**: Intrinsic motivation from prediction improvement
- **Meta-Learning System**: Learns from its own learning process for continuous improvement
- **Enhanced Sleep System**: Automatic object encoding and memory consolidation during offline periods
- **Embedded Memory**: Differentiable Neural Computer for integrated memory
- **Energy & Death**: Survival pressure through limited resources
- **Goal Invention**: Self-generated objectives from high-learning-progress experiences

## Project Structure

```
adaptive-learning-agent/
├── src/
│   ├── core/              # Core agent components (includes meta-learning)
│   ├── environment/       # Simulation environments
│   ├── memory/           # Memory systems (DNC)
│   ├── goals/            # Goal invention and management
│   ├── monitoring/       # Debugging and introspection
│   └── utils/            # Utilities and helpers
├── tests/                # Organized test suite
│   ├── unit/             # Unit tests for individual components
│   ├── integration/      # Integration tests
│   └── system/           # System-level tests
├── agi_puzzles/          # AGI evaluation puzzles
├── configs/              # Configuration files
├── experiments/          # Experiment scripts and results
├── docs/                 # Centralized documentation
└── research_results/     # Research evaluation results
```

## Development Phases

- **Phase 0** (Weeks 1-4): Component isolation and validation 
- **Phase 1** (Weeks 5-12): Integrated system development 
- **Phase 1+** (Current): Meta-learning integration and enhanced sleep system **CURRENT**
- **Phase 2** (Weeks 13-24): Complexity scaling and advanced features
- **Phase 3** (Month 7+): Multi-agent and emergent behaviors

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Test basic functionality
python tests/unit/test_basic_functionality.py

# Test meta-learning system
python tests/unit/test_meta_learning_simple.py

# Test enhanced sleep system
python tests/integration/test_enhanced_sleep_system.py

# Run Phase 1 training
python src/main_training.py

# Run with custom configuration
python src/main_training.py --config configs/phase1_config.yaml

# Test agent on AGI puzzles
python tests/integration/test_agent_on_puzzles.py
```

## Key Features

### Core Systems 
- **Robust Learning Progress Drive** with stability validation
- **Differentiable Neural Computer** for embedded memory
- **Bootstrap protection** for newborn agents
- **Comprehensive monitoring** and introspection tools
- **Phased development** with validation at each stage
- **Phase 1 Survival Environment** with adaptive complexity
- **Integrated Goal System** with survival objectives

### Advanced Features 
- **Meta-Learning System**: Learns from learning experiences, extracts insights, and applies knowledge across contexts
- **Enhanced Sleep System**: Automatic object encoding and memory consolidation during offline periods
- **Object Recognition**: Visual pattern clustering and encoding during sleep phases
- **Context-Aware Memory**: Meta-learning informed memory consolidation strategies
- **AGI Puzzle Integration**: Comprehensive evaluation on cognitive benchmarks

### Testing & Organization 
- **Organized Test Suite**: Unit, integration, and system tests
- **AGI Puzzle Validation**: Performance testing on cognitive challenges
- **Centralized Documentation**: All docs organized in `/docs` folder

## Recent Achievements

### Meta-Learning Integration
- **Episodic Memory**: Records complete learning episodes with contextual information
- **Learning Insights**: Extracts patterns from successful learning experiences
- **Experience Consolidation**: Processes experiences into generalizable knowledge
- **Context-Aware Retrieval**: Applies relevant past insights to current situations

### Enhanced Sleep System
- **Automatic Object Encoding**: Learns visual object representations during sleep
- **Memory Consolidation**: Strengthens important memories, prunes irrelevant ones
- **Meta-Learning Integration**: Uses insights to guide consolidation strategies
- **Dream Generation**: Creates synthetic experiences for additional learning

### AGI Puzzle Performance
- **Hidden Cause (Baby Physics)**: Tests causal reasoning
- **Object Permanence**: Validates object tracking capabilities
- **Cooperation & Deception**: Multi-agent interaction scenarios
- **Tool Use**: Problem-solving with environmental objects
- **Deferred Gratification**: Long-term planning and impulse control

## Research Goals

This system validates the hypothesis that intelligence emerges from the right environmental conditions and internal drives, not from explicit programming. The meta-learning capabilities demonstrate how agents can develop increasingly sophisticated cognitive strategies through self-reflection and experience consolidation.

## Architecture Highlights

### Meta-Learning Loop
1. **Experience Recording**: All agent interactions recorded with context
2. **Pattern Recognition**: Identify successful learning strategies
3. **Insight Extraction**: Generalize patterns into reusable knowledge
4. **Application**: Apply insights to improve future learning

### Sleep-Wake Cycle
1. **Active Learning**: Normal agent operation with experience collection
2. **Sleep Triggers**: Low energy, high boredom, or memory pressure
3. **Object Encoding**: Visual pattern analysis and clustering
4. **Memory Consolidation**: Strengthen/prune memories using meta-insights
5. **Dream Generation**: Synthetic experience creation
6. **Wake Up**: Return to active learning with enhanced capabilities

## Documentation

For detailed documentation, see the `/docs` folder:
- **Implementation Status**: Current development progress
- **Research Findings**: Detailed analysis and results
- **AGI Puzzle Summary**: Evaluation results and insights
- **Phase 1 Implementation**: Technical specifications