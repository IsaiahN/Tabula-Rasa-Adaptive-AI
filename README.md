# Tabula Rasa: Self-Learning Adaptive Learning Agent

A sophisticated AI system that develops intelligence through intrinsic motivation, curiosity, and survival pressure rather than external programming. Features advanced meta-learning capabilities, automatic object encoding during sleep phases, and **real ARC-AGI-3 integration** for competitive reasoning evaluation.

## Overview

This project implements a "digital childhood" paradigm where an agent learns through:
- **Predictive Core**: Recurrent world-model that predicts sensory input
- **Learning Progress Drive**: Intrinsic motivation from prediction improvement
- **Meta-Learning System**: Learns from its own learning process for continuous improvement
- **Enhanced Sleep System**: Automatic object encoding and memory consolidation during offline periods
- **Embedded Memory**: Differentiable Neural Computer for integrated memory
- **Energy & Death**: Survival pressure through limited resources
- **Goal Invention**: Self-generated objectives from high-learning-progress experiences
- **ARC-3 Integration**: Real competition evaluation on abstract reasoning tasks

## Project Structure

```
tabula-rasa/
├── src/
│   ├── core/              # Core agent components (includes meta-learning)
│   ├── environment/       # Simulation environments
│   ├── memory/           # Memory systems (DNC)
│   ├── goals/            # Goal invention and management
│   ├── arc_integration/  # ARC-AGI-3 competition integration
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
├── examples/             # Demo scripts and comparisons
└── research_results/     # Research evaluation results
```

## Development Phases

- **Phase 0** (Weeks 1-4): Component isolation and validation ✅
- **Phase 1** (Weeks 5-12): Integrated system development ✅
- **Phase 1+** (Current): Meta-learning integration and enhanced sleep system ✅
- **Phase 2** (Weeks 13-24): ARC-3 competition integration and advanced reasoning **CURRENT**
- **Phase 3** (Month 7+): Multi-agent and emergent behaviors

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables (copy .env.template to .env and configure)
cp .env.template .env
# Add your ARC-3 API key and paths to .env
```

## Quick Start

### Basic System Testing
```bash
# Test basic functionality
python tests/unit/test_basic_functionality.py

# Test meta-learning system
python tests/unit/test_meta_learning_simple.py

# Test enhanced sleep system
python tests/integration/test_enhanced_sleep_system.py

# Run Phase 1 training
python src/main_training.py

# Test agent on AGI puzzles
python tests/integration/test_agent_on_puzzles.py
```

### ARC-3 Competition Integration
```bash
# Quick ARC-3 integration test (3 random tasks, ~30 minutes)
python run_continuous_learning.py --mode demo

# Comprehensive training on all 24 ARC tasks (hours to days)
python run_continuous_learning.py --mode full_training

# Compare memory management strategies (scientific analysis)
python run_continuous_learning.py --mode comparison

# Demo salience modes comparison
python examples/salience_modes_demo.py

# Run salience mode comparison framework
python examples/salience_mode_comparison.py
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
- **Dual Salience Mode System**: Choose between lossless memory preservation or intelligent decay/compression
- **Salience-Driven Memory Management**: Experience importance determined by agent's own drives (Learning Progress + Energy/Survival)
- **Adaptive Memory Compression**: Low-salience memories compressed into abstract concepts with merging capabilities
- **Meta-Learning Parameter Optimization**: Automatic adjustment of decay rates based on performance metrics
- **Object Recognition**: Visual pattern clustering and encoding during sleep phases
- **Context-Aware Memory**: Meta-learning informed memory consolidation strategies
- **AGI Puzzle Integration**: Comprehensive evaluation on cognitive benchmarks

### ARC-3 Competition Features ✨
- **Real API Integration**: Direct connection to official ARC-AGI-3 servers
- **Global Task Configuration**: Smart task selection with randomization to prevent overtraining
- **Three Training Modes**:
  - **Demo Mode**: Quick integration test (3 random tasks)
  - **Full Training Mode**: Comprehensive mastery training (all 24 tasks)
  - **Comparison Mode**: Scientific analysis of memory strategies (4 random tasks)
- **Scorecard Generation**: Real competition scorecards from https://three.arcprize.org
- **Performance Tracking**: Win rates, learning efficiency, knowledge transfer metrics
- **Clean Architecture**: Unified system without code duplication

### Testing & Organization 
- **Organized Test Suite**: Unit, integration, and system tests
- **AGI Puzzle Validation**: Performance testing on cognitive challenges
- **Centralized Documentation**: All docs organized in `/docs` folder

## Recent Achievements

### ARC-3 Competition Integration 🏆
- **Official API Integration**: Real-time connection to ARC-AGI-3 servers
- **Adaptive Learning Agent**: Custom agent that uses tabula-rasa's self-learning capabilities
- **Global Task Management**: Prevents overtraining through randomized task selection
- **Performance Analytics**: Comprehensive metrics tracking and learning insights
- **Code Architecture Cleanup**: Eliminated duplication, proper separation of concerns
- **Three Training Modes**: Demo, full training, and scientific comparison modes

### Meta-Learning Integration
- **Episodic Memory**: Records complete learning episodes with contextual information
- **Learning Insights**: Extracts patterns from successful learning experiences
- **Experience Consolidation**: Processes experiences into generalizable knowledge
- **Context-Aware Retrieval**: Applies relevant past insights to current situations

### Enhanced Sleep System with Dual Salience Modes
- **Automatic Object Encoding**: Learns visual object representations during sleep
- **Salience-Based Memory Consolidation**: Strengthens high-salience memories, processes low-salience ones
- **Dual Mode Operation**: 
  - **Lossless Mode**: Preserves all memories without decay (optimal for critical learning phases)
  - **Decay/Compression Mode**: Applies exponential decay and compresses low-salience memories into abstract concepts
- **Intelligent Memory Compression**: Converts detailed experiences into high-level concepts (e.g., "food_found_here", "energy_loss_event")
- **Memory Merging**: Combines multiple low-salience memories into single compressed representations
- **Meta-Learning Integration**: Uses insights to guide consolidation strategies and optimize decay parameters
- **Dream Generation**: Creates synthetic experiences for additional learning

### AGI Puzzle Performance
- **Hidden Cause (Baby Physics)**: Tests causal reasoning
- **Object Permanence**: Validates object tracking capabilities
- **Cooperation & Deception**: Multi-agent interaction scenarios
- **Tool Use**: Problem-solving with environmental objects
- **Deferred Gratification**: Long-term planning and impulse control

## ARC-3 Training Modes

### 🧪 Demo Mode
- **Purpose**: Quick verification that your system works with real ARC-3 servers
- **Tasks**: 3 randomly selected ARC tasks
- **Episodes**: 10 per task (30 total episodes)
- **Duration**: ~15-30 minutes
- **Use When**: Testing integration, verifying scorecard URL generation

### 🔥 Full Training Mode  
- **Purpose**: Comprehensive training until agent masters all ARC tasks
- **Tasks**: All 24 ARC-3 evaluation tasks
- **Episodes**: Up to 50 per task (up to 1,200 total episodes)
- **Target**: 90% win rate, 85+ average score (mastery level)
- **Duration**: Several hours to days depending on performance
- **Use When**: Achieving human-level or superhuman performance

### 🔬 Comparison Mode
- **Purpose**: Scientific comparison of memory management strategies
- **Tasks**: 4 randomly selected tasks
- **Episodes**: 15 per task in each mode (120 total episodes)
- **Analysis**: Tests both LOSSLESS and DECAY_COMPRESSION salience modes
- **Output**: Performance comparison and optimization recommendations
- **Use When**: Optimizing memory system, understanding trade-offs

## Research Goals

This system validates the hypothesis that intelligence emerges from the right environmental conditions and internal drives, not from explicit programming. The meta-learning capabilities demonstrate how agents can develop increasingly sophisticated cognitive strategies through self-reflection and experience consolidation.

**ARC-3 Integration** extends this research by testing the system on one of the most challenging AI benchmarks for abstract reasoning, providing objective measurement of emergent intelligence.

## Architecture Highlights

### Meta-Learning Loop
1. **Experience Recording**: All agent interactions recorded with context
2. **Pattern Recognition**: Identify successful learning strategies
3. **Insight Extraction**: Generalize patterns into reusable knowledge
4. **Application**: Apply insights to improve future learning

### Sleep-Wake Cycle with Salience Processing
1. **Active Learning**: Normal agent operation with experience collection and salience calculation
2. **Sleep Triggers**: Low energy, high boredom, or memory pressure
3. **Memory Decay & Compression**: Apply exponential decay and compress low-salience memories (if enabled)
4. **Salience-Weighted Replay**: Prioritize high-salience experiences for learning
5. **Object Encoding**: Visual pattern analysis and clustering
6. **Memory Consolidation**: Strengthen/prune memories using salience values and meta-insights
7. **Dream Generation**: Synthetic experience creation
8. **Wake Up**: Return to active learning with optimized memory and enhanced capabilities

### ARC-3 Competition Pipeline
1. **Task Selection**: Global configuration with randomization to prevent overtraining
2. **Agent Deployment**: Self-learning tabula-rasa agent connects to ARC-AGI-3 framework
3. **Real API Calls**: Direct communication with official competition servers
4. **Performance Tracking**: Win rates, scores, learning efficiency, knowledge transfer
5. **Scorecard Generation**: Official competition URLs for result verification
6. **Meta-Learning Application**: Insights from previous tasks improve performance on new ones

### Salience Mode Selection
- **Automatic Discovery**: Agent can test both modes and choose optimal strategy
- **Performance-Based Switching**: Meta-learning optimizes mode selection based on task requirements
- **Context-Aware Adaptation**: Different modes for different learning phases or resource constraints

## Configuration

### Environment Variables (.env)
```bash
# ARC-3 API Configuration
ARC_API_KEY=your_arc_api_key_here                    # Get from https://three.arcprize.org
ARC_AGENTS_PATH=/path/to/ARC-AGI-3-Agents           # Optional: path to ARC-AGI-3-Agents repo

# Training Configuration  
TARGET_WIN_RATE=0.90                                 # Target win rate for mastery
TARGET_AVG_SCORE=85.0                               # Target average score
MAX_EPISODES_PER_GAME=50                            # Maximum episodes per task

# ARC-AGI-3-Agents Server Configuration
DEBUG=False
RECORDINGS_DIR=recordings
SCHEME=https
HOST=three.arcprize.org
PORT=443
WDM_LOG=0
```

## Getting Started with ARC-3

1. **Register** at https://three.arcprize.org and get your API key
2. **Clone** the ARC-AGI-3-Agents repository: `git clone https://github.com/arcprize/ARC-AGI-3-Agents`
3. **Configure** your `.env` file with your API key
4. **Run** a quick demo: `python run_continuous_learning.py --mode demo`
5. **View** your results at the official ARC-3 scoreboard: https://arcprize.org/leaderboard

## Documentation

For detailed documentation, see the `/docs` folder:
- **Implementation Status**: Current development progress
- **Research Findings**: Detailed analysis and results
- **AGI Puzzle Summary**: Evaluation results and insights
- **Phase 1 Implementation**: Technical specifications