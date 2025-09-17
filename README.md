# Tabula Rasa: Adaptive AI Architecture

A novel AI system that implements **meta-cognitive supervision** and **simulation-driven intelligence** for adaptive reasoning and autonomous evolution. Built for the [ARC-AGI-3](https://arcprize.org/arc-agi/3/) challenge with unique architectural approaches to consciousness, memory management, and self-improvement.

<img width="1309" height="553" alt="image" src="https://github.com/user-attachments/assets/5ebe7ffc-e73c-41e3-b7c6-14a9a01aa81e" />

## 🧠 Novel Architectural Contributions

### Meta-Cognitive Supervision
Unlike traditional AI systems that operate as black boxes, Tabula Rasa implements explicit meta-cognitive layers:

- **Governor System**: Runtime supervisor with 37 system monitors that actively manages cognitive resources and decision-making
- **Architect System**: Autonomous evolution engine that modifies its own architecture based on performance analysis
- **Recursive Self-Improvement**: Governor reports to Architect, which generates evolutionary directives back to Governor
- **Advanced Learning Integration**: Elastic Weight Consolidation, Residual Learning, and Extreme Learning Machines for enhanced learning capabilities

### Multi-Phase Memory Optimization
A unique approach to memory management that goes beyond simple storage:

1. **Pattern Recognition**: Detects memory access patterns and optimization opportunities
2. **Hierarchical Clustering**: Groups memories by causality, temporal patterns, and semantic similarity  
3. **Architect Evolution**: Uses memory analysis to drive architectural improvements
4. **Performance Optimization**: Real-time performance maximization using cross-phase intelligence

### Simulation-Driven Intelligence
Transforms reactive AI into proactive planning:

- **Multi-Step Planning**: Thinks 5-10 steps ahead instead of single-step decisions
- **Imagination Engine**: Runs internal simulations before taking real actions
- **Strategy Memory**: Stores successful action sequences as reusable strategies
- **Hypothesis Generation**: Creates "what-if" scenarios for strategic planning

## 🔍 How It Contrasts with Standard AI Approaches

### 1. **Autonomous Architecture Modification**
- Most AI systems have fixed architectures
- Tabula Rasa can modify its own structure based on performance feedback
- Implements safe evolutionary changes with rollback capabilities

### 2. **Meta-Cognitive Awareness**
- Traditional AI lacks self-awareness of its own cognitive processes
- Tabula Rasa monitors its own decision-making and resource allocation
- Provides transparent reasoning chains and decision justifications

### 3. **Cross-Session Learning with Persistence**
- Most systems start fresh each session
- Tabula Rasa maintains persistent learning state across sessions
- Implements breakthrough-only memory preservation to avoid catastrophic forgetting

### 4. **Simulation Before Action**
- Current AI is largely reactive
- Tabula Rasa simulates multiple futures before acting
- Uses internal drives to (energy, learning drive, boredom) to evaluate outcomes

### 5. **Hierarchical Memory with Semantic Understanding**
- Traditional systems treat all memories equally
- Tabula Rasa implements strategic memory decay with prioritization
- Mainly preserves memories when achieving new personal bests

## 🚀 Quick Start

### Installation
```bash
# Clone and install dependencies
git clone <repository-url>
cd tabula-rasa
pip install -r requirements.txt

# Set up environment
python setup_env.py
# Edit .env file with your ARC-3 API key

# Database is automatically initialized on first run
# Template database (tabula_rasa_template.db) available for fresh starts
```

### Basic Usage

#### Windows Users (Recommended)
```bash
# Intelligent 9-hour training with automatic resource optimization
start_intelligent_training.bat

# Interactive launcher with multiple training modes
launch_trainer.bat
```

#### Advanced Users
```bash
# Direct access to advanced 9-hour training scripts
python run_9hour_scaled_training.py    # Intelligent parallel training
python run_9hour_simple_training.py    # Sequential training
```

#### Manual Usage
```bash
# Intelligent 9-hour scaled training (Recommended)
python run_9hour_scaled_training.py

# Simple 9-hour sequential training
python run_9hour_simple_training.py

# Legacy master trainer (basic functionality)
python master_arc_trainer.py
```

### Get Your API Key
1. Visit [https://three.arcprize.org](https://three.arcprize.org)
2. Sign up for an account
3. Get your API key from your profile
4. Add it to your `.env` file

## 🏗️ System Architecture

### Core Components

```
Tabula Rasa
├── Entry Points
│   ├── run_9hour_scaled_training.py (Primary)
│   ├── run_9hour_simple_training.py (Alternative)
│   └── master_arc_trainer.py (Legacy)
├── Database Layer (NEW!)
│   ├── SQLite Database (High-Performance Storage)
│   ├── Real-Time API (Director/LLM Interface)
│   ├── System Integration (Backward Compatibility)
│   └── Data Migration (File-to-Database)
├── Meta-Cognitive Layer
│   ├── Governor System (Runtime Supervisor)
│   ├── Architect System (Self-Improvement)
│   ├── Director System (LLM Executive)
│   └── Recursive Self-Improvement Loop
├── Cognitive Layer
│   ├── Differentiable Neural Computer
│   ├── Phased Memory Optimization
│   └── Simulation-Driven Intelligence
└── Application Layer
    ├── ARC-AGI-3 Integration
    ├── Action Intelligence
    └── Cross-Session Learning
```

### Database Architecture (NEW!)
```
Database Layer
├── SQLite Database (tabula_rasa.db)
│   ├── training_sessions (Session Management)
│   ├── game_results (Game Performance)
│   ├── action_effectiveness (Action Intelligence)
│   ├── coordinate_intelligence (Coordinate Learning)
│   ├── learned_patterns (Pattern Recognition)
│   ├── system_logs (Structured Logging)
│   └── performance_metrics (Analytics)
├── Director Commands API
│   ├── Real-Time System Status
│   ├── Learning Analysis
│   ├── System Health Monitoring
│   └── Performance Analytics
└── System Integration
    ├── Backward Compatibility
    ├── File I/O Replacement
    └── Multi-System Access
```

### Data Structure (Database-Driven)
```
tabula_rasa.db           # SQLite Database (Primary Storage)
├── training_sessions    # Session management and metrics
├── game_results         # Game performance data
├── action_effectiveness # Action intelligence and success patterns
├── coordinate_intelligence # Coordinate learning data
├── learned_patterns     # Pattern recognition and memory
├── system_logs          # Structured logging and events
├── global_counters      # Real-time system state
└── performance_metrics  # Analytics and monitoring

src/database/            # Database Layer
├── api.py              # High-level database API
├── director_commands.py # Director/LLM interface
├── system_integration.py # Backward compatibility layer
└── schema.sql          # Database schema definition
```

## ⚙️ Configuration

### Key Parameters
- `--max-actions`: Maximum actions per game (default: 500)
- `--mode`: Training mode (maximum-intelligence, quick-validation, meta-cognitive-training, continuous, etc.)
- `--enable-governor`: Enable meta-cognitive supervision (default: True)
- `--enable-architect`: Enable autonomous evolution (default: True)

### Environment Variables (.env)
```bash
ARC_API_KEY=your_arc_api_key_here
TARGET_WIN_RATE=0.90
TARGET_AVG_SCORE=85.0
MAX_EPISODES_PER_GAME=50
```

## 🧪 Training Modes

### Maximum Intelligence (Default)
- All advanced features enabled with optimal settings
- Governor + Architect supervision with 37 cognitive subsystems
- Autonomous evolution with self-improvement capabilities

### Quick Validation
- Fast testing with essential features only
- 2-3 cycles for rapid feedback
- Perfect for testing setup and basic functionality

### Meta-Cognitive Training
- Full meta-cognitive supervision with cross-session learning
- Persistent state management across sessions
- Advanced memory optimization

### Continuous Training
- Extended training sessions with persistent state
- Progressive difficulty with adaptive curriculum
- Cross-task knowledge transfer

### Research Lab
- Scientific analysis and system comparison
- Performance monitoring and optimization
- Meta-cognitive effectiveness measurement

## 🗄️ Database Architecture (NEW!)

### High-Performance SQLite Database
Tabula Rasa now features a comprehensive database architecture that replaces file-based storage with a robust SQLite database:

- **10-100x faster queries** than traditional file I/O
- **Concurrent access** for multiple systems (Director, Governor, Architect)
- **Real-time data sharing** across all components
- **ACID transactions** for data integrity
- **Optimized indexing** for heavy usage scenarios

### Director Commands API
The Director (LLM) can now interface with system data through a comprehensive command interface:

```python
from src.database.director_commands import get_director_commands

director = get_director_commands()

# Get real-time system status
status = await director.get_system_overview()

# Analyze learning progress
learning = await director.get_learning_analysis()

# Check system health
health = await director.analyze_system_health()

# Get action effectiveness analysis
actions = await director.get_action_effectiveness()

# Get coordinate intelligence analysis
coordinates = await director.get_coordinate_intelligence()
```

### Key Database Features
- **Real-Time Monitoring**: Instant system status and performance metrics
- **Learning Analytics**: Advanced analysis of learning patterns and effectiveness
- **System Health**: Comprehensive health monitoring with recommendations
- **Performance Tracking**: Detailed performance metrics and trends
- **Action Intelligence**: Analysis of action effectiveness and success patterns
- **Coordinate Learning**: Intelligence on successful coordinate patterns

### Complete Database Migration
All file-based storage has been successfully migrated to the database:
- ✅ **166 training sessions** migrated to `training_sessions` table
- ✅ **166 game results** migrated to `game_results` table
- ✅ **539,943+ system logs** migrated to `system_logs` table
- ✅ **Action intelligence data** migrated to `action_effectiveness` table
- ✅ **Coordinate patterns** migrated to `coordinate_intelligence` table
- ✅ **Global counters** migrated to `global_counters` table
- ✅ **All file I/O operations** replaced with database calls
- ✅ **Legacy data directories** cleaned up and removed
- ✅ **System fully database-driven** with no file dependencies

## 🧠 Meta-Cognitive Systems

### Governor System
Runtime supervisor that provides:
- **37 System Monitors**: Real-time cognitive system health tracking
- **Outcome Tracking**: Cross-session learning and performance persistence
- **Adaptive Configuration**: Dynamic parameter optimization
- **Resource Management**: Intelligent allocation of computational resources
- **Residual Learning**: Skip connections and gradient flow optimization for deep learning

### Architect System
Autonomous evolution engine that enables:
- **Self-Improvement Cycles**: Autonomous architecture modification
- **Git Integration**: Version control for evolutionary changes
- **Mutation Testing**: Safe experimentation with system modifications
- **Performance-Driven Evolution**: Data-driven architectural improvements
- **Elastic Weight Consolidation**: Prevents catastrophic forgetting during continuous learning

### Director System
Central executive that orchestrates:
- **Meta-Cognitive Oversight**: High-level strategic decision making
- **Extreme Learning Machines**: Fast single-hidden layer neural networks for rapid adaptation
- **Narrative Engine**: Internal monologue and reasoning transparency
- **Drive Management**: Intrinsic motivation and goal prioritization

### Recursive Self-Improvement Loop
Orchestrates the complete cycle:
1. Governor monitors system performance and makes runtime decisions
2. Governor reports session data to Architect
3. Architect analyzes reports and generates evolutionary directives
4. Director synthesizes meta-cognitive insights and strategic direction
5. Governor implements directives and continues operation


## 📊 Performance Monitoring

### Real-time Tracking
- Action counter with proper limit enforcement
- Score tracking and improvement metrics
- Memory utilization and system health
- Learning progress indicators
- Advanced learning paradigm effectiveness metrics

### Database-Driven Output
- **Real-time data access**: All data stored in SQLite database
- **Structured logging**: System events and performance metrics
- **Performance analytics**: Advanced querying and analysis capabilities
- **Cross-session persistence**: Learning state maintained across sessions
- **Director interface**: LLM can query and analyze all system data

## 🔧 Advanced Configuration

### Advanced Training
```bash
# Intelligent scaled training with resource optimization
python run_9hour_scaled_training.py

# Simple sequential training for stability
python run_9hour_simple_training.py

# Legacy trainer with custom parameters
python master_arc_trainer.py --memory-size 1024 --memory-read-heads 8
```

## 🤝 Contributing

This project explores novel approaches to AI architecture and meta-cognition. Contributions welcome in:

- Meta-cognitive systems: Governor, Architect, and Director enhancements
- Memory management: 4-phase optimization algorithms
- Simulation intelligence: Multi-step planning improvements
- Cross-session learning: Persistent state management
- Autonomous evolution: Safe architectural modification
- Advanced learning paradigms: EWC, Residual Learning, ELM optimizations
- Testing framework: Validation and monitoring tools

## 🎯 Current Project Status

### ✅ **PROJECT FULLY MIGRATED TO DATABASE**
The Tabula Rasa project has been completely transformed from file-based storage to a high-performance database architecture:

- **🗄️ Database-Driven**: All data now stored in SQLite database (`tabula_rasa.db`)
- **🧹 Clean Workspace**: Root directory cleaned of all temporary and redundant files
- **⚡ High Performance**: 10-100x faster data access compared to file I/O
- **🔄 Real-Time Access**: Director/LLM can query system data in real-time
- **📊 Complete Migration**: All historical data successfully migrated
- **🛠️ Streamlined Code**: All file references updated to use database API

### 📁 **Clean Project Structure**
```
tabula-rasa/
├── src/                    # Source code
│   ├── database/          # Database layer
│   ├── arc_integration/   # Core systems
│   └── ...
├── tabula_rasa.db         # Main database (117 MB)
├── tabula_rasa_template.db # Template database (164 KB)
├── run_9hour_*.py         # Training scripts
├── master_arc_trainer.py  # Legacy trainer
├── README.md              # This file
└── requirements.txt       # Dependencies
```

## 📚 Recent Updates

### Complete Database Migration (v5.1) - COMPLETED!
Revolutionary database architecture that transforms system performance and capabilities:

- **SQLite Database**: High-performance database replacing all file-based storage
- **10-100x Performance Improvement**: Dramatically faster queries and data access
- **Real-Time API**: Director/LLM can now interface with system data in real-time
- **Concurrent Access**: Multiple systems (Director, Governor, Architect) can access data simultaneously
- **Complete Migration**: Successfully migrated all historical data to database
- **File System Cleanup**: Removed all legacy data directories and temporary files
- **Director Commands**: Comprehensive command interface for system analysis and control
- **System Integration**: All file I/O operations replaced with database calls
- **Clean Workspace**: Root directory cleaned of temporary and redundant files

### Advanced Learning Integration (v4.0)
Successfully integrated cutting-edge learning paradigms:

- **Elastic Weight Consolidation (EWC)**: Prevents catastrophic forgetting in the Architect system during continuous learning
- **Residual Learning**: Implements skip connections in the Governor system for improved gradient flow and deeper learning
- **Extreme Learning Machines (ELMs)**: Fast single-hidden layer neural networks in the Director for rapid adaptation
- **Meta-Cognitive Integration**: All learning paradigms work together through the central Director system

### System Stability Improvements
- **Error Elimination**: Resolved all governor-related errors and parameter mismatches
- **Performance Optimization**: Enhanced action counting and scorecard linking
- **Robust Operation**: 9-hour training sessions running error-free with full functionality
- **Learning Verification**: Demonstrated score improvement from 0 to 1 with active pattern learning
- **Database Integration**: All systems now use database for data persistence
- **Clean Architecture**: Removed all temporary files and legacy data directories
- **Streamlined Codebase**: Updated all file references to use database API

## 📚 Future State

The system now includes a Large Language Model as a "Director" that synthesizes the cognitive subsystems into a unified "self". 

This adds the missing pillars of simulated consciousness through specialized agent modules: 
- a NarrativeEngine for internal monologue 
- an AffectiveAgent for emotional context 
- a DriveAgent for intrinsic motivation 
- a SocialSimulant for cultural learning
- a CodingAgent to help self-refactor and test core architecture 

The LLM serves as the central executive that translates technical data into context-based data points, enabling proactive rather than reactive behavior to obstacles in expanded sandbox environments. 

Beyond ARC AGI puzzles, this architecture could eventually operate in rich simulation environment Games with Open World Models, where the agent could develop theory of mind, negotiate with NPCs, and pursue self-generated goals based on intrinsic drives rather than external rewards. 

## 📄 License

MIT License - See LICENSE file for details.

