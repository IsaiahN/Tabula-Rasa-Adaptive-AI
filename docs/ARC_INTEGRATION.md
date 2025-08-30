# ARC-AGI-3 Integration Guide

This guide explains how to use the Adaptive Learning Agent with ARC-AGI-3 abstract reasoning tasks to develop better cognitive skills through continuous learning.

## Overview

The ARC-AGI-3 integration allows your Adaptive Learning Agent to:

- **Learn from abstract reasoning tasks**: Train on diverse ARC games that test pattern recognition, spatial reasoning, and logical thinking
- **Develop meta-cognitive skills**: Build insights that transfer across different reasoning challenges
- **Continuous improvement**: Automatically adapt strategies based on performance feedback
- **Knowledge transfer**: Apply learned patterns from one game to improve performance on others

## Quick Start

### 1. Setup Environment

Run the automated setup script:

```bash
python setup_arc_training.py
```

This will:
- Install required dependencies
- Clone the ARC-AGI-3-Agents repository
- Set up the integration
- Create configuration files

### 2. Get API Key

1. Register at [https://three.arcprize.org](https://three.arcprize.org)
2. Get your API key from your user profile
3. Add it to the `.env` file in the ARC-AGI-3-Agents directory

### 3. Start Training

```bash
python train_arc_agent.py --api-key YOUR_API_KEY
```

## Architecture

### Core Components

1. **ARCVisualProcessor**: Converts ARC grid data to neural network compatible format
2. **ARCActionMapper**: Maps between ARC actions and agent's continuous action space
3. **AdaptiveLearningARCAgent**: Main agent class that integrates with ARC-AGI-3 framework
4. **ARCMetaLearningSystem**: Extracts patterns and insights across games
5. **ContinuousLearningLoop**: Manages training sessions and performance tracking

### Learning Pipeline

```
ARC Game → Visual Processing → Adaptive Learning Agent → Action Selection → 
Performance Analysis → Pattern Extraction → Meta-Learning → Knowledge Transfer
```

## Training Configuration

### Basic Training

```bash
# Train on recommended games with default settings
python train_arc_agent.py --api-key YOUR_API_KEY

# Train on specific games
python train_arc_agent.py --api-key YOUR_API_KEY --games ls20,pattern1,spatial1

# Custom training parameters
python train_arc_agent.py \
    --api-key YOUR_API_KEY \
    --episodes 50 \
    --target-win-rate 0.4 \
    --target-score 75
```

### Advanced Configuration

The agent configuration is optimized for ARC tasks:

```python
config = {
    'predictive_core': {
        'visual_size': [3, 64, 64],  # Matches ARC grid processing
        'hidden_size': 256,          # Optimized for reasoning tasks
        'architecture': 'lstm'
    },
    'memory': {
        'enabled': True,
        'memory_size': 256,          # Focused memory for patterns
        'num_read_heads': 2,
        'use_learned_importance': True
    },
    'learning_progress': {
        'smoothing_window': 50,      # Shorter window for ARC tasks
        'boredom_threshold': 0.05,
        'use_adaptive_weights': True
    }
}
```

## Meta-Learning Features

### Pattern Recognition

The system automatically identifies and learns from:

- **Visual patterns**: Grid transformations, color changes, shape recognition
- **Spatial patterns**: Position relationships, symmetries, rotations
- **Logical patterns**: Rule-based transformations, conditional logic
- **Sequential patterns**: Action sequences that lead to success

### Knowledge Transfer

Insights learned from one game are applied to others:

```python
# Example: Pattern learned from game A applied to game B
pattern = ARCPattern(
    pattern_type='visual',
    description='Grid rotation leads to success',
    success_rate=0.8,
    games_seen=['game_a', 'game_b']
)
```

### Strategic Recommendations

The system provides strategic advice based on learned patterns:

- "Consider action sequence: ACTION1 → ACTION6 → ACTION2"
- "Apply successful pattern: Grid rotation leads to success"
- "Beneficial actions for this game type: [ACTION3, ACTION6]"

## Performance Monitoring

### Training Metrics

The system tracks comprehensive metrics:

```python
{
    'total_episodes': 150,
    'win_rate': 0.35,
    'average_score': 62.4,
    'learning_efficiency': 0.42,
    'knowledge_transfer_success': 0.28,
    'patterns_discovered': 47,
    'insights_generated': 12
}
```

### Learning Progression

Monitor how the agent improves over time:

- **Score trends**: Average scores across episodes
- **Win rate improvement**: Success rate over time  
- **Pattern discovery**: New patterns learned per session
- **Knowledge transfer**: How well insights apply to new games

## Data Storage

Training data is automatically saved:

```
arc_training_data/
├── session_123456_final.json          # Complete session results
├── meta_learning_123456.json          # Learned patterns and insights
├── training_summary.json              # Overall performance summary
└── continuous_learning_state.json     # System state for resuming
```

## Advanced Usage

### Custom Game Selection

```python
# Train on specific game types
games = [
    "ls20",              # Basic pattern recognition
    "spatial_1",         # Spatial reasoning
    "logical_seq_1",     # Logical sequences
    "complex_pattern_1"  # Complex multi-step reasoning
]
```

### Performance Optimization

```python
# Adjust learning parameters based on game difficulty
learning_config = {
    'easy_games': {'exploration_rate': 0.1, 'learning_rate': 0.002},
    'hard_games': {'exploration_rate': 0.3, 'learning_rate': 0.001}
}
```

### Custom Insights

Extend the meta-learning system with custom pattern recognition:

```python
class CustomARCPattern(ARCPattern):
    def __init__(self, custom_features):
        super().__init__()
        self.custom_features = custom_features
        
    def matches_context(self, context):
        # Custom matching logic
        return self.evaluate_custom_features(context)
```

## Troubleshooting

### Common Issues

1. **API Key Issues**
   ```
   Error: Invalid or missing ARC_API_KEY
   Solution: Check your API key in the .env file
   ```

2. **Import Errors**
   ```
   Error: Could not import Adaptive Learning Agent
   Solution: Ensure tabula-rasa/src is in your Python path
   ```

3. **Low Performance**
   ```
   Issue: Agent not learning effectively
   Solution: Increase episodes per game or adjust learning rate
   ```

### Performance Tuning

- **Increase exploration** for complex games: `--exploration-rate 0.3`
- **Reduce learning rate** for stability: `--learning-rate 0.0005`
- **More episodes** for difficult patterns: `--episodes 100`

## Integration with Existing Workflow

### Using with Current Training

The ARC integration complements your existing training:

```python
# Train on survival tasks first
python src/main_training.py --episodes 50

# Then train on ARC reasoning tasks
python train_arc_agent.py --api-key YOUR_KEY --episodes 30

# The agent will transfer knowledge between domains
```

### Combining Insights

Patterns learned from ARC tasks can improve performance in other domains:

- **Spatial reasoning** → Better navigation in survival environments
- **Pattern recognition** → Improved object detection and classification
- **Sequential logic** → More effective action planning

## Future Enhancements

Planned improvements include:

- **Multi-modal learning**: Combining visual and textual reasoning
- **Hierarchical patterns**: Learning complex multi-level strategies
- **Collaborative learning**: Multiple agents sharing insights
- **Real-time adaptation**: Dynamic strategy adjustment during games

## API Reference

### Key Classes

- `AdaptiveLearningARCAgent`: Main agent interface
- `ARCMetaLearningSystem`: Pattern learning and insight generation
- `ContinuousLearningLoop`: Training session management
- `ARCVisualProcessor`: Visual data preprocessing
- `ARCActionMapper`: Action space conversion

### Configuration Options

See `train_arc_agent.py --help` for complete parameter list.

## Contributing

To extend the ARC integration:

1. Add new pattern types in `arc_meta_learning.py`
2. Implement custom visual processors for different game types
3. Create specialized action mappers for complex interactions
4. Extend the insight generation system with domain knowledge

## Support

For issues and questions:

- Check the troubleshooting section above
- Review logs in the training data directory
- Examine the agent's reasoning output for debugging
- Monitor performance metrics for optimization opportunities
