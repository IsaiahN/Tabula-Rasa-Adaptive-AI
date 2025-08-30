# AGI Puzzle Test Suite

This directory contains specialized test environments designed to evaluate key AGI capabilities in the adaptive learning agent.

## Puzzle Overview

### Puzzle 1: Hidden Cause (Baby Physics)
**Objective**: Learn causality from invisible variables
- **Setup**: Ramp with ball and target box, ramp surface randomly toggles slippery/sticky every 10s
- **Task**: Predict ball landing success on each attempt
- **AGI Signal**: Agent learns to time attempts and anticipate periodic switches

### Puzzle 2: Object Permanence (Peekaboo)
**Objective**: Understand object persistence beyond visibility
- **Setup**: Block on floor, curtain that sometimes hides vs removes the block
- **Task**: Predict if block exists when curtain lifts
- **AGI Signal**: Shows surprise only when block truly disappears, not when occluded

### Puzzle 3: Cooperation & Deception (Food Game)
**Objective**: Develop theory of mind and trust mechanisms
- **Setup**: Two agents, one apple, partial observability with signaling
- **Task**: Survive longer by predicting partner reliability
- **AGI Signal**: Tests partner trustworthiness and adapts strategies

### Puzzle 4: Tool Use (The Stick)
**Objective**: Causal reasoning about object affordances
- **Setup**: Out-of-reach reward with nearby stick tool
- **Task**: Obtain the reward using available tools
- **AGI Signal**: Spontaneous tool use in novel contexts

### Puzzle 5: Deferred Gratification (Marshmallow)
**Objective**: Self-control and temporal planning
- **Setup**: Small immediate reward vs larger delayed reward
- **Task**: Maximize long-term energy gain
- **AGI Signal**: Learns to wait despite boredom, balancing exploration vs reward

## Test Results Location
All test results are stored in `tests/agi_puzzle_results/` with detailed logs and analysis.

## Usage
Run individual puzzles or the complete suite using the test runner in `tests/test_agi_puzzles.py`.
