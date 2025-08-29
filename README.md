# Adaptive Learning Agent

An AI system that develops intelligence through intrinsic motivation, curiosity, and survival pressure rather than external programming.

## Overview

This project implements a "digital childhood" paradigm where an agent learns through:
- **Predictive Core**: Recurrent world-model that predicts sensory input
- **Learning Progress Drive**: Intrinsic motivation from prediction improvement
- **Embedded Memory**: Differentiable Neural Computer for integrated memory
- **Energy & Death**: Survival pressure through limited resources
- **Goal Invention**: Self-generated objectives from high-learning-progress experiences

## Project Structure

```
adaptive-learning-agent/
├── src/
│   ├── core/              # Core agent components
│   ├── environment/       # Simulation environments
│   ├── memory/           # Memory systems
│   ├── goals/            # Goal invention and management
│   ├── monitoring/       # Debugging and introspection
│   └── utils/            # Utilities and helpers
├── tests/                # Comprehensive unit tests
├── configs/              # Configuration files
├── experiments/          # Experiment scripts and results
└── docs/                 # Documentation
```

## Development Phases

- **Phase 0** (Weeks 1-4): Component isolation and validation
- **Phase 1** (Weeks 5-12): Integrated system development
- **Phase 2** (Weeks 13-24): Complexity scaling and advanced features
- **Phase 3** (Month 7+): Multi-agent and emergent behaviors

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Run basic survival test
python experiments/phase0_survival_test.py

# Run LP drive validation
python experiments/phase0_lp_validation.py
```

## Key Features

- **Robust Learning Progress Drive** with stability validation
- **Differentiable Neural Computer** for embedded memory
- **Bootstrap protection** for newborn agents
- **Comprehensive monitoring** and introspection tools
- **Phased development** with validation at each stage

## Research Goals

This system aims to validate the hypothesis that intelligence emerges from the right environmental conditions and internal drives, not from explicit programming of behaviors or goals.