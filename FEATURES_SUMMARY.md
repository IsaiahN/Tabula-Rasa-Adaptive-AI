# Tabula Rasa — Features Summary (TB-Seed DNA)

This document is an inventory of the major features and modules present in the TB-Seed codebase. Treat this as the "DNA" summary — the canonical list of features that child branches can choose to express.

## High-level features

- ARC-AGI (ARC3) API integration
  - Files: `train.py`, `src/api`, calls in `src/training/core/continuous_learning_loop.py`
  - Behavior: discover available games, create/reset games, take actions, submit scores, manage scorecards and rate limits.

- Continuous learning & training orchestration
  - Files: `src/training/core/continuous_learning_loop.py`, `train.py`
  - Behavior: main loop, session management, real-time and batch training modes, game runner, game outcome analysis.

- Vision pipeline (frame analysis & object detection)
  - Files: `src/vision/*`, `src/vision/enhanced/*`
  - Behavior: frame analyzers, enhanced object/feature detectors, coordinate selection for ACTION6.

- Real-time learning engine (Phase 1.1)
  - Files: `src/core/real_time_learner.py`, `src/core/mid_game_pattern_detector.py`, `src/core/dynamic_strategy_adjuster.py`
  - Behavior: process action outcomes live, detect patterns, adjust strategy mid-game.

- Strategy discovery / pattern learners
  - Files: `src/learning`, `src/training/*`, `src/analysis/*`
  - Behavior: discover winning sequences, store strategies, replay and analyze game states.

- NEAT-based architect system
  - Files: `src/core/neat_based_architect.py`, integrated in `continuous_learning_loop.py`
  - Behavior: evolve architecture/modules, module pruning/creation, add fitness observers.

- Context-dependent fitness evolution
  - Files: `src/core/context_dependent_fitness_evolution.py`
  - Behavior: evaluate contextual fitness, provide priorities for attention allocation.

- Bayesian inference engine
  - Files: `src/core/bayesian_inference_engine.py`
  - Behavior: hypothesis creation, evidence addition, predictions for action outcomes.

- Enhanced graph traversal / decision-space modeling
  - Files: `src/core/enhanced_graph_traversal.py`
  - Behavior: create game-state graphs, find optimal paths, visualize decision spaces.

- Attention & communication system
  - Files: `src/core/central_attention_controller.py`, `src/core/weighted_communication_system.py`
  - Behavior: allocate compute/attention resources, route messages between subsystems.

- Governor / meta-cognitive controller
  - Files: `src/governor/*`, `src/governor/meta_cognitive.py`
  - Behavior: high-level decisions, when to reflect, manage resource usage and run policies.

- Memory and persistence
  - Files: `src/memory/*`, `src/database/*`, `data/*.db`, `scripts/create_empty_db.py`
  - Behavior: store patterns, sequences, button priorities, game/session persistence.

- Performance monitoring & metrics
  - Files: `src/core/unified_performance_monitor.py`, `src/performance/*`
  - Behavior: monitor memory, CPU, events, metrics collection.

- Utilities & dev scripts
  - Files: `scripts/*`, `tools/*`, `docs/*`
  - Behavior: DB tools, cleanup, analysis helpers, experiment harnesses.

## Notable cross-cutting concerns

- Heavy use of synchronous + asynchronous mixing inside orchestration (async API calls + sync initializers).
- Multiple optional components (tiers 1-3) that are conditional on DB connectivity/flags.
- Several systems duplicate responsibilities (pattern discovery, strategy discovery, real-time learner overlap).
- The codebase contains guardrails and many fallback paths — good for robustness, increases complexity.

## Suggested places to look first during refactoring

- `src/training/core/continuous_learning_loop.py` — central orchestrator (largest single file).
- `src/vision/enhanced/` — vision subsystem imports and lazy init; often causes import cycles.
- `src/core/` — contains many experimental components (NEAT, Bayesian, Graph traversal) that can be toggled.
- `src/database/` — persistence and integration code; central to enabling advanced modules.


*This inventory is intentionally high-level. The next step (SIMPLIFICATIONS.md) will analyze pros/cons and propose simplifications.*
