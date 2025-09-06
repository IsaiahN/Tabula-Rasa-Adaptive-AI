Migration: Removed compatibility shims

Goal

This change intentionally removes several legacy compatibility shims and deprecated aliases to simplify and modernize the public API. Tests and downstream code that relied on these shims will need small updates.

Scope of removals

- `hidden_dim` legacy constructor kwarg on `AdaptiveLearningAgent`
  - Removed. The agent now expects configuration to contain predictive core size under `config['predictive_core']['hidden_size']`.
  - Replacement: read `agent.predictive_core.hidden_size` or set `config['predictive_core']['hidden_size']` when constructing the agent.

- `SalienceSystem` alias and `process_memory` wrapper in `core.salience_system`
  - Removed. The canonical API is now `SalienceCalculator` and `SalienceWeightedReplayBuffer`.
  - Replacement: import `SalienceCalculator` and call `calculate_salience(...)` or create `create_salient_experience(...)`. For prioritized replay use `SalienceWeightedReplayBuffer`.

- `process_memory(memories, importance_scores)` compatibility wrapper
  - Removed. Instead, explicitly compute salience per experience and feed results into replay buffer / compression flows.
  - Example:
    from core.salience_system import SalienceCalculator
    sc = SalienceCalculator()
    sal = sc.calculate_salience(lp, energy_change, current_energy)

- `success_multiplier` and `_calculate_episode_effectiveness` stub in `ContinuousLearningLoop`
  - Removed. Episode effectiveness should be computed from explicit episode metrics (wins, actions, score) or derived via salience metrics.
  - Replacement: compute effectiveness via:
    effectiveness = wins / max(1, actions_taken)
  - For prioritized storage, use `SalienceCalculator` to compute salience for episode-level experiences and then weight them via `SalienceWeightedReplayBuffer`.

Why this change?

- The compatibility shims masked API design issues and increased the test/maintenance burden.
- Removing aliases clarifies the canonical API surface and encourages callers/tests to use explicit, stable APIs.

Quick test-update guide (how to update failing tests)

1) Tests asserting `agent.hidden_dim == <n>`
   - Replace with `assert agent.predictive_core.hidden_size == <n>`
   - Or explicitly pass `config={'predictive_core': {'hidden_size': <n>, ...}}` when constructing the agent.

2) Tests importing `SalienceSystem`:
   - Replace `from core.salience_system import SalienceSystem` with `from core.salience_system import SalienceCalculator as SalienceSystem` or preferably use `SalienceCalculator` directly.
   - Update tests that call `process_memory(...)` to manually compute salience per memory and assert on resulting objects or use `SalienceWeightedReplayBuffer` sampling behavior.

3) Tests checking for text snippets like `"success_multiplier = 10.0"` or `_calculate_episode_effectiveness`:
   - Update assertions to validate computation instead. Example: check `effectiveness == wins / actions` or derive salience and assert > threshold for prioritized episodes.

Suggested migration patch snippets

- Replace hidden_dim assertions (pytest example):

    agent = AdaptiveLearningAgent(config={'predictive_core': {'hidden_size': 64}})
    assert agent.predictive_core.hidden_size == 64

- Replace SalienceSystem usage:

    from core.salience_system import SalienceCalculator
    sc = SalienceCalculator()
    sal = sc.calculate_salience(learning_progress=0.2, energy_change=1.0, current_energy=50.0)
    assert 0.0 <= sal <= 1.0

Risks & follow-ups

- Tests that relied on implicit compatibility layers will fail until migrated. This is expected.
- Minor follow-up: update docs and README to reference canonical API names.
- Optional: provide a short adapter module that tests can import during migration (`core/compat.py`) that maps old names to new ones for a short deprecation window.

Contact/Verification

If you want, I can either:
- Apply automated test updates for the simplest replacements (hidden_dim -> predictive_core.hidden_size, SalienceSystem -> SalienceCalculator) and run the test suite, or
- Leave tests untouched and produce a smaller adapter module `core/compat.py` that re-exports old names with warnings so downstream code can migrate at leisure.

Choose which path to take next.
