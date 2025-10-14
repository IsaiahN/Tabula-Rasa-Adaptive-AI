# TB-Seed Core Contract

This file defines the core "DNA" contract shared across all child branches. Keep this minimal and stable.

## Purpose
Defines the minimal API, data shapes, and behaviors that must be preserved across branches so experiments remain compatible.

## Core responsibilities

1. ARC3 API integration (external service)
   - Responsibilities: discover games, reset game, take action, get game state, create/close scorecards, submit scores.
   - Expected method signatures (high level):
     - APIManager.get_available_games() -> List[Dict]
     - APIManager.reset_game(game_id, scorecard_id=None) -> GameState
     - APIManager.take_action(game_id, action, scorecard_id, guid) -> GameState/dict
     - APIManager.create_scorecard(name, description) -> scorecard_id
     - APIManager.close_scorecard(scorecard_id)

2. GameState shape
   - Must include at least: `state` (str), `score` (float), `available_actions` (list), `frame` (optional), `guid` (optional)

    - Frame data (core):
       - The `frame` field is considered a first-class, core part of the GameState. It must convey the current visual frame representation the agent sees (pixel grid, compressed image bytes, or an agreed structured representation).
       - The seed contract requires branches to preserve a `frame` payload (or a clear shim/dummy provider in minimal modes) because frame data is essential to understand the live game context and to drive vision-based action selection (e.g., ACTION6 coordinate selection).

    - Object detection (core):
       - Object detection is treated as a core capability built on top of `frame` data. Branches must either provide an object-detection component (real or stubbed) or expose a stable interface that accepts `frame` and returns detected objects/regions.
       - Expected object-detection output shape: list of detections, each with at minimum `{ 'label': str, 'confidence': float, 'bbox': [x, y, w, h] }` or a simple coordinate tuple for actionable objects.


3. Action shape
   - Minimal: `{'id': int}` or `{'id': int, 'x': int, 'y': int}` for coordinate actions

4. Persistence
   - Minimal DB schema with sessions, games, strategies, and basic metrics. Implementations may extend but must support baseline tables.

5. Runner contract
   - A GameRunner class with methods:
     - `create_session(session_id)`
     - `run_game(game_id, max_actions)` -> returns result dict with `score`, `win`, `actions_taken`

## Compatibility rules

- Branches may extend the seed but must maintain the above method names and data shapes for interoperability (especially strategy artifacts and stored sequences).
- Feature flags should use a stable config key (e.g., `TB_FEATURES['neat_architect'] = True/False`).

## Testing & CI

- All branches must provide a `--minimal` mode that runs against a local stub API for CI.

## Operational constraints (important)

- No simulated ARC3 API for core runs: The TB-Seed contract requires that developer and production runs use the real ARC-AGI-3 (ARC3) API endpoints whenever running "real" experiments. Local or in-repo API simulations are allowed only for CI smoke-tests and must be explicitly opt-in via a clearly named flag (for example `--use-stub-api-for-ci`) and must never be used for research/production experiments. This prevents drift between simulated behavior and the real system and ensures experiments remain grounded in the real API semantics.

- Database-backed persistence required: The system must use a real database for persistence (SQLite is acceptable for local/dev and is the default). Logging to files/console is **not** a substitute for persistence. The DB must provide at minimum the following baseline tables (names are suggestions and may be extended):
   - `sessions` (session_id, start_time, end_time, status, metadata)
   - `games` (game_id, guid, scorecard_id, start_time, end_time, final_score, final_state)
   - `actions` (action_id, session_id, game_id, action_payload, result_payload, timestamp)
   - `strategies` (strategy_id, game_id, action_sequence, efficiency, metadata)
   - `metrics` (metric_name, metric_value, timestamp, context)

   The default local DB path is `tabula_rasa.db` at the repo root (this repo already includes DB files for examples). Branches may change the DB backend (Postgres, etc.), but must provide a migration plan and keep the baseline schema compatible or provide a translation layer.

- Minimal mode behavior: The `--minimal` or `tb-minimal` branch should still require a DB and the real ARC3 API unless explicitly running in CI with the opt-in stub flag. Minimal mode can use the `DummyDetector` or a frame shim (for low-resource runs), but all game/session/state writes and learned artifacts must be persisted to the DB, not just logged to stdout.

---

Keep the seed small but operationally strict: the goal is to avoid silent drift when teams run experiments with different assumptions about persistence or API correctness.


---

Keep this seed small. Child branches will add additional documentation describing their behavior.
