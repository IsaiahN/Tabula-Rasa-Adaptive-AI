-- Enhanced Systems Database Schema Updates

-- Pattern Detection System Tables
CREATE TABLE IF NOT EXISTS detected_patterns (
    pattern_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    pattern_type TEXT NOT NULL,
    pattern_data TEXT NOT NULL,  -- JSON
    confidence REAL NOT NULL,
    detection_timestamp TEXT NOT NULL,
    metadata TEXT,  -- JSON
    FOREIGN KEY (game_id) REFERENCES games(id)
);

CREATE INDEX IF NOT EXISTS idx_detected_patterns_game ON detected_patterns(game_id);
CREATE INDEX IF NOT EXISTS idx_detected_patterns_type ON detected_patterns(pattern_type);

-- Causal Analysis System Tables
CREATE TABLE IF NOT EXISTS action_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    action_id INTEGER NOT NULL,
    coordinates TEXT,  -- JSON [x, y]
    score_change REAL NOT NULL,
    before_state TEXT NOT NULL,  -- JSON
    after_state TEXT NOT NULL,  -- JSON
    timestamp TEXT NOT NULL,
    FOREIGN KEY (game_id) REFERENCES games(id)
);

CREATE TABLE IF NOT EXISTS causal_relations (
    game_id TEXT NOT NULL,
    relation_id TEXT PRIMARY KEY,
    cause_type TEXT NOT NULL,
    cause_data TEXT NOT NULL,  -- JSON
    effect_type TEXT NOT NULL,
    effect_data TEXT NOT NULL,  -- JSON
    confidence REAL NOT NULL,
    support_count INTEGER NOT NULL,
    last_observed TEXT NOT NULL,
    context_conditions TEXT,  -- JSON
    FOREIGN KEY (game_id) REFERENCES games(id)
);

CREATE INDEX IF NOT EXISTS idx_action_outcomes_game ON action_outcomes(game_id);
CREATE INDEX IF NOT EXISTS idx_causal_relations_game ON causal_relations(game_id);
CREATE INDEX IF NOT EXISTS idx_causal_relations_confidence ON causal_relations(confidence);

-- Knowledge Transfer System Tables
CREATE TABLE IF NOT EXISTS transferable_patterns (
    pattern_id TEXT PRIMARY KEY,
    pattern_type TEXT NOT NULL,
    features TEXT NOT NULL,  -- JSON
    abstraction_level INTEGER NOT NULL,
    confidence REAL NOT NULL,
    success_rate REAL NOT NULL,
    usage_count INTEGER NOT NULL,
    last_used TEXT NOT NULL,
    game_ids TEXT NOT NULL  -- JSON array
);

CREATE TABLE IF NOT EXISTS adaptation_rules (
    rule_id TEXT PRIMARY KEY,
    condition TEXT NOT NULL,  -- JSON
    transformation TEXT NOT NULL,  -- JSON
    success_rate REAL NOT NULL,
    usage_count INTEGER NOT NULL,
    last_used TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_transferable_patterns_type ON transferable_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_transferable_patterns_confidence ON transferable_patterns(confidence);
CREATE INDEX IF NOT EXISTS idx_adaptation_rules_success ON adaptation_rules(success_rate);

-- Strategic Learning System Tables
CREATE TABLE IF NOT EXISTS strategies (
    strategy_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    components TEXT NOT NULL,  -- JSON
    state_conditions TEXT NOT NULL,  -- JSON
    action_weights TEXT NOT NULL,  -- JSON
    success_rate REAL NOT NULL,
    confidence REAL NOT NULL,
    usage_count INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    last_updated TEXT
);

CREATE TABLE IF NOT EXISTS strategy_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id TEXT NOT NULL,
    success INTEGER NOT NULL,  -- 0 or 1
    score_change REAL NOT NULL,
    action_sequence TEXT NOT NULL,  -- JSON array
    state_changes TEXT NOT NULL,  -- JSON
    duration REAL NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
);

CREATE INDEX IF NOT EXISTS idx_strategies_success ON strategies(success_rate);
CREATE INDEX IF NOT EXISTS idx_strategies_confidence ON strategies(confidence);
CREATE INDEX IF NOT EXISTS idx_strategy_outcomes_strategy ON strategy_outcomes(strategy_id);

-- Integration Support Tables
CREATE TABLE IF NOT EXISTS system_stats (
    stat_id TEXT PRIMARY KEY,
    system_name TEXT NOT NULL,
    stat_name TEXT NOT NULL,
    stat_value REAL NOT NULL,
    last_updated TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS system_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    system_name TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_data TEXT NOT NULL,  -- JSON
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_system_stats_name ON system_stats(system_name, stat_name);
CREATE INDEX IF NOT EXISTS idx_system_events_type ON system_events(system_name, event_type);