-- TABULA RASA DATABASE SCHEMA
-- Comprehensive SQLite database for ARC-AGI-3 training system
-- Designed for heavy usage with optimized indexing and relationships

-- ============================================================================
-- CORE SYSTEM TABLES
-- ============================================================================

-- System sessions and training runs
CREATE TABLE IF NOT EXISTS training_sessions (
    session_id TEXT PRIMARY KEY,
    game_id TEXT, -- Optional game_id for session context
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    mode TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    total_actions INTEGER DEFAULT 0,
    total_wins INTEGER DEFAULT 0,
    total_games INTEGER DEFAULT 0,
    win_rate REAL DEFAULT 0.0,
    avg_score REAL DEFAULT 0.0,
    energy_level REAL DEFAULT 100.0,
    memory_operations INTEGER DEFAULT 0,
    sleep_cycles INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Individual game results within sessions
CREATE TABLE IF NOT EXISTS game_results (
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    status TEXT NOT NULL, -- 'completed', 'failed', 'timeout', 'cancelled'
    final_score REAL DEFAULT 0.0,
    total_actions INTEGER DEFAULT 0,
    actions_taken TEXT, -- JSON array of action numbers
    win_detected BOOLEAN DEFAULT FALSE,
    level_completions INTEGER DEFAULT 0,
    frame_changes INTEGER DEFAULT 0,
    coordinate_attempts INTEGER DEFAULT 0,
    coordinate_successes INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (game_id, session_id),
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- ============================================================================
-- ACTION INTELLIGENCE TABLES
-- ============================================================================

-- Action effectiveness tracking
CREATE TABLE IF NOT EXISTS action_effectiveness (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    action_number INTEGER NOT NULL,
    attempts INTEGER DEFAULT 0,
    successes INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0.0,
    avg_score_impact REAL DEFAULT 0.0,
    last_used TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(game_id, action_number)
);

-- Coordinate intelligence for Action 6
CREATE TABLE IF NOT EXISTS coordinate_intelligence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    attempts INTEGER DEFAULT 0,
    successes INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0.0,
    frame_changes INTEGER DEFAULT 0,
    last_used TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(game_id, x, y)
);

-- Note: winning_sequences table was removed (data now in learned_patterns)

-- Note: action_transitions table was removed (data now in learned_patterns)

-- ============================================================================
-- LEARNING AND PATTERN TABLES
-- ============================================================================

-- Learned patterns and strategies
CREATE TABLE IF NOT EXISTS learned_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_type TEXT NOT NULL, -- 'coordinate', 'sequence', 'strategy', 'heuristic'
    pattern_data TEXT NOT NULL, -- JSON data
    confidence REAL DEFAULT 0.0,
    frequency INTEGER DEFAULT 1,
    success_rate REAL DEFAULT 0.0,
    game_context TEXT, -- Game ID or pattern context
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Meta-learning session data
CREATE TABLE IF NOT EXISTS training_sessions (
    session_id TEXT PRIMARY KEY,
    learning_type TEXT NOT NULL, -- 'coordinate_optimization', 'action_sequencing', 'strategy_refinement'
    input_data TEXT, -- JSON input data
    output_data TEXT, -- JSON output/learned data
    improvement_metrics TEXT, -- JSON performance metrics
    success BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- LOGGING AND TRACING TABLES
-- ============================================================================

-- Action traces for detailed analysis
CREATE TABLE IF NOT EXISTS action_traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    game_id TEXT NOT NULL,
    action_number INTEGER DEFAULT 0, -- Allow 0 for unknown actions
    coordinates TEXT, -- JSON coordinates for Action 6
    timestamp TIMESTAMP NOT NULL,
    frame_before TEXT, -- JSON frame data
    frame_after TEXT, -- JSON frame data
    frame_changed BOOLEAN DEFAULT FALSE,
    score_before REAL DEFAULT 0.0,
    score_after REAL DEFAULT 0.0,
    score_change REAL DEFAULT 0.0,
    response_data TEXT, -- JSON API response
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- System logs with structured data
CREATE TABLE IF NOT EXISTS system_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    log_level TEXT NOT NULL, -- 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    component TEXT NOT NULL, -- 'governor', 'architect', 'director', 'learning_loop'
    message TEXT NOT NULL,
    data TEXT, -- JSON additional data
    session_id TEXT,
    game_id TEXT,
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- Alias for backward compatibility
CREATE VIEW IF NOT EXISTS logs AS SELECT * FROM system_logs;

-- System logs (replaces governor_decisions and other log tables)
CREATE TABLE IF NOT EXISTS system_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    log_level TEXT NOT NULL,
    component TEXT NOT NULL,
    message TEXT NOT NULL,
    data TEXT, -- JSON data
    session_id TEXT,
    game_id TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- ARCHITECTURE AND EVOLUTION TABLES
-- ============================================================================

-- System architecture evolution
CREATE TABLE IF NOT EXISTS system_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation INTEGER NOT NULL,
    evolution_type TEXT NOT NULL, -- 'mutation', 'crossover', 'optimization'
    changes TEXT NOT NULL, -- JSON changes made
    performance_impact REAL DEFAULT 0.0,
    success BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Note: experiments and task_performance tables were removed (data now in system_logs)

-- ============================================================================
-- SYSTEM STATE AND MONITORING TABLES
-- ============================================================================

-- Global system counters and state
CREATE TABLE IF NOT EXISTS global_counters (
    counter_name TEXT PRIMARY KEY,
    counter_value INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

-- System performance metrics
CREATE TABLE IF NOT EXISTS training_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metric_type TEXT NOT NULL, -- 'counter', 'rate', 'percentage', 'score'
    session_id TEXT,
    game_id TEXT,
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- Reset debug information
CREATE TABLE IF NOT EXISTS system_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    game_id TEXT NOT NULL,
    reset_reason TEXT NOT NULL,
    debug_data TEXT NOT NULL, -- JSON debug information
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Primary performance indexes
CREATE INDEX IF NOT EXISTS idx_game_results_session ON game_results(session_id);
CREATE INDEX IF NOT EXISTS idx_game_results_game ON game_results(game_id);
CREATE INDEX IF NOT EXISTS idx_game_results_status ON game_results(status);
CREATE INDEX IF NOT EXISTS idx_game_results_timestamp ON game_results(start_time);

CREATE INDEX IF NOT EXISTS idx_action_effectiveness_game ON action_effectiveness(game_id);
CREATE INDEX IF NOT EXISTS idx_action_effectiveness_action ON action_effectiveness(action_number);
CREATE INDEX IF NOT EXISTS idx_action_effectiveness_success_rate ON action_effectiveness(success_rate);

CREATE INDEX IF NOT EXISTS idx_coordinate_intelligence_game ON coordinate_intelligence(game_id);
CREATE INDEX IF NOT EXISTS idx_coordinate_intelligence_coords ON coordinate_intelligence(x, y);
CREATE INDEX IF NOT EXISTS idx_coordinate_intelligence_success ON coordinate_intelligence(success_rate);

CREATE INDEX IF NOT EXISTS idx_action_traces_session ON action_traces(session_id);
CREATE INDEX IF NOT EXISTS idx_action_traces_game ON action_traces(game_id);
CREATE INDEX IF NOT EXISTS idx_action_traces_timestamp ON action_traces(timestamp);

CREATE INDEX IF NOT EXISTS idx_system_logs_component ON system_logs(component);
CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(log_level);
CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp);

CREATE INDEX IF NOT EXISTS idx_system_logs_session ON system_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_system_logs_component ON system_logs(component);
CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp);

-- Performance optimization indexes
CREATE INDEX IF NOT EXISTS idx_training_sessions_status ON training_sessions(status);
CREATE INDEX IF NOT EXISTS idx_training_sessions_mode ON training_sessions(mode);
CREATE INDEX IF NOT EXISTS idx_training_sessions_start_time ON training_sessions(start_time);

CREATE INDEX IF NOT EXISTS idx_learned_patterns_type ON learned_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_learned_patterns_confidence ON learned_patterns(confidence);
CREATE INDEX IF NOT EXISTS idx_learned_patterns_success_rate ON learned_patterns(success_rate);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Real-time system status view
CREATE VIEW IF NOT EXISTS system_status AS
SELECT 
    ts.session_id,
    ts.mode,
    ts.status,
    ts.total_actions,
    ts.total_wins,
    ts.total_games,
    ts.win_rate,
    ts.avg_score,
    ts.energy_level,
    ts.memory_operations,
    ts.sleep_cycles,
    ts.start_time,
    ts.updated_at
FROM training_sessions ts
WHERE ts.status = 'running'
ORDER BY ts.updated_at DESC;

-- Action effectiveness summary view
CREATE VIEW IF NOT EXISTS action_effectiveness_summary AS
SELECT 
    action_number,
    COUNT(*) as game_count,
    SUM(attempts) as total_attempts,
    SUM(successes) as total_successes,
    AVG(success_rate) as avg_success_rate,
    MAX(updated_at) as last_updated
FROM action_effectiveness
GROUP BY action_number
ORDER BY avg_success_rate DESC;

-- Coordinate intelligence summary view
CREATE VIEW IF NOT EXISTS coordinate_intelligence_summary AS
SELECT 
    game_id,
    COUNT(*) as coordinate_count,
    SUM(attempts) as total_attempts,
    SUM(successes) as total_successes,
    AVG(success_rate) as avg_success_rate,
    MAX(updated_at) as last_updated
FROM coordinate_intelligence
GROUP BY game_id
ORDER BY avg_success_rate DESC;

-- Recent performance trends view
CREATE VIEW IF NOT EXISTS recent_performance AS
SELECT 
    DATE(ts.start_time) as date,
    COUNT(*) as sessions,
    AVG(ts.win_rate) as avg_win_rate,
    AVG(ts.avg_score) as avg_score,
    SUM(ts.total_actions) as total_actions,
    SUM(ts.total_wins) as total_wins
FROM training_sessions ts
WHERE ts.start_time >= datetime('now', '-7 days')
GROUP BY DATE(ts.start_time)
ORDER BY date DESC;

-- Director Self-Model Persistence
CREATE TABLE IF NOT EXISTS director_self_model (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    type TEXT NOT NULL CHECK (type IN ('identity', 'trait', 'memory', 'reflection')),
    content TEXT NOT NULL,
    session_id INTEGER,
    importance INTEGER DEFAULT 1 CHECK (importance >= 1 AND importance <= 5),
    metadata TEXT DEFAULT '{}',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for director_self_model
CREATE INDEX IF NOT EXISTS idx_director_self_model_type ON director_self_model(type);
CREATE INDEX IF NOT EXISTS idx_director_self_model_session ON director_self_model(session_id);
CREATE INDEX IF NOT EXISTS idx_director_self_model_importance ON director_self_model(importance);
CREATE INDEX IF NOT EXISTS idx_director_self_model_created ON director_self_model(created_at);

-- View for recent self-model entries
CREATE VIEW IF NOT EXISTS recent_self_model AS
SELECT 
    type,
    content,
    importance,
    session_id,
    created_at,
    json_extract(metadata, '$.insight_type') as insight_type
FROM director_self_model
WHERE created_at >= datetime('now', '-7 days')
ORDER BY created_at DESC, importance DESC;
