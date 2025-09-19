-- TABULA RASA DATABASE SCHEMA
-- Comprehensive SQLite database for ARC-AGI-3 training system
-- Generated from current tabula_rasa.db structure
-- Last updated: $(date)

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

-- Coordinate intelligence tracking
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

-- Action tracking for pattern analysis
CREATE TABLE IF NOT EXISTS action_tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    action_type TEXT,
    action_sequence TEXT, -- JSON array of actions
    effectiveness REAL,
    context TEXT, -- JSON context data
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Coordinate tracking for spatial analysis
CREATE TABLE IF NOT EXISTS coordinate_tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    coordinate_x INTEGER,
    coordinate_y INTEGER,
    action_type TEXT,
    success BOOLEAN,
    context TEXT, -- JSON context data
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

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

-- Winning sequences analysis
CREATE TABLE IF NOT EXISTS winning_sequences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    sequence TEXT NOT NULL, -- JSON array of action numbers
    frequency INTEGER DEFAULT 1,
    avg_score REAL DEFAULT 0.0,
    success_rate REAL DEFAULT 1.0,
    last_used TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(game_id, sequence)
);

-- ============================================================================
-- PERFORMANCE AND ANALYTICS TABLES
-- ============================================================================

-- Performance history tracking
CREATE TABLE IF NOT EXISTS performance_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    game_id TEXT,
    score REAL,
    win_rate REAL,
    learning_efficiency REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT, -- JSON metadata
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Session history tracking
CREATE TABLE IF NOT EXISTS session_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    game_id TEXT,
    status TEXT,
    duration_seconds INTEGER,
    actions_taken INTEGER,
    score REAL,
    win BOOLEAN,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT, -- JSON metadata
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Score history tracking
CREATE TABLE IF NOT EXISTS score_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    session_id TEXT,
    score REAL,
    score_type TEXT, -- 'current', 'best', 'average', etc.
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Frame tracking for stagnation detection
CREATE TABLE IF NOT EXISTS frame_tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    frame_hash TEXT,
    frame_analysis TEXT, -- JSON analysis data
    stagnation_detected BOOLEAN,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- SYSTEM MANAGEMENT TABLES
-- ============================================================================

-- System logs
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

-- Global counters for system state
CREATE TABLE IF NOT EXISTS global_counters (
    counter_name TEXT PRIMARY KEY,
    counter_value INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

-- Error logging with deduplication
CREATE TABLE IF NOT EXISTS error_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    error_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    error_hash TEXT NOT NULL UNIQUE,
    stack_trace TEXT,
    context TEXT,
    occurrence_count INTEGER DEFAULT 1,
    first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
    resolved BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- CONFIGURATION AND EXPERIMENT TABLES
-- ============================================================================

-- Reward cap configuration
CREATE TABLE IF NOT EXISTS reward_cap_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_key TEXT NOT NULL UNIQUE,
    config_value TEXT NOT NULL,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,        
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Experiment tracking
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name TEXT NOT NULL,
    experiment_type TEXT NOT NULL,
    parameters TEXT NOT NULL, -- JSON experiment parameters
    results TEXT NOT NULL, -- JSON experiment results
    success BOOLEAN DEFAULT FALSE,
    duration REAL DEFAULT 0.0, -- Duration in seconds
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Task performance tracking
CREATE TABLE IF NOT EXISTS task_performance (
    task_id TEXT PRIMARY KEY,
    training_sessions TEXT NOT NULL, -- JSON performance data
    learning_progress TEXT NOT NULL, -- JSON learning progress
    success_rate REAL DEFAULT 0.0,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- DIRECTOR SELF-MODEL TABLE
-- ============================================================================

-- Director self-model persistence
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

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Training sessions indexes
CREATE INDEX IF NOT EXISTS idx_training_sessions_mode ON training_sessions(mode);
CREATE INDEX IF NOT EXISTS idx_training_sessions_status ON training_sessions(status);
CREATE INDEX IF NOT EXISTS idx_training_sessions_start_time ON training_sessions(start_time);

-- Game results indexes
CREATE INDEX IF NOT EXISTS idx_game_results_session_id ON game_results(session_id);
CREATE INDEX IF NOT EXISTS idx_game_results_status ON game_results(status);
CREATE INDEX IF NOT EXISTS idx_game_results_final_score ON game_results(final_score);

-- Action effectiveness indexes
CREATE INDEX IF NOT EXISTS idx_action_effectiveness_game_id ON action_effectiveness(game_id);
CREATE INDEX IF NOT EXISTS idx_action_effectiveness_action_number ON action_effectiveness(action_number);
CREATE INDEX IF NOT EXISTS idx_action_effectiveness_success_rate ON action_effectiveness(success_rate);

-- Coordinate intelligence indexes
CREATE INDEX IF NOT EXISTS idx_coordinate_intelligence_game_id ON coordinate_intelligence(game_id);
CREATE INDEX IF NOT EXISTS idx_coordinate_intelligence_coords ON coordinate_intelligence(x, y);
CREATE INDEX IF NOT EXISTS idx_coordinate_intelligence_success_rate ON coordinate_intelligence(success_rate);

-- Action traces indexes
CREATE INDEX IF NOT EXISTS idx_action_traces_session_id ON action_traces(session_id);
CREATE INDEX IF NOT EXISTS idx_action_traces_game_id ON action_traces(game_id);
CREATE INDEX IF NOT EXISTS idx_action_traces_timestamp ON action_traces(timestamp);

-- System logs indexes
CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(log_level);
CREATE INDEX IF NOT EXISTS idx_system_logs_component ON system_logs(component);
CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_logs_session_id ON system_logs(session_id);

-- Performance history indexes
CREATE INDEX IF NOT EXISTS idx_performance_history_session_id ON performance_history(session_id);
CREATE INDEX IF NOT EXISTS idx_performance_history_timestamp ON performance_history(timestamp);

-- Learned patterns indexes
CREATE INDEX IF NOT EXISTS idx_learned_patterns_type ON learned_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_learned_patterns_confidence ON learned_patterns(confidence);
CREATE INDEX IF NOT EXISTS idx_learned_patterns_success_rate ON learned_patterns(success_rate);

-- Director self-model indexes
CREATE INDEX IF NOT EXISTS idx_director_self_model_type ON director_self_model(type);
CREATE INDEX IF NOT EXISTS idx_director_self_model_importance ON director_self_model(importance);
CREATE INDEX IF NOT EXISTS idx_director_self_model_created_at ON director_self_model(created_at);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Recent training performance view
CREATE VIEW IF NOT EXISTS recent_training_performance AS
SELECT 
    ts.session_id,
    ts.mode,
    ts.status,
    ts.total_actions,
    ts.total_wins,
    ts.total_games,
    ts.win_rate,
    ts.avg_score,
    ts.start_time,
    ts.end_time
FROM training_sessions ts
WHERE ts.start_time >= datetime('now', '-7 days')
ORDER BY ts.start_time DESC;

-- Action effectiveness summary view
CREATE VIEW IF NOT EXISTS action_effectiveness_summary AS
SELECT 
    action_number,
    COUNT(*) as game_count,
    SUM(attempts) as total_attempts,
    SUM(successes) as total_successes,
    AVG(success_rate) as avg_success_rate,
    AVG(avg_score_impact) as avg_score_impact
FROM action_effectiveness
GROUP BY action_number
ORDER BY avg_success_rate DESC;

-- Coordinate intelligence summary view
CREATE VIEW IF NOT EXISTS coordinate_intelligence_summary AS
SELECT 
    x,
    y,
    COUNT(*) as game_count,
    SUM(attempts) as total_attempts,
    SUM(successes) as total_successes,
    AVG(success_rate) as avg_success_rate
FROM coordinate_intelligence
GROUP BY x, y
ORDER BY avg_success_rate DESC;

-- System health view
CREATE VIEW IF NOT EXISTS system_health AS
SELECT 
    component,
    log_level,
    COUNT(*) as log_count,
    MAX(timestamp) as last_log
FROM system_logs
WHERE timestamp >= datetime('now', '-1 hour')
GROUP BY component, log_level
ORDER BY component, log_level;

-- ============================================================================
-- GAN SYSTEM TABLES
-- ============================================================================

-- GAN training sessions and performance tracking
CREATE TABLE IF NOT EXISTS gan_training_sessions (
    session_id TEXT PRIMARY KEY,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'running',
    generator_loss REAL DEFAULT 0.0,
    discriminator_loss REAL DEFAULT 0.0,
    pattern_accuracy REAL DEFAULT 0.0,
    synthetic_quality_score REAL DEFAULT 0.0,
    total_generated_states INTEGER DEFAULT 0,
    total_training_steps INTEGER DEFAULT 0,
    convergence_epoch INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Generated synthetic game states
CREATE TABLE IF NOT EXISTS gan_generated_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    game_id TEXT,
    state_data TEXT NOT NULL, -- JSON encoded game state
    pattern_context TEXT, -- JSON encoded pattern context
    quality_score REAL DEFAULT 0.0,
    discriminator_score REAL DEFAULT 0.0,
    pattern_consistency_score REAL DEFAULT 0.0,
    generation_method TEXT DEFAULT 'gan',
    is_validated BOOLEAN DEFAULT FALSE,
    validation_score REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES gan_training_sessions(session_id)
);

-- GAN model checkpoints and weights
CREATE TABLE IF NOT EXISTS gan_model_checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    generator_weights TEXT NOT NULL, -- JSON encoded model weights
    discriminator_weights TEXT NOT NULL, -- JSON encoded model weights
    generator_loss REAL NOT NULL,
    discriminator_loss REAL NOT NULL,
    pattern_accuracy REAL NOT NULL,
    synthetic_quality REAL NOT NULL,
    model_size_bytes INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES gan_training_sessions(session_id)
);

-- GAN pattern learning integration
CREATE TABLE IF NOT EXISTS gan_pattern_learning (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    pattern_id TEXT NOT NULL,
    pattern_type TEXT NOT NULL, -- 'visual', 'action', 'reasoning'
    synthetic_generation_count INTEGER DEFAULT 0,
    pattern_accuracy REAL DEFAULT 0.0,
    learning_effectiveness REAL DEFAULT 0.0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES gan_training_sessions(session_id)
);

-- GAN reverse engineering results
CREATE TABLE IF NOT EXISTS gan_reverse_engineering (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    game_id TEXT NOT NULL,
    discovered_rules TEXT, -- JSON encoded discovered game rules
    rule_confidence REAL DEFAULT 0.0,
    rule_accuracy REAL DEFAULT 0.0,
    mechanics_understood REAL DEFAULT 0.0,
    reverse_engineering_method TEXT DEFAULT 'adversarial_learning',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES gan_training_sessions(session_id)
);

-- GAN synthetic data validation results
CREATE TABLE IF NOT EXISTS gan_validation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    generated_state_id INTEGER NOT NULL,
    validation_type TEXT NOT NULL, -- 'pattern_consistency', 'game_logic', 'visual_quality'
    validation_score REAL NOT NULL,
    validation_details TEXT, -- JSON encoded validation details
    is_passed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES gan_training_sessions(session_id),
    FOREIGN KEY (generated_state_id) REFERENCES gan_generated_states(id)
);

-- GAN performance metrics and analytics
CREATE TABLE IF NOT EXISTS gan_performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metric_type TEXT NOT NULL, -- 'loss', 'accuracy', 'quality', 'convergence'
    epoch INTEGER DEFAULT 0,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES gan_training_sessions(session_id)
);

-- GAN indexes for performance
CREATE INDEX IF NOT EXISTS idx_gan_generated_states_session ON gan_generated_states(session_id);
CREATE INDEX IF NOT EXISTS idx_gan_generated_states_quality ON gan_generated_states(quality_score);
CREATE INDEX IF NOT EXISTS idx_gan_model_checkpoints_session ON gan_model_checkpoints(session_id);
CREATE INDEX IF NOT EXISTS idx_gan_pattern_learning_session ON gan_pattern_learning(session_id);
CREATE INDEX IF NOT EXISTS idx_gan_reverse_engineering_session ON gan_reverse_engineering(session_id);
CREATE INDEX IF NOT EXISTS idx_gan_performance_metrics_session ON gan_performance_metrics(session_id);