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

-- Enhanced coordinate penalty and decay system
CREATE TABLE IF NOT EXISTS coordinate_penalties (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    penalty_score REAL DEFAULT 0.0,
    penalty_reason TEXT DEFAULT 'no_improvement',
    zero_progress_streak INTEGER DEFAULT 0,
    last_penalty_applied TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_success TIMESTAMP,
    decay_rate REAL DEFAULT 0.1,
    is_stuck_coordinate BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(game_id, x, y)
);

-- Coordinate diversity tracking
CREATE TABLE IF NOT EXISTS coordinate_diversity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    usage_frequency INTEGER DEFAULT 1,
    avoidance_score REAL DEFAULT 0.0,
    recent_attempts TEXT, -- JSON array of recent attempts
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(game_id, x, y)
);

-- Failure learning system
CREATE TABLE IF NOT EXISTS failure_learning (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    coordinate_x INTEGER,
    coordinate_y INTEGER,
    action_type TEXT,
    failure_type TEXT NOT NULL, -- 'no_improvement', 'score_decrease', 'stuck_loop'
    failure_context TEXT, -- JSON context data
    failure_count INTEGER DEFAULT 1,
    last_failure TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    recovery_attempts INTEGER DEFAULT 0,
    learned_insights TEXT, -- JSON insights learned from failure
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    pattern_type TEXT NOT NULL, -- 'coordinate', 'sequence', 'strategy', 'heuristic', 'game_type_profile', 'button_priority'                                                                    
    pattern_data TEXT NOT NULL, -- JSON data
    confidence REAL DEFAULT 0.0,
    frequency INTEGER DEFAULT 1,
    success_rate REAL DEFAULT 0.0,
    game_context TEXT, -- Game ID or pattern context
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Button priorities for Action 6 centric games
CREATE TABLE IF NOT EXISTS button_priorities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_type TEXT NOT NULL,
    coordinate_x INTEGER NOT NULL,
    coordinate_y INTEGER NOT NULL,
    button_type TEXT NOT NULL, -- 'score_button', 'action_button', 'visual_button'
    confidence REAL DEFAULT 0.0,
    success_count INTEGER DEFAULT 1,
    score_changes INTEGER DEFAULT 0,
    action_unlocks INTEGER DEFAULT 0,
    test_count INTEGER DEFAULT 0,
    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(game_type, coordinate_x, coordinate_y, button_type)
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

-- GAN training data - Frame-Action pairs for GAN training
CREATE TABLE IF NOT EXISTS gan_training_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    session_id TEXT,
    action_number INTEGER NOT NULL,
    coordinates TEXT, -- JSON format [x, y] for action 6
    previous_frame TEXT, -- JSON compressed frame data
    current_frame TEXT, -- JSON compressed frame data
    frame_changes TEXT, -- JSON array of change coordinates
    score_change INTEGER DEFAULT 0,
    is_button_candidate BOOLEAN DEFAULT FALSE,
    button_candidate_confidence REAL DEFAULT 0.0,
    frame_comparison_data TEXT, -- JSON metadata about frame comparison
    timestamp REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- ============================================================================
-- ADVANCED ACTION SYSTEM TABLES
-- ============================================================================

-- Strategy discovery and replication system
CREATE TABLE IF NOT EXISTS winning_strategies (
    strategy_id TEXT PRIMARY KEY,
    game_type TEXT NOT NULL,
    game_id TEXT NOT NULL,
    action_sequence TEXT NOT NULL, -- JSON array of action numbers
    score_progression TEXT NOT NULL, -- JSON array of score values
    total_score_increase REAL NOT NULL,
    efficiency REAL NOT NULL, -- Score per action
    discovery_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    replication_attempts INTEGER DEFAULT 0,
    successful_replications INTEGER DEFAULT 0,
    refinement_level INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Strategy refinement tracking
CREATE TABLE IF NOT EXISTS strategy_refinements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id TEXT NOT NULL,
    refinement_attempt INTEGER NOT NULL,
    original_efficiency REAL NOT NULL,
    new_efficiency REAL NOT NULL,
    improvement REAL NOT NULL,
    action_sequence TEXT NOT NULL, -- JSON array of refined actions
    refinement_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (strategy_id) REFERENCES winning_strategies(strategy_id)
);

-- Strategy replication attempts
CREATE TABLE IF NOT EXISTS strategy_replications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id TEXT NOT NULL,
    game_id TEXT NOT NULL,
    replication_attempt INTEGER NOT NULL,
    expected_efficiency REAL NOT NULL,
    actual_efficiency REAL,
    success BOOLEAN DEFAULT FALSE,
    replication_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (strategy_id) REFERENCES winning_strategies(strategy_id)
);

-- Win condition analysis and tracking
CREATE TABLE IF NOT EXISTS win_conditions (
    condition_id TEXT PRIMARY KEY,
    game_type TEXT NOT NULL,
    game_id TEXT,
    condition_type TEXT NOT NULL, -- 'action_pattern', 'score_threshold', 'sequence_timing', 'level_completion'
    condition_data TEXT NOT NULL, -- JSON data describing the condition
    frequency INTEGER DEFAULT 1,   -- How often this condition leads to wins
    success_rate REAL DEFAULT 1.0, -- Success rate when this condition is met
    first_observed REAL,           -- Timestamp when first observed
    last_observed REAL,            -- Timestamp when last observed
    total_games_observed INTEGER DEFAULT 1,
    strategy_id TEXT,              -- Link to associated strategy if any
    created_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (strategy_id) REFERENCES winning_strategies(strategy_id)
);

-- Advanced stagnation detection
CREATE TABLE IF NOT EXISTS stagnation_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    stagnation_type TEXT NOT NULL, -- 'score_regression', 'action_repetition', 'no_frame_changes', 'coordinate_stuck'
    severity REAL NOT NULL, -- 0.0 to 1.0
    consecutive_count INTEGER NOT NULL,
    stagnation_context TEXT, -- JSON encoded context data
    recovery_action TEXT, -- Action taken to recover
    recovery_successful BOOLEAN DEFAULT FALSE,
    detection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- Frame change analysis and classification
CREATE TABLE IF NOT EXISTS frame_change_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    action_number INTEGER NOT NULL,
    coordinates_x INTEGER,
    coordinates_y INTEGER,
    change_type TEXT NOT NULL, -- 'major_movement', 'object_movement', 'small_movement', 'visual_change', 'minor_change'
    num_pixels_changed INTEGER NOT NULL,
    change_percentage REAL NOT NULL,
    movement_detected BOOLEAN DEFAULT FALSE,
    change_locations TEXT, -- JSON array of (x,y) coordinates
    classification_confidence REAL DEFAULT 0.0,
    analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Systematic exploration phases
CREATE TABLE IF NOT EXISTS exploration_phases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    phase_name TEXT NOT NULL, -- 'corners', 'center', 'edges', 'random'
    phase_attempts INTEGER DEFAULT 0,
    successful_attempts INTEGER DEFAULT 0,
    coordinates_tried TEXT, -- JSON array of (x,y) coordinates
    phase_start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    phase_end_time TIMESTAMP,
    phase_success_rate REAL DEFAULT 0.0,
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- Emergency override events
CREATE TABLE IF NOT EXISTS emergency_overrides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    override_type TEXT NOT NULL, -- 'action_loop_break', 'coordinate_stuck_break', 'stagnation_break'
    trigger_reason TEXT NOT NULL,
    actions_before_override INTEGER NOT NULL,
    override_action INTEGER NOT NULL,
    override_successful BOOLEAN DEFAULT FALSE,
    override_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- Visual-interactive Action6 targeting
CREATE TABLE IF NOT EXISTS visual_targets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    target_x INTEGER NOT NULL,
    target_y INTEGER NOT NULL,
    target_type TEXT NOT NULL, -- 'button', 'object', 'anomaly', 'interactive_element'
    confidence REAL NOT NULL,
    detection_method TEXT NOT NULL, -- 'opencv', 'frame_analysis', 'pattern_matching'
    interaction_successful BOOLEAN DEFAULT FALSE,
    frame_changes_detected BOOLEAN DEFAULT FALSE,
    score_impact REAL DEFAULT 0.0,
    detection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Action effectiveness detailed tracking
CREATE TABLE IF NOT EXISTS action_effectiveness_detailed (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    action_number INTEGER NOT NULL,
    coordinates_x INTEGER,
    coordinates_y INTEGER,
    frame_changes INTEGER DEFAULT 0,
    movement_detected INTEGER DEFAULT 0,
    score_changes INTEGER DEFAULT 0,
    action_unlocks INTEGER DEFAULT 0,
    stagnation_breaks INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0.0,
    efficiency_score REAL DEFAULT 0.0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Governor integration for new features
CREATE TABLE IF NOT EXISTS governor_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    decision_type TEXT NOT NULL, -- 'stagnation_recovery', 'strategy_replication', 'emergency_override', 'exploration_phase'
    context_data TEXT NOT NULL, -- JSON encoded decision context
    governor_confidence REAL NOT NULL,
    decision_outcome TEXT, -- JSON encoded decision result
    decision_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- ============================================================================
-- INDEXES FOR NEW TABLES
-- ============================================================================

-- Strategy discovery indexes
CREATE INDEX IF NOT EXISTS idx_winning_strategies_game_type ON winning_strategies(game_type);
CREATE INDEX IF NOT EXISTS idx_winning_strategies_efficiency ON winning_strategies(efficiency);
CREATE INDEX IF NOT EXISTS idx_strategy_refinements_strategy ON strategy_refinements(strategy_id);
CREATE INDEX IF NOT EXISTS idx_strategy_replications_strategy ON strategy_replications(strategy_id);

-- Win conditions indexes
CREATE INDEX IF NOT EXISTS idx_win_conditions_game_type ON win_conditions(game_type);
CREATE INDEX IF NOT EXISTS idx_win_conditions_type ON win_conditions(condition_type);
CREATE INDEX IF NOT EXISTS idx_win_conditions_success_rate ON win_conditions(success_rate);
CREATE INDEX IF NOT EXISTS idx_win_conditions_strategy ON win_conditions(strategy_id);

-- Stagnation detection indexes
CREATE INDEX IF NOT EXISTS idx_stagnation_events_game ON stagnation_events(game_id);
CREATE INDEX IF NOT EXISTS idx_stagnation_events_type ON stagnation_events(stagnation_type);
CREATE INDEX IF NOT EXISTS idx_stagnation_events_timestamp ON stagnation_events(detection_timestamp);

-- Frame analysis indexes
CREATE INDEX IF NOT EXISTS idx_frame_change_analysis_game ON frame_change_analysis(game_id);
CREATE INDEX IF NOT EXISTS idx_frame_change_analysis_action ON frame_change_analysis(action_number);
CREATE INDEX IF NOT EXISTS idx_frame_change_analysis_type ON frame_change_analysis(change_type);

-- Exploration indexes
CREATE INDEX IF NOT EXISTS idx_exploration_phases_game ON exploration_phases(game_id);
CREATE INDEX IF NOT EXISTS idx_exploration_phases_phase ON exploration_phases(phase_name);

-- Emergency override indexes
CREATE INDEX IF NOT EXISTS idx_emergency_overrides_game ON emergency_overrides(game_id);
CREATE INDEX IF NOT EXISTS idx_emergency_overrides_type ON emergency_overrides(override_type);

-- Visual targeting indexes
CREATE INDEX IF NOT EXISTS idx_visual_targets_game ON visual_targets(game_id);
CREATE INDEX IF NOT EXISTS idx_visual_targets_coordinates ON visual_targets(target_x, target_y);
CREATE INDEX IF NOT EXISTS idx_visual_targets_type ON visual_targets(target_type);

-- Action effectiveness indexes
CREATE INDEX IF NOT EXISTS idx_action_effectiveness_game ON action_effectiveness_detailed(game_id);
CREATE INDEX IF NOT EXISTS idx_action_effectiveness_action ON action_effectiveness_detailed(action_number);

-- Governor decision indexes
CREATE INDEX IF NOT EXISTS idx_governor_decisions_session ON governor_decisions(session_id);
CREATE INDEX IF NOT EXISTS idx_governor_decisions_type ON governor_decisions(decision_type);

-- ============================================================================
-- LOSING STREAK DETECTION AND ANTI-PATTERN LEARNING TABLES
-- ============================================================================

-- Losing streak tracking for cross-attempt failure analysis
CREATE TABLE IF NOT EXISTS losing_streaks (
    streak_id TEXT PRIMARY KEY,
    game_type TEXT NOT NULL,
    game_id TEXT NOT NULL,
    level_identifier TEXT, -- For games with distinct levels
    consecutive_failures INTEGER NOT NULL DEFAULT 1,
    total_attempts INTEGER NOT NULL DEFAULT 1,
    first_failure_timestamp REAL NOT NULL,
    last_failure_timestamp REAL NOT NULL,
    failure_types TEXT, -- JSON array of failure types (timeout, score_stagnation, etc.)
    escalation_level INTEGER DEFAULT 0, -- 0=none, 1=mild, 2=moderate, 3=aggressive
    last_escalation_timestamp REAL,
    intervention_attempts INTEGER DEFAULT 0,
    successful_intervention BOOLEAN DEFAULT FALSE,
    streak_broken BOOLEAN DEFAULT FALSE,
    break_timestamp REAL,
    break_method TEXT, -- Description of what finally worked
    created_at REAL DEFAULT (strftime('%s', 'now')),
    updated_at REAL DEFAULT (strftime('%s', 'now'))
);

-- Anti-pattern learning to identify consistently failing approaches
CREATE TABLE IF NOT EXISTS anti_patterns (
    pattern_id TEXT PRIMARY KEY,
    game_type TEXT NOT NULL,
    game_id TEXT,
    pattern_type TEXT NOT NULL, -- 'action_sequence', 'coordinate_cluster', 'timing_pattern', 'score_approach'
    pattern_data TEXT NOT NULL, -- JSON data describing the anti-pattern
    failure_count INTEGER DEFAULT 1,
    total_encounters INTEGER DEFAULT 1,
    failure_rate REAL DEFAULT 1.0,
    first_observed REAL NOT NULL,
    last_observed REAL NOT NULL,
    severity REAL DEFAULT 0.5, -- 0.0 to 1.0, how bad this pattern is
    confidence REAL DEFAULT 0.5, -- 0.0 to 1.0, confidence in pattern identification
    context_data TEXT, -- JSON context when pattern occurs
    alternative_suggestions TEXT, -- JSON array of suggested alternatives
    created_at REAL DEFAULT (strftime('%s', 'now')),
    updated_at REAL DEFAULT (strftime('%s', 'now'))
);

-- Escalated intervention tracking
CREATE TABLE IF NOT EXISTS escalated_interventions (
    intervention_id TEXT PRIMARY KEY,
    streak_id TEXT NOT NULL,
    game_id TEXT NOT NULL,
    escalation_level INTEGER NOT NULL,
    intervention_type TEXT NOT NULL, -- 'randomization', 'exploration_boost', 'strategy_override', 'pattern_avoidance'
    intervention_data TEXT NOT NULL, -- JSON data describing the intervention
    applied_timestamp REAL NOT NULL,
    success BOOLEAN DEFAULT FALSE,
    outcome_data TEXT, -- JSON data about the outcome
    duration_seconds REAL,
    recovery_actions INTEGER DEFAULT 0,
    created_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (streak_id) REFERENCES losing_streaks(streak_id)
);

-- ============================================================================
-- REAL-TIME LEARNING ENGINE TABLES (Phase 1.1)
-- ============================================================================

-- Real-time pattern detection during gameplay
CREATE TABLE IF NOT EXISTS real_time_patterns (
    pattern_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    pattern_type TEXT NOT NULL, -- 'emerging_sequence', 'coordinate_cluster', 'score_momentum', 'action_effectiveness'
    pattern_data TEXT NOT NULL, -- JSON data describing the detected pattern
    confidence REAL DEFAULT 0.0, -- Confidence in pattern detection (0.0 to 1.0)
    detection_timestamp REAL NOT NULL,
    game_action_count INTEGER NOT NULL, -- Number of actions taken when pattern was detected
    pattern_strength REAL DEFAULT 0.0, -- How strong/clear the pattern is
    immediate_feedback TEXT, -- JSON immediate feedback data
    pattern_evolution TEXT, -- JSON tracking how pattern has evolved
    is_active BOOLEAN DEFAULT TRUE,
    last_updated REAL DEFAULT (strftime('%s', 'now')),
    created_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- Dynamic strategy adjustments made during gameplay
CREATE TABLE IF NOT EXISTS real_time_strategy_adjustments (
    adjustment_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    trigger_pattern_id TEXT, -- Pattern that triggered this adjustment
    adjustment_type TEXT NOT NULL, -- 'action_priority_change', 'coordinate_focus_shift', 'exploration_boost', 'pattern_avoidance'
    adjustment_data TEXT NOT NULL, -- JSON data describing the adjustment
    applied_at_action INTEGER NOT NULL, -- Game action count when adjustment was applied
    immediate_effect TEXT, -- JSON immediate effect observed
    effectiveness_score REAL DEFAULT 0.0, -- How effective was this adjustment
    duration_actions INTEGER DEFAULT 0, -- How many actions the adjustment remained active
    success BOOLEAN DEFAULT FALSE,
    created_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id),
    FOREIGN KEY (trigger_pattern_id) REFERENCES real_time_patterns(pattern_id)
);

-- Action outcome tracking for immediate learning
CREATE TABLE IF NOT EXISTS action_outcome_tracking (
    outcome_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    action_number INTEGER NOT NULL,
    coordinates_x INTEGER,
    coordinates_y INTEGER,
    action_timestamp REAL NOT NULL,
    score_before REAL NOT NULL,
    score_after REAL NOT NULL,
    score_delta REAL NOT NULL,
    frame_changes_detected BOOLEAN DEFAULT FALSE,
    movement_detected BOOLEAN DEFAULT FALSE,
    new_elements_detected BOOLEAN DEFAULT FALSE,
    immediate_classification TEXT NOT NULL, -- 'highly_effective', 'effective', 'neutral', 'negative', 'harmful'
    confidence_level REAL DEFAULT 0.0,
    context_data TEXT, -- JSON context at time of action
    learning_triggers TEXT, -- JSON array of learning events this outcome triggered
    created_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- Mid-game learning events and insights
CREATE TABLE IF NOT EXISTS mid_game_learning_events (
    event_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL, -- 'pattern_discovery', 'strategy_insight', 'coordinate_revelation', 'sequence_breakthrough'
    event_data TEXT NOT NULL, -- JSON data describing the learning event
    trigger_action INTEGER NOT NULL, -- Action count that triggered the learning
    confidence REAL DEFAULT 0.0,
    immediate_application BOOLEAN DEFAULT FALSE, -- Was this insight immediately applied?
    application_data TEXT, -- JSON data about how insight was applied
    validation_actions INTEGER DEFAULT 0, -- Actions taken to validate the insight
    validation_success BOOLEAN DEFAULT FALSE,
    insight_value REAL DEFAULT 0.0, -- Estimated value of this insight (0.0 to 1.0)
    created_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- Real-time attention focus tracking
CREATE TABLE IF NOT EXISTS real_time_attention_focus (
    focus_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    focus_type TEXT NOT NULL, -- 'coordinate_region', 'action_sequence', 'score_optimization', 'pattern_completion'
    focus_data TEXT NOT NULL, -- JSON data describing current focus
    focus_start_action INTEGER NOT NULL,
    focus_end_action INTEGER,
    focus_duration_actions INTEGER DEFAULT 0,
    focus_intensity REAL DEFAULT 0.5, -- How intensely focused (0.0 to 1.0)
    focus_effectiveness REAL DEFAULT 0.0, -- How effective was this focus
    interruption_count INTEGER DEFAULT 0,
    interruption_reasons TEXT, -- JSON array of reasons focus was interrupted
    created_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- Real-time hypothesis formation and testing
CREATE TABLE IF NOT EXISTS real_time_hypotheses (
    hypothesis_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    hypothesis_type TEXT NOT NULL, -- 'action_effect', 'coordinate_behavior', 'sequence_outcome', 'score_mechanism'
    hypothesis_statement TEXT NOT NULL, -- Natural language description of hypothesis
    hypothesis_data TEXT NOT NULL, -- JSON formal hypothesis data
    formation_action INTEGER NOT NULL, -- Action count when hypothesis was formed
    confidence REAL DEFAULT 0.5,
    test_actions TEXT, -- JSON array of actions taken to test hypothesis
    test_results TEXT, -- JSON results of hypothesis testing
    status TEXT DEFAULT 'active', -- 'active', 'confirmed', 'refuted', 'inconclusive'
    confirmation_score REAL DEFAULT 0.0,
    created_at REAL DEFAULT (strftime('%s', 'now')),
    updated_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- ============================================================================
-- INDEXES FOR REAL-TIME LEARNING TABLES
-- ============================================================================

-- Real-time patterns indexes
CREATE INDEX IF NOT EXISTS idx_real_time_patterns_game ON real_time_patterns(game_id);
CREATE INDEX IF NOT EXISTS idx_real_time_patterns_type ON real_time_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_real_time_patterns_confidence ON real_time_patterns(confidence);
CREATE INDEX IF NOT EXISTS idx_real_time_patterns_active ON real_time_patterns(is_active);

-- Strategy adjustments indexes
CREATE INDEX IF NOT EXISTS idx_real_time_adjustments_game ON real_time_strategy_adjustments(game_id);
CREATE INDEX IF NOT EXISTS idx_real_time_adjustments_type ON real_time_strategy_adjustments(adjustment_type);
CREATE INDEX IF NOT EXISTS idx_real_time_adjustments_success ON real_time_strategy_adjustments(success);

-- Action outcome tracking indexes
CREATE INDEX IF NOT EXISTS idx_action_outcomes_game ON action_outcome_tracking(game_id);
CREATE INDEX IF NOT EXISTS idx_action_outcomes_action ON action_outcome_tracking(action_number);
CREATE INDEX IF NOT EXISTS idx_action_outcomes_classification ON action_outcome_tracking(immediate_classification);

-- Mid-game learning events indexes
CREATE INDEX IF NOT EXISTS idx_mid_game_learning_game ON mid_game_learning_events(game_id);
CREATE INDEX IF NOT EXISTS idx_mid_game_learning_type ON mid_game_learning_events(event_type);
CREATE INDEX IF NOT EXISTS idx_mid_game_learning_confidence ON mid_game_learning_events(confidence);

-- Attention focus indexes
CREATE INDEX IF NOT EXISTS idx_attention_focus_game ON real_time_attention_focus(game_id);
CREATE INDEX IF NOT EXISTS idx_attention_focus_type ON real_time_attention_focus(focus_type);
CREATE INDEX IF NOT EXISTS idx_attention_focus_effectiveness ON real_time_attention_focus(focus_effectiveness);

-- Real-time hypotheses indexes
CREATE INDEX IF NOT EXISTS idx_real_time_hypotheses_game ON real_time_hypotheses(game_id);
CREATE INDEX IF NOT EXISTS idx_real_time_hypotheses_type ON real_time_hypotheses(hypothesis_type);
CREATE INDEX IF NOT EXISTS idx_real_time_hypotheses_status ON real_time_hypotheses(status);

-- ============================================================================
-- CAUSAL MODEL BUILDER TABLES (Phase 1.2)
-- ============================================================================

-- Causal relationships: "action X in context Y causes outcome Z"
CREATE TABLE IF NOT EXISTS causal_relationships (
    relationship_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    action_type TEXT NOT NULL, -- 'single_action', 'action_sequence', 'coordinate_action'
    action_data TEXT NOT NULL, -- JSON data describing the action(s)
    context_type TEXT NOT NULL, -- 'game_state', 'score_range', 'level_context', 'pattern_context'
    context_data TEXT NOT NULL, -- JSON data describing the context
    outcome_type TEXT NOT NULL, -- 'score_change', 'state_transition', 'unlock_event', 'level_completion'
    outcome_data TEXT NOT NULL, -- JSON data describing the outcome
    causality_strength REAL DEFAULT 0.0, -- 0.0 to 1.0, strength of causal relationship
    confidence_level REAL DEFAULT 0.0, -- 0.0 to 1.0, confidence in this relationship
    observation_count INTEGER DEFAULT 1, -- How many times this relationship was observed
    success_count INTEGER DEFAULT 0, -- How many times the expected outcome occurred
    failure_count INTEGER DEFAULT 0, -- How many times the expected outcome did NOT occur
    first_observed REAL NOT NULL,
    last_observed REAL NOT NULL,
    context_specificity REAL DEFAULT 0.5, -- How specific/general the context is
    transferability_score REAL DEFAULT 0.5, -- How transferable this relationship is to other situations
    created_at REAL DEFAULT (strftime('%s', 'now')),
    updated_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- Causal chains: sequences of causal relationships
CREATE TABLE IF NOT EXISTS causal_chains (
    chain_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    chain_type TEXT NOT NULL, -- 'linear_sequence', 'branching_tree', 'feedback_loop'
    chain_data TEXT NOT NULL, -- JSON array of relationship_ids in sequence
    chain_length INTEGER NOT NULL, -- Number of relationships in chain
    chain_strength REAL DEFAULT 0.0, -- Overall strength of the causal chain
    chain_reliability REAL DEFAULT 0.0, -- How reliably this chain produces expected results
    start_context TEXT NOT NULL, -- JSON initial context
    end_outcome TEXT NOT NULL, -- JSON final outcome
    intermediate_steps TEXT, -- JSON array of intermediate states
    validation_attempts INTEGER DEFAULT 0,
    successful_validations INTEGER DEFAULT 0,
    discovered_timestamp REAL NOT NULL,
    last_validated REAL,
    practical_value REAL DEFAULT 0.0, -- How valuable this chain is for achieving goals
    created_at REAL DEFAULT (strftime('%s', 'now')),
    updated_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- Context patterns that influence causal relationships
CREATE TABLE IF NOT EXISTS causal_contexts (
    context_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    context_type TEXT NOT NULL, -- 'score_threshold', 'game_state_pattern', 'action_history', 'temporal_context'
    context_signature TEXT NOT NULL, -- Unique signature identifying this context
    context_data TEXT NOT NULL, -- JSON detailed context information
    influence_strength REAL DEFAULT 0.0, -- How strongly this context influences outcomes
    occurrence_frequency REAL DEFAULT 0.0, -- How often this context occurs
    associated_relationships TEXT, -- JSON array of relationship_ids influenced by this context
    predictive_power REAL DEFAULT 0.0, -- How well this context predicts outcomes
    generalization_level REAL DEFAULT 0.5, -- How general vs specific this context is
    first_observed REAL NOT NULL,
    last_observed REAL NOT NULL,
    observation_count INTEGER DEFAULT 1,
    created_at REAL DEFAULT (strftime('%s', 'now')),
    updated_at REAL DEFAULT (strftime('%s', 'now'))
);

-- Outcome patterns and their characteristics
CREATE TABLE IF NOT EXISTS causal_outcomes (
    outcome_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    outcome_type TEXT NOT NULL, -- 'immediate_effect', 'delayed_effect', 'compound_effect', 'emergent_effect'
    outcome_signature TEXT NOT NULL, -- Unique signature identifying this outcome
    outcome_data TEXT NOT NULL, -- JSON detailed outcome information
    magnitude REAL DEFAULT 0.0, -- Strength/size of the outcome
    duration_estimate REAL DEFAULT 0.0, -- How long the effect lasts (in actions)
    reversibility BOOLEAN DEFAULT TRUE, -- Whether this outcome can be undone
    prerequisite_contexts TEXT, -- JSON array of contexts that enable this outcome
    associated_relationships TEXT, -- JSON array of relationship_ids that produce this outcome
    desirability_score REAL DEFAULT 0.0, -- How desirable this outcome is (-1.0 to 1.0)
    predictability REAL DEFAULT 0.0, -- How predictable this outcome is
    first_observed REAL NOT NULL,
    last_observed REAL NOT NULL,
    observation_count INTEGER DEFAULT 1,
    created_at REAL DEFAULT (strftime('%s', 'now')),
    updated_at REAL DEFAULT (strftime('%s', 'now'))
);

-- Causal experiments and their results
CREATE TABLE IF NOT EXISTS causal_experiments (
    experiment_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    experiment_type TEXT NOT NULL, -- 'hypothesis_test', 'causal_intervention', 'counterfactual_test'
    hypothesis_relationship_id TEXT, -- Relationship being tested
    experimental_context TEXT NOT NULL, -- JSON context where experiment was conducted
    experimental_action TEXT NOT NULL, -- JSON action(s) taken during experiment
    predicted_outcome TEXT NOT NULL, -- JSON what we expected to happen
    actual_outcome TEXT NOT NULL, -- JSON what actually happened
    experiment_success BOOLEAN DEFAULT FALSE, -- Whether prediction matched reality
    confidence_before REAL DEFAULT 0.5, -- Confidence before experiment
    confidence_after REAL DEFAULT 0.5, -- Confidence after experiment
    learning_gained REAL DEFAULT 0.0, -- How much was learned from this experiment
    experiment_timestamp REAL NOT NULL,
    analysis_notes TEXT, -- JSON notes about the experiment and results
    follow_up_experiments TEXT, -- JSON array of experiment_ids for follow-ups
    created_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id),
    FOREIGN KEY (hypothesis_relationship_id) REFERENCES causal_relationships(relationship_id)
);

-- Causal insights derived from the model
CREATE TABLE IF NOT EXISTS causal_insights (
    insight_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    insight_type TEXT NOT NULL, -- 'causal_discovery', 'pattern_generalization', 'strategy_implication', 'contradiction_resolution'
    insight_description TEXT NOT NULL, -- Natural language description of the insight
    insight_data TEXT NOT NULL, -- JSON formal representation of the insight
    supporting_relationships TEXT, -- JSON array of relationship_ids supporting this insight
    supporting_experiments TEXT, -- JSON array of experiment_ids supporting this insight
    confidence_level REAL DEFAULT 0.0,
    practical_importance REAL DEFAULT 0.0, -- How important this insight is for gameplay
    generalization_scope TEXT, -- JSON description of where this insight applies
    derived_timestamp REAL NOT NULL,
    validation_status TEXT DEFAULT 'unvalidated', -- 'unvalidated', 'validated', 'refuted'
    validation_data TEXT, -- JSON validation results
    application_count INTEGER DEFAULT 0, -- How many times this insight has been applied
    application_success_rate REAL DEFAULT 0.0,
    created_at REAL DEFAULT (strftime('%s', 'now')),
    updated_at REAL DEFAULT (strftime('%s', 'now'))
);

-- ============================================================================
-- INDEXES FOR CAUSAL MODEL BUILDER TABLES
-- ============================================================================

-- Causal relationships indexes
CREATE INDEX IF NOT EXISTS idx_causal_relationships_game ON causal_relationships(game_id);
CREATE INDEX IF NOT EXISTS idx_causal_relationships_action_type ON causal_relationships(action_type);
CREATE INDEX IF NOT EXISTS idx_causal_relationships_context_type ON causal_relationships(context_type);
CREATE INDEX IF NOT EXISTS idx_causal_relationships_outcome_type ON causal_relationships(outcome_type);
CREATE INDEX IF NOT EXISTS idx_causal_relationships_strength ON causal_relationships(causality_strength);
CREATE INDEX IF NOT EXISTS idx_causal_relationships_confidence ON causal_relationships(confidence_level);

-- Causal chains indexes
CREATE INDEX IF NOT EXISTS idx_causal_chains_game ON causal_chains(game_id);
CREATE INDEX IF NOT EXISTS idx_causal_chains_type ON causal_chains(chain_type);
CREATE INDEX IF NOT EXISTS idx_causal_chains_strength ON causal_chains(chain_strength);
CREATE INDEX IF NOT EXISTS idx_causal_chains_reliability ON causal_chains(chain_reliability);

-- Causal contexts indexes
CREATE INDEX IF NOT EXISTS idx_causal_contexts_game ON causal_contexts(game_id);
CREATE INDEX IF NOT EXISTS idx_causal_contexts_type ON causal_contexts(context_type);
CREATE INDEX IF NOT EXISTS idx_causal_contexts_influence ON causal_contexts(influence_strength);

-- Causal outcomes indexes
CREATE INDEX IF NOT EXISTS idx_causal_outcomes_game ON causal_outcomes(game_id);
CREATE INDEX IF NOT EXISTS idx_causal_outcomes_type ON causal_outcomes(outcome_type);
CREATE INDEX IF NOT EXISTS idx_causal_outcomes_desirability ON causal_outcomes(desirability_score);

-- Causal experiments indexes
CREATE INDEX IF NOT EXISTS idx_causal_experiments_game ON causal_experiments(game_id);
CREATE INDEX IF NOT EXISTS idx_causal_experiments_type ON causal_experiments(experiment_type);
CREATE INDEX IF NOT EXISTS idx_causal_experiments_success ON causal_experiments(experiment_success);

-- Causal insights indexes
CREATE INDEX IF NOT EXISTS idx_causal_insights_game ON causal_insights(game_id);
CREATE INDEX IF NOT EXISTS idx_causal_insights_type ON causal_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_causal_insights_confidence ON causal_insights(confidence_level);
CREATE INDEX IF NOT EXISTS idx_causal_insights_importance ON causal_insights(practical_importance);

-- ============================================================================
-- ENHANCED ATTENTION + COMMUNICATION SYSTEM TABLES (TIER 1)
-- ============================================================================

-- Attention allocation tracking for computational resource management
CREATE TABLE IF NOT EXISTS attention_allocations (
    allocation_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    subsystem_name TEXT NOT NULL, -- 'real_time_learning', 'strategy_discovery', 'losing_streak_detection', etc.
    allocated_priority REAL NOT NULL, -- 0.0 to 1.0, how much attention this subsystem gets
    processing_load REAL NOT NULL, -- Current computational load estimate
    requested_priority REAL DEFAULT 0.5, -- What the subsystem requested
    allocation_timestamp REAL NOT NULL,
    allocation_duration REAL DEFAULT 0.0, -- How long this allocation lasted
    effectiveness_score REAL DEFAULT 0.0, -- How effective this allocation was
    resource_utilization REAL DEFAULT 0.0, -- How well resources were used
    context_data TEXT, -- JSON context that influenced this allocation
    created_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- Communication pathway management for inter-subsystem messaging
CREATE TABLE IF NOT EXISTS communication_pathways (
    pathway_id TEXT PRIMARY KEY,
    sender_system TEXT NOT NULL, -- Which subsystem is sending
    receiver_system TEXT NOT NULL, -- Which subsystem is receiving
    message_type TEXT NOT NULL, -- Type of communication
    pathway_weight REAL DEFAULT 1.0, -- Weight/importance of this pathway (0.0 to 2.0)
    speed_multiplier REAL DEFAULT 1.0, -- "Myelination" factor for faster transmission
    priority_level INTEGER DEFAULT 1, -- 1=low, 2=medium, 3=high, 4=critical
    message_count INTEGER DEFAULT 0, -- Total messages sent through this pathway
    success_count INTEGER DEFAULT 0, -- Successfully processed messages
    failure_count INTEGER DEFAULT 0, -- Failed message processing
    average_latency REAL DEFAULT 0.0, -- Average processing time
    last_weight_adjustment REAL, -- When weight was last adjusted
    last_message_timestamp REAL, -- When last message was sent
    pathway_effectiveness REAL DEFAULT 0.0, -- Overall effectiveness score
    auto_adjust_enabled BOOLEAN DEFAULT TRUE, -- Whether to auto-adjust weights
    created_at REAL DEFAULT (strftime('%s', 'now')),
    updated_at REAL DEFAULT (strftime('%s', 'now'))
);

-- Resource usage monitoring for subsystem performance tracking
CREATE TABLE IF NOT EXISTS resource_usage_monitoring (
    usage_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    subsystem_name TEXT NOT NULL,
    cpu_usage_estimate REAL DEFAULT 0.0, -- Estimated CPU usage (0.0 to 1.0)
    memory_usage_estimate REAL DEFAULT 0.0, -- Estimated memory usage in MB
    processing_time REAL DEFAULT 0.0, -- Time taken for last operation in seconds
    queue_depth INTEGER DEFAULT 0, -- Number of pending operations
    throughput_rate REAL DEFAULT 0.0, -- Operations per second
    error_rate REAL DEFAULT 0.0, -- Error rate (0.0 to 1.0)
    performance_impact REAL DEFAULT 0.0, -- Impact on overall system performance
    bottleneck_detected BOOLEAN DEFAULT FALSE, -- Whether this subsystem is a bottleneck
    monitoring_timestamp REAL NOT NULL,
    monitoring_duration REAL DEFAULT 1.0, -- Duration of this monitoring period
    optimization_suggestions TEXT, -- JSON suggestions for optimization
    created_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- Communication message logs for debugging and analysis
CREATE TABLE IF NOT EXISTS communication_message_logs (
    message_id TEXT PRIMARY KEY,
    pathway_id TEXT NOT NULL,
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    sender_system TEXT NOT NULL,
    receiver_system TEXT NOT NULL,
    message_type TEXT NOT NULL,
    message_data TEXT NOT NULL, -- JSON message content
    message_priority INTEGER DEFAULT 1,
    sent_timestamp REAL NOT NULL,
    received_timestamp REAL,
    processed_timestamp REAL,
    processing_success BOOLEAN DEFAULT FALSE,
    processing_error TEXT, -- Error message if processing failed
    latency_ms REAL DEFAULT 0.0, -- Total processing latency
    message_size_bytes INTEGER DEFAULT 0,
    created_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (pathway_id) REFERENCES communication_pathways(pathway_id),
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- Attention controller decisions and reasoning
CREATE TABLE IF NOT EXISTS attention_controller_decisions (
    decision_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    decision_type TEXT NOT NULL, -- 'resource_allocation', 'priority_adjustment', 'load_balancing'
    decision_timestamp REAL NOT NULL,
    current_context TEXT NOT NULL, -- JSON game context that influenced decision
    subsystem_demands TEXT NOT NULL, -- JSON demands from each subsystem
    allocation_strategy TEXT NOT NULL, -- JSON decided allocation strategy
    reasoning_data TEXT, -- JSON explanation of decision reasoning
    predicted_outcome TEXT, -- JSON what we expect to happen
    actual_outcome TEXT, -- JSON what actually happened (filled later)
    decision_effectiveness REAL DEFAULT 0.0, -- How effective this decision was
    adjustment_magnitude REAL DEFAULT 0.0, -- How big a change this was
    created_at REAL DEFAULT (strftime('%s', 'now')),
    updated_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- ============================================================================
-- INDEXES FOR ATTENTION + COMMUNICATION SYSTEM TABLES
-- ============================================================================

-- Attention allocation indexes
CREATE INDEX IF NOT EXISTS idx_attention_allocations_game ON attention_allocations(game_id);
CREATE INDEX IF NOT EXISTS idx_attention_allocations_subsystem ON attention_allocations(subsystem_name);
CREATE INDEX IF NOT EXISTS idx_attention_allocations_priority ON attention_allocations(allocated_priority);
CREATE INDEX IF NOT EXISTS idx_attention_allocations_effectiveness ON attention_allocations(effectiveness_score);
CREATE INDEX IF NOT EXISTS idx_attention_allocations_timestamp ON attention_allocations(allocation_timestamp);

-- Communication pathway indexes
CREATE INDEX IF NOT EXISTS idx_communication_pathways_sender ON communication_pathways(sender_system);
CREATE INDEX IF NOT EXISTS idx_communication_pathways_receiver ON communication_pathways(receiver_system);
CREATE INDEX IF NOT EXISTS idx_communication_pathways_type ON communication_pathways(message_type);
CREATE INDEX IF NOT EXISTS idx_communication_pathways_weight ON communication_pathways(pathway_weight);
CREATE INDEX IF NOT EXISTS idx_communication_pathways_effectiveness ON communication_pathways(pathway_effectiveness);

-- Resource usage monitoring indexes
CREATE INDEX IF NOT EXISTS idx_resource_usage_game ON resource_usage_monitoring(game_id);
CREATE INDEX IF NOT EXISTS idx_resource_usage_subsystem ON resource_usage_monitoring(subsystem_name);
CREATE INDEX IF NOT EXISTS idx_resource_usage_timestamp ON resource_usage_monitoring(monitoring_timestamp);
CREATE INDEX IF NOT EXISTS idx_resource_usage_bottleneck ON resource_usage_monitoring(bottleneck_detected);

-- Communication message logs indexes
CREATE INDEX IF NOT EXISTS idx_message_logs_pathway ON communication_message_logs(pathway_id);
CREATE INDEX IF NOT EXISTS idx_message_logs_game ON communication_message_logs(game_id);
CREATE INDEX IF NOT EXISTS idx_message_logs_sender ON communication_message_logs(sender_system);
CREATE INDEX IF NOT EXISTS idx_message_logs_receiver ON communication_message_logs(receiver_system);
CREATE INDEX IF NOT EXISTS idx_message_logs_timestamp ON communication_message_logs(sent_timestamp);
CREATE INDEX IF NOT EXISTS idx_message_logs_success ON communication_message_logs(processing_success);

-- Attention controller decisions indexes
CREATE INDEX IF NOT EXISTS idx_attention_decisions_game ON attention_controller_decisions(game_id);
CREATE INDEX IF NOT EXISTS idx_attention_decisions_type ON attention_controller_decisions(decision_type);
CREATE INDEX IF NOT EXISTS idx_attention_decisions_timestamp ON attention_controller_decisions(decision_timestamp);
CREATE INDEX IF NOT EXISTS idx_attention_decisions_effectiveness ON attention_controller_decisions(decision_effectiveness);

-- ============================================================================
-- CONTEXT-DEPENDENT FITNESS EVOLUTION SYSTEM TABLES (TIER 2)
-- ============================================================================

-- Dynamic fitness criteria that evolve based on learning context
CREATE TABLE IF NOT EXISTS fitness_criteria_evolution (
    criteria_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    criteria_type TEXT NOT NULL, -- 'exploration_depth', 'pattern_discovery', 'strategy_innovation', 'learning_efficiency', 'adaptation_speed'
    criteria_name TEXT NOT NULL, -- Human-readable name for this criteria
    criteria_definition TEXT NOT NULL, -- JSON definition of how to measure this criteria
    base_weight REAL DEFAULT 1.0, -- Base importance weight (0.0 to 2.0)
    current_weight REAL DEFAULT 1.0, -- Current evolved weight
    context_modifiers TEXT, -- JSON context-based weight adjustments
    evolution_history TEXT, -- JSON history of how this criteria has evolved
    performance_correlation REAL DEFAULT 0.0, -- How well this criteria correlates with success
    last_weight_update REAL NOT NULL,
    criteria_effectiveness REAL DEFAULT 0.5, -- How effective this criteria is at driving improvement
    adaptation_rate REAL DEFAULT 0.1, -- How quickly this criteria can evolve (0.0 to 1.0)
    stability_threshold REAL DEFAULT 0.05, -- Minimum change needed to update weight
    created_at REAL DEFAULT (strftime('%s', 'now')),
    updated_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- Context-aware fitness evaluations with dynamic criteria
CREATE TABLE IF NOT EXISTS contextual_fitness_evaluations (
    evaluation_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    evaluation_timestamp REAL NOT NULL,
    context_snapshot TEXT NOT NULL, -- JSON snapshot of current learning context
    fitness_criteria_used TEXT NOT NULL, -- JSON array of criteria_ids and their weights used
    individual_scores TEXT NOT NULL, -- JSON object mapping criteria_id to score
    composite_fitness_score REAL NOT NULL, -- Final weighted fitness score
    context_type TEXT NOT NULL, -- 'early_exploration', 'pattern_learning', 'strategy_optimization', 'mastery_phase'
    learning_phase TEXT NOT NULL, -- 'initialization', 'exploration', 'exploitation', 'refinement', 'mastery'
    performance_indicators TEXT, -- JSON performance metrics that influenced this evaluation
    predicted_improvement_areas TEXT, -- JSON areas where improvement is most needed
    fitness_trend_analysis TEXT, -- JSON analysis of fitness evolution trend
    evaluation_confidence REAL DEFAULT 0.5, -- Confidence in this evaluation (0.0 to 1.0)
    created_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- Fitness evolution triggers and environmental factors
CREATE TABLE IF NOT EXISTS fitness_evolution_triggers (
    trigger_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    trigger_type TEXT NOT NULL, -- 'performance_plateau', 'new_pattern_discovered', 'context_shift', 'learning_breakthrough'
    trigger_timestamp REAL NOT NULL,
    trigger_context TEXT NOT NULL, -- JSON context when trigger occurred
    affected_criteria TEXT NOT NULL, -- JSON array of criteria_ids that should be adjusted
    suggested_adjustments TEXT NOT NULL, -- JSON suggested weight/importance adjustments
    trigger_strength REAL DEFAULT 0.5, -- How strong this trigger is (0.0 to 1.0)
    automatic_adjustment BOOLEAN DEFAULT FALSE, -- Whether adjustment was applied automatically
    manual_review_required BOOLEAN DEFAULT FALSE, -- Whether human/advanced review is needed
    adjustment_applied BOOLEAN DEFAULT FALSE, -- Whether any adjustment was actually applied
    adjustment_results TEXT, -- JSON results of applying the adjustment
    trigger_effectiveness REAL DEFAULT 0.0, -- How effective this trigger was at improving performance
    created_at REAL DEFAULT (strftime('%s', 'now')),
    updated_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- Context-dependent fitness function definitions
CREATE TABLE IF NOT EXISTS adaptive_fitness_functions (
    function_id TEXT PRIMARY KEY,
    game_type TEXT NOT NULL,
    game_id TEXT,
    function_name TEXT NOT NULL, -- 'exploration_fitness', 'learning_fitness', 'efficiency_fitness', 'innovation_fitness'
    function_category TEXT NOT NULL, -- 'exploration', 'exploitation', 'adaptation', 'innovation'
    function_definition TEXT NOT NULL, -- JSON mathematical definition of the function
    input_parameters TEXT NOT NULL, -- JSON parameters this function requires
    output_range TEXT NOT NULL, -- JSON min/max output range
    context_applicability TEXT NOT NULL, -- JSON contexts where this function applies
    function_complexity REAL DEFAULT 0.5, -- Computational complexity (0.0 to 1.0)
    stability_over_time REAL DEFAULT 0.5, -- How stable this function's outputs are
    sensitivity_to_context REAL DEFAULT 0.5, -- How much context affects this function
    validation_results TEXT, -- JSON results from validating this function
    usage_frequency INTEGER DEFAULT 0, -- How often this function has been used
    average_effectiveness REAL DEFAULT 0.0, -- Average effectiveness when used
    last_used_timestamp REAL,
    created_at REAL DEFAULT (strftime('%s', 'now')),
    updated_at REAL DEFAULT (strftime('%s', 'now'))
);

-- Performance-based fitness adaptation learning
CREATE TABLE IF NOT EXISTS fitness_adaptation_learning (
    learning_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    adaptation_timestamp REAL NOT NULL,
    pre_adaptation_context TEXT NOT NULL, -- JSON context before adaptation
    post_adaptation_context TEXT NOT NULL, -- JSON context after adaptation
    criteria_adjustments TEXT NOT NULL, -- JSON what criteria weights were changed
    performance_before TEXT NOT NULL, -- JSON performance metrics before adaptation
    performance_after TEXT NOT NULL, -- JSON performance metrics after adaptation
    adaptation_type TEXT NOT NULL, -- 'weight_adjustment', 'criteria_addition', 'criteria_removal', 'function_swap'
    improvement_detected BOOLEAN DEFAULT FALSE, -- Whether adaptation led to improvement
    improvement_magnitude REAL DEFAULT 0.0, -- How much improvement occurred
    adaptation_confidence REAL DEFAULT 0.5, -- Confidence in this adaptation
    learning_insights TEXT, -- JSON insights gained from this adaptation
    rollback_recommended BOOLEAN DEFAULT FALSE, -- Whether this adaptation should be reversed
    stability_impact REAL DEFAULT 0.0, -- Impact on system stability (-1.0 to 1.0)
    generalization_potential REAL DEFAULT 0.5, -- How generalizable this adaptation is
    created_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- Multi-dimensional fitness optimization tracking
CREATE TABLE IF NOT EXISTS multidimensional_fitness_tracking (
    tracking_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    tracking_timestamp REAL NOT NULL,
    fitness_dimensions TEXT NOT NULL, -- JSON array of fitness dimensions being tracked
    dimension_scores TEXT NOT NULL, -- JSON mapping dimension to current score
    dimension_weights TEXT NOT NULL, -- JSON mapping dimension to current weight
    pareto_frontier_analysis TEXT, -- JSON analysis of pareto-optimal solutions
    trade_off_analysis TEXT, -- JSON analysis of trade-offs between dimensions
    optimization_direction TEXT NOT NULL, -- JSON preferred optimization direction for each dimension
    conflict_resolution_strategy TEXT, -- JSON strategy for resolving conflicting objectives
    convergence_status TEXT NOT NULL, -- 'diverging', 'converging', 'oscillating', 'stable'
    optimization_efficiency REAL DEFAULT 0.0, -- How efficiently we're optimizing
    dimension_correlation_matrix TEXT, -- JSON correlation matrix between dimensions
    prediction_accuracy REAL DEFAULT 0.0, -- How accurate our fitness predictions are
    created_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- Integration with attention system for fitness-driven resource allocation
CREATE TABLE IF NOT EXISTS fitness_attention_integration (
    integration_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    integration_timestamp REAL NOT NULL,
    current_fitness_priorities TEXT NOT NULL, -- JSON fitness criteria prioritization
    attention_allocation_request TEXT NOT NULL, -- JSON requested attention allocation based on fitness
    attention_allocation_received TEXT NOT NULL, -- JSON actual attention allocation received
    fitness_improvement_targets TEXT NOT NULL, -- JSON specific areas fitness system wants to improve
    resource_utilization_efficiency REAL DEFAULT 0.0, -- How efficiently allocated resources were used
    attention_fitness_correlation REAL DEFAULT 0.0, -- Correlation between attention and fitness improvement
    bottleneck_analysis TEXT, -- JSON analysis of attention bottlenecks affecting fitness
    optimization_suggestions TEXT, -- JSON suggestions for optimizing attention for fitness
    integration_effectiveness REAL DEFAULT 0.0, -- How effective this integration instance was
    created_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- ============================================================================
-- INDEXES FOR CONTEXT-DEPENDENT FITNESS EVOLUTION TABLES
-- ============================================================================

-- Fitness criteria evolution indexes
CREATE INDEX IF NOT EXISTS idx_fitness_criteria_game ON fitness_criteria_evolution(game_id);
CREATE INDEX IF NOT EXISTS idx_fitness_criteria_type ON fitness_criteria_evolution(criteria_type);
CREATE INDEX IF NOT EXISTS idx_fitness_criteria_weight ON fitness_criteria_evolution(current_weight);
CREATE INDEX IF NOT EXISTS idx_fitness_criteria_effectiveness ON fitness_criteria_evolution(criteria_effectiveness);
CREATE INDEX IF NOT EXISTS idx_fitness_criteria_updated ON fitness_criteria_evolution(last_weight_update);

-- Contextual fitness evaluations indexes
CREATE INDEX IF NOT EXISTS idx_contextual_fitness_game ON contextual_fitness_evaluations(game_id);
CREATE INDEX IF NOT EXISTS idx_contextual_fitness_timestamp ON contextual_fitness_evaluations(evaluation_timestamp);
CREATE INDEX IF NOT EXISTS idx_contextual_fitness_context_type ON contextual_fitness_evaluations(context_type);
CREATE INDEX IF NOT EXISTS idx_contextual_fitness_phase ON contextual_fitness_evaluations(learning_phase);
CREATE INDEX IF NOT EXISTS idx_contextual_fitness_score ON contextual_fitness_evaluations(composite_fitness_score);

-- Fitness evolution triggers indexes
CREATE INDEX IF NOT EXISTS idx_fitness_triggers_game ON fitness_evolution_triggers(game_id);
CREATE INDEX IF NOT EXISTS idx_fitness_triggers_type ON fitness_evolution_triggers(trigger_type);
CREATE INDEX IF NOT EXISTS idx_fitness_triggers_timestamp ON fitness_evolution_triggers(trigger_timestamp);
CREATE INDEX IF NOT EXISTS idx_fitness_triggers_strength ON fitness_evolution_triggers(trigger_strength);
CREATE INDEX IF NOT EXISTS idx_fitness_triggers_applied ON fitness_evolution_triggers(adjustment_applied);

-- Adaptive fitness functions indexes
CREATE INDEX IF NOT EXISTS idx_adaptive_fitness_game_type ON adaptive_fitness_functions(game_type);
CREATE INDEX IF NOT EXISTS idx_adaptive_fitness_category ON adaptive_fitness_functions(function_category);
CREATE INDEX IF NOT EXISTS idx_adaptive_fitness_effectiveness ON adaptive_fitness_functions(average_effectiveness);
CREATE INDEX IF NOT EXISTS idx_adaptive_fitness_usage ON adaptive_fitness_functions(usage_frequency);

-- Fitness adaptation learning indexes
CREATE INDEX IF NOT EXISTS idx_fitness_adaptation_game ON fitness_adaptation_learning(game_id);
CREATE INDEX IF NOT EXISTS idx_fitness_adaptation_timestamp ON fitness_adaptation_learning(adaptation_timestamp);
CREATE INDEX IF NOT EXISTS idx_fitness_adaptation_type ON fitness_adaptation_learning(adaptation_type);
CREATE INDEX IF NOT EXISTS idx_fitness_adaptation_improvement ON fitness_adaptation_learning(improvement_detected);

-- Multidimensional fitness tracking indexes
CREATE INDEX IF NOT EXISTS idx_multidim_fitness_game ON multidimensional_fitness_tracking(game_id);
CREATE INDEX IF NOT EXISTS idx_multidim_fitness_timestamp ON multidimensional_fitness_tracking(tracking_timestamp);
CREATE INDEX IF NOT EXISTS idx_multidim_fitness_convergence ON multidimensional_fitness_tracking(convergence_status);
CREATE INDEX IF NOT EXISTS idx_multidim_fitness_efficiency ON multidimensional_fitness_tracking(optimization_efficiency);

-- Fitness attention integration indexes
CREATE INDEX IF NOT EXISTS idx_fitness_attention_game ON fitness_attention_integration(game_id);
CREATE INDEX IF NOT EXISTS idx_fitness_attention_timestamp ON fitness_attention_integration(integration_timestamp);
CREATE INDEX IF NOT EXISTS idx_fitness_attention_effectiveness ON fitness_attention_integration(integration_effectiveness);
CREATE INDEX IF NOT EXISTS idx_fitness_attention_correlation ON fitness_attention_integration(attention_fitness_correlation);

-- ============================================================================
-- NEAT-BASED ARCHITECT SYSTEM TABLES WITH SAFETY GUARDRAILS (TIER 2)
-- ============================================================================

-- Architect system phase progression and permissions
CREATE TABLE IF NOT EXISTS architect_phases (
    phase_id TEXT PRIMARY KEY,
    phase_number INTEGER NOT NULL, -- 1=Learner's Permit, 2=Provisional License, 3=Full License
    phase_name TEXT NOT NULL, -- 'learners_permit', 'provisional_license', 'full_license'
    activation_timestamp REAL NOT NULL,
    activation_criteria TEXT NOT NULL, -- JSON criteria that triggered this phase
    permissions TEXT NOT NULL, -- JSON permissions granted in this phase
    restrictions TEXT NOT NULL, -- JSON restrictions still in place
    performance_requirements TEXT NOT NULL, -- JSON requirements to advance to next phase
    current_performance_metrics TEXT, -- JSON current performance against requirements
    phase_active BOOLEAN DEFAULT TRUE, -- Whether this phase is currently active
    advancement_eligible BOOLEAN DEFAULT FALSE, -- Whether ready to advance to next phase
    regression_count INTEGER DEFAULT 0, -- Number of times system regressed and had to restart phase
    human_override_reason TEXT, -- Reason if human manually changed phase
    created_at REAL DEFAULT (strftime('%s', 'now')),
    updated_at REAL DEFAULT (strftime('%s', 'now'))
);

-- Change proposals generated by the architect (Phase 1 & 2)
CREATE TABLE IF NOT EXISTS architect_change_proposals (
    proposal_id TEXT PRIMARY KEY,
    generation_timestamp REAL NOT NULL,
    architect_phase INTEGER NOT NULL, -- Which phase generated this proposal
    proposal_type TEXT NOT NULL, -- 'module_addition', 'module_modification', 'architecture_refactor', 'parameter_tuning'
    proposal_title TEXT NOT NULL, -- Human-readable title
    proposal_description TEXT NOT NULL, -- Detailed human-readable description
    proposed_changes TEXT NOT NULL, -- JSON specific code changes or architectural adjustments
    justification TEXT NOT NULL, -- Data-driven reasoning for the change
    supporting_data TEXT NOT NULL, -- JSON performance data supporting the proposal
    impact_analysis TEXT NOT NULL, -- JSON analysis of which subsystems will be affected
    rollback_plan TEXT NOT NULL, -- JSON detailed procedure to revert the change
    risk_assessment TEXT NOT NULL, -- JSON assessment of potential risks
    predicted_improvements TEXT NOT NULL, -- JSON predicted performance improvements
    estimated_implementation_time REAL DEFAULT 0.0, -- Hours estimated for implementation
    complexity_score REAL DEFAULT 0.5, -- Complexity rating (0.0 to 1.0)
    confidence_level REAL DEFAULT 0.5, -- Architect's confidence in this proposal (0.0 to 1.0)
    proposal_status TEXT DEFAULT 'pending', -- 'pending', 'approved', 'rejected', 'implemented', 'rolled_back'
    human_review_notes TEXT, -- Notes from human reviewer
    approval_timestamp REAL, -- When approved/rejected
    implementation_timestamp REAL, -- When implementation started
    completion_timestamp REAL, -- When implementation completed
    actual_performance_delta TEXT, -- JSON actual performance change after implementation
    created_at REAL DEFAULT (strftime('%s', 'now')),
    updated_at REAL DEFAULT (strftime('%s', 'now'))
);

-- Human feedback and approval workflow
CREATE TABLE IF NOT EXISTS architect_human_feedback (
    feedback_id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL,
    reviewer_id TEXT DEFAULT 'human_reviewer', -- Identifier for the human reviewer
    review_timestamp REAL NOT NULL,
    approval_decision TEXT NOT NULL, -- 'approved', 'rejected', 'needs_revision'
    feedback_text TEXT NOT NULL, -- Human-readable feedback
    concerns_raised TEXT, -- JSON specific concerns or issues identified
    suggestions TEXT, -- JSON suggestions for improvement
    confidence_in_decision REAL DEFAULT 0.8, -- Reviewer's confidence (0.0 to 1.0)
    time_spent_reviewing REAL DEFAULT 0.0, -- Minutes spent on review
    follow_up_required BOOLEAN DEFAULT FALSE, -- Whether additional review/discussion needed
    created_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (proposal_id) REFERENCES architect_change_proposals(proposal_id)
);

-- Stability metrics and regression detection
CREATE TABLE IF NOT EXISTS architect_stability_metrics (
    metric_id TEXT PRIMARY KEY,
    measurement_timestamp REAL NOT NULL,
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    metric_type TEXT NOT NULL, -- 'performance_consistency', 'error_rate', 'latency', 'memory_usage', 'win_rate_stability'
    baseline_value REAL NOT NULL, -- Baseline measurement before changes
    current_value REAL NOT NULL, -- Current measurement
    change_percentage REAL NOT NULL, -- Percentage change from baseline
    stability_score REAL NOT NULL, -- Overall stability score (0.0 to 1.0, higher is more stable)
    regression_detected BOOLEAN DEFAULT FALSE, -- Whether a regression was detected
    regression_severity TEXT, -- 'minor', 'moderate', 'major', 'critical'
    related_proposal_id TEXT, -- Change proposal that may have caused this change
    measurement_context TEXT, -- JSON context of measurement
    corrective_action_needed BOOLEAN DEFAULT FALSE, -- Whether corrective action is required
    created_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id),
    FOREIGN KEY (related_proposal_id) REFERENCES architect_change_proposals(proposal_id)
);

-- Performance benchmarks for tracking improvements/regressions
CREATE TABLE IF NOT EXISTS architect_performance_benchmarks (
    benchmark_id TEXT PRIMARY KEY,
    benchmark_name TEXT NOT NULL, -- 'game_completion_rate', 'average_score', 'learning_efficiency', etc.
    benchmark_category TEXT NOT NULL, -- 'learning', 'performance', 'stability', 'efficiency'
    measurement_timestamp REAL NOT NULL,
    game_type TEXT, -- Specific game type if applicable
    baseline_value REAL NOT NULL, -- Established baseline
    current_value REAL NOT NULL, -- Current measurement
    target_value REAL, -- Target improvement goal
    measurement_confidence REAL DEFAULT 0.8, -- Confidence in measurement accuracy
    trend_direction TEXT DEFAULT 'stable', -- 'improving', 'declining', 'stable', 'volatile'
    benchmark_context TEXT, -- JSON additional context
    last_improvement_timestamp REAL, -- When this benchmark last improved
    regression_threshold REAL DEFAULT 0.95, -- Ratio below baseline that triggers regression alert
    critical_threshold REAL DEFAULT 0.85, -- Ratio that triggers critical regression
    created_at REAL DEFAULT (strftime('%s', 'now'))
);

-- Rollback procedures and execution logs
CREATE TABLE IF NOT EXISTS architect_rollback_procedures (
    rollback_id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL,
    rollback_plan TEXT NOT NULL, -- JSON detailed rollback procedure
    rollback_trigger TEXT NOT NULL, -- What triggered the rollback
    execution_timestamp REAL,
    rollback_status TEXT DEFAULT 'planned', -- 'planned', 'executing', 'completed', 'failed'
    execution_steps TEXT, -- JSON steps taken during rollback
    pre_rollback_state TEXT, -- JSON system state before rollback
    post_rollback_state TEXT, -- JSON system state after rollback
    rollback_success BOOLEAN DEFAULT FALSE, -- Whether rollback was successful
    issues_encountered TEXT, -- JSON any issues during rollback
    verification_results TEXT, -- JSON verification that rollback worked correctly
    rollback_duration_seconds REAL DEFAULT 0.0,
    created_at REAL DEFAULT (strftime('%s', 'now')),
    updated_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (proposal_id) REFERENCES architect_change_proposals(proposal_id)
);

-- Big Red Button failsafe logs
CREATE TABLE IF NOT EXISTS architect_failsafe_logs (
    failsafe_id TEXT PRIMARY KEY,
    activation_timestamp REAL NOT NULL,
    failsafe_type TEXT NOT NULL, -- 'emergency_stop', 'phase_regression', 'system_rollback', 'full_reset'
    trigger_reason TEXT NOT NULL, -- Why the failsafe was activated
    activated_by TEXT NOT NULL, -- 'human_operator', 'automatic_system', 'regression_detector'
    system_state_before TEXT NOT NULL, -- JSON system state before activation
    actions_taken TEXT NOT NULL, -- JSON actions performed by failsafe
    system_state_after TEXT, -- JSON system state after failsafe actions
    restoration_plan TEXT, -- JSON plan for restoring normal operation
    restoration_timestamp REAL, -- When normal operation was restored
    lessons_learned TEXT, -- JSON insights from this incident
    prevention_measures TEXT, -- JSON measures to prevent similar incidents
    created_at REAL DEFAULT (strftime('%s', 'now'))
);

-- Transparency logs for all architect decisions and reasoning
CREATE TABLE IF NOT EXISTS architect_transparency_logs (
    log_id TEXT PRIMARY KEY,
    log_timestamp REAL NOT NULL,
    component TEXT NOT NULL, -- Which architect component generated this log
    decision_type TEXT NOT NULL, -- 'analysis', 'proposal_generation', 'risk_assessment', 'performance_evaluation'
    decision_context TEXT NOT NULL, -- JSON context that led to this decision
    reasoning_process TEXT NOT NULL, -- Human-readable step-by-step reasoning
    data_sources TEXT NOT NULL, -- JSON data sources consulted
    alternatives_considered TEXT, -- JSON alternative approaches considered
    confidence_factors TEXT, -- JSON factors affecting confidence in decision
    uncertainty_factors TEXT, -- JSON sources of uncertainty
    decision_outcome TEXT NOT NULL, -- What decision was made
    expected_consequences TEXT, -- JSON expected consequences of the decision
    actual_consequences TEXT, -- JSON actual consequences (filled in later)
    decision_quality_score REAL, -- Retrospective quality assessment (0.0 to 1.0)
    created_at REAL DEFAULT (strftime('%s', 'now')),
    updated_at REAL DEFAULT (strftime('%s', 'now'))
);

-- Evolution reports for Phase 3 oversight
CREATE TABLE IF NOT EXISTS architect_evolution_reports (
    report_id TEXT PRIMARY KEY,
    reporting_period_start REAL NOT NULL,
    reporting_period_end REAL NOT NULL,
    generation_timestamp REAL NOT NULL,
    report_type TEXT DEFAULT 'regular', -- 'regular', 'milestone', 'incident', 'phase_transition'
    changes_summary TEXT NOT NULL, -- JSON summary of all changes made
    performance_deltas TEXT NOT NULL, -- JSON performance changes across benchmarks
    unexpected_consequences TEXT, -- JSON any unexpected consequences and mitigations
    stability_assessment TEXT NOT NULL, -- JSON overall stability assessment
    risk_factors TEXT, -- JSON current risk factors
    future_plans TEXT, -- JSON planned future evolution directions
    human_attention_required BOOLEAN DEFAULT FALSE, -- Whether human attention is needed
    recommendations TEXT, -- JSON recommendations for human oversight
    architect_self_assessment TEXT, -- JSON architect's assessment of its own performance
    created_at REAL DEFAULT (strftime('%s', 'now'))
);

-- System modules with enhanced safety tracking
CREATE TABLE IF NOT EXISTS system_modules (
    module_id TEXT PRIMARY KEY,
    module_name TEXT NOT NULL, -- 'real_time_learner', 'pattern_detector', 'strategy_adjuster', etc.
    module_type TEXT NOT NULL, -- 'core', 'analysis', 'learning', 'coordination', 'optimization'
    module_category TEXT NOT NULL, -- 'essential', 'enhancement', 'experimental', 'deprecated'
    current_version TEXT DEFAULT '1.0.0',
    module_definition TEXT NOT NULL, -- JSON definition of module structure and interfaces
    dependency_modules TEXT, -- JSON array of module_ids this module depends on
    dependent_modules TEXT, -- JSON array of module_ids that depend on this module
    resource_requirements TEXT, -- JSON resource requirements (CPU, memory, etc.)
    performance_metrics TEXT, -- JSON current performance metrics
    usage_frequency INTEGER DEFAULT 0, -- How often this module is used
    effectiveness_score REAL DEFAULT 0.5, -- Overall effectiveness (0.0 to 1.0)
    stability_score REAL DEFAULT 0.8, -- NEW: Stability metric for safety
    last_used_timestamp REAL,
    activation_threshold REAL DEFAULT 0.3, -- Minimum usage/effectiveness to remain active
    pruning_candidate BOOLEAN DEFAULT FALSE, -- Whether this module is candidate for pruning
    safety_critical BOOLEAN DEFAULT FALSE, -- NEW: Whether this module is safety-critical
    regression_risk_level TEXT DEFAULT 'low', -- NEW: 'low', 'medium', 'high', 'critical'
    last_safety_evaluation REAL, -- NEW: When safety was last evaluated
    is_active BOOLEAN DEFAULT TRUE, -- Whether module is currently active
    created_at REAL DEFAULT (strftime('%s', 'now')),
    updated_at REAL DEFAULT (strftime('%s', 'now'))
);

-- NEAT genome representation for system architectures
CREATE TABLE IF NOT EXISTS neat_genomes (
    genome_id TEXT PRIMARY KEY,
    generation INTEGER NOT NULL, -- Which generation this genome belongs to
    species_id TEXT, -- Which species this genome belongs to
    parent_genome_ids TEXT, -- JSON array of parent genome IDs (for crossover tracking)
    fitness_score REAL DEFAULT 0.0, -- Overall fitness of this architecture
    adjusted_fitness REAL DEFAULT 0.0, -- Fitness adjusted for species sharing
    nodes_count INTEGER DEFAULT 0, -- Number of nodes in the network
    connections_count INTEGER DEFAULT 0, -- Number of connections in the network
    genome_structure TEXT NOT NULL, -- JSON representation of the full genome structure
    phenotype_definition TEXT, -- JSON definition of the expressed phenotype (actual system)
    innovation_numbers TEXT, -- JSON array of innovation numbers for tracking evolution
    complexity_score REAL DEFAULT 0.0, -- Complexity measure for this genome
    specialization_score REAL DEFAULT 0.0, -- How specialized this genome is
    robustness_score REAL DEFAULT 0.0, -- How robust this genome is to perturbations
    age INTEGER DEFAULT 0, -- How many generations this genome has existed
    last_improvement_generation INTEGER DEFAULT 0, -- Last generation this genome improved
    is_champion BOOLEAN DEFAULT FALSE, -- Whether this is the best genome in its species
    is_active BOOLEAN DEFAULT TRUE, -- Whether this genome is still in the population
    created_at REAL DEFAULT (strftime('%s', 'now')),
    updated_at REAL DEFAULT (strftime('%s', 'now'))
);

-- NEAT species for organizing similar genomes
CREATE TABLE IF NOT EXISTS neat_species (
    species_id TEXT PRIMARY KEY,
    generation INTEGER NOT NULL, -- Generation when this species was formed
    representative_genome_id TEXT, -- Genome that represents this species
    species_name TEXT, -- Human-readable name for this species
    genome_count INTEGER DEFAULT 0, -- Number of genomes in this species
    average_fitness REAL DEFAULT 0.0, -- Average fitness of genomes in this species
    max_fitness REAL DEFAULT 0.0, -- Best fitness achieved by this species
    fitness_stagnation_count INTEGER DEFAULT 0, -- Generations without improvement
    allowed_offspring INTEGER DEFAULT 0, -- Number of offspring this species gets next generation
    elitism_threshold REAL DEFAULT 0.8, -- Fitness threshold for elite preservation
    compatibility_threshold REAL DEFAULT 3.0, -- Threshold for species membership
    species_traits TEXT, -- JSON description of what makes this species unique
    extinction_risk REAL DEFAULT 0.0, -- Risk of extinction (0.0 to 1.0)
    is_extinct BOOLEAN DEFAULT FALSE, -- Whether this species has gone extinct
    extinction_generation INTEGER, -- Generation when species went extinct
    created_at REAL DEFAULT (strftime('%s', 'now')),
    updated_at REAL DEFAULT (strftime('%s', 'now'))
);

-- NEAT innovations for tracking structural changes
CREATE TABLE IF NOT EXISTS neat_innovations (
    innovation_id INTEGER PRIMARY KEY, -- Global innovation number
    innovation_type TEXT NOT NULL, -- 'add_node', 'add_connection', 'modify_weight', 'remove_connection'
    source_node_id TEXT, -- Source node for connections
    target_node_id TEXT, -- Target node for connections
    weight_value REAL, -- Weight value for connections
    innovation_description TEXT, -- Human-readable description
    first_genome_id TEXT, -- First genome to have this innovation
    generation_introduced INTEGER NOT NULL, -- Generation when this innovation first appeared
    usage_frequency INTEGER DEFAULT 1, -- How many genomes have adopted this innovation
    average_fitness_impact REAL DEFAULT 0.0, -- Average fitness impact of this innovation
    is_beneficial BOOLEAN DEFAULT NULL, -- Whether this innovation is generally beneficial
    created_at REAL DEFAULT (strftime('%s', 'now'))
);

-- Module usage tracking for "use it or lose it" functionality
CREATE TABLE IF NOT EXISTS module_usage_tracking (
    usage_id TEXT PRIMARY KEY,
    module_id TEXT NOT NULL,
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    usage_timestamp REAL NOT NULL,
    usage_context TEXT NOT NULL, -- JSON context when module was used
    input_data_size INTEGER DEFAULT 0, -- Size of input data processed
    output_data_size INTEGER DEFAULT 0, -- Size of output data produced
    processing_time_ms REAL DEFAULT 0.0, -- Time taken to process
    success BOOLEAN DEFAULT TRUE, -- Whether the module operation was successful
    effectiveness_score REAL DEFAULT 0.5, -- How effective this usage was
    impact_on_performance REAL DEFAULT 0.0, -- Impact on overall system performance
    resource_consumption TEXT, -- JSON resource usage during this operation
    interactions_with_modules TEXT, -- JSON modules this interacted with during usage
    user_feedback_score REAL, -- Optional user/system feedback on this usage
    created_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (module_id) REFERENCES system_modules(module_id),
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
);

-- Architectural evolution experiments and results
CREATE TABLE IF NOT EXISTS architectural_evolution_experiments (
    experiment_id TEXT PRIMARY KEY,
    experiment_name TEXT NOT NULL,
    experiment_type TEXT NOT NULL, -- 'module_addition', 'module_removal', 'connection_modification', 'full_evolution'
    baseline_genome_id TEXT, -- Starting genome for the experiment
    experimental_genome_id TEXT, -- Resulting genome after experiment
    experiment_parameters TEXT NOT NULL, -- JSON parameters for the experiment
    experiment_timestamp REAL NOT NULL,
    experiment_duration_seconds REAL DEFAULT 0.0,
    baseline_performance TEXT, -- JSON baseline performance metrics
    experimental_performance TEXT, -- JSON experimental performance metrics
    performance_improvement REAL DEFAULT 0.0, -- Improvement over baseline (-1.0 to +inf)
    statistical_significance REAL DEFAULT 0.0, -- P-value or confidence measure
    experiment_success BOOLEAN DEFAULT FALSE, -- Whether experiment met success criteria
    rollback_performed BOOLEAN DEFAULT FALSE, -- Whether changes were rolled back
    lessons_learned TEXT, -- JSON insights gained from the experiment
    follow_up_experiments TEXT, -- JSON array of follow-up experiment_ids
    created_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (baseline_genome_id) REFERENCES neat_genomes(genome_id),
    FOREIGN KEY (experimental_genome_id) REFERENCES neat_genomes(genome_id)
);

-- Module pruning decisions and outcomes
CREATE TABLE IF NOT EXISTS module_pruning_decisions (
    pruning_id TEXT PRIMARY KEY,
    module_id TEXT NOT NULL,
    decision_timestamp REAL NOT NULL,
    decision_type TEXT NOT NULL, -- 'prune', 'preserve', 'modify', 'merge'
    decision_reason TEXT NOT NULL, -- JSON detailed reasoning for the decision
    usage_statistics TEXT NOT NULL, -- JSON usage stats that influenced decision
    performance_impact_analysis TEXT, -- JSON analysis of expected performance impact
    dependency_analysis TEXT, -- JSON analysis of dependencies and dependents
    alternative_modules TEXT, -- JSON array of modules that could replace this one
    pruning_scheduled BOOLEAN DEFAULT FALSE, -- Whether pruning is scheduled
    pruning_executed BOOLEAN DEFAULT FALSE, -- Whether pruning was actually executed
    execution_timestamp REAL, -- When pruning was executed
    post_pruning_performance TEXT, -- JSON performance after pruning
    rollback_required BOOLEAN DEFAULT FALSE, -- Whether rollback is needed
    rollback_timestamp REAL, -- When rollback was performed
    decision_effectiveness REAL DEFAULT 0.0, -- How effective this decision was
    created_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (module_id) REFERENCES system_modules(module_id)
);

-- Genetic algorithm populations and generations
CREATE TABLE IF NOT EXISTS neat_populations (
    population_id TEXT PRIMARY KEY,
    generation INTEGER NOT NULL, -- Current generation number
    population_size INTEGER DEFAULT 50, -- Number of genomes in population
    species_count INTEGER DEFAULT 0, -- Number of species in this generation
    champion_genome_id TEXT, -- Best genome in this generation
    average_fitness REAL DEFAULT 0.0, -- Average fitness of population
    max_fitness REAL DEFAULT 0.0, -- Best fitness in population
    fitness_variance REAL DEFAULT 0.0, -- Variance in fitness scores
    diversity_measure REAL DEFAULT 0.0, -- Measure of genetic diversity
    stagnation_count INTEGER DEFAULT 0, -- Generations without improvement
    extinction_events INTEGER DEFAULT 0, -- Number of species extinctions this generation
    innovation_rate REAL DEFAULT 0.0, -- Rate of new innovations
    population_parameters TEXT, -- JSON algorithm parameters for this population
    environmental_pressure REAL DEFAULT 0.5, -- Selection pressure (0.0 to 1.0)
    mutation_rate REAL DEFAULT 0.1, -- Current mutation rate
    crossover_rate REAL DEFAULT 0.7, -- Current crossover rate
    elitism_percentage REAL DEFAULT 0.1, -- Percentage of elite genomes preserved
    generation_timestamp REAL NOT NULL,
    generation_duration_seconds REAL DEFAULT 0.0,
    created_at REAL DEFAULT (strftime('%s', 'now'))
);

-- Integration with attention system for architectural priorities
CREATE TABLE IF NOT EXISTS architectural_attention_integration (
    integration_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    integration_timestamp REAL NOT NULL,
    current_architecture_id TEXT, -- Current active genome/architecture
    architectural_priorities TEXT NOT NULL, -- JSON priorities for architectural evolution
    attention_allocation_request TEXT NOT NULL, -- JSON requested attention for evolution
    attention_allocation_received TEXT NOT NULL, -- JSON actual attention allocation received
    evolution_targets TEXT NOT NULL, -- JSON specific architectural areas to evolve
    resource_allocation_efficiency REAL DEFAULT 0.0, -- How efficiently resources were used
    evolution_progress_rate REAL DEFAULT 0.0, -- Rate of architectural evolution progress
    bottleneck_analysis TEXT, -- JSON analysis of architectural bottlenecks
    optimization_suggestions TEXT, -- JSON suggestions for architectural optimization
    integration_effectiveness REAL DEFAULT 0.0, -- How effective this integration was
    created_at REAL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id),
    FOREIGN KEY (current_architecture_id) REFERENCES neat_genomes(genome_id)
);

-- ============================================================================
-- INDEXES FOR NEAT-BASED ARCHITECT SYSTEM WITH SAFETY GUARDRAILS
-- ============================================================================

-- Architect phases indexes
CREATE INDEX IF NOT EXISTS idx_architect_phases_number ON architect_phases(phase_number);
CREATE INDEX IF NOT EXISTS idx_architect_phases_name ON architect_phases(phase_name);
CREATE INDEX IF NOT EXISTS idx_architect_phases_active ON architect_phases(phase_active);
CREATE INDEX IF NOT EXISTS idx_architect_phases_advancement ON architect_phases(advancement_eligible);
CREATE INDEX IF NOT EXISTS idx_architect_phases_regression ON architect_phases(regression_count);

-- Change proposals indexes
CREATE INDEX IF NOT EXISTS idx_change_proposals_phase ON architect_change_proposals(architect_phase);
CREATE INDEX IF NOT EXISTS idx_change_proposals_type ON architect_change_proposals(proposal_type);
CREATE INDEX IF NOT EXISTS idx_change_proposals_status ON architect_change_proposals(proposal_status);
CREATE INDEX IF NOT EXISTS idx_change_proposals_timestamp ON architect_change_proposals(generation_timestamp);
CREATE INDEX IF NOT EXISTS idx_change_proposals_complexity ON architect_change_proposals(complexity_score);
CREATE INDEX IF NOT EXISTS idx_change_proposals_confidence ON architect_change_proposals(confidence_level);

-- Human feedback indexes
CREATE INDEX IF NOT EXISTS idx_human_feedback_proposal ON architect_human_feedback(proposal_id);
CREATE INDEX IF NOT EXISTS idx_human_feedback_decision ON architect_human_feedback(approval_decision);
CREATE INDEX IF NOT EXISTS idx_human_feedback_timestamp ON architect_human_feedback(review_timestamp);
CREATE INDEX IF NOT EXISTS idx_human_feedback_followup ON architect_human_feedback(follow_up_required);

-- Stability metrics indexes
CREATE INDEX IF NOT EXISTS idx_stability_metrics_game ON architect_stability_metrics(game_id);
CREATE INDEX IF NOT EXISTS idx_stability_metrics_type ON architect_stability_metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_stability_metrics_timestamp ON architect_stability_metrics(measurement_timestamp);
CREATE INDEX IF NOT EXISTS idx_stability_metrics_regression ON architect_stability_metrics(regression_detected);
CREATE INDEX IF NOT EXISTS idx_stability_metrics_severity ON architect_stability_metrics(regression_severity);
CREATE INDEX IF NOT EXISTS idx_stability_metrics_proposal ON architect_stability_metrics(related_proposal_id);

-- Performance benchmarks indexes
CREATE INDEX IF NOT EXISTS idx_performance_benchmarks_name ON architect_performance_benchmarks(benchmark_name);
CREATE INDEX IF NOT EXISTS idx_performance_benchmarks_category ON architect_performance_benchmarks(benchmark_category);
CREATE INDEX IF NOT EXISTS idx_performance_benchmarks_timestamp ON architect_performance_benchmarks(measurement_timestamp);
CREATE INDEX IF NOT EXISTS idx_performance_benchmarks_trend ON architect_performance_benchmarks(trend_direction);

-- Rollback procedures indexes
CREATE INDEX IF NOT EXISTS idx_rollback_procedures_proposal ON architect_rollback_procedures(proposal_id);
CREATE INDEX IF NOT EXISTS idx_rollback_procedures_status ON architect_rollback_procedures(rollback_status);
CREATE INDEX IF NOT EXISTS idx_rollback_procedures_success ON architect_rollback_procedures(rollback_success);
CREATE INDEX IF NOT EXISTS idx_rollback_procedures_timestamp ON architect_rollback_procedures(execution_timestamp);

-- Failsafe logs indexes
CREATE INDEX IF NOT EXISTS idx_failsafe_logs_type ON architect_failsafe_logs(failsafe_type);
CREATE INDEX IF NOT EXISTS idx_failsafe_logs_timestamp ON architect_failsafe_logs(activation_timestamp);
CREATE INDEX IF NOT EXISTS idx_failsafe_logs_activated_by ON architect_failsafe_logs(activated_by);

-- Transparency logs indexes
CREATE INDEX IF NOT EXISTS idx_transparency_logs_component ON architect_transparency_logs(component);
CREATE INDEX IF NOT EXISTS idx_transparency_logs_decision_type ON architect_transparency_logs(decision_type);
CREATE INDEX IF NOT EXISTS idx_transparency_logs_timestamp ON architect_transparency_logs(log_timestamp);
CREATE INDEX IF NOT EXISTS idx_transparency_logs_quality ON architect_transparency_logs(decision_quality_score);

-- Evolution reports indexes
CREATE INDEX IF NOT EXISTS idx_evolution_reports_type ON architect_evolution_reports(report_type);
CREATE INDEX IF NOT EXISTS idx_evolution_reports_timestamp ON architect_evolution_reports(generation_timestamp);
CREATE INDEX IF NOT EXISTS idx_evolution_reports_attention ON architect_evolution_reports(human_attention_required);

-- System modules indexes (enhanced with safety fields)
CREATE INDEX IF NOT EXISTS idx_system_modules_name ON system_modules(module_name);
CREATE INDEX IF NOT EXISTS idx_system_modules_type ON system_modules(module_type);
CREATE INDEX IF NOT EXISTS idx_system_modules_category ON system_modules(module_category);
CREATE INDEX IF NOT EXISTS idx_system_modules_effectiveness ON system_modules(effectiveness_score);
CREATE INDEX IF NOT EXISTS idx_system_modules_stability ON system_modules(stability_score);
CREATE INDEX IF NOT EXISTS idx_system_modules_usage ON system_modules(usage_frequency);
CREATE INDEX IF NOT EXISTS idx_system_modules_active ON system_modules(is_active);
CREATE INDEX IF NOT EXISTS idx_system_modules_pruning ON system_modules(pruning_candidate);
CREATE INDEX IF NOT EXISTS idx_system_modules_safety_critical ON system_modules(safety_critical);
CREATE INDEX IF NOT EXISTS idx_system_modules_risk_level ON system_modules(regression_risk_level);

-- NEAT genomes indexes
CREATE INDEX IF NOT EXISTS idx_neat_genomes_generation ON neat_genomes(generation);
CREATE INDEX IF NOT EXISTS idx_neat_genomes_species ON neat_genomes(species_id);
CREATE INDEX IF NOT EXISTS idx_neat_genomes_fitness ON neat_genomes(fitness_score);
CREATE INDEX IF NOT EXISTS idx_neat_genomes_adjusted_fitness ON neat_genomes(adjusted_fitness);
CREATE INDEX IF NOT EXISTS idx_neat_genomes_complexity ON neat_genomes(complexity_score);
CREATE INDEX IF NOT EXISTS idx_neat_genomes_champion ON neat_genomes(is_champion);
CREATE INDEX IF NOT EXISTS idx_neat_genomes_active ON neat_genomes(is_active);

-- NEAT species indexes
CREATE INDEX IF NOT EXISTS idx_neat_species_generation ON neat_species(generation);
CREATE INDEX IF NOT EXISTS idx_neat_species_fitness ON neat_species(average_fitness);
CREATE INDEX IF NOT EXISTS idx_neat_species_max_fitness ON neat_species(max_fitness);
CREATE INDEX IF NOT EXISTS idx_neat_species_stagnation ON neat_species(fitness_stagnation_count);
CREATE INDEX IF NOT EXISTS idx_neat_species_extinct ON neat_species(is_extinct);

-- NEAT innovations indexes
CREATE INDEX IF NOT EXISTS idx_neat_innovations_type ON neat_innovations(innovation_type);
CREATE INDEX IF NOT EXISTS idx_neat_innovations_generation ON neat_innovations(generation_introduced);
CREATE INDEX IF NOT EXISTS idx_neat_innovations_usage ON neat_innovations(usage_frequency);
CREATE INDEX IF NOT EXISTS idx_neat_innovations_impact ON neat_innovations(average_fitness_impact);

-- Module usage tracking indexes
CREATE INDEX IF NOT EXISTS idx_module_usage_module ON module_usage_tracking(module_id);
CREATE INDEX IF NOT EXISTS idx_module_usage_game ON module_usage_tracking(game_id);
CREATE INDEX IF NOT EXISTS idx_module_usage_timestamp ON module_usage_tracking(usage_timestamp);
CREATE INDEX IF NOT EXISTS idx_module_usage_effectiveness ON module_usage_tracking(effectiveness_score);
CREATE INDEX IF NOT EXISTS idx_module_usage_success ON module_usage_tracking(success);

-- Architectural evolution experiments indexes
CREATE INDEX IF NOT EXISTS idx_arch_experiments_type ON architectural_evolution_experiments(experiment_type);
CREATE INDEX IF NOT EXISTS idx_arch_experiments_timestamp ON architectural_evolution_experiments(experiment_timestamp);
CREATE INDEX IF NOT EXISTS idx_arch_experiments_success ON architectural_evolution_experiments(experiment_success);
CREATE INDEX IF NOT EXISTS idx_arch_experiments_improvement ON architectural_evolution_experiments(performance_improvement);

-- Module pruning decisions indexes
CREATE INDEX IF NOT EXISTS idx_module_pruning_module ON module_pruning_decisions(module_id);
CREATE INDEX IF NOT EXISTS idx_module_pruning_timestamp ON module_pruning_decisions(decision_timestamp);
CREATE INDEX IF NOT EXISTS idx_module_pruning_type ON module_pruning_decisions(decision_type);
CREATE INDEX IF NOT EXISTS idx_module_pruning_executed ON module_pruning_decisions(pruning_executed);

-- NEAT populations indexes
CREATE INDEX IF NOT EXISTS idx_neat_populations_generation ON neat_populations(generation);
CREATE INDEX IF NOT EXISTS idx_neat_populations_fitness ON neat_populations(max_fitness);
CREATE INDEX IF NOT EXISTS idx_neat_populations_diversity ON neat_populations(diversity_measure);
CREATE INDEX IF NOT EXISTS idx_neat_populations_stagnation ON neat_populations(stagnation_count);

-- Architectural attention integration indexes
CREATE INDEX IF NOT EXISTS idx_arch_attention_game ON architectural_attention_integration(game_id);
CREATE INDEX IF NOT EXISTS idx_arch_attention_timestamp ON architectural_attention_integration(integration_timestamp);
CREATE INDEX IF NOT EXISTS idx_arch_attention_effectiveness ON architectural_attention_integration(integration_effectiveness);
CREATE INDEX IF NOT EXISTS idx_arch_attention_architecture ON architectural_attention_integration(current_architecture_id);

-- ============================================================================
-- INDEXES FOR LOSING STREAK DETECTION TABLES
-- ============================================================================

-- Losing streaks indexes
CREATE INDEX IF NOT EXISTS idx_losing_streaks_game_type ON losing_streaks(game_type);
CREATE INDEX IF NOT EXISTS idx_losing_streaks_game_id ON losing_streaks(game_id);
CREATE INDEX IF NOT EXISTS idx_losing_streaks_consecutive ON losing_streaks(consecutive_failures);
CREATE INDEX IF NOT EXISTS idx_losing_streaks_escalation ON losing_streaks(escalation_level);
CREATE INDEX IF NOT EXISTS idx_losing_streaks_active ON losing_streaks(streak_broken);

-- Anti-patterns indexes
CREATE INDEX IF NOT EXISTS idx_anti_patterns_game_type ON anti_patterns(game_type);
CREATE INDEX IF NOT EXISTS idx_anti_patterns_type ON anti_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_anti_patterns_failure_rate ON anti_patterns(failure_rate);
CREATE INDEX IF NOT EXISTS idx_anti_patterns_severity ON anti_patterns(severity);

-- Escalated interventions indexes
CREATE INDEX IF NOT EXISTS idx_escalated_interventions_streak ON escalated_interventions(streak_id);
CREATE INDEX IF NOT EXISTS idx_escalated_interventions_level ON escalated_interventions(escalation_level);
CREATE INDEX IF NOT EXISTS idx_escalated_interventions_type ON escalated_interventions(intervention_type);
CREATE INDEX IF NOT EXISTS idx_escalated_interventions_success ON escalated_interventions(success);

-- GAN indexes for performance
CREATE INDEX IF NOT EXISTS idx_gan_generated_states_session ON gan_generated_states(session_id);
CREATE INDEX IF NOT EXISTS idx_gan_generated_states_quality ON gan_generated_states(quality_score);
CREATE INDEX IF NOT EXISTS idx_gan_model_checkpoints_session ON gan_model_checkpoints(session_id);
CREATE INDEX IF NOT EXISTS idx_gan_pattern_learning_session ON gan_pattern_learning(session_id);
CREATE INDEX IF NOT EXISTS idx_gan_reverse_engineering_session ON gan_reverse_engineering(session_id);
CREATE INDEX IF NOT EXISTS idx_gan_performance_metrics_session ON gan_performance_metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_gan_training_data_game ON gan_training_data(game_id);
CREATE INDEX IF NOT EXISTS idx_gan_training_data_action ON gan_training_data(action_number);
CREATE INDEX IF NOT EXISTS idx_gan_training_data_button_candidate ON gan_training_data(is_button_candidate);
CREATE INDEX IF NOT EXISTS idx_gan_training_data_timestamp ON gan_training_data(timestamp);-- ============================================================================
-- TIER 3 SYSTEM SCHEMAS: BAYESIAN INFERENCE ENGINE + ENHANCED GRAPH TRAVERSAL
-- ============================================================================

-- ============================================================================
-- BAYESIAN INFERENCE ENGINE TABLES
-- ============================================================================

-- Bayesian hypotheses for probabilistic reasoning
CREATE TABLE IF NOT EXISTS bayesian_hypotheses (
    hypothesis_id TEXT PRIMARY KEY,
    hypothesis_type TEXT NOT NULL, -- 'action_outcome', 'sequence_pattern', 'conditional_rule', etc.
    description TEXT NOT NULL,
    prior_probability REAL NOT NULL,
    posterior_probability REAL NOT NULL,
    evidence_count INTEGER DEFAULT 0,
    supporting_evidence INTEGER DEFAULT 0,
    refuting_evidence INTEGER DEFAULT 0,
    confidence_lower REAL DEFAULT 0.0,
    confidence_upper REAL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    last_updated TEXT NOT NULL,
    context_conditions TEXT, -- JSON
    validation_threshold REAL DEFAULT 0.8,
    is_active BOOLEAN DEFAULT 1
);

-- Evidence for Bayesian hypothesis validation
CREATE TABLE IF NOT EXISTS bayesian_evidence (
    evidence_id TEXT PRIMARY KEY,
    hypothesis_id TEXT NOT NULL,
    evidence_type TEXT NOT NULL, -- 'direct_observation', 'pattern_match', 'statistical_correlation', etc.
    supports_hypothesis BOOLEAN NOT NULL,
    strength REAL NOT NULL, -- 0.0 to 1.0
    context TEXT, -- JSON
    observed_at TEXT NOT NULL,
    game_id TEXT,
    session_id TEXT,
    FOREIGN KEY (hypothesis_id) REFERENCES bayesian_hypotheses (hypothesis_id)
);

-- Bayesian belief networks for complex relationships
CREATE TABLE IF NOT EXISTS bayesian_belief_networks (
    network_id TEXT PRIMARY KEY,
    game_type TEXT,
    nodes TEXT NOT NULL, -- JSON
    edges TEXT NOT NULL, -- JSON
    conditional_probabilities TEXT, -- JSON
    evidence_propagation_rules TEXT, -- JSON
    created_at TEXT NOT NULL,
    last_updated TEXT NOT NULL,
    accuracy_score REAL DEFAULT 0.5
);

-- Bayesian predictions for tracking accuracy
CREATE TABLE IF NOT EXISTS bayesian_predictions (
    prediction_id TEXT PRIMARY KEY,
    predicted_action TEXT NOT NULL, -- JSON
    success_probability REAL NOT NULL,
    confidence_level REAL NOT NULL,
    supporting_hypotheses TEXT, -- JSON
    uncertainty_factors TEXT, -- JSON
    evidence_summary TEXT, -- JSON
    context_conditions TEXT, -- JSON
    created_at TEXT NOT NULL,
    actual_outcome TEXT, -- JSON (filled after observation)
    prediction_accuracy REAL, -- 0.0 to 1.0 (filled after observation)
    game_id TEXT,
    session_id TEXT
);

-- ============================================================================
-- ENHANCED GRAPH TRAVERSAL TABLES
-- ============================================================================

-- Graph structures for pattern navigation
CREATE TABLE IF NOT EXISTS graph_structures (
    graph_id TEXT PRIMARY KEY,
    graph_type TEXT NOT NULL, -- 'game_state_graph', 'decision_tree', 'pattern_space', etc.
    properties TEXT, -- JSON
    created_at TEXT NOT NULL,
    last_updated TEXT NOT NULL,
    node_count INTEGER DEFAULT 0,
    edge_count INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT 1
);

-- Graph nodes for state representation
CREATE TABLE IF NOT EXISTS graph_nodes (
    node_id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    node_type TEXT NOT NULL, -- 'state_node', 'action_node', 'pattern_node', etc.
    properties TEXT, -- JSON
    coordinates_x REAL,
    coordinates_y REAL,
    created_at TEXT NOT NULL,
    last_visited TEXT,
    visit_count INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0.0,
    FOREIGN KEY (graph_id) REFERENCES graph_structures (graph_id)
);

-- Graph edges for transitions and relationships
CREATE TABLE IF NOT EXISTS graph_edges (
    edge_id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    from_node TEXT NOT NULL,
    to_node TEXT NOT NULL,
    weight REAL NOT NULL,
    edge_type TEXT DEFAULT 'default',
    properties TEXT, -- JSON
    created_at TEXT NOT NULL,
    traversal_count INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0.0,
    FOREIGN KEY (graph_id) REFERENCES graph_structures (graph_id),
    FOREIGN KEY (from_node) REFERENCES graph_nodes (node_id),
    FOREIGN KEY (to_node) REFERENCES graph_nodes (node_id)
);

-- Graph traversal results and performance tracking
CREATE TABLE IF NOT EXISTS graph_traversals (
    traversal_id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    algorithm TEXT NOT NULL, -- 'breadth_first', 'depth_first', 'dijkstra', 'a_star', etc.
    start_node TEXT NOT NULL,
    end_node TEXT NOT NULL,
    path_nodes TEXT, -- JSON array
    path_edges TEXT, -- JSON array
    total_weight REAL,
    success BOOLEAN NOT NULL,
    computation_time REAL,
    nodes_explored INTEGER,
    heuristic_score REAL,
    created_at TEXT NOT NULL,
    game_id TEXT,
    session_id TEXT,
    FOREIGN KEY (graph_id) REFERENCES graph_structures (graph_id)
);

-- Path optimization for improving traversal efficiency
CREATE TABLE IF NOT EXISTS path_optimizations (
    optimization_id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    original_path TEXT, -- JSON
    optimized_path TEXT, -- JSON
    improvement_factor REAL,
    optimization_strategy TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (graph_id) REFERENCES graph_structures (graph_id)
);

-- ============================================================================
-- TIER 3 INTEGRATION TABLES
-- ============================================================================

-- Bayesian-Graph integration for probabilistic path finding
CREATE TABLE IF NOT EXISTS bayesian_graph_integration (
    integration_id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    hypothesis_id TEXT NOT NULL,
    integration_type TEXT NOT NULL, -- 'path_probability', 'node_confidence', 'edge_reliability'
    probability_influence REAL NOT NULL, -- How much Bayesian reasoning affects graph decisions
    integration_effectiveness REAL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    last_updated TEXT NOT NULL,
    game_id TEXT,
    session_id TEXT,
    FOREIGN KEY (graph_id) REFERENCES graph_structures (graph_id),
    FOREIGN KEY (hypothesis_id) REFERENCES bayesian_hypotheses (hypothesis_id)
);

-- Tier 3 attention coordination for resource allocation
CREATE TABLE IF NOT EXISTS tier3_attention_coordination (
    coordination_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    bayesian_attention_priority REAL DEFAULT 0.3,
    graph_traversal_attention_priority REAL DEFAULT 0.3,
    reasoning_complexity REAL NOT NULL,
    traversal_complexity REAL NOT NULL,
    attention_allocation_result TEXT, -- JSON
    coordination_effectiveness REAL DEFAULT 0.5,
    coordination_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tier 3 performance analytics
CREATE TABLE IF NOT EXISTS tier3_performance_analytics (
    analytics_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    bayesian_hypotheses_active INTEGER DEFAULT 0,
    bayesian_prediction_accuracy REAL DEFAULT 0.5,
    graphs_active INTEGER DEFAULT 0,
    avg_traversal_efficiency REAL DEFAULT 0.5,
    combined_system_effectiveness REAL DEFAULT 0.5,
    reasoning_accuracy REAL DEFAULT 0.5,
    navigation_success_rate REAL DEFAULT 0.5,
    analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Persistence helpers debug logs
CREATE TABLE IF NOT EXISTS persistence_debug_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP NOT NULL,
    log_level TEXT NOT NULL, -- 'DEBUG', 'INFO', 'ERROR'
    logger_name TEXT NOT NULL, -- 'tabula_rasa.persistence_helpers'
    function_name TEXT NOT NULL, -- 'persist_winning_sequence', 'persist_button_priorities', 'persist_governor_decision'
    operation_type TEXT NOT NULL, -- 'CALLED', 'Success', 'Commit attempted', 'Exception', etc.
    message TEXT NOT NULL, -- Full log message
    parameters TEXT, -- JSON string of function parameters
    db_info TEXT, -- Database connection info
    sql_query TEXT, -- SQL query if applicable
    result_info TEXT, -- Result information
    session_id TEXT,
    game_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES FOR TIER 3 SYSTEMS
-- ============================================================================

-- Bayesian Inference Engine indexes
CREATE INDEX IF NOT EXISTS idx_bayesian_hypotheses_type ON bayesian_hypotheses (hypothesis_type);
CREATE INDEX IF NOT EXISTS idx_bayesian_hypotheses_active ON bayesian_hypotheses (is_active);
CREATE INDEX IF NOT EXISTS idx_bayesian_hypotheses_probability ON bayesian_hypotheses (posterior_probability);
CREATE INDEX IF NOT EXISTS idx_bayesian_hypotheses_updated ON bayesian_hypotheses (last_updated);

CREATE INDEX IF NOT EXISTS idx_bayesian_evidence_hypothesis ON bayesian_evidence (hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_bayesian_evidence_type ON bayesian_evidence (evidence_type);
CREATE INDEX IF NOT EXISTS idx_bayesian_evidence_game ON bayesian_evidence (game_id, session_id);
CREATE INDEX IF NOT EXISTS idx_bayesian_evidence_observed ON bayesian_evidence (observed_at);

CREATE INDEX IF NOT EXISTS idx_bayesian_networks_game_type ON bayesian_belief_networks (game_type);
CREATE INDEX IF NOT EXISTS idx_bayesian_networks_accuracy ON bayesian_belief_networks (accuracy_score);

CREATE INDEX IF NOT EXISTS idx_bayesian_predictions_game ON bayesian_predictions (game_id, session_id);
CREATE INDEX IF NOT EXISTS idx_bayesian_predictions_accuracy ON bayesian_predictions (prediction_accuracy);
CREATE INDEX IF NOT EXISTS idx_bayesian_predictions_probability ON bayesian_predictions (success_probability);

-- Enhanced Graph Traversal indexes
CREATE INDEX IF NOT EXISTS idx_graph_structures_type ON graph_structures (graph_type);
CREATE INDEX IF NOT EXISTS idx_graph_structures_active ON graph_structures (is_active);
CREATE INDEX IF NOT EXISTS idx_graph_structures_updated ON graph_structures (last_updated);

CREATE INDEX IF NOT EXISTS idx_graph_nodes_graph ON graph_nodes (graph_id);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes (node_type);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_success ON graph_nodes (success_rate);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_visited ON graph_nodes (last_visited);

CREATE INDEX IF NOT EXISTS idx_graph_edges_graph ON graph_edges (graph_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_nodes ON graph_edges (from_node, to_node);
CREATE INDEX IF NOT EXISTS idx_graph_edges_weight ON graph_edges (weight);
CREATE INDEX IF NOT EXISTS idx_graph_edges_success ON graph_edges (success_rate);

CREATE INDEX IF NOT EXISTS idx_graph_traversals_graph ON graph_traversals (graph_id);
CREATE INDEX IF NOT EXISTS idx_graph_traversals_algorithm ON graph_traversals (algorithm);
CREATE INDEX IF NOT EXISTS idx_graph_traversals_success ON graph_traversals (success);
CREATE INDEX IF NOT EXISTS idx_graph_traversals_game ON graph_traversals (game_id, session_id);

CREATE INDEX IF NOT EXISTS idx_path_optimizations_graph ON path_optimizations (graph_id);
CREATE INDEX IF NOT EXISTS idx_path_optimizations_improvement ON path_optimizations (improvement_factor);

-- Tier 3 integration indexes
CREATE INDEX IF NOT EXISTS idx_bayesian_graph_integration_graph ON bayesian_graph_integration (graph_id);
CREATE INDEX IF NOT EXISTS idx_bayesian_graph_integration_hypothesis ON bayesian_graph_integration (hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_bayesian_graph_integration_game ON bayesian_graph_integration (game_id, session_id);

CREATE INDEX IF NOT EXISTS idx_tier3_attention_game ON tier3_attention_coordination (game_id, session_id);
CREATE INDEX IF NOT EXISTS idx_tier3_attention_complexity ON tier3_attention_coordination (reasoning_complexity, traversal_complexity);

CREATE INDEX IF NOT EXISTS idx_tier3_analytics_game ON tier3_performance_analytics (game_id, session_id);
CREATE INDEX IF NOT EXISTS idx_tier3_analytics_effectiveness ON tier3_performance_analytics (combined_system_effectiveness);
CREATE INDEX IF NOT EXISTS idx_tier3_analytics_timestamp ON tier3_performance_analytics (analysis_timestamp);