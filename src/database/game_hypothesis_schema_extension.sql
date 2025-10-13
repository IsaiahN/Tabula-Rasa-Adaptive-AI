-- ============================================================================
-- GAME-SPECIFIC HYPOTHESIS GENERATION AND TESTING SYSTEM SCHEMA EXTENSION
-- ============================================================================
--
-- This schema extension adds tables to support the Game-Specific Hypothesis
-- Generation and Testing System that automatically generates testable hypotheses
-- about winning strategies, tests them systematically, and provides explicit
-- reasoning for each action.
--
-- Key Components:
-- - Game hypotheses with database integration
-- - Detailed test experiment results
-- - Individual action records with reasoning
-- - Game mechanics classifications and profiles
-- - Multi-level learning support (micro, meso, macro)
--
-- ============================================================================

-- ============================================================================
-- GAME HYPOTHESIS TABLES
-- ============================================================================

-- Main table for storing generated game strategy hypotheses
CREATE TABLE IF NOT EXISTS game_hypotheses (
    hypothesis_id TEXT PRIMARY KEY,
    game_type TEXT NOT NULL, -- Game type from GameTypeClassifier (e.g., 'lp85', 'vc33')
    hypothesis_type TEXT NOT NULL, -- 'coordinate_sequence', 'pattern_completion', 'object_manipulation', etc.
    source TEXT NOT NULL, -- 'pattern_analysis', 'database_retrieval', 'game_type_knowledge', 'hybrid_approach'
    description TEXT NOT NULL,
    hypothesis_data TEXT NOT NULL, -- JSON: predicted_coordinates, expected_actions, game_mechanics, etc.
    confidence REAL NOT NULL, -- 0.0 to 1.0
    success_rate REAL DEFAULT 0.0, -- Updated as hypothesis is tested
    test_count INTEGER DEFAULT 0,
    reasoning TEXT NOT NULL, -- Human-readable explanation
    supporting_evidence TEXT, -- JSON: evidence used to generate hypothesis
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    is_active BOOLEAN DEFAULT 1
);

-- Table for storing detailed hypothesis test experiment results
CREATE TABLE IF NOT EXISTS hypothesis_test_results (
    experiment_id TEXT PRIMARY KEY,
    hypothesis_id TEXT NOT NULL,
    outcome TEXT NOT NULL, -- 'success', 'failure', 'partial_success', 'inconclusive', 'error'
    experiment_data TEXT NOT NULL, -- JSON: complete experiment details
    total_score_change REAL NOT NULL,
    test_duration REAL NOT NULL, -- in seconds
    actions_count INTEGER NOT NULL,
    success_actions_count INTEGER NOT NULL,
    learning_insights TEXT, -- JSON array of insights
    failure_reasons TEXT, -- JSON array of failure reasons
    success_factors TEXT, -- JSON array of success factors
    recommendations TEXT, -- JSON array of recommendations
    created_at TEXT NOT NULL,
    game_id TEXT,
    session_id TEXT,
    FOREIGN KEY (hypothesis_id) REFERENCES game_hypotheses (hypothesis_id)
);

-- Table for storing individual action records with detailed reasoning
CREATE TABLE IF NOT EXISTS action_reasoning_log (
    action_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    hypothesis_id TEXT NOT NULL,
    coordinate_x INTEGER NOT NULL,
    coordinate_y INTEGER NOT NULL,
    action_type TEXT NOT NULL,
    reasoning TEXT NOT NULL, -- Detailed explanation of why action was taken
    reason_category TEXT NOT NULL, -- 'hypothesis_prediction', 'evidence_gathering', 'pattern_exploration', etc.
    expected_outcome TEXT NOT NULL,
    actual_outcome TEXT NOT NULL,
    confidence_before REAL NOT NULL,
    confidence_after REAL NOT NULL,
    score_change REAL NOT NULL,
    success BOOLEAN NOT NULL,
    evidence_collected TEXT, -- JSON: evidence gathered from this action
    created_at TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES hypothesis_test_results (experiment_id),
    FOREIGN KEY (hypothesis_id) REFERENCES game_hypotheses (hypothesis_id)
);

-- ============================================================================
-- GAME MECHANICS CLASSIFICATION TABLES
-- ============================================================================

-- Table for storing game mechanics profiles from GamePatternAnalyzer
CREATE TABLE IF NOT EXISTS game_mechanics_profiles (
    profile_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    game_type TEXT NOT NULL,
    primary_mechanic TEXT NOT NULL, -- 'pattern_completion', 'physics_simulation', 'spatial_puzzle', etc.
    secondary_mechanics TEXT, -- JSON array of secondary mechanics
    mechanic_confidence TEXT NOT NULL, -- JSON: confidence scores for each mechanic
    visual_patterns TEXT NOT NULL, -- JSON: detected visual patterns
    grid_features TEXT NOT NULL, -- JSON: grid analysis features
    complexity_score REAL NOT NULL, -- 0.0 to 1.0
    analysis_timestamp TEXT NOT NULL,
    screenshot_hash TEXT, -- Hash of analyzed screenshot for caching
    created_at TEXT NOT NULL
);

-- Table for storing visual patterns detected by GamePatternAnalyzer
CREATE TABLE IF NOT EXISTS visual_patterns (
    pattern_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    pattern_type TEXT NOT NULL, -- 'circle', 'rectangle', 'color_region', 'row_sequence', etc.
    confidence REAL NOT NULL,
    location_x INTEGER NOT NULL,
    location_y INTEGER NOT NULL,
    size_width INTEGER NOT NULL,
    size_height INTEGER NOT NULL,
    features TEXT, -- JSON: pattern-specific features
    created_at TEXT NOT NULL,
    FOREIGN KEY (profile_id) REFERENCES game_mechanics_profiles (profile_id)
);

-- ============================================================================
-- HYPOTHESIS GENERATION TRACKING TABLES
-- ============================================================================

-- Table for tracking hypothesis generation sessions and performance
CREATE TABLE IF NOT EXISTS hypothesis_generation_sessions (
    session_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    game_type TEXT NOT NULL,
    hypotheses_generated INTEGER NOT NULL,
    generation_strategies_used TEXT NOT NULL, -- JSON array
    pattern_analysis_time REAL NOT NULL, -- seconds
    database_query_time REAL NOT NULL, -- seconds
    total_generation_time REAL NOT NULL, -- seconds
    top_hypothesis_confidence REAL NOT NULL,
    average_hypothesis_confidence REAL NOT NULL,
    session_timestamp TEXT NOT NULL,
    screenshot_analyzed BOOLEAN DEFAULT 0
);

-- Table for tracking multi-level learning insights
CREATE TABLE IF NOT EXISTS multi_level_learning (
    learning_id TEXT PRIMARY KEY,
    learning_level TEXT NOT NULL, -- 'micro', 'meso', 'macro'
    game_id TEXT,
    game_type TEXT,
    session_id TEXT,
    learning_context TEXT NOT NULL, -- 'within_game', 'across_attempts', 'across_games'
    insight_type TEXT NOT NULL, -- 'pattern_recognition', 'strategy_effectiveness', 'mechanic_understanding'
    insight_description TEXT NOT NULL,
    supporting_data TEXT, -- JSON: data supporting the insight
    confidence REAL NOT NULL,
    impact_score REAL NOT NULL, -- How much this insight affects future decisions
    created_at TEXT NOT NULL,
    applied_count INTEGER DEFAULT 0, -- How many times this insight has been applied
    success_when_applied INTEGER DEFAULT 0 -- How many times it led to success
);

-- ============================================================================
-- INTEGRATION TABLES
-- ============================================================================

-- Table for tracking integration with Action6Coordinator
CREATE TABLE IF NOT EXISTS action6_integration_log (
    integration_id TEXT PRIMARY KEY,
    hypothesis_id TEXT NOT NULL,
    action_id TEXT NOT NULL,
    action6_coordinate_x INTEGER NOT NULL,
    action6_coordinate_y INTEGER NOT NULL,
    action6_result TEXT NOT NULL, -- JSON: result from Action6Coordinator
    hypothesis_prediction TEXT NOT NULL, -- What hypothesis predicted
    prediction_accuracy REAL NOT NULL, -- How accurate was the prediction
    integration_effectiveness REAL NOT NULL, -- How well integration worked
    created_at TEXT NOT NULL,
    FOREIGN KEY (hypothesis_id) REFERENCES game_hypotheses (hypothesis_id),
    FOREIGN KEY (action_id) REFERENCES action_reasoning_log (action_id)
);

-- Table for tracking Enhanced Gameplay integration
CREATE TABLE IF NOT EXISTS enhanced_gameplay_integration (
    integration_id TEXT PRIMARY KEY,
    hypothesis_id TEXT NOT NULL,
    gameplay_phase TEXT NOT NULL, -- Which phase of enhanced gameplay
    hypothesis_contribution TEXT NOT NULL, -- How hypothesis contributed
    gameplay_improvement REAL NOT NULL, -- 0.0 to 1.0
    feedback_data TEXT, -- JSON: feedback from Enhanced Gameplay
    adaptation_made BOOLEAN DEFAULT 0, -- Whether hypothesis was adapted based on feedback
    created_at TEXT NOT NULL,
    FOREIGN KEY (hypothesis_id) REFERENCES game_hypotheses (hypothesis_id)
);

-- ============================================================================
-- PERFORMANCE ANALYTICS TABLES
-- ============================================================================

-- Table for tracking overall system performance
CREATE TABLE IF NOT EXISTS hypothesis_system_analytics (
    analytics_id TEXT PRIMARY KEY,
    game_type TEXT NOT NULL,
    session_id TEXT,
    hypotheses_generated INTEGER NOT NULL,
    hypotheses_tested INTEGER NOT NULL,
    successful_hypotheses INTEGER NOT NULL,
    average_confidence REAL NOT NULL,
    average_test_duration REAL NOT NULL,
    total_score_improvement REAL NOT NULL,
    pattern_analysis_accuracy REAL NOT NULL,
    database_retrieval_effectiveness REAL NOT NULL,
    action_reasoning_quality REAL NOT NULL, -- Based on success correlation
    multi_level_learning_impact REAL NOT NULL,
    analytics_timestamp TEXT NOT NULL
);

-- Table for storing hypothesis effectiveness by game mechanics
CREATE TABLE IF NOT EXISTS hypothesis_effectiveness_by_mechanic (
    effectiveness_id TEXT PRIMARY KEY,
    game_mechanic TEXT NOT NULL,
    hypothesis_type TEXT NOT NULL,
    total_tests INTEGER NOT NULL,
    successful_tests INTEGER NOT NULL,
    success_rate REAL NOT NULL,
    average_score_change REAL NOT NULL,
    average_confidence REAL NOT NULL,
    best_performing_strategy TEXT, -- JSON: details of best strategy
    common_failure_reasons TEXT, -- JSON array
    last_updated TEXT NOT NULL
);

-- ============================================================================
-- CACHING AND OPTIMIZATION TABLES
-- ============================================================================

-- Table for caching pattern analysis results
CREATE TABLE IF NOT EXISTS pattern_analysis_cache (
    cache_id TEXT PRIMARY KEY,
    screenshot_hash TEXT UNIQUE NOT NULL,
    game_type TEXT NOT NULL,
    analysis_result TEXT NOT NULL, -- JSON: complete GameMechanicsProfile
    cache_hit_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    expires_at TEXT NOT NULL -- For cache invalidation
);

-- Table for caching successful hypothesis patterns
CREATE TABLE IF NOT EXISTS successful_hypothesis_cache (
    cache_id TEXT PRIMARY KEY,
    game_type TEXT NOT NULL,
    mechanic_signature TEXT NOT NULL, -- Hash of key mechanics features
    successful_hypothesis TEXT NOT NULL, -- JSON: hypothesis that worked
    success_count INTEGER NOT NULL,
    average_score_improvement REAL NOT NULL,
    last_successful_use TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE OPTIMIZATION
-- ============================================================================

-- Game hypotheses indexes
CREATE INDEX IF NOT EXISTS idx_game_hypotheses_game_type ON game_hypotheses (game_type);
CREATE INDEX IF NOT EXISTS idx_game_hypotheses_type ON game_hypotheses (hypothesis_type);
CREATE INDEX IF NOT EXISTS idx_game_hypotheses_source ON game_hypotheses (source);
CREATE INDEX IF NOT EXISTS idx_game_hypotheses_confidence ON game_hypotheses (confidence);
CREATE INDEX IF NOT EXISTS idx_game_hypotheses_success_rate ON game_hypotheses (success_rate);
CREATE INDEX IF NOT EXISTS idx_game_hypotheses_active ON game_hypotheses (is_active);
CREATE INDEX IF NOT EXISTS idx_game_hypotheses_updated ON game_hypotheses (updated_at);

-- Hypothesis test results indexes
CREATE INDEX IF NOT EXISTS idx_hypothesis_test_results_hypothesis ON hypothesis_test_results (hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_hypothesis_test_results_outcome ON hypothesis_test_results (outcome);
CREATE INDEX IF NOT EXISTS idx_hypothesis_test_results_score ON hypothesis_test_results (total_score_change);
CREATE INDEX IF NOT EXISTS idx_hypothesis_test_results_game ON hypothesis_test_results (game_id, session_id);
CREATE INDEX IF NOT EXISTS idx_hypothesis_test_results_created ON hypothesis_test_results (created_at);

-- Action reasoning log indexes
CREATE INDEX IF NOT EXISTS idx_action_reasoning_experiment ON action_reasoning_log (experiment_id);
CREATE INDEX IF NOT EXISTS idx_action_reasoning_hypothesis ON action_reasoning_log (hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_action_reasoning_coordinate ON action_reasoning_log (coordinate_x, coordinate_y);
CREATE INDEX IF NOT EXISTS idx_action_reasoning_type ON action_reasoning_log (action_type);
CREATE INDEX IF NOT EXISTS idx_action_reasoning_category ON action_reasoning_log (reason_category);
CREATE INDEX IF NOT EXISTS idx_action_reasoning_success ON action_reasoning_log (success);
CREATE INDEX IF NOT EXISTS idx_action_reasoning_created ON action_reasoning_log (created_at);

-- Game mechanics profiles indexes
CREATE INDEX IF NOT EXISTS idx_game_mechanics_profiles_game ON game_mechanics_profiles (game_id);
CREATE INDEX IF NOT EXISTS idx_game_mechanics_profiles_type ON game_mechanics_profiles (game_type);
CREATE INDEX IF NOT EXISTS idx_game_mechanics_profiles_mechanic ON game_mechanics_profiles (primary_mechanic);
CREATE INDEX IF NOT EXISTS idx_game_mechanics_profiles_complexity ON game_mechanics_profiles (complexity_score);
CREATE INDEX IF NOT EXISTS idx_game_mechanics_profiles_hash ON game_mechanics_profiles (screenshot_hash);
CREATE INDEX IF NOT EXISTS idx_game_mechanics_profiles_timestamp ON game_mechanics_profiles (analysis_timestamp);

-- Visual patterns indexes
CREATE INDEX IF NOT EXISTS idx_visual_patterns_profile ON visual_patterns (profile_id);
CREATE INDEX IF NOT EXISTS idx_visual_patterns_type ON visual_patterns (pattern_type);
CREATE INDEX IF NOT EXISTS idx_visual_patterns_confidence ON visual_patterns (confidence);
CREATE INDEX IF NOT EXISTS idx_visual_patterns_location ON visual_patterns (location_x, location_y);

-- Hypothesis generation sessions indexes
CREATE INDEX IF NOT EXISTS idx_hypothesis_generation_game ON hypothesis_generation_sessions (game_id);
CREATE INDEX IF NOT EXISTS idx_hypothesis_generation_type ON hypothesis_generation_sessions (game_type);
CREATE INDEX IF NOT EXISTS idx_hypothesis_generation_timestamp ON hypothesis_generation_sessions (session_timestamp);
CREATE INDEX IF NOT EXISTS idx_hypothesis_generation_confidence ON hypothesis_generation_sessions (top_hypothesis_confidence);

-- Multi-level learning indexes
CREATE INDEX IF NOT EXISTS idx_multi_level_learning_level ON multi_level_learning (learning_level);
CREATE INDEX IF NOT EXISTS idx_multi_level_learning_context ON multi_level_learning (learning_context);
CREATE INDEX IF NOT EXISTS idx_multi_level_learning_type ON multi_level_learning (insight_type);
CREATE INDEX IF NOT EXISTS idx_multi_level_learning_game ON multi_level_learning (game_id, game_type);
CREATE INDEX IF NOT EXISTS idx_multi_level_learning_confidence ON multi_level_learning (confidence);
CREATE INDEX IF NOT EXISTS idx_multi_level_learning_impact ON multi_level_learning (impact_score);
CREATE INDEX IF NOT EXISTS idx_multi_level_learning_applied ON multi_level_learning (applied_count, success_when_applied);

-- Integration indexes
CREATE INDEX IF NOT EXISTS idx_action6_integration_hypothesis ON action6_integration_log (hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_action6_integration_action ON action6_integration_log (action_id);
CREATE INDEX IF NOT EXISTS idx_action6_integration_accuracy ON action6_integration_log (prediction_accuracy);
CREATE INDEX IF NOT EXISTS idx_action6_integration_effectiveness ON action6_integration_log (integration_effectiveness);

CREATE INDEX IF NOT EXISTS idx_enhanced_gameplay_integration_hypothesis ON enhanced_gameplay_integration (hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_enhanced_gameplay_integration_phase ON enhanced_gameplay_integration (gameplay_phase);
CREATE INDEX IF NOT EXISTS idx_enhanced_gameplay_integration_improvement ON enhanced_gameplay_integration (gameplay_improvement);

-- Analytics indexes
CREATE INDEX IF NOT EXISTS idx_hypothesis_system_analytics_type ON hypothesis_system_analytics (game_type);
CREATE INDEX IF NOT EXISTS idx_hypothesis_system_analytics_session ON hypothesis_system_analytics (session_id);
CREATE INDEX IF NOT EXISTS idx_hypothesis_system_analytics_timestamp ON hypothesis_system_analytics (analytics_timestamp);
CREATE INDEX IF NOT EXISTS idx_hypothesis_system_analytics_score ON hypothesis_system_analytics (total_score_improvement);

CREATE INDEX IF NOT EXISTS idx_hypothesis_effectiveness_mechanic ON hypothesis_effectiveness_by_mechanic (game_mechanic);
CREATE INDEX IF NOT EXISTS idx_hypothesis_effectiveness_type ON hypothesis_effectiveness_by_mechanic (hypothesis_type);
CREATE INDEX IF NOT EXISTS idx_hypothesis_effectiveness_success ON hypothesis_effectiveness_by_mechanic (success_rate);
CREATE INDEX IF NOT EXISTS idx_hypothesis_effectiveness_updated ON hypothesis_effectiveness_by_mechanic (last_updated);

-- Cache indexes
CREATE INDEX IF NOT EXISTS idx_pattern_analysis_cache_hash ON pattern_analysis_cache (screenshot_hash);
CREATE INDEX IF NOT EXISTS idx_pattern_analysis_cache_type ON pattern_analysis_cache (game_type);
CREATE INDEX IF NOT EXISTS idx_pattern_analysis_cache_hits ON pattern_analysis_cache (cache_hit_count);
CREATE INDEX IF NOT EXISTS idx_pattern_analysis_cache_expires ON pattern_analysis_cache (expires_at);

CREATE INDEX IF NOT EXISTS idx_successful_hypothesis_cache_type ON successful_hypothesis_cache (game_type);
CREATE INDEX IF NOT EXISTS idx_successful_hypothesis_cache_signature ON successful_hypothesis_cache (mechanic_signature);
CREATE INDEX IF NOT EXISTS idx_successful_hypothesis_cache_success ON successful_hypothesis_cache (success_count);
CREATE INDEX IF NOT EXISTS idx_successful_hypothesis_cache_last_use ON successful_hypothesis_cache (last_successful_use);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View for hypothesis performance summary
CREATE VIEW IF NOT EXISTS hypothesis_performance_summary AS
SELECT
    h.hypothesis_id,
    h.game_type,
    h.hypothesis_type,
    h.source,
    h.confidence,
    h.success_rate,
    h.test_count,
    COUNT(htr.experiment_id) as experiments_count,
    AVG(htr.total_score_change) as avg_score_change,
    AVG(htr.test_duration) as avg_test_duration,
    SUM(CASE WHEN htr.outcome = 'success' THEN 1 ELSE 0 END) as successful_experiments,
    h.created_at,
    h.updated_at
FROM game_hypotheses h
LEFT JOIN hypothesis_test_results htr ON h.hypothesis_id = htr.hypothesis_id
WHERE h.is_active = 1
GROUP BY h.hypothesis_id;

-- View for action reasoning analysis
CREATE VIEW IF NOT EXISTS action_reasoning_analysis AS
SELECT
    arl.reason_category,
    arl.action_type,
    COUNT(*) as total_actions,
    SUM(CASE WHEN arl.success = 1 THEN 1 ELSE 0 END) as successful_actions,
    ROUND(AVG(CASE WHEN arl.success = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) as success_rate_percent,
    AVG(arl.score_change) as avg_score_change,
    AVG(arl.confidence_after - arl.confidence_before) as avg_confidence_change
FROM action_reasoning_log arl
GROUP BY arl.reason_category, arl.action_type
ORDER BY success_rate_percent DESC, avg_score_change DESC;

-- View for game mechanics effectiveness
CREATE VIEW IF NOT EXISTS game_mechanics_effectiveness AS
SELECT
    gmp.primary_mechanic,
    gmp.game_type,
    COUNT(DISTINCT gmp.game_id) as games_analyzed,
    AVG(gmp.complexity_score) as avg_complexity,
    COUNT(DISTINCT h.hypothesis_id) as hypotheses_generated,
    AVG(h.success_rate) as avg_hypothesis_success_rate,
    SUM(CASE WHEN htr.outcome = 'success' THEN 1 ELSE 0 END) as successful_tests,
    COUNT(htr.experiment_id) as total_tests
FROM game_mechanics_profiles gmp
LEFT JOIN game_hypotheses h ON gmp.game_type = h.game_type
LEFT JOIN hypothesis_test_results htr ON h.hypothesis_id = htr.hypothesis_id
GROUP BY gmp.primary_mechanic, gmp.game_type
ORDER BY avg_hypothesis_success_rate DESC, successful_tests DESC;

-- ============================================================================
-- TRIGGERS FOR DATA CONSISTENCY
-- ============================================================================

-- Trigger to update game_hypotheses success_rate when test results are inserted
CREATE TRIGGER IF NOT EXISTS update_hypothesis_success_rate
AFTER INSERT ON hypothesis_test_results
BEGIN
    UPDATE game_hypotheses
    SET
        success_rate = (
            SELECT
                CAST(SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) AS REAL) / COUNT(*)
            FROM hypothesis_test_results
            WHERE hypothesis_id = NEW.hypothesis_id
        ),
        test_count = (
            SELECT COUNT(*)
            FROM hypothesis_test_results
            WHERE hypothesis_id = NEW.hypothesis_id
        ),
        updated_at = datetime('now')
    WHERE hypothesis_id = NEW.hypothesis_id;
END;

-- Trigger to update cache hit counts
CREATE TRIGGER IF NOT EXISTS update_cache_hit_count
AFTER UPDATE ON pattern_analysis_cache
WHEN NEW.last_accessed > OLD.last_accessed
BEGIN
    UPDATE pattern_analysis_cache
    SET cache_hit_count = cache_hit_count + 1
    WHERE cache_id = NEW.cache_id;
END;

-- ============================================================================
-- INITIALIZATION COMMENTS
-- ============================================================================

-- This schema extension provides comprehensive support for:
-- 1. Game-specific hypothesis generation and storage
-- 2. Detailed experiment tracking with action reasoning
-- 3. Visual pattern analysis and game mechanics classification
-- 4. Multi-level learning insights (micro, meso, macro)
-- 5. Integration tracking with existing systems
-- 6. Performance analytics and optimization
-- 7. Intelligent caching for improved performance
--
-- The schema is designed to work alongside existing database tables
-- and provides the foundation for the Game-Specific Hypothesis
-- Generation and Testing System.