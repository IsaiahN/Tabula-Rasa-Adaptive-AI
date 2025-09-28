-- ============================================================================
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