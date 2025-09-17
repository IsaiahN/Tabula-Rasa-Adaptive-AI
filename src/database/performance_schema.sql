-- Database schema for storing performance and session data
-- This replaces the growing in-memory data structures

-- Performance history table
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

-- Session history table
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

-- Action tracking table
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

-- Score history table (replaces the growing score_history lists)
CREATE TABLE IF NOT EXISTS score_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    session_id TEXT,
    score REAL,
    score_type TEXT, -- 'current', 'best', 'average', etc.
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Coordinate tracking table (replaces growing coordinate dictionaries)
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

-- Frame tracking table (replaces growing frame dictionaries)
CREATE TABLE IF NOT EXISTS frame_tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    frame_hash TEXT,
    frame_analysis TEXT, -- JSON analysis data
    stagnation_detected BOOLEAN,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_performance_session_id ON performance_history(session_id);
CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_session_session_id ON session_history(session_id);
CREATE INDEX IF NOT EXISTS idx_session_timestamp ON session_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_action_game_id ON action_tracking(game_id);
CREATE INDEX IF NOT EXISTS idx_action_timestamp ON action_tracking(timestamp);
CREATE INDEX IF NOT EXISTS idx_score_game_id ON score_history(game_id);
CREATE INDEX IF NOT EXISTS idx_score_timestamp ON score_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_coordinate_game_id ON coordinate_tracking(game_id);
CREATE INDEX IF NOT EXISTS idx_coordinate_timestamp ON coordinate_tracking(timestamp);
CREATE INDEX IF NOT EXISTS idx_frame_game_id ON frame_tracking(game_id);
CREATE INDEX IF NOT EXISTS idx_frame_timestamp ON frame_tracking(timestamp);

-- Cleanup policy: Keep only last 30 days of data by default
CREATE TRIGGER IF NOT EXISTS cleanup_old_performance_data
    AFTER INSERT ON performance_history
    BEGIN
        DELETE FROM performance_history 
        WHERE created_at < datetime('now', '-30 days');
    END;

CREATE TRIGGER IF NOT EXISTS cleanup_old_session_data
    AFTER INSERT ON session_history
    BEGIN
        DELETE FROM session_history 
        WHERE created_at < datetime('now', '-30 days');
    END;

CREATE TRIGGER IF NOT EXISTS cleanup_old_action_data
    AFTER INSERT ON action_tracking
    BEGIN
        DELETE FROM action_tracking 
        WHERE created_at < datetime('now', '-30 days');
    END;

CREATE TRIGGER IF NOT EXISTS cleanup_old_score_data
    AFTER INSERT ON score_history
    BEGIN
        DELETE FROM score_history 
        WHERE created_at < datetime('now', '-30 days');
    END;

CREATE TRIGGER IF NOT EXISTS cleanup_old_coordinate_data
    AFTER INSERT ON coordinate_tracking
    BEGIN
        DELETE FROM coordinate_tracking 
        WHERE created_at < datetime('now', '-30 days');
    END;

CREATE TRIGGER IF NOT EXISTS cleanup_old_frame_data
    AFTER INSERT ON frame_tracking
    BEGIN
        DELETE FROM frame_tracking 
        WHERE created_at < datetime('now', '-30 days');
    END;
