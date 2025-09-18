# Director Analytics Queries
## Comprehensive Database Queries for Deep Gameplay Insights

This document contains powerful SQL queries designed to give the Director deep understanding of gameplay state, progression patterns, and strategic improvement opportunities. These queries are organized by category and purpose.

---

## 🎯 **GAMEPLAY STATE & PROGRESSION**

### Current Session Overview
```sql
-- Get real-time session status and performance
SELECT 
    ts.session_id,
    ts.mode,
    ts.status,
    ts.total_actions,
    ts.total_wins,
    ts.total_games,
    ROUND(ts.win_rate * 100, 2) as win_rate_percent,
    ROUND(ts.avg_score, 2) as avg_score,
    ts.energy_level,
    ts.memory_operations,
    ts.sleep_cycles,
    datetime(ts.start_time) as session_start,
    CASE 
        WHEN ts.end_time IS NOT NULL 
        THEN ROUND((julianday(ts.end_time) - julianday(ts.start_time)) * 24 * 60, 2)
        ELSE ROUND((julianday('now') - julianday(ts.start_time)) * 24 * 60, 2)
    END as duration_minutes
FROM training_sessions ts
WHERE ts.status = 'running'
ORDER BY ts.start_time DESC
LIMIT 10;
```

### Game-Specific Progress Analysis
```sql
-- Analyze progress for specific games
SELECT 
    gr.game_id,
    gr.session_id,
    gr.status,
    gr.final_score,
    gr.total_actions,
    gr.win_detected,
    gr.level_completions,
    gr.coordinate_attempts,
    gr.coordinate_successes,
    ROUND(CAST(gr.coordinate_successes AS FLOAT) / NULLIF(gr.coordinate_attempts, 0) * 100, 2) as coordinate_success_rate,
    gr.frame_changes,
    datetime(gr.start_time) as game_start,
    CASE 
        WHEN gr.end_time IS NOT NULL 
        THEN ROUND((julianday(gr.end_time) - julianday(gr.start_time)) * 24 * 60, 2)
        ELSE ROUND((julianday('now') - julianday(gr.start_time)) * 24 * 60, 2)
    END as duration_minutes
FROM game_results gr
WHERE gr.game_id LIKE '%sp80%' OR gr.game_id LIKE '%vc33%'  -- Filter for specific game types
ORDER BY gr.start_time DESC
LIMIT 20;
```

### Performance Trends Over Time
```sql
-- Track performance trends across sessions
SELECT 
    DATE(ts.start_time) as session_date,
    COUNT(*) as sessions_count,
    ROUND(AVG(ts.win_rate) * 100, 2) as avg_win_rate,
    ROUND(AVG(ts.avg_score), 2) as avg_score,
    ROUND(AVG(ts.total_actions), 0) as avg_actions,
    ROUND(AVG(ts.energy_level), 1) as avg_energy,
    SUM(ts.total_wins) as total_wins,
    SUM(ts.total_games) as total_games
FROM training_sessions ts
WHERE ts.start_time >= datetime('now', '-7 days')
GROUP BY DATE(ts.start_time)
ORDER BY session_date DESC;
```

---

## 🧠 **ACTION INTELLIGENCE & STRATEGY**

### Most Effective Actions by Game Type
```sql
-- Find the most effective actions for different game types
SELECT 
    SUBSTR(ae.game_id, 1, 4) as game_type,
    ae.action_number,
    COUNT(*) as games_played,
    SUM(ae.attempts) as total_attempts,
    SUM(ae.successes) as total_successes,
    ROUND(AVG(ae.success_rate) * 100, 2) as avg_success_rate,
    ROUND(AVG(ae.avg_score_impact), 2) as avg_score_impact,
    MAX(ae.last_used) as last_used
FROM action_effectiveness ae
WHERE ae.attempts > 0
GROUP BY SUBSTR(ae.game_id, 1, 4), ae.action_number
HAVING total_attempts >= 5  -- Minimum attempts for statistical significance
ORDER BY game_type, avg_success_rate DESC;
```

### Coordinate Intelligence Analysis
```sql
-- Analyze coordinate effectiveness patterns
SELECT 
    SUBSTR(ci.game_id, 1, 4) as game_type,
    ci.x,
    ci.y,
    COUNT(*) as games_played,
    SUM(ci.attempts) as total_attempts,
    SUM(ci.successes) as total_successes,
    ROUND(AVG(ci.success_rate) * 100, 2) as avg_success_rate,
    ROUND(AVG(ci.frame_changes), 1) as avg_frame_changes,
    MAX(ci.last_used) as last_used
FROM coordinate_intelligence ci
WHERE ci.attempts > 0
GROUP BY SUBSTR(ci.game_id, 1, 4), ci.x, ci.y
HAVING total_attempts >= 3
ORDER BY game_type, avg_success_rate DESC, total_attempts DESC
LIMIT 50;
```

### Winning Sequence Analysis
```sql
-- Find the most successful action sequences
SELECT 
    ws.game_id,
    ws.sequence,
    ws.frequency,
    ROUND(ws.avg_score, 2) as avg_score,
    ROUND(ws.success_rate * 100, 2) as success_rate_percent,
    ws.last_used,
    LENGTH(ws.sequence) - LENGTH(REPLACE(ws.sequence, ',', '')) + 1 as sequence_length
FROM winning_sequences ws
WHERE ws.frequency > 1
ORDER BY ws.success_rate DESC, ws.frequency DESC, ws.avg_score DESC
LIMIT 20;
```

### Action Pattern Evolution
```sql
-- Track how action effectiveness changes over time
SELECT 
    ae.action_number,
    DATE(ae.updated_at) as date,
    COUNT(*) as games_count,
    ROUND(AVG(ae.success_rate) * 100, 2) as avg_success_rate,
    ROUND(AVG(ae.avg_score_impact), 2) as avg_score_impact,
    SUM(ae.attempts) as total_attempts,
    SUM(ae.successes) as total_successes
FROM action_effectiveness ae
WHERE ae.updated_at >= datetime('now', '-7 days')
GROUP BY ae.action_number, DATE(ae.updated_at)
ORDER BY ae.action_number, date DESC;
```

---

## 🔍 **ERROR DETECTION & DEBUGGING**

### Critical Error Analysis
```sql
-- Find the most frequent and critical errors
SELECT 
    el.error_type,
    el.error_message,
    el.occurrence_count,
    el.first_seen,
    el.last_seen,
    el.resolved,
    ROUND((julianday('now') - julianday(el.first_seen)) * 24, 1) as hours_since_first,
    ROUND((julianday('now') - julianday(el.last_seen)) * 24, 1) as hours_since_last
FROM error_logs el
WHERE el.resolved = FALSE
ORDER BY el.occurrence_count DESC, el.last_seen DESC
LIMIT 20;
```

### System Health Monitoring
```sql
-- Monitor system component health
SELECT 
    sl.component,
    sl.log_level,
    COUNT(*) as log_count,
    MAX(sl.timestamp) as last_log,
    ROUND((julianday('now') - julianday(MAX(sl.timestamp))) * 24 * 60, 1) as minutes_since_last_log
FROM system_logs sl
WHERE sl.timestamp >= datetime('now', '-1 hour')
GROUP BY sl.component, sl.log_level
ORDER BY sl.component, 
    CASE sl.log_level 
        WHEN 'CRITICAL' THEN 1 
        WHEN 'ERROR' THEN 2 
        WHEN 'WARNING' THEN 3 
        WHEN 'INFO' THEN 4 
        WHEN 'DEBUG' THEN 5 
    END;
```

### Performance Degradation Detection
```sql
-- Detect performance degradation patterns
SELECT 
    gr.game_id,
    gr.session_id,
    gr.final_score,
    gr.total_actions,
    gr.win_detected,
    ROUND(CAST(gr.coordinate_successes AS FLOAT) / NULLIF(gr.coordinate_attempts, 0) * 100, 2) as coordinate_success_rate,
    gr.frame_changes,
    datetime(gr.start_time) as game_start,
    CASE 
        WHEN gr.final_score = 0 AND gr.total_actions > 50 THEN 'High Actions, No Score'
        WHEN gr.coordinate_attempts > 0 AND gr.coordinate_successes = 0 THEN 'Coordinate Failures'
        WHEN gr.frame_changes = 0 AND gr.total_actions > 10 THEN 'No Frame Changes'
        ELSE 'Normal'
    END as issue_type
FROM game_results gr
WHERE gr.start_time >= datetime('now', '-24 hours')
    AND (gr.final_score = 0 AND gr.total_actions > 50)
    OR (gr.coordinate_attempts > 0 AND gr.coordinate_successes = 0)
    OR (gr.frame_changes = 0 AND gr.total_actions > 10)
ORDER BY gr.start_time DESC;
```

---

## 📊 **LEARNING & PATTERN ANALYSIS**

### Learned Pattern Effectiveness
```sql
-- Analyze the effectiveness of learned patterns
SELECT 
    lp.pattern_type,
    COUNT(*) as pattern_count,
    ROUND(AVG(lp.confidence), 3) as avg_confidence,
    ROUND(AVG(lp.success_rate), 3) as avg_success_rate,
    SUM(lp.frequency) as total_usage,
    MAX(lp.updated_at) as last_updated
FROM learned_patterns lp
GROUP BY lp.pattern_type
ORDER BY avg_success_rate DESC, total_usage DESC;
```

### Learning Progress Tracking
```sql
-- Track learning progress over time
SELECT 
    DATE(ph.timestamp) as date,
    COUNT(DISTINCT ph.session_id) as sessions,
    ROUND(AVG(ph.win_rate) * 100, 2) as avg_win_rate,
    ROUND(AVG(ph.learning_efficiency), 3) as avg_learning_efficiency,
    ROUND(AVG(ph.score), 2) as avg_score,
    COUNT(*) as data_points
FROM performance_history ph
WHERE ph.timestamp >= datetime('now', '-7 days')
GROUP BY DATE(ph.timestamp)
ORDER BY date DESC;
```

### Stagnation Detection
```sql
-- Detect games that might be stuck or stagnating
SELECT 
    ft.game_id,
    ft.stagnation_detected,
    COUNT(*) as stagnation_events,
    MAX(ft.timestamp) as last_stagnation,
    ROUND((julianday('now') - julianday(MAX(ft.timestamp))) * 24 * 60, 1) as minutes_since_stagnation
FROM frame_tracking ft
WHERE ft.timestamp >= datetime('now', '-1 hour')
    AND ft.stagnation_detected = TRUE
GROUP BY ft.game_id
ORDER BY stagnation_events DESC, last_stagnation DESC;
```

---

## 🎮 **GAME-SPECIFIC INSIGHTS**

### Game Difficulty Analysis
```sql
-- Analyze game difficulty based on success rates
SELECT 
    SUBSTR(gr.game_id, 1, 4) as game_type,
    COUNT(*) as total_games,
    SUM(CASE WHEN gr.win_detected = TRUE THEN 1 ELSE 0 END) as wins,
    ROUND(CAST(SUM(CASE WHEN gr.win_detected = TRUE THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100, 2) as win_rate_percent,
    ROUND(AVG(gr.final_score), 2) as avg_score,
    ROUND(AVG(gr.total_actions), 1) as avg_actions,
    ROUND(AVG(gr.coordinate_successes), 1) as avg_coordinate_successes,
    ROUND(AVG(gr.frame_changes), 1) as avg_frame_changes
FROM game_results gr
WHERE gr.start_time >= datetime('now', '-7 days')
GROUP BY SUBSTR(gr.game_id, 1, 4)
ORDER BY win_rate_percent ASC, avg_actions DESC;
```

### Action Sequence Success Patterns
```sql
-- Find successful action sequences by game type
SELECT 
    SUBSTR(at.game_id, 1, 4) as game_type,
    at.action_sequence,
    COUNT(*) as usage_count,
    ROUND(AVG(at.effectiveness), 3) as avg_effectiveness,
    MAX(at.timestamp) as last_used
FROM action_tracking at
WHERE at.effectiveness > 0.5
    AND at.timestamp >= datetime('now', '-7 days')
GROUP BY SUBSTR(at.game_id, 1, 4), at.action_sequence
HAVING usage_count >= 2
ORDER BY game_type, avg_effectiveness DESC, usage_count DESC
LIMIT 30;
```

### Coordinate Hotspots
```sql
-- Find coordinate hotspots (frequently used coordinates)
SELECT 
    SUBSTR(ct.game_id, 1, 4) as game_type,
    ct.coordinate_x,
    ct.coordinate_y,
    COUNT(*) as usage_count,
    SUM(CASE WHEN ct.success = TRUE THEN 1 ELSE 0 END) as success_count,
    ROUND(CAST(SUM(CASE WHEN ct.success = TRUE THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100, 2) as success_rate_percent,
    MAX(ct.timestamp) as last_used
FROM coordinate_tracking ct
WHERE ct.timestamp >= datetime('now', '-7 days')
GROUP BY SUBSTR(ct.game_id, 1, 4), ct.coordinate_x, ct.coordinate_y
HAVING usage_count >= 3
ORDER BY game_type, success_rate_percent DESC, usage_count DESC
LIMIT 50;
```

---

## 🔧 **SYSTEM OPTIMIZATION QUERIES**

### Memory Usage Analysis
```sql
-- Analyze memory operations and efficiency
SELECT 
    ts.session_id,
    ts.memory_operations,
    ts.sleep_cycles,
    ts.total_actions,
    ROUND(CAST(ts.memory_operations AS FLOAT) / NULLIF(ts.total_actions, 0), 2) as memory_per_action,
    ROUND(CAST(ts.sleep_cycles AS FLOAT) / NULLIF(ts.total_actions, 0), 4) as sleep_per_action,
    ts.energy_level,
    ts.win_rate
FROM training_sessions ts
WHERE ts.start_time >= datetime('now', '-24 hours')
ORDER BY ts.start_time DESC;
```

### Energy Management Analysis
```sql
-- Analyze energy usage patterns
SELECT 
    ts.session_id,
    ts.energy_level,
    ts.total_actions,
    ts.sleep_cycles,
    ROUND(ts.energy_level / NULLIF(ts.total_actions, 0), 3) as energy_per_action,
    ts.win_rate,
    ts.avg_score
FROM training_sessions ts
WHERE ts.start_time >= datetime('now', '-24 hours')
    AND ts.status = 'running'
ORDER BY ts.energy_level ASC, ts.start_time DESC;
```

### Learning Efficiency Metrics
```sql
-- Calculate learning efficiency metrics
SELECT 
    ph.session_id,
    ph.game_id,
    ph.win_rate,
    ph.learning_efficiency,
    ph.score,
    ROUND(ph.learning_efficiency * ph.win_rate, 3) as efficiency_score,
    ph.timestamp
FROM performance_history ph
WHERE ph.timestamp >= datetime('now', '-24 hours')
ORDER BY efficiency_score DESC, ph.timestamp DESC
LIMIT 20;
```

---

## 🚀 **STRATEGIC IMPROVEMENT QUERIES**

### Underperforming Actions
```sql
-- Find actions that are underperforming and need attention
SELECT 
    ae.action_number,
    SUBSTR(ae.game_id, 1, 4) as game_type,
    COUNT(*) as games_played,
    SUM(ae.attempts) as total_attempts,
    SUM(ae.successes) as total_successes,
    ROUND(AVG(ae.success_rate) * 100, 2) as success_rate_percent,
    ROUND(AVG(ae.avg_score_impact), 2) as avg_score_impact
FROM action_effectiveness ae
WHERE ae.attempts >= 10  -- Minimum attempts for statistical significance
GROUP BY ae.action_number, SUBSTR(ae.game_id, 1, 4)
HAVING success_rate_percent < 20  -- Low success rate threshold
ORDER BY success_rate_percent ASC, total_attempts DESC;
```

### High-Potential Coordinates
```sql
-- Find coordinates with high success rates but low usage (untapped potential)
SELECT 
    ci.x,
    ci.y,
    SUBSTR(ci.game_id, 1, 4) as game_type,
    COUNT(*) as games_played,
    SUM(ci.attempts) as total_attempts,
    SUM(ci.successes) as total_successes,
    ROUND(AVG(ci.success_rate) * 100, 2) as success_rate_percent,
    ROUND(AVG(ci.frame_changes), 1) as avg_frame_changes
FROM coordinate_intelligence ci
WHERE ci.attempts >= 3
GROUP BY ci.x, ci.y, SUBSTR(ci.game_id, 1, 4)
HAVING success_rate_percent > 70  -- High success rate
    AND total_attempts < 20  -- Low usage
ORDER BY success_rate_percent DESC, total_attempts ASC
LIMIT 30;
```

### Learning Acceleration Opportunities
```sql
-- Find patterns that could accelerate learning
SELECT 
    lp.pattern_type,
    lp.confidence,
    lp.success_rate,
    lp.frequency,
    ROUND(lp.confidence * lp.success_rate * lp.frequency, 3) as learning_potential,
    lp.game_context,
    lp.updated_at
FROM learned_patterns lp
WHERE lp.confidence > 0.7
    AND lp.success_rate > 0.6
    AND lp.frequency > 1
ORDER BY learning_potential DESC, lp.updated_at DESC
LIMIT 20;
```

---

## 📈 **REAL-TIME MONITORING QUERIES**

### Live Session Dashboard
```sql
-- Real-time dashboard for active sessions
SELECT 
    ts.session_id,
    ts.mode,
    ts.total_actions,
    ts.total_wins,
    ts.total_games,
    ROUND(ts.win_rate * 100, 2) as win_rate_percent,
    ts.energy_level,
    ts.memory_operations,
    ROUND((julianday('now') - julianday(ts.start_time)) * 24 * 60, 1) as minutes_running,
    COUNT(gr.game_id) as active_games
FROM training_sessions ts
LEFT JOIN game_results gr ON ts.session_id = gr.session_id AND gr.end_time IS NULL
WHERE ts.status = 'running'
GROUP BY ts.session_id
ORDER BY ts.start_time DESC;
```

### Recent Performance Summary
```sql
-- Quick performance summary for the last hour
SELECT 
    'Last Hour' as period,
    COUNT(DISTINCT ts.session_id) as active_sessions,
    SUM(ts.total_actions) as total_actions,
    SUM(ts.total_wins) as total_wins,
    SUM(ts.total_games) as total_games,
    ROUND(CAST(SUM(ts.total_wins) AS FLOAT) / NULLIF(SUM(ts.total_games), 0) * 100, 2) as overall_win_rate,
    ROUND(AVG(ts.avg_score), 2) as avg_score,
    ROUND(AVG(ts.energy_level), 1) as avg_energy
FROM training_sessions ts
WHERE ts.start_time >= datetime('now', '-1 hour')
    AND ts.status = 'running';
```

---

## 🎯 **DIRECTOR SELF-MODEL QUERIES**

### Recent Reflections and Insights
```sql
-- Get recent Director reflections and insights
SELECT 
    dsm.type,
    dsm.content,
    dsm.importance,
    dsm.session_id,
    dsm.created_at,
    ROUND((julianday('now') - julianday(dsm.created_at)) * 24, 1) as hours_ago
FROM director_self_model dsm
WHERE dsm.type IN ('reflection', 'memory', 'trait')
ORDER BY dsm.importance DESC, dsm.created_at DESC
LIMIT 20;
```

### Learning Pattern Evolution
```sql
-- Track how the Director's learning patterns have evolved
SELECT 
    DATE(dsm.created_at) as date,
    dsm.type,
    COUNT(*) as entries,
    AVG(dsm.importance) as avg_importance
FROM director_self_model dsm
WHERE dsm.created_at >= datetime('now', '-7 days')
GROUP BY DATE(dsm.created_at), dsm.type
ORDER BY date DESC, dsm.type;
```

---

## 🔧 **MAINTENANCE & CLEANUP QUERIES**

### Database Health Check
```sql
-- Check database health and data distribution
SELECT 
    'training_sessions' as table_name,
    COUNT(*) as record_count,
    MAX(created_at) as latest_record,
    MIN(created_at) as oldest_record
FROM training_sessions
UNION ALL
SELECT 
    'game_results' as table_name,
    COUNT(*) as record_count,
    MAX(created_at) as latest_record,
    MIN(created_at) as oldest_record
FROM game_results
UNION ALL
SELECT 
    'action_traces' as table_name,
    COUNT(*) as record_count,
    MAX(created_at) as latest_record,
    MIN(created_at) as oldest_record
FROM action_traces
UNION ALL
SELECT 
    'system_logs' as table_name,
    COUNT(*) as record_count,
    MAX(created_at) as latest_record,
    MIN(created_at) as oldest_record
FROM system_logs
ORDER BY table_name;
```

### Data Cleanup Recommendations
```sql
-- Find old data that could be cleaned up
SELECT 
    'Old Error Logs' as cleanup_type,
    COUNT(*) as records_to_clean,
    'DELETE FROM error_logs WHERE resolved = TRUE AND last_seen < datetime("now", "-30 days")' as cleanup_query
FROM error_logs
WHERE resolved = TRUE AND last_seen < datetime('now', '-30 days')
UNION ALL
SELECT 
    'Old Performance Data' as cleanup_type,
    COUNT(*) as records_to_clean,
    'DELETE FROM performance_history WHERE created_at < datetime("now", "-30 days")' as cleanup_query
FROM performance_history
WHERE created_at < datetime('now', '-30 days')
UNION ALL
SELECT 
    'Old Action Traces' as cleanup_type,
    COUNT(*) as records_to_clean,
    'DELETE FROM action_traces WHERE created_at < datetime("now", "-7 days")' as cleanup_query
FROM action_traces
WHERE created_at < datetime('now', '-7 days');
```

---

## 📝 **USAGE NOTES**

### Query Categories:
- **🎯 Gameplay State**: Real-time session monitoring and game progress
- **🧠 Action Intelligence**: Action effectiveness and strategy analysis
- **🔍 Error Detection**: System health and debugging insights
- **📊 Learning Analysis**: Pattern recognition and learning progress
- **🎮 Game-Specific**: Game type analysis and difficulty assessment
- **🔧 System Optimization**: Performance and resource management
- **🚀 Strategic Improvement**: Opportunities for system enhancement
- **📈 Real-Time Monitoring**: Live dashboard and status queries
- **🎯 Director Self-Model**: AI reflection and learning tracking
- **🔧 Maintenance**: Database health and cleanup operations

### Performance Tips:
- Use `LIMIT` clauses for large result sets
- Add appropriate `WHERE` clauses to filter data by time ranges
- Consider creating indexes for frequently queried columns
- Use `EXPLAIN QUERY PLAN` to optimize complex queries

### Integration with Director Commands:
These queries can be integrated with the Director Commands API for automated analysis and decision-making. Consider creating wrapper functions that execute these queries and return structured data for the Director's cognitive processes.

---

*This document should be regularly updated as new tables and analysis needs emerge. The queries are designed to provide comprehensive insights while maintaining good performance.*
