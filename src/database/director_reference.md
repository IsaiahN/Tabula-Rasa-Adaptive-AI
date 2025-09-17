# DIRECTOR COMMAND REFERENCE

## Overview
This document provides a comprehensive reference for Director/LLM commands to interact with the Tabula Rasa database system. All commands are designed for easy use and provide rich data for analysis and decision-making.

## Quick Start

```python
from src.database.director_commands import get_director_commands

# Get Director commands instance
director = get_director_commands()

# Get system status
status = await director.get_system_overview()

# Get learning analysis
learning = await director.get_learning_analysis()

# Get system health
health = await director.analyze_system_health()
```

## System Status Commands

### `get_system_overview()`
Get comprehensive system overview with all key metrics.

**Returns:**
```python
{
    "system_status": {
        "active_sessions": 2,
        "total_actions": 1500,
        "total_wins": 45,
        "total_games": 200,
        "avg_win_rate": 0.225,
        "avg_score": 67.5
    },
    "recent_trends": [...],
    "action_effectiveness": [...],
    "global_counters": {...},
    "timestamp": "2025-09-16T16:30:00"
}
```

### `get_performance_summary(hours=24)`
Get performance summary for specified time period.

**Parameters:**
- `hours` (int): Number of hours to look back (default: 24)

**Returns:**
```python
{
    "time_period_hours": 24,
    "total_sessions": 5,
    "total_games": 200,
    "total_wins": 45,
    "win_rate": 0.225,
    "recent_sessions": [...],
    "recent_games": [...],
    "analysis_timestamp": "2025-09-16T16:30:00"
}
```

### `get_active_sessions()`
Get all currently active training sessions.

**Returns:**
```python
{
    "active_sessions": [
        {
            "session_id": "session_123",
            "mode": "maximum-intelligence",
            "status": "running",
            "total_actions": 750,
            "total_wins": 22,
            "win_rate": 0.293
        }
    ],
    "count": 1,
    "timestamp": "2025-09-16T16:30:00"
}
```

## Learning Analysis Commands

### `get_learning_analysis(game_id=None)`
Get learning analysis and insights.

**Parameters:**
- `game_id` (str, optional): Specific game to analyze

**Returns:**
```python
{
    "coordinate_insights": [
        {
            "game_id": "vc33-6ae7bf49eea5",
            "coordinate_count": 15,
            "total_attempts": 200,
            "total_successes": 45,
            "avg_success_rate": 0.225
        }
    ],
    "winning_sequences": [
        {
            "game_id": "vc33-6ae7bf49eea5",
            "sequence": "[6, 1, 2, 3]",
            "frequency": 5,
            "success_rate": 0.8
        }
    ],
    "recent_patterns": [...],
    "learning_effectiveness": {
        "coordinate_learning": 15,
        "sequence_learning": 8,
        "pattern_learning": 25,
        "overall_effectiveness": 16.0
    },
    "recommendations": [
        "Increase coordinate exploration to improve Action 6 effectiveness",
        "Focus on identifying and learning winning action sequences"
    ]
}
```

### `get_learning_progress(game_id=None)`
Get learning progress analysis.

**Parameters:**
- `game_id` (str, optional): Specific game to analyze

**Returns:**
```python
{
    "learning_metrics": {
        "coordinate_learning": 15,
        "sequence_learning": 8,
        "pattern_learning": 25,
        "action_learning": 6,
        "overall_progress": "GOOD"
    },
    "progress_trends": {
        "coordinate_trend": "IMPROVING",
        "sequence_trend": "IMPROVING",
        "pattern_trend": "IMPROVING"
    },
    "improvement_areas": [
        "Coordinate learning - need more successful coordinate patterns"
    ],
    "next_steps": [
        "Focus on Action 6 coordinate exploration to build coordinate intelligence"
    ]
}
```

## Action Intelligence Commands

### `get_action_effectiveness(game_id=None, action_number=None)`
Get action effectiveness analysis.

**Parameters:**
- `game_id` (str, optional): Specific game to analyze
- `action_number` (int, optional): Specific action to analyze

**Returns:**
```python
{
    "effectiveness_data": [
        {
            "game_id": "vc33-6ae7bf49eea5",
            "action_number": 6,
            "attempts": 200,
            "successes": 45,
            "success_rate": 0.225,
            "avg_score_impact": 2.5
        }
    ],
    "summary": {
        "total_actions": 6,
        "total_attempts": 1200,
        "total_successes": 180,
        "avg_success_rate": 0.15,
        "best_action": 6,
        "best_success_rate": 0.225
    },
    "recommendations": [
        "Action 6 is performing well - increase usage",
        "Actions 1, 2, 3 have low success rates - investigate and improve"
    ]
}
```

### `get_coordinate_intelligence(game_id=None, min_success_rate=0.1)`
Get coordinate intelligence analysis.

**Parameters:**
- `game_id` (str, optional): Specific game to analyze
- `min_success_rate` (float): Minimum success rate to include (default: 0.1)

**Returns:**
```python
{
    "coordinate_data": [
        {
            "game_id": "vc33-6ae7bf49eea5",
            "x": 32,
            "y": 32,
            "attempts": 50,
            "successes": 15,
            "success_rate": 0.3,
            "frame_changes": 12
        }
    ],
    "summary": {
        "total_coordinates": 15,
        "total_attempts": 200,
        "total_successes": 45,
        "avg_success_rate": 0.225,
        "best_coordinate": [32, 32],
        "best_success_rate": 0.3
    },
    "hotspots": [
        {
            "coordinate": [32, 32],
            "success_rate": 0.3,
            "attempts": 50,
            "game_id": "vc33-6ae7bf49eea5"
        }
    ],
    "recommendations": [
        "Found 5 high-success coordinates - prioritize these areas",
        "Found 3 low-success coordinates - avoid these areas"
    ]
}
```

## System Control Commands

### `create_training_session(mode="maximum-intelligence", session_id=None)`
Create a new training session.

**Parameters:**
- `mode` (str): Training mode (default: "maximum-intelligence")
- `session_id` (str, optional): Custom session ID

**Returns:**
```python
{
    "success": True,
    "session_id": "director_session_1758057489",
    "mode": "maximum-intelligence",
    "created_at": "2025-09-16T16:30:00"
}
```

### `update_session_status(session_id, status, updates=None)`
Update training session status.

**Parameters:**
- `session_id` (str): Session to update
- `status` (str): New status
- `updates` (dict, optional): Additional updates

**Returns:**
```python
{
    "success": True,
    "session_id": "session_123",
    "status": "completed",
    "updated_at": "2025-09-16T16:30:00"
}
```

## System Health Commands

### `analyze_system_health()`
Analyze overall system health.

**Returns:**
```python
{
    "health_score": 0.75,
    "status": "GOOD",
    "issues": [
        "Very low win rate - system may be struggling"
    ],
    "recommendations": [
        "Investigate why win rate is low - check action effectiveness and coordinate selection"
    ],
    "overview": {...},
    "performance": {...},
    "analysis_timestamp": "2025-09-16T16:30:00"
}
```

## Real-Time Query Commands

### `get_system_status()`
Get real-time system status (quick access).

**Returns:**
```python
{
    "active_sessions": [...],
    "recent_performance": [...],
    "action_effectiveness": [...],
    "global_counters": {
        "total_memory_operations": 12312,
        "total_sleep_cycles": 45,
        "total_actions": 6162
    },
    "timestamp": "2025-09-16T16:30:00"
}
```

### `get_learning_insights(game_id=None)`
Get learning insights for Director analysis (quick access).

**Returns:**
```python
{
    "coordinate_insights": [...],
    "winning_sequences": [...],
    "recent_patterns": [...],
    "analysis_timestamp": "2025-09-16T16:30:00"
}
```

## Global Counter Commands

### `update_global_counter(counter_name, value, description=None)`
Update global counter.

**Parameters:**
- `counter_name` (str): Counter name
- `value` (int): Counter value
- `description` (str, optional): Counter description

**Returns:**
```python
{
    "success": True,
    "counter_name": "total_actions",
    "value": 6500,
    "updated_at": "2025-09-16T16:30:00"
}
```

### `get_global_counters()`
Get all global counters.

**Returns:**
```python
{
    "total_memory_operations": 12312,
    "total_sleep_cycles": 45,
    "total_actions": 6162,
    "total_wins": 45,
    "total_games": 200
}
```

## Error Handling

All commands return structured responses with error information:

```python
{
    "success": False,
    "error": "Database connection failed",
    "error_code": "DB_CONNECTION_ERROR",
    "timestamp": "2025-09-16T16:30:00"
}
```

## Best Practices

1. **Always check success status** before using returned data
2. **Use specific game_id** when analyzing specific games
3. **Monitor system health** regularly with `analyze_system_health()`
4. **Check learning progress** with `get_learning_progress()`
5. **Use real-time queries** for immediate status updates

## Example Usage Patterns

### Daily System Check
```python
# Get overall system status
status = await director.get_system_overview()

# Check system health
health = await director.analyze_system_health()

# Get learning progress
progress = await director.get_learning_progress()

# Check for issues
if health["status"] == "CRITICAL":
    print("System needs immediate attention!")
```

### Game-Specific Analysis
```python
# Analyze specific game
game_analysis = await director.get_learning_analysis("vc33-6ae7bf49eea5")

# Get action effectiveness for game
actions = await director.get_action_effectiveness("vc33-6ae7bf49eea5")

# Get coordinate intelligence for game
coordinates = await director.get_coordinate_intelligence("vc33-6ae7bf49eea5")
```

### Performance Monitoring
```python
# Get 24-hour performance
performance = await director.get_performance_summary(24)

# Get 7-day performance
weekly_performance = await director.get_performance_summary(168)

# Check recent trends
trends = performance["recent_trends"]
```

## Integration with Other Systems

The Director commands integrate seamlessly with:
- **Governor**: Uses same database for decision-making
- **Architect**: Accesses evolution data and patterns
- **Learning Loop**: Provides real-time data for training
- **Coordinate System**: Shares intelligence data

All systems use the same database API, ensuring data consistency and real-time updates across all components.
