# Log Rotation System

## Overview

The Tabula Rasa system now includes an intelligent log rotation system that automatically manages log file sizes to prevent them from growing too large and consuming excessive disk space.

## Features

- **Automatic Rotation**: Logs are automatically rotated when new games start
- **Configurable Limits**: Line limits are easily adjustable
- **Backup Creation**: Old logs are backed up before rotation
- **Governor Integration**: Log cleanup is managed by the Meta-Cognitive Governor
- **Real-time Monitoring**: Log statistics are available for monitoring

## Configuration

### Default Settings

- **Maximum Lines**: 100,000 lines per log file
- **Backup Lines**: 50,000 lines kept in backup
- **Log Files Managed**:
  - `data/logs/master_arc_trainer.log`
  - `data/logs/master_arc_trainer_output.log`

### Adjusting Line Limits

#### Using the Utility Script

```bash
# Show current log statistics
python adjust_log_limit.py --show-stats

# Change line limit to 50,000
python adjust_log_limit.py --max-lines 50000

# Change line limit and rotate immediately
python adjust_log_limit.py --max-lines 75000 --rotate-now
```

#### Programmatically

```python
from src.core.meta_cognitive_governor import MetaCognitiveGovernor
from src.config.log_config import LogConfig

# Set line limit
config = LogConfig()
config.set_max_lines(75000)

# Or through Governor
governor = MetaCognitiveGovernor()
governor.set_log_line_limit(75000)
```

## How It Works

### Automatic Rotation

1. **New Game Detection**: When a new game starts, the Governor checks if logs need rotation
2. **Size Check**: If either log file exceeds the configured limit, rotation is triggered
3. **Backup Creation**: Current logs are backed up to `.backup` files
4. **Trimming**: Only the most recent lines are kept in the main log files
5. **Cleanup**: Old backup files are removed after a configurable period

### Manual Rotation

```python
from src.core.meta_cognitive_governor import MetaCognitiveGovernor

governor = MetaCognitiveGovernor()
result = governor.rotate_logs()
print(result)
```

## File Structure

```
data/logs/
├── master_arc_trainer.log              # Main training log (≤100k lines)
├── master_arc_trainer_output.log       # Main output log (≤100k lines)
├── master_arc_trainer_backup.log       # Backup of training log
├── master_arc_trainer_output_backup.log # Backup of output log
└── archive/                            # Archived old logs
```

## Monitoring

### Check Log Statistics

```python
from src.core.meta_cognitive_governor import MetaCognitiveGovernor

governor = MetaCognitiveGovernor()
stats = governor.get_log_stats()
print(stats)
```

### Example Output

```json
{
  "master_arc_trainer": {
    "lines": 100000,
    "size_mb": 14.6,
    "path": "data/logs/master_arc_trainer.log"
  },
  "master_arc_trainer_output": {
    "lines": 100000,
    "size_mb": 14.6,
    "path": "data/logs/master_arc_trainer_output.log"
  }
}
```

## Benefits

- **Disk Space Management**: Prevents log files from consuming excessive disk space
- **Performance**: Smaller log files improve system performance
- **Maintainability**: Easier to analyze recent logs without overwhelming data
- **Automation**: No manual intervention required
- **Flexibility**: Configurable limits for different use cases

## Troubleshooting

### Logs Not Rotating

1. Check if Governor is initialized: `governor.log_rotator is not None`
2. Verify log files exist and are accessible
3. Check file permissions in `data/logs/` directory

### Configuration Not Saving

1. Ensure `src/config/log_config.py` is accessible
2. Check that the Governor is properly initialized
3. Verify no import errors in the log rotation modules

### Manual Cleanup

If automatic rotation fails, you can manually clean logs:

```bash
# Keep only last 50,000 lines
tail -n 50000 data/logs/master_arc_trainer.log > temp.log
mv temp.log data/logs/master_arc_trainer.log

# Same for output log
tail -n 50000 data/logs/master_arc_trainer_output.log > temp.log
mv temp.log data/logs/master_arc_trainer_output.log
```

## Advanced Configuration

### Custom Log Files

To add more log files to the rotation system, modify `src/config/log_config.py`:

```python
class LogConfig:
    # Add your custom log files
    CUSTOM_LOG = "data/logs/custom.log"
    CUSTOM_LOG_BACKUP = "data/logs/custom_backup.log"
```

### Custom Rotation Logic

Override the rotation logic in `src/utils/log_rotation.py` for specialized needs.

## Integration Points

- **Governor System**: Manages log rotation decisions
- **Continuous Learning Loop**: Triggers rotation on new games
- **Configuration System**: Centralized settings management
- **Monitoring System**: Provides log statistics and health checks
