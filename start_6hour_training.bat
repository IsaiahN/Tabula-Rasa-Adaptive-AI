@echo off
echo ========================================
echo TABULA RASA - 6 HOUR CONTINUOUS TRAINING
echo ========================================
echo.
echo Starting enhanced meta-cognitive training session...
echo Duration: 6 hours (360 minutes)
echo Mode: Continuous with all advanced features enabled
echo.
echo Press Ctrl+C to stop gracefully
echo.

REM Set environment variables for optimal performance
set ARC_API_KEY=%ARC_API_KEY%
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

REM Run the master trainer with MAXIMUM INTELLIGENCE for 6-hour continuous training
python master_arc_trainer.py --mode maximum-intelligence --session-duration 360 --max-actions 1000 --max-cycles 100 --target-score 90.0 --enable-detailed-monitoring --salience-threshold 0.4 --salience-decay 0.95 --memory-size 1024 --memory-word-size 128 --memory-read-heads 8 --memory-write-heads 2 --dashboard console --verbose

echo.
echo Training session completed or stopped.
pause
