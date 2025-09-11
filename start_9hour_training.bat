@echo off
echo ========================================
echo TABULA RASA - 9 HOUR CONTINUOUS TRAINING
echo ========================================
echo.
echo Starting enhanced meta-cognitive training session...
echo Duration: 9 hours (540 minutes)
echo Mode: Maximum Intelligence with all advanced features enabled
echo.
echo Press Ctrl+C to stop gracefully
echo.

REM Check if API key is set
if "%ARC_API_KEY%"=="" (
    echo ❌ ERROR: ARC_API_KEY environment variable is not set!
    echo.
    echo Please set your API key first:
    echo   set ARC_API_KEY=your_api_key_here
    echo.
    echo Then run this script again.
    echo.
    pause
    exit /b 1
)

echo ✅ API key found: %ARC_API_KEY:~0,8%...%ARC_API_KEY:~-4%
echo.

REM Set environment variables for optimal performance
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

REM Test API connection first
echo 🔍 Testing API connection...
python test_api_connection.py
if errorlevel 1 (
    echo.
    echo ❌ API connection test failed!
    echo Please check your API key and network connection.
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ API connection test passed!
echo 🚀 Starting 9-hour training session...
echo.

REM Run the master trainer with MAXIMUM INTELLIGENCE for 9-hour continuous training
python master_arc_trainer.py --mode maximum-intelligence --session-duration 540 --max-actions 1000 --max-cycles 100 --target-score 90.0 --enable-detailed-monitoring --salience-threshold 0.4 --salience-decay 0.95 --memory-size 1024 --memory-word-size 128 --memory-read-heads 8 --memory-write-heads 2 --dashboard console --verbose

echo.
echo Training session completed or stopped.
pause
