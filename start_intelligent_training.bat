@echo off
echo ========================================
echo TABULA RASA - INTELLIGENT 9 HOUR TRAINING
echo ========================================
echo.
echo Starting intelligent parallel training session...
echo Duration: 9 hours (540 minutes)
echo Mode: Intelligent parallel with dynamic resource optimization
echo Features: RAM-aware scaling, CPU optimization, adaptive learning
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
echo.
echo Choose your training mode:
echo 1. Intelligent Parallel Training (Recommended)
echo    - Multiple concurrent games based on your RAM
echo    - Dynamic resource optimization
echo    - Maximum learning speed
echo.
echo 2. Simple Sequential Training
echo    - One game at a time
echo    - Maximum stability
echo    - No resource conflicts
echo.
set /p choice="Enter your choice (1 or 2): "

if "%choice%"=="1" (
    echo.
    echo 🚀 Starting Intelligent Parallel Training...
    echo The system will automatically detect your RAM and CPU
    echo and run the optimal number of concurrent games.
    echo.
    python run_9hour_scaled_training.py
) else if "%choice%"=="2" (
    echo.
    echo 🚀 Starting Simple Sequential Training...
    echo This will run games one at a time for maximum stability.
    echo.
    python run_9hour_simple_training.py
) else (
    echo.
    echo ❌ Invalid choice! Please run the script again and choose 1 or 2.
    echo.
    pause
    exit /b 1
)

echo.
echo Training session completed or stopped.
pause
