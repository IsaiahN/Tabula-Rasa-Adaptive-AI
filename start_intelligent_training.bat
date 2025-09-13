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

REM Load API key from .env file if not already set
if "%ARC_API_KEY%"=="" (
    echo 🔍 Loading API key from .env file...
    for /f "usebackq tokens=1,2 delims==" %%a in ("%cd%\.env") do (
        if "%%a"=="ARC_API_KEY" set ARC_API_KEY=%%b
    )
)

REM Check if API key is now available
if "%ARC_API_KEY%"=="" (
    echo ❌ ERROR: ARC_API_KEY not found in environment or .env file!
    echo.
    echo Please either:
    echo 1. Set the environment variable: set ARC_API_KEY=your_api_key_here
    echo 2. Or add it to your .env file: ARC_API_KEY=your_api_key_here
    echo.
    echo Then run this script again.
    echo.
    pause
    exit /b 1
)

echo ✅ API key loaded: %ARC_API_KEY:~0,8%...%ARC_API_KEY:~-4%
echo.

REM Set environment variables for optimal performance
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

REM Check and install required dependencies
echo 🔍 Checking required dependencies...
python -c "import psutil" 2>nul
if errorlevel 1 (
    echo ⚠️ Required dependencies not found, installing...
    echo Installing from requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies!
        echo Please install them manually: pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
    echo ✅ Dependencies installed successfully!
) else (
    echo ✅ All required dependencies are available!
)

REM Test API connection first
echo 🔍 Testing API connection...
python tests/test_api_connection.py
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
