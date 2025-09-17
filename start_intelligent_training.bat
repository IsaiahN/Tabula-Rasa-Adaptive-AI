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

REM Check if .env file exists, if not run setup
if not exist ".env" (
    echo .env file not found. Running environment setup...
    python setup_env.py
    if errorlevel 1 (
        echo Failed to create .env file. Please run 'python setup_env.py' manually.
        pause
        exit /b 1
    )
)

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
echo 3. Custom Config Training
echo    - Choose specific training modes and parameters
echo    - Quick validation, meta-cognitive, continuous, research lab
echo    - Customizable cycles and duration
echo.
set /p choice="Enter your choice (1, 2, or 3): "

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
) else if "%choice%"=="3" (
    goto CUSTOM_CONFIG
) else (
    echo.
    echo ❌ Invalid choice! Please run the script again and choose 1, 2, or 3.
    echo.
    pause
    exit /b 1
)

goto END

:CUSTOM_CONFIG
cls
echo ===============================================
echo  TABULA RASA - Custom Config Training
echo ===============================================
echo.
echo Select Training Mode:
echo.
echo 1. Quick Validation (fast testing)
echo 2. Meta-Cognitive Training (comprehensive)
echo 3. Continuous Training (long-running)
echo 4. Research Lab (experimentation)
echo 5. Maximum Intelligence (full power)
echo 6. Custom mode (specify parameters)
echo 7. Back to main menu
echo.
set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" goto QUICK
if "%choice%"=="2" goto META
if "%choice%"=="3" goto CONTINUOUS
if "%choice%"=="4" goto RESEARCH
if "%choice%"=="5" goto MAXIMUM
if "%choice%"=="6" goto CUSTOM
if "%choice%"=="7" goto MAIN_MENU
echo Invalid choice. Please try again.
pause
goto CUSTOM_CONFIG

:QUICK
echo Launching Quick Validation...
python master_arc_trainer.py --mode quick-validation --verbose --max-cycles 2 --session-duration 3
goto DONE

:META
echo Launching Meta-Cognitive Training...
python master_arc_trainer.py --mode meta-cognitive-training --verbose --max-cycles 5 --session-duration 30
goto DONE

:CONTINUOUS
echo Launching Continuous Training...
python master_arc_trainer.py --mode continuous-training --dashboard gui --max-cycles 100 --session-duration 240
goto DONE

:RESEARCH
echo Launching Research Lab Mode...
python master_arc_trainer.py --mode research-lab --verbose --max-cycles 3 --session-duration 15
goto DONE

:MAXIMUM
echo Launching Maximum Intelligence Mode...
python master_arc_trainer.py --mode maximum-intelligence --verbose --max-cycles 10 --session-duration 60
goto DONE

:CUSTOM
echo.
set /p mode="Enter mode (quick-validation, meta-cognitive-training, continuous-training, research-lab, maximum-intelligence): "
set /p cycles="Enter max cycles (default 5): "
set /p duration="Enter session duration in minutes (default 30): "
if "%cycles%"=="" set cycles=5
if "%duration%"=="" set duration=30
echo Launching Custom Mode: %mode%
python master_arc_trainer.py --mode %mode% --verbose --max-cycles %cycles% --session-duration %duration%
goto DONE

:DONE
echo.
echo Training session completed.
echo.
set /p restart="Run another session? (y/n): "
if /i "%restart%"=="y" goto CUSTOM_CONFIG

:MAIN_MENU
goto :eof

:END
echo.
echo Training session completed or stopped.
pause
