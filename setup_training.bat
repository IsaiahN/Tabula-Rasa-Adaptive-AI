@echo off
echo ========================================
echo TABULA RASA - TRAINING SETUP
echo ========================================
echo.
echo This script will help you set up the 6-hour training session.
echo.

REM Check if API key is already set
if not "%ARC_API_KEY%"=="" (
    echo ✅ ARC_API_KEY is already set: %ARC_API_KEY:~0,8%...%ARC_API_KEY:~-4%
    echo.
    echo Would you like to:
    echo   1. Use the existing API key
    echo   2. Set a new API key
    echo   3. Test the connection
    echo.
    set /p choice="Enter your choice (1-3): "
    
    if "%choice%"=="1" goto :test_connection
    if "%choice%"=="2" goto :set_new_key
    if "%choice%"=="3" goto :test_connection
    goto :invalid_choice
) else (
    echo ❌ ARC_API_KEY is not set.
    echo.
    goto :set_new_key
)

:set_new_key
echo.
echo Please enter your ARC API key:
echo (You can get this from https://arcprize.org/)
echo.
set /p new_key="API Key: "

if "%new_key%"=="" (
    echo ❌ No API key entered. Exiting.
    pause
    exit /b 1
)

echo.
echo Setting API key...
set ARC_API_KEY=%new_key%
echo ✅ API key set: %ARC_API_KEY:~0,8%...%ARC_API_KEY:~-4%

REM Make it persistent for this session
echo set ARC_API_KEY=%new_key% > set_api_key.bat
echo ✅ API key saved to set_api_key.bat for future use

:test_connection
echo.
echo 🔍 Testing API connection...
python test_api_connection.py
if errorlevel 1 (
    echo.
    echo ❌ API connection test failed!
    echo Please check your API key and try again.
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ API connection test passed!
echo.
echo 🚀 Ready to start training!
echo.
echo Choose an option:
echo   1. Start 6-hour training immediately
echo   2. Start 6-hour training with connection test
echo   3. Exit
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo 🚀 Starting 6-hour training...
    python master_arc_trainer.py --mode maximum-intelligence --session-duration 360 --max-actions 1000 --max-cycles 100 --target-score 90.0 --enable-detailed-monitoring --salience-threshold 0.4 --salience-decay 0.95 --memory-size 1024 --memory-word-size 128 --memory-read-heads 8 --memory-write-heads 2 --dashboard console --verbose
) else if "%choice%"=="2" (
    echo.
    echo 🚀 Starting 6-hour training with connection test...
    call start_6hour_training.bat
) else if "%choice%"=="3" (
    echo.
    echo 👋 Goodbye!
    exit /b 0
) else (
    goto :invalid_choice
)

echo.
echo Training completed or stopped.
pause
exit /b 0

:invalid_choice
echo.
echo ❌ Invalid choice. Please try again.
pause
goto :test_connection
