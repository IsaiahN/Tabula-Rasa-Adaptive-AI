@echo off
REM Multi-Mode Master ARC Trainer Launcher
REM Provides easy access to different training modes

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

:MENU
cls
echo ===============================================
echo  TABULA RASA - Master ARC Trainer Launcher
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
echo 7. Exit
echo.
set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" goto QUICK
if "%choice%"=="2" goto META
if "%choice%"=="3" goto CONTINUOUS
if "%choice%"=="4" goto RESEARCH
if "%choice%"=="5" goto MAXIMUM
if "%choice%"=="6" goto CUSTOM
if "%choice%"=="7" goto EXIT
echo Invalid choice. Please try again.
pause
goto MENU

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
if /i "%restart%"=="y" goto MENU

:EXIT
echo Thank you for using Master ARC Trainer!
pause
