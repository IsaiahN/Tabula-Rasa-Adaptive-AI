@echo off
REM Master ARC Trainer - Default Training Script Launcher
REM This batch file provides easy access to the unified training system

echo ===============================================
echo  TABULA RASA - Master ARC Trainer
echo ===============================================
echo.
echo Starting unified training system...
echo Mode: continuous-training (default)
echo.

REM Check if ARC_API_KEY is set
if not defined ARC_API_KEY (
    echo ERROR: ARC_API_KEY environment variable is not set.
    echo Please set it in your environment or .env file.
    echo.
    pause
    exit /b 1
)

REM Run the master trainer with default continuous training mode
python master_arc_trainer.py --mode continuous-training --dashboard gui --max-cycles 10 --session-duration 60 --verbose

echo.
echo Training session completed.
pause
