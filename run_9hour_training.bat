@echo off
echo ========================================
echo TABULA RASA - 9 HOUR CONTINUOUS TRAINING
echo ========================================
echo.
echo Starting enhanced meta-cognitive training session...
echo Duration: 9 hours (540 minutes)
echo Mode: Continuous with all advanced features enabled
echo.

REM Record start time
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "start_time=%dt%"
echo Started: %start_time:~0,4%-%start_time:~4,2%-%start_time:~6,2% %start_time:~8,2%:%start_time:~10,2%:%start_time:~12,2%
echo.
echo Press Ctrl+C to stop gracefully
echo.

REM Set environment variables for optimal performance
set ARC_API_KEY=%ARC_API_KEY%
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

REM Calculate 9 hours in seconds (9 * 60 * 60 = 32400)
set /a total_duration=32400
set /a session_count=0

:loop
REM Get current time
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "current_time=%dt%"

REM Calculate elapsed seconds (improved calculation)
REM Remove leading zeros and handle time overflow
set /a start_hour=1%start_time:~8,2%-100
set /a start_min=1%start_time:~10,2%-100
set /a start_sec=1%start_time:~12,2%-100
set /a current_hour=1%current_time:~8,2%-100
set /a current_min=1%current_time:~10,2%-100
set /a current_sec=1%current_time:~12,2%-100

REM Calculate elapsed time in seconds
set /a elapsed_hours=%current_hour%-%start_hour%
set /a elapsed_minutes=%current_min%-%start_min%
set /a elapsed_seconds=%current_sec%-%start_sec%

REM Convert to total seconds
set /a elapsed_total=%elapsed_hours%*3600+%elapsed_minutes%*60+%elapsed_seconds%

REM Handle negative values (day rollover)
if %elapsed_total% lss 0 set /a elapsed_total=%elapsed_total%+86400

REM Check if 9 hours have elapsed
if %elapsed_total% geq %total_duration% (
    echo.
    echo 🎉 9 HOUR TRAINING COMPLETE!
    echo Total duration: %elapsed_hours% hours
    echo Total sessions: %session_count%
    echo Completed at: %current_time:~0,4%-%current_time:~4,2%-%current_time:~6,2% %current_time:~8,2%:%current_time:~10,2%:%current_time:~12,2%
    goto :end
)

REM Increment session count
set /a session_count+=1

REM Calculate remaining time
set /a remaining_seconds=%total_duration%-%elapsed_total%
set /a remaining_hours=%remaining_seconds%/3600

echo ========================================
echo TRAINING SESSION #%session_count%
echo ========================================
echo Time remaining: %remaining_hours% hours
echo Session started: %current_time:~0,4%-%current_time:~4,2%-%current_time:~6,2% %current_time:~8,2%:%current_time:~10,2%:%current_time:~12,2%
echo.
echo 🚀 Launching continuous training...
echo.

REM Run the master trainer with MAXIMUM INTELLIGENCE for 9-hour continuous training
python master_arc_trainer.py ^
    --mode maximum-intelligence ^
    --session-duration 540 ^
    --max-actions 5000 ^
    --max-cycles 2000 ^
    --target-score 90.0 ^
    --enable-detailed-monitoring ^
    --salience-threshold 0.4 ^
    --salience-decay 0.95 ^
    --memory-size 1024 ^
    --memory-word-size 128 ^
    --memory-read-heads 8 ^
    --memory-write-heads 2 ^
    --dashboard console ^
    --verbose

echo.
echo ✅ Training session #%session_count% completed!

REM Check if we still have time remaining
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "current_time=%dt%"

REM Calculate elapsed time (reuse the same logic)
set /a start_hour=1%start_time:~8,2%-100
set /a start_min=1%start_time:~10,2%-100
set /a start_sec=1%start_time:~12,2%-100
set /a current_hour=1%current_time:~8,2%-100
set /a current_min=1%current_time:~10,2%-100
set /a current_sec=1%current_time:~12,2%-100

set /a elapsed_hours=%current_hour%-%start_hour%
set /a elapsed_minutes=%current_min%-%start_min%
set /a elapsed_seconds=%current_sec%-%start_sec%
set /a elapsed_total=%elapsed_hours%*3600+%elapsed_minutes%*60+%elapsed_seconds%

REM Handle negative values (day rollover)
if %elapsed_total% lss 0 set /a elapsed_total=%elapsed_total%+86400

if %elapsed_total% lss %total_duration% (
    set /a remaining_seconds=%total_duration%-%elapsed_total%
    set /a remaining_hours=%remaining_seconds%/3600
    echo.
    echo ⏰ Time remaining: %remaining_hours% hours
    echo 🔄 Restarting training session...
    echo.
    timeout /t 2 /nobreak >nul
    goto :loop
) else (
    echo.
    echo 🎉 9 HOUR TRAINING COMPLETE!
    echo Total duration: %elapsed_hours% hours
    echo Total sessions: %session_count%
    echo Completed at: %current_time:~0,4%-%current_time:~4,2%-%current_time:~6,2% %current_time:~8,2%:%current_time:~10,2%:%current_time:~12,2%
)

:end
echo.
echo Training session ended.
pause
