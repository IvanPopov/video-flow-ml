@echo off
echo Running Velocity Test - DEBUG Mode (Quick Test)
echo =================================================
echo.

cd /d "%~dp0"
python velocity_test.py --speed slow --frames 5 --fps 30

echo.
echo Quick debug test completed. Check temp/ directory for results.
pause 