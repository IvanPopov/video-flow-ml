@echo off
echo Running Velocity Test - FAST Motion
echo ====================================
echo.

cd /d "%~dp0"
python velocity_test.py --speed fast --frames 60 --fps 30

echo.
echo Test completed. Check temp/ directory for results.
pause 