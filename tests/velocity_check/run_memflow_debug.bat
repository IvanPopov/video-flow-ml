@echo off
echo Running MemFlow Debug Tests (Quick)
echo ====================================
echo.

cd /d "%~dp0"

echo [1/1] Running FAST motion test (MemFlow only - short sequence)...
python velocity_test.py --speed fast --frames 10 --fps 30 --memflow-only
echo.

echo MemFlow debug test completed! Check temp/ directory for results.
echo Results file: results_fast.json
pause 