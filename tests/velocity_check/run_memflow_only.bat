@echo off
echo Running MemFlow-Only Velocity Tests
echo ====================================
echo.

cd /d "%~dp0"

echo [1/3] Running SLOW motion test (MemFlow only)...
python velocity_test.py --speed slow --frames 60 --fps 30 --memflow-only
echo.

echo [2/3] Running MEDIUM motion test (MemFlow only)...
python velocity_test.py --speed medium --frames 60 --fps 30 --memflow-only
echo.

echo [3/3] Running FAST motion test (MemFlow only)...
python velocity_test.py --speed fast --frames 60 --fps 30 --memflow-only
echo.

echo All MemFlow tests completed! Check temp/ directory for results.
echo Results files:
echo - results_slow.json
echo - results_medium.json
echo - results_fast.json
pause 