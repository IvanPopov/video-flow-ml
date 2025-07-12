@echo off
echo Running Quick MemFlow Sequence Length Test
echo ==========================================
echo.

cd /d "%~dp0"

echo Testing sequence lengths 3,5,10 with medium motion (quick test)...
python seq_length_test.py --motion medium --frames 90 --seq-lengths 3,5,10
echo.

echo Quick test completed! Check temp/ directory for results.
echo Results file: seq_length_results_medium_90f.json
pause 