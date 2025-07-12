@echo off
echo Running Full MemFlow Sequence Length Test
echo ==========================================
echo.

cd /d "%~dp0"

echo Testing sequence lengths 3,5,10 with medium motion (full test)...
python seq_length_test.py --motion medium --frames 120 --seq-lengths 3,5,10
echo.

echo Full test completed! Check temp/ directory for results.
echo Results file: seq_length_results_medium_120f.json
pause 