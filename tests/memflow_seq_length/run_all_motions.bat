@echo off
echo Running MemFlow Sequence Length Test - All Motions
echo ===================================================
echo.

cd /d "%~dp0"

echo [1/3] Testing SLOW motion with sequence lengths 3,5,10,15,25,50...
python seq_length_test.py --motion slow --frames 120 --seq-lengths 3,5,10,15,25,50
echo.

echo [2/3] Testing MEDIUM motion with sequence lengths 3,5,10,15,25,50...
python seq_length_test.py --motion medium --frames 120 --seq-lengths 3,5,10,15,25,50
echo.

echo [3/3] Testing FAST motion with sequence lengths 3,5,10,15,25,50...
python seq_length_test.py --motion fast --frames 120 --seq-lengths 3,5,10,15,25,50
echo.

echo All motion tests completed! Check temp/ directory for results.
echo Results files:
echo - seq_length_results_slow_120f.json
echo - seq_length_results_medium_120f.json
echo - seq_length_results_fast_120f.json
pause 