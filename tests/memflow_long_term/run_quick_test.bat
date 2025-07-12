@echo off
echo Running MemFlow Long-Term Memory Quick Test...
echo.

cd /d "%~dp0"

REM Activate virtual environment if it exists
if exist "..\..\activate.bat" (
    call "..\..\activate.bat"
)

REM Run quick test with fewer frames
python long_term_test.py --frames 60 --fps 30 --dataset sintel --seq-length 5

echo.
echo Quick test completed!
pause 