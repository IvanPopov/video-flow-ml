@echo off
echo Running MemFlow Long-Term Memory Full Test...
echo.

cd /d "%~dp0"

REM Activate virtual environment if it exists
if exist "..\..\activate.bat" (
    call "..\..\activate.bat"
)

REM Run full test with more frames
python long_term_test.py --frames 180 --fps 30 --dataset sintel --seq-length 5

echo.
echo Full test completed!
pause 