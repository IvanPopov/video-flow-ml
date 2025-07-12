@echo off
echo Running MemFlow Long-Term Memory Tests with Different Motions...
echo.

cd /d "%~dp0"

REM Activate virtual environment if it exists
if exist "..\..\activate.bat" (
    call "..\..\activate.bat"
)

echo Testing slow motion scenario...
python long_term_test.py --frames 120 --fps 30 --dataset sintel --seq-length 5

echo.
echo Testing medium motion scenario...
python long_term_test.py --frames 180 --fps 30 --dataset sintel --seq-length 5

echo.
echo Testing fast motion scenario...
python long_term_test.py --frames 240 --fps 30 --dataset sintel --seq-length 5

echo.
echo All motion tests completed!
pause 