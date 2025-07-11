@echo off
rem Activate VideoFlow virtual environment
echo Activating VideoFlow virtual environment...

if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first to create the environment.
    pause
    exit /b 1
)

echo Virtual environment activated. You can now run:
echo - python flow_processor.py input_video.mp4
echo - python gui_runner.py
echo - python check_cuda.py
echo.

call .venv\Scripts\activate.bat
cmd /k 