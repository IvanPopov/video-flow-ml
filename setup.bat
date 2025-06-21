@echo off
echo Setting up VideoFlow ML with CUDA support...
echo.

REM Check if Python is installed
python --version >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Python not found! Please install Python 3.8+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python found, checking for NVIDIA GPU...

REM Check if nvidia-smi is available
nvidia-smi >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo NVIDIA GPU not detected or drivers not installed.
    echo This script will install CUDA version, but it may not work.
    echo.
    echo Press any key to continue with CUDA installation anyway...
    pause >nul
)

echo Creating virtual environment...
if exist venv_video_flow (
    echo Removing existing virtual environment...
    rmdir /s /q venv_video_flow
)

python -m venv venv_video_flow
if %ERRORLEVEL% NEQ 0 (
    echo Failed to create virtual environment!
    pause
    exit /b 1
)

echo Virtual environment created.
echo.

echo Activating virtual environment and installing packages...
call venv_video_flow\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo Installing remaining dependencies...
pip install -r requirements.txt

if %ERRORLEVEL% NEQ 0 (
    echo Failed to install packages!
    pause
    exit /b 1
)

echo.
echo Testing CUDA installation...
python check_cuda.py

echo.
echo Setup completed!
echo.
echo To activate the environment, run:
echo    venv_video_flow\Scripts\activate
echo.  
echo To test the installation, run:
echo    python flow_processor.py --help
echo.
echo For usage examples, see README.md
echo.
pause 