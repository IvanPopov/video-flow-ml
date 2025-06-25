@echo off
echo VideoFlow ML Setup Script
echo =========================

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo WARNING: NVIDIA GPU not detected. Installing CPU version.
    set CUDA_SUPPORT=false
) else (
    echo NVIDIA GPU detected. Installing CUDA version.
    set CUDA_SUPPORT=true
)

echo.
echo Creating virtual environment...
if not exist "venv_video_flow" (
    python -m venv venv_video_flow
)

echo.
echo Activating virtual environment...
call venv_video_flow\Scripts\activate.bat

echo.
echo Updating submodules...
git submodule init
git submodule update

echo.
echo Installing dependencies...
if "%CUDA_SUPPORT%"=="true" (
    echo Installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo Installing PyTorch CPU version...
    pip install torch torchvision torchaudio
)

echo Installing remaining dependencies...
pip install -r requirements.txt

echo.
echo Testing installation...
python -c "import torch, torchvision, numpy, cv2; print('Dependencies OK'); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo.
echo Setup complete! To activate the environment, run:
echo call venv_video_flow\Scripts\activate.bat
echo.
echo To process a video, run:
echo python flow_processor.py input_video.mp4
pause 