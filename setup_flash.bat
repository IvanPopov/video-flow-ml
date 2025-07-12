@echo OFF
ECHO "--- Starting Flash Attention Setup ---"

ECHO "1. Activating conda environment: flash_attn_env"
CALL %CONDA_EXE% activate flash_attn_env || CALL C:\Users\%USERNAME%\AppData\Local\miniconda3\Scripts\conda.exe activate flash_attn_env
IF %ERRORLEVEL% NEQ 0 (
    ECHO "Failed to activate conda environment. Please activate it manually and run the commands."
    GOTO :EOF
)

ECHO "2. Installing PyTorch for CUDA 12.8..."
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
IF %ERRORLEVEL% NEQ 0 (
    ECHO "Failed to install PyTorch."
    GOTO :EOF
)

SET "WHEEL_NAME=flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp311-cp311-win_amd64.whl"
SET "WHEEL_URL=https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/%WHEEL_NAME%"

ECHO "3. Downloading Flash Attention wheel: %WHEEL_NAME%"
IF NOT EXIST "%WHEEL_NAME%" (
    powershell -Command "Invoke-WebRequest -Uri %WHEEL_URL% -OutFile %WHEEL_NAME%"
    IF %ERRORLEVEL% NEQ 0 (
        ECHO "Failed to download the wheel."
        GOTO :EOF
    )
) ELSE (
    ECHO "Wheel already exists."
)

ECHO "4. Installing Flash Attention..."
pip install "%WHEEL_NAME%"
IF %ERRORLEVEL% NEQ 0 (
    ECHO "Failed to install Flash Attention."
    GOTO :EOF
)

ECHO "5. Verifying installation..."
python -c "import sys, torch, flash_attn; print('--- Verification ---'); print(f'Python version: {sys.version}'); print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Flash Attention version: {flash_attn.__version__}'); print('--- SUCCESS ---')"
IF %ERRORLEVEL% NEQ 0 (
    ECHO "Verification script failed."
    GOTO :EOF
)

ECHO "--- Setup Finished Successfully ---" 