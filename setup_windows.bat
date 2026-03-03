@echo off
echo ============================================================
echo Speech Transcription POC - Windows Setup Script
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.11 from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Create virtual environment
echo Creating virtual environment...
if exist venv (
    echo [INFO] Virtual environment already exists
) else (
    python -m venv venv
    echo [OK] Virtual environment created
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install PyTorch (CPU version)
echo Installing PyTorch (CPU version)...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
echo.

REM Install other dependencies
echo Installing other dependencies...
pip install -r requirements.txt
echo.

REM Run test script
echo.
echo ============================================================
echo Running installation test...
echo ============================================================
python simple_test.py
echo.

echo ============================================================
echo Setup complete!
echo ============================================================
echo.
echo Next steps:
echo 1. Set your Hugging Face token:
echo    set HF_TOKEN=your_token_here
echo.
echo 2. List audio devices:
echo    python main.py --list-devices
echo.
echo 3. Start real-time transcription:
echo    python main.py
echo.
pause
