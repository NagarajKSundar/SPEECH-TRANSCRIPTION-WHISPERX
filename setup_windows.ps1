# Speech Transcription POC - Windows PowerShell Setup Script
# Run this script in PowerShell

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Speech Transcription POC - Windows Setup Script" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.11 from https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "[INFO] Virtual environment already exists" -ForegroundColor Cyan
} else {
    python -m venv venv
    Write-Host "[OK] Virtual environment created" -ForegroundColor Green
}

Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

Write-Host ""

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

Write-Host ""

# Install PyTorch (CPU version)
Write-Host "Installing PyTorch (CPU version)..." -ForegroundColor Yellow
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

Write-Host ""

# Install other dependencies
Write-Host "Installing other dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Running installation test..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

python simple_test.py

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Set your Hugging Face token:" -ForegroundColor White
Write-Host '   $env:HF_TOKEN = "your_token_here"' -ForegroundColor Cyan
Write-Host ""
Write-Host "2. List audio devices:" -ForegroundColor White
Write-Host "   python main.py --list-devices" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. Start real-time transcription:" -ForegroundColor White
Write-Host "   python main.py" -ForegroundColor Cyan
Write-Host ""
