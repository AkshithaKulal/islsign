$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment .venv ..."
    python -m venv .venv
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe src\check_features.py

Write-Host ""
Write-Host "Setup done. Activate with:  .\.venv\Scripts\Activate.ps1"
Write-Host "Record a check set with:    python src\record.py --label hello --count 5"
