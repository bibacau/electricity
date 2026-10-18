$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path '.venv\Scripts\python.exe')) { throw 'Run scripts\setup.ps1 first.' }
Write-Host 'Personal AI is starting on port 8765. Press Ctrl+C to stop.'
& .venv\Scripts\python.exe -m personal_ai --data data
if ($LASTEXITCODE -ne 0) { throw 'Workbench exited with an error.' }
