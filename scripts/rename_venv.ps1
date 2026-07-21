# Переименование venv EnergyCraft -> .venv (один раз)
# Запуск: .\scripts\rename_venv.ps1
# Остановите обучение перед переименованием, если папка занята процессом.

$root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $root

if (Test-Path ".venv") {
    Write-Host ".venv already exists — nothing to do."
    exit 0
}

if (-not (Test-Path "EnergyCraft")) {
    Write-Host "Neither EnergyCraft nor .venv found. Create venv: python -m venv .venv"
    exit 1
}

Rename-Item -Path "EnergyCraft" -NewName ".venv"
Write-Host "Done: EnergyCraft -> .venv"
Write-Host "Activate: .\.venv\Scripts\Activate.ps1"
