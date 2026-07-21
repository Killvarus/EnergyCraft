# Установка NVIDIA PhysicsNeMo diffusion_fwi
# Запуск из корня проекта (PowerShell):
#   .\diffusion_fwi\setup_physicsnemo.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Target = Join-Path $Root "diffusion_fwi" "physicsnemo"

Write-Host "Cloning NVIDIA PhysicsNeMo..."
if (-not (Test-Path $Target)) {
    git clone --depth 1 https://github.com/NVIDIA/physicsnemo.git $Target
}

$ExampleDir = Join-Path $Target "examples" "geophysics" "diffusion_fwi"
if (-not (Test-Path $ExampleDir)) {
    Write-Error "diffusion_fwi example not found in PhysicsNeMo repo"
}

Write-Host "Installing PhysicsNeMo..."
pip install -e $Target

Write-Host "Installing diffusion_fwi requirements..."
pip install -r (Join-Path $ExampleDir "requirements.txt")

Write-Host ""
Write-Host "PhysicsNeMo diffusion_fwi installed."
Write-Host "Example directory: $ExampleDir"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Download E-FWI dataset (see RUNNING.md)"
Write-Host "  2. Or use local dataset: python diffusion_fwi/prepare_dataset.py"
Write-Host "  3. Train local baseline: python diffusion_fwi/local_diffusion/train.py"
