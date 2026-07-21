@echo off
REM Запуск обучения NVIDIA PhysicsNeMo diffusion_fwi (требует GPU и E-FWI датасет)
REM Использование:
REM   diffusion_fwi\run_physicsnemo_train.bat <path_to_efwi_dataset>

setlocal
set EXAMPLE_DIR=%~dp0physicsnemo\examples\geophysics\diffusion_fwi
set DATASET_DIR=%1

if "%DATASET_DIR%"=="" (
  echo Usage: run_physicsnemo_train.bat ^<path_to_dataset_directory^>
  echo Example: run_physicsnemo_train.bat C:\data\efwi\all
  exit /b 1
)

if not exist "%EXAMPLE_DIR%\train.py" (
  echo PhysicsNeMo not installed. Run: powershell -File diffusion_fwi\setup_physicsnemo.ps1
  exit /b 1
)

cd /d "%EXAMPLE_DIR%"
python train.py --config-name=config_train ++dataset.directory="%DATASET_DIR%" ++training.batch_size_per_device=8
