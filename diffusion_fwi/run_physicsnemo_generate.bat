@echo off
REM Генерация (sampling) NVIDIA PhysicsNeMo diffusion_fwi
REM Использование:
REM   diffusion_fwi\run_physicsnemo_generate.bat <dataset_dir> <checkpoint_path> [physics_informed:true|false]

setlocal
set EXAMPLE_DIR=%~dp0physicsnemo\examples\geophysics\diffusion_fwi
set DATASET_DIR=%1
set CHECKPOINT=%2
set PHYSICS=%3

if "%DATASET_DIR%"=="" (
  echo Usage: run_physicsnemo_generate.bat ^<dataset_dir^> ^<checkpoint_path^> [physics_informed]
  exit /b 1
)

if "%CHECKPOINT%"=="" (
  echo Checkpoint path required
  exit /b 1
)

if "%PHYSICS%"=="" set PHYSICS=false

cd /d "%EXAMPLE_DIR%"
python generate.py --config-name=config_generate ++dataset.directory="%DATASET_DIR%" ++model.checkpoint_path="%CHECKPOINT%" ++generation.sampler.physics_informed=%PHYSICS% ++generation.num_ensembles=4
