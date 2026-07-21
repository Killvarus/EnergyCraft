"""
Конфигурация проекта.

Все параметры настраиваются через глобальные переменные в этом файле.
После изменения любого значения — просто перезапустите скрипт.
"""
from pathlib import Path
import torch

# =============================================================================
# ПУТИ
# =============================================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Исходные SEG-Y (для конвертации)
DATA_DIR: Path = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

# Legacy-пути (SEG-Y) — используются только скриптом конвертации
SEGY_PATH: str = str(DATA_DIR / "Vp_zoloto_101shot_501rec_1000ms.sgy")
VP_SEGY_PATH: str = str(DATA_DIR / "Vp_zoloto_50sm-50sm.sgy")
VP_MODEL_SEGY_PATH: str = str(DATA_DIR / "Vp_zoloto_50sm-50sm_MODEL.sgy")

# Шаг сетки модели по X [м]
VP_DX_METERS: float = 0.5

# =============================================================================
# ВЫБОР ДАТАСЕТОВ (NumPy / manifest)
# =============================================================================
# Пресет определяет пару «сейсмика + модель Vp» и размеры окон.
#   "zoloto"  — Vp_zoloto_* (две сейсмики к одной модели)
#   "domanic" — Domanic_Vp_2-2 (сейсмика) + Domanic_Vp_2-2_MODEL (Vp)
ACTIVE_DATASET: str = "zoloto"

DATASET_PRESETS: dict = {
  "zoloto": {
    "seismic": "Vp_zoloto_101shot_501rec_1000ms",
    "seismic_alt": "Vp_zoloto_50sm-50sm",
    "velocity_model": "Vp_zoloto_50sm-50sm_MODEL",
    "height": 501,
    "width": 201,
    "vp_dx_meters": 0.5,
    "note": (
      "Две сейсмики (101 shot и 21 shot) к одной модели Vp_zoloto_50sm-50sm_MODEL, "
      "разные параметры синтетической генерации."
    ),
  },
  "domanic": {
    "seismic": "Domanic_Vp_2-2",
    "seismic_alt": None,
    "velocity_model": "Domanic_Vp_2-2_MODEL",
    "height": 1001,
    "width": 201,
    "vp_dx_meters": 0.5,
    "note": "Domanic_Vp_2-2 — сейсмограммы; Domanic_Vp_2-2_MODEL — скоростная модель к ним.",
  },
}

# Все пары «сейсмика → модель Vp» для совместного обучения
DATASET_PAIRS: list = [
  {
    "id": "zoloto_101shot",
    "family": "zoloto",
    "seismic": "Vp_zoloto_101shot_501rec_1000ms",
    "velocity_model": "Vp_zoloto_50sm-50sm_MODEL",
    "vp_dx_meters": 0.5,
  },
  {
    "id": "zoloto_21shot",
    "family": "zoloto",
    "seismic": "Vp_zoloto_50sm-50sm",
    "velocity_model": "Vp_zoloto_50sm-50sm_MODEL",
    "vp_dx_meters": 0.5,
  },
  {
    "id": "domanic",
    "family": "domanic",
    "seismic": "Domanic_Vp_2-2",
    "velocity_model": "Domanic_Vp_2-2_MODEL",
    "vp_dx_meters": 0.5,
  },
]

# True = обучать на всех парах из DATASET_PAIRS (рекомендуется)
USE_ALL_DATASETS: bool = True

ACTIVE_SEISMIC_DATASET: str = "Vp_zoloto_101shot_501rec_1000ms"
ACTIVE_SEISMIC_ALT: str = "Vp_zoloto_50sm-50sm"
ACTIVE_VELOCITY_MODEL: str = "Vp_zoloto_50sm-50sm_MODEL"


def apply_dataset_preset(name: str | None = None) -> dict:
  """
  Применяет пресет датасета к ACTIVE_* и размерам окон (HEIGHT, WIDTH).
  Вызывается автоматически в run-скриптах; можно вызвать вручную после смены ACTIVE_DATASET.
  """
  global ACTIVE_SEISMIC_DATASET, ACTIVE_SEISMIC_ALT, ACTIVE_VELOCITY_MODEL
  global HEIGHT, WIDTH, VP_DX_METERS

  key = name or ACTIVE_DATASET
  if key not in DATASET_PRESETS:
    raise ValueError(f"Unknown dataset preset: {key}. Available: {list(DATASET_PRESETS)}")

  preset = DATASET_PRESETS[key]
  ACTIVE_SEISMIC_DATASET = preset["seismic"]
  ACTIVE_SEISMIC_ALT = preset.get("seismic_alt") or preset["seismic"]
  ACTIVE_VELOCITY_MODEL = preset["velocity_model"]
  HEIGHT = preset["height"]
  WIDTH = preset["width"]
  VP_DX_METERS = preset.get("vp_dx_meters", VP_DX_METERS)
  return preset


OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
MODELS_DIR: Path = OUTPUT_DIR / "models"
PLOTS_DIR: Path = OUTPUT_DIR / "plots"
LOGS_DIR: Path = OUTPUT_DIR / "logs"
TENSORBOARD_DIR: Path = OUTPUT_DIR / "tensorboard"

# =============================================================================
# ПАРАМЕТРЫ ДАННЫХ
# =============================================================================
HEIGHT: int = 501                # отсчётов по времени (глубине)
WIDTH: int = 201                 # ширина окна по X (трассы)
VP_MIN: float = 1500.0           # минимальная скорость Vp [м/с]
VP_MAX: float = 4500.0           # максимальная скорость Vp [м/с]

# Режим разбиения окон:
#   "spatial" — блоками по X (честная оценка, без утечки; рекомендуется)
#   "random"  — вперемешку (эксперимент: сеть видит паттерны со всего профиля,
#               но метрики завышены — соседние окна перекрываются)
SPLIT_MODE: str = "spatial"

# Разбиение набора: train/val/test = 70/20/10
SPLIT_TRAIN_RATIO: float = 0.7
SPLIT_VAL_RATIO: float = 0.2
SPLIT_TEST_RATIO: float = 0.1
SPLIT_RANDOM_STATE: int = 42

# =============================================================================
# УСТРОЙСТВО
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# АРХИТЕКТУРА MLP
# =============================================================================
# Кортеж размеров скрытых слоёв. Каждое число — количество нейронов в слое.
# Примеры:
#   (512, 1024, 512)         — 3 скрытых слоя, ~104M параметров
#   (256, 512, 256)          — 3 скрытых слоя, ~52M параметров
#   (128, 256, 128)          — 3 скрытых слоя, ~26M параметров
#   (64,)                    — 1 скрытый слой, лёгкая модель
# Архитектура baseline: "cnn" (рекомендуется) или "mlp" (legacy flatten-MLP)
BASELINE_ARCH: str = "cnn"

# Базовое число каналов CNN (encoder-decoder); параметров ~1.5M при 32
CNN_BASE_CHANNELS: int = 32

# При ~20-60 обучающих окнах большие сети (100M+) гарантированно переобучаются.
MLP_HIDDEN_DIMS: tuple = (256, 256)

# Функция активации: "relu", "leaky_relu", "gelu", "tanh", "sigmoid"
MLP_ACTIVATION: str = "relu"

# Использовать LayerNorm между слоями
MLP_USE_LAYERNORM: bool = True

# Dropout между слоями (0.0 = без dropout)
MLP_DROPOUT: float = 0.0

# =============================================================================
# АРХИТЕКТУРА PINN
# =============================================================================
# Новый формат: явный кортеж скрытых слоёв (как у MLP)
PINN_HIDDEN_DIMS: tuple = (256, 512, 256)

# Legacy-параметры оставлены для обратной совместимости
PINN_HIDDEN_DIM: int = 128      # нейронов в каждом скрытом слое (legacy)
PINN_NUM_LAYERS: int = 4        # всего слоёв (legacy)
PINN_ACTIVATION: str = "tanh"   # "tanh", "relu", "gelu", "sin"

# =============================================================================
# ОБУЧЕНИЕ — ОБЩИЕ ПАРАМЕТРЫ
# =============================================================================

# --- Оптимизатор ---
# "adam", "sgd", "adamw", "rmsprop"
OPTIMIZER: str = "adam"

# Learning rate
LR: float = 0.001

# Momentum (только для SGD; для Adam игнорируется)
MOMENTUM: float = 0.9

# Weight decay (L2-регуляризация)
WEIGHT_DECAY: float = 0.0

# --- Early stopping (по val_loss) ---
# PATIENCE — мин. относительное улучшение val_loss за эпоху (доля, напр. 0.003 = 0.3%)
# N_EPOCH   — сколько эпох подряд без улучшения >= PATIENCE → остановка
# MAX_EPOCHS — верхний лимит эпох; 0 = без лимита (только early stopping)
PATIENCE: float = 0.003
N_EPOCH: int = 250
MAX_EPOCHS: int = 100000

# --- Эпохи и батч ---
MLP_BATCH_SIZE: int = 16

# Batch-режим MLP (несколько независимых прогонов)
MLP_N_ITERATIONS: int = 1        # количество итераций обучения (CNN на CPU ~8-10 мин/итерация)
MLP_RANDOM_STATE: int = 42       # базовое зерно
MLP_RANDOM_STATE_STEP: int = 1   # шаг изменения зерна по итерациям
MLP_SELECTION_METRIC: str = "rmse"  # метрика отчёта на test; выбор итерации — по best_val_loss
MLP_SAVE_EACH_ITER_CHECKPOINT: bool = False

PINN_MAX_EPOCHS: int = 100000         # 0 = только early stopping на окне

# --- Функция потерь ---
# "mse" или "l1" (MAE)
LOSS_FN: str = "mse"

# --- Gradient clipping ---
# 0.0 = без клиппинга; рекомендовано 1.0 для PINN
GRAD_CLIP: float = 0.0

# --- Scheduler (планировщик lr) ---
# "none", "step", "cosine", "plateau"
# StepLR: каждые SCHEDULER_STEP_SIZE эпох lr *= SCHEDULER_GAMMA
# CosineAnnealingLR: T_max = MAX_EPOCHS или 10000 при MAX_EPOCHS=0
# ReduceLROnPlateau: снижение при плато
SCHEDULER_TYPE: str = "none"
SCHEDULER_STEP_SIZE: int = 20
SCHEDULER_GAMMA: float = 0.5

# =============================================================================
# PINN — СПЕЦИФИЧНЫЕ ПАРАМЕТРЫ
# =============================================================================
PINN_LAMBDA_PHYS: float = 0.1    # вес PDE-невязки
PINN_LAMBDA_DATA: float = 1.0    # вес невязки с наблюдённой сейсмикой
PINN_LAMBDA_VP: float = 0.01     # вес регуляризации скорости (гладкость)
PINN_LAMBDA_BC: float = 0.05     # вес открытых граничных условий
PINN_N_COLLOC: int = 4096        # коллокационных точек PDE
PINN_N_BOUNDARY: int = 512       # точек на границах
PINN_N_RECEIVERS: int = 2048     # точек наблюдения (сейсмика)
PINN_DT: float = 0.002           # шаг по времени [с] (2 мс)
PINN_DX: float = 0.5             # шаг по X [м]
PINN_DZ: float = 2.0             # шаг по Z [м] (приближённо: глубина/HEIGHT)
PINN_SOURCE_FREQ: float = 15.0   # частота Ricker-источника [Гц]
PINN_VP_INIT: float = 2500.0     # начальная скорость [м/с]

# =============================================================================
# DIFFUSION (локальный baseline)
# =============================================================================
DIFFUSION_BASE_CHANNELS: int = 32
DIFFUSION_TIMESTEPS: int = 100
DIFFUSION_MAX_EPOCHS: int = 0
DIFFUSION_BATCH_SIZE: int = 32
DIFFUSION_LR: float = 1e-4

# Интервал логирования (каждые N эпох); эпоха 1 логируется всегда
LOG_INTERVAL: int = 5

# TensorBoard (встроенное логирование в цикле обучения)
TENSORBOARD_ENABLED: bool = True
TENSORBOARD_LOG_BATCH_EVERY: int = 25

# Сохранять ли графики
SAVE_PLOTS: bool = True

# =============================================================================
# ЗАПУСК
# =============================================================================
# "cnn" | "pinn" | "all"
MODEL_MODE: str = "all"

# ===========================================================================
# ИНИЦИАЛИЗАЦИЯ
# ===========================================================================
def setup_directories() -> None:
    """Создание всех необходимых директорий."""
    for d in [OUTPUT_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# Применить пресет по умолчанию после определения всех переменных
apply_dataset_preset(ACTIVE_DATASET)
