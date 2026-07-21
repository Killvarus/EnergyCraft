# EnergyCraft — инструкция по запуску

Проект решает **обратную задачу полноволновой инверсии (FWI)**: по сейсмическим данным восстанавливается модель скорости Vp. Реализованы три подхода:


| Подход        | Скрипт                | Описание                                     |
| ------------- | --------------------- | -------------------------------------------- |
| CNN baseline  | `run_cnn_baseline.py` | Supervised encoder-decoder (лучший baseline) |
| PINN          | `run_pinn_fwi.py`     | Physics-Informed NN с волновым уравнением    |
| Diffusion FWI | `diffusion_fwi/`      | NVIDIA PhysicsNeMo + локальный baseline      |


---

## 1. Установка

```powershell
cd D:\Desktop\EnergyCraft
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Если venv раньше назывался `EnergyCraft/` (как папка проекта) — переименуйте в `.venv`. Подробнее: [GITHUB.md](GITHUB.md).

**Требования:** Python 3.10+, PyTorch. GPU рекомендуется для CNN и diffusion.

---



## 2. Подготовка данных (SEG-Y → NumPy)

Все исходные SEG-Y лежат в `data/`. Перед обучением их нужно конвертировать:

```powershell
pip install -r requirements.txt
```



### Структура датасета Zoloto


| Файл                                  | Тип            | Описание                                    |
| ------------------------------------- | -------------- | ------------------------------------------- |
| `Vp_zoloto_50sm-50sm_MODEL.sgy`       | **Модель Vp**  | Целевая скоростная модель (4001×801)        |
| `Vp_zoloto_101shot_501rec_1000ms.sgy` | **Сейсмика A** | 101 shot, dt=2 ms, 501–1001 приёмников/shot |
| `Vp_zoloto_50sm-50sm.sgy`             | **Сейсмика B** | 21 shot, dt=1 ms, 201 приёмник/shot         |


**Важно:** оба файла `101shot` и `50sm-50sm` — это **сейсмограммы к одной и той же модели Vp**, но с разными параметрами синтетической генерации. Модель скорости — только `*_MODEL.sgy`.

### Структура датасета Domanic


| Файл                       | Тип           | Описание                               |
| -------------------------- | ------------- | -------------------------------------- |
| `Domanic_Vp_2-2_MODEL.sgy` | **Модель Vp** | Скоростная модель (93361×1001)         |
| `Domanic_Vp_2-2.sgy`       | **Сейсмика**  | Сейсмограммы к этой модели (8298×1776) |




### Объединённое обучение (по умолчанию)

`USE_ALL_DATASETS = True` в `src/config.py` — модель обучается на **всех** парах:


| ID               | Сейсмика                        | Модель Vp                 |
| ---------------- | ------------------------------- | ------------------------- |
| `zoloto_101shot` | Vp_zoloto_101shot_501rec_1000ms | Vp_zoloto_50sm-50sm_MODEL |
| `zoloto_21shot`  | Vp_zoloto_50sm-50sm             | Vp_zoloto_50sm-50sm_MODEL |
| `domanic`        | Domanic_Vp_2-2                  | Domanic_Vp_2-2_MODEL      |


Окна из всех источников объединяются в один пул; split делается **внутри каждого источника** (spatial), затем train/val/test склеиваются.

```powershell
python run_cnn_baseline.py              # все 3 пары
python run_cnn_baseline.py --dataset zoloto   # 2 сейсмики Zoloto
python run_cnn_baseline.py --single         # одна пара из ACTIVE_DATASET
```

Отключить объединение: `USE_ALL_DATASETS = False` в config.

### Переключение между датасетами (один источник)

В `src/config.py` задайте пресет:

```python
ACTIVE_DATASET = "zoloto"    # или "domanic"
```

Пресет автоматически выставляет сейсмику, модель Vp и размер окон (`HEIGHT`×`WIDTH`).

Или через CLI:

```powershell
python run_cnn_baseline.py --dataset domanic
python run_pinn_fwi.py --dataset domanic
```

Для Zoloto можно дополнительно выбрать альтернативную сейсмику:

```python
ACTIVE_SEISMIC_DATASET = "Vp_zoloto_50sm-50sm"   # вместо 101shot
```

```powershell
python run_pinn_fwi.py --seismic Vp_zoloto_50sm-50sm
```

После смены датасета перезапустите обучение (и при необходимости `convert_sgy_to_numpy.py`).

Результат конвертации:

```
data/processed/
  manifest.json
  Vp_zoloto_*.npy / Domanic_Vp_*.npy
  *_headers.npz
```

**Почему NumPy, а не CSV:** массивы 2D (сотни МБ), бинарный формат быстрее и без потери точности.

Конвертация использует **obspy** (как в `sgy_to_np.ipynb`) + **segyio** для заголовков shot/receiver.

---



## 2.1. Логирование

Все скрипты пишут подробные логи в `outputs/logs/` и дублируют в консоль:


| Файл                  | Содержание                                   |
| --------------------- | -------------------------------------------- |
| `cnn_baseline.log`    | подготовка данных, split, метрики по эпохам  |
| `pinn_fwi.log`        | окна PINN, компоненты loss (data/phys/bc/vp) |
| `diffusion_local.log` | эпохи diffusion                              |


Интервал логирования эпох задаётся в `src/config.py`:

```python
LOG_INTERVAL = 5   # каждые 5 эпох (+ эпоха 1 всегда)
```

Для CNN дополнительно выводится прогресс каждые 25 train-батчей. На CPU одна
эпоха объединённого набора содержит около 347 батчей размером 501×201 и может
занимать продолжительное время. Строка `validation started` означает, что
train-часть эпохи завершена и начался проход по val-выборке.

Пример строки CNN (обучение без фиксированного лимита эпох — остановка по val):

```
Epoch    5 | train=1.23e-02 | val=1.45e-02 | best_val=1.40e-02 (ep 4) | rel_imp=3.57e-03 | no_improve=1/15 | lr=1.00e-03
```

Пример строки PINN:

```
PINN epoch   10 | total=3.21e+02 | data=1.05e-01 | phys=2.10e+00 | bc=4.32e-02 | vp=1.23e-03 | best=3.15e+02 | rel_imp=1.87e-03 | no_improve=2/15
```

### TensorBoard (встроенный)

Метрики пишутся **напрямую из цикла обучения** в `outputs/tensorboard/` (без парсинга логов).

```powershell
tensorboard --logdir outputs/tensorboard --port 6006
```

Откройте http://localhost:6006. В Colab: `%tensorboard --logdir outputs/tensorboard`.

Отключить: `TENSORBOARD_ENABLED = False` в `src/config.py`.

---



## 3. CNN Baseline (рекомендуется начать с этого)

```powershell
python run_cnn_baseline.py
```

**Что происходит:**

1. Загрузка NumPy из `data/processed/` (все пары или одна — см. `USE_ALL_DATASETS`)
2. Построение мозаики сейсмики на сетке модели Vp
3. Нарезка перекрывающихся окон 501×201
4. Spatial split 70/20/10 **внутри каждого источника** (train / val / test)
5. Обучение FWICNN с **ранней остановкой по val_loss** (см. §8)
6. Лучшая итерация выбирается по `best_val_loss`; test — только для финальной оценки
7. Метрики и графики в `outputs/`

**Результаты:**

- `outputs/models/cnn_baseline_best.pt`
- `outputs/summary_cnn_baseline_spatial.json`
- `outputs/plots/cnn_test_window_*.png`

---



## 4. PINN (Physics-Informed Neural Network)

```powershell
python run_pinn_fwi.py
python run_pinn_fwi.py --max-windows 3 --epochs 100
```

**Физическое уравнение в функции потерь:**

```
(1 / v(x,z)²) · ∂²u/∂t²  −  ∂²u/∂x²  −  ∂²u/∂z²  =  f(x,z,t)
```

**Компоненты loss:**

- `L_data` — невязка волнового поля с наблюдённой сейсмикой в точках приёмников
- `L_phys` — невязка PDE в коллокационных точках
- `L_bc` — открытые граничные условия
- `L_vp` — гладкость скорости

**Архитектура:** P-Net (u) + V-Net (v) с Fourier features (по мотивам arXiv:2601.16068).

Параметры в `src/config.py` (секция `PINN_*`).

**Результаты:** `outputs/summary_pinn_fwi_spatial.json`, графики в `outputs/plots/`.

> PINN на одном окне обучается дольше CNN. Для быстрого теста используйте `--max-windows 2 --epochs 50`.

---



## 5. Diffusion Model for FWI (NVIDIA PhysicsNeMo)



### 5.1. Установка PhysicsNeMo

```powershell
powershell -ExecutionPolicy Bypass -File diffusion_fwi\setup_physicsnemo.ps1
```

Скрипт клонирует [NVIDIA/physicsnemo](https://github.com/NVIDIA/physicsnemo) и устанавливает зависимости примера `diffusion_fwi`.

### 5.2. Вариант A — E-FWI датасет (оригинальный пример NVIDIA)

Официальный пайплайн использует [E-FWI dataset](https://github.com/YangFangShu/EFWI) (лицензия CC BY-NC-SA 4.0, 100+ GB):

```powershell
cd diffusion_fwi\physicsnemo\examples\geophysics\diffusion_fwi\data
python download_data.py --download --reorganize --clean --shuffle --name all
python generate_data.py --in_dir ./all --out_dir C:\data\efwi_generated
python compute_stats.py --dir C:\data\efwi_generated --batch_size 512 --num_workers 4
```

**Обучение:**

```powershell
diffusion_fwi\run_physicsnemo_train.bat C:\data\efwi_generated
```

**Генерация (zero-shot):**

```powershell
diffusion_fwi\run_physicsnemo_generate.bat C:\data\efwi_generated C:\path\to\checkpoint.mdlus false
```

**Physics-informed sampling (DPS):**

```powershell
diffusion_fwi\run_physicsnemo_generate.bat C:\data\efwi_generated C:\path\to\checkpoint.mdlus true
```

Документация: [https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/geophysics/diffusion_fwi/README.html](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/geophysics/diffusion_fwi/README.html)

### 5.3. Вариант B — локальный датасет EnergyCraft

Подготовка окон в формате `.npz`:

```powershell
python diffusion_fwi\prepare_dataset.py --output diffusion_fwi\data\energycraft_windows
```

**Локальный diffusion baseline** (без E-FWI, на ваших данных):

```powershell
python diffusion_fwi\local_diffusion\train.py --epochs 50 --batch-size 4
```

Чекпоинт: `outputs/models/local_diffusion_fwi.pt`

---



## 6. Запуск всего пайплайна

```powershell
# Конвертация + CNN + PINN
python main.py --convert

# Только CNN
python main.py --mode cnn

# Только PINN
python main.py --mode pinn
```

---



## 7. Структура проекта

```
EnergyCraft/
├── data/                          # исходные SEG-Y
│   └── processed/                 # NumPy после конвертации
├── scripts/
│   └── convert_sgy_to_numpy.py    # SEG-Y → NumPy
├── src/
│   ├── config.py                  # все гиперпараметры
│   ├── data/                      # загрузка, мозаика, нормализация
│   ├── models/                    # CNN, PINN, physics
│   ├── training/                  # циклы обучения
│   ├── evaluation/                # метрики
│   └── visualization/             # графики
├── diffusion_fwi/                 # NVIDIA + локальный diffusion
├── run_cnn_baseline.py
├── run_pinn_fwi.py
├── main.py
└── outputs/                       # модели, логи, графики
```

---



## 8. Настройка параметров (общие)

Все параметры — в `src/config.py`:


| Параметр                 | Описание                                                  | По умолчанию   |
| ------------------------ | --------------------------------------------------------- | -------------- |
| `USE_ALL_DATASETS`       | объединять все пары                                       | `True`         |
| `ACTIVE_SEISMIC_DATASET` | активная сейсмика (задаётся пресетом)                     | см. preset     |
| `HEIGHT`, `WIDTH`        | размер окна (T × X)                                       | 501 × 201      |
| `SPLIT_MODE`             | `"spatial"` или `"random"`                                | spatial        |
| `LOG_INTERVAL`           | лог каждые N эпох                                         | 5              |
| `PATIENCE`               | мин. **относительное** улучшение val_loss за эпоху (доля) | `0.003` (0.3%) |
| `N_EPOCH`                | эпох подряд без улучшения ≥ `PATIENCE` → стоп             | `15`           |
| `MAX_EPOCHS`             | верхний лимит эпох CNN; `0` = без лимита                  | `0`            |
| `PINN_MAX_EPOCHS`        | то же для PINN на окно                                    | `0`            |
| `DIFFUSION_MAX_EPOCHS`   | то же для локального diffusion                            | `0`            |




### Ранняя остановка (val-driven)

Обучение CNN и diffusion **не ограничено фиксированным числом эпох** (при `MAX_EPOCHS=0`). Каждую эпоху считается `val_loss` на валидационной выборке. Улучшение засчитывается, если:

```
(val_best - val_current) / |val_best|  >=  PATIENCE
```

Если такого улучшения нет `N_EPOCH` эпох подряд — обучение останавливается, загружается чекпоинт с лучшим `val_loss`.

- **Train** — только для градиентного шага
- **Val** — критерий остановки и выбор лучшей модели
- **Test** — финальная оценка после обучения (не участвует в подборе)

При нескольких итерациях (`MLP_N_ITERATIONS > 1`) лучшая итерация выбирается по минимальному `best_val_loss`.

Опционально задайте `MAX_EPOCHS > 0` как жёсткий потолок (страховка от бесконечного цикла).

---



## 9. Fine-tuning: гиперпараметры нейросетей

Все архитектурные параметры задаются в `src/config.py`. После изменения — перезапустите соответствующий скрипт.

### 9.1. CNN Baseline (`FWICNN`)


| Параметр            | Файл   | Что меняет                            | Рекомендации                                      |
| ------------------- | ------ | ------------------------------------- | ------------------------------------------------- |
| `BASELINE_ARCH`     | config | `"cnn"` или `"mlp"`                   | оставьте `"cnn"`                                  |
| `CNN_BASE_CHANNELS` | config | ширина encoder-decoder (32→64→128)    | 32 для малых выборок; 64 если >100 окон           |
| `MLP_HIDDEN_DIMS`   | config | слои MLP (если `BASELINE_ARCH="mlp"`) | `(256,256)` — default; уменьшите при переобучении |
| `MLP_DROPOUT`       | config | dropout в MLP                         | 0.0–0.2 при переобучении                          |
| `MLP_BATCH_SIZE`    | config | размер батча                          | 2–8 (зависит от GPU RAM)                          |
| `LR`                | config | learning rate                         | `1e-3` default; `1e-4` для fine-tune с чекпоинта  |
| `PATIENCE`          | config | мин. относительное улучшение val_loss | `0.001`–`0.01`; меньше → дольше обучение          |
| `N_EPOCH`           | config | эпох без улучшения до стопа           | `10`–`30`                                         |
| `MAX_EPOCHS`        | config | потолок эпох (`0` = без лимита)       | `0`                                               |
| `GRAD_CLIP`         | config | clipping градиента                    | 0.0 для CNN; 1.0 если loss скачет                 |
| `SCHEDULER_TYPE`    | config | `"none"`, `"cosine"`, `"plateau"`     | `"cosine"` для длинного обучения                  |


**Пример — более строгая остановка:**

```python
PATIENCE: float = 0.005
N_EPOCH: int = 20
MAX_EPOCHS: int = 0          # только early stopping
CNN_BASE_CHANNELS: int = 64
LR: float = 0.0005
SCHEDULER_TYPE: str = "cosine"
```

**Загрузка чекпоинта для дообучения** (в Python):

```python
import torch
from src.models import FWICNN
model = FWICNN(base_channels=32)
ckpt = torch.load("outputs/models/cnn_baseline_best.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
# затем train_mlp(model, ...) с меньшим LR
```



### 9.2. PINN (`FWI_PINN`)


| Параметр                        | Что меняет                            | Рекомендации                                        |
| ------------------------------- | ------------------------------------- | --------------------------------------------------- |
| `PINN_HIDDEN_DIMS`              | размеры скрытых слоёв P-Net и V-Net   | `(128,256,128)` — быстрее; `(256,512,256)` — точнее |
| `PINN_ACTIVATION`               | `"tanh"`, `"relu"`, `"gelu"`, `"sin"` | `"tanh"` или `"sin"` для PINN                       |
| `PINN_LAMBDA_DATA`              | вес невязки с сейсмикой               | 1.0 (основной сигнал)                               |
| `PINN_LAMBDA_PHYS`              | вес PDE-невязки                       | 0.01–0.5; уменьшите если phys loss взрывается       |
| `PINN_LAMBDA_BC`                | вес граничных условий                 | 0.01–0.1                                            |
| `PINN_LAMBDA_VP`                | гладкость скорости                    | 0.001–0.1                                           |
| `PINN_N_COLLOC`                 | точек PDE на эпоху                    | 2048–8192                                           |
| `PINN_N_RECEIVERS`              | точек сейсмики на эпоху               | 1024–4096                                           |
| `PINN_SOURCE_FREQ`              | частота Ricker [Гц]                   | подстройте под ваши данные (10–25)                  |
| `PINN_DT`, `PINN_DX`, `PINN_DZ` | физические шаги сетки                 | должны соответствовать данным                       |
| `GRAD_CLIP`                     | clipping                              | **1.0** рекомендуется для PINN                      |
| `LR`                            | learning rate                         | `1e-3` → `1e-4` при нестабильности                  |


**Пример — стабилизировать PINN:**

```python
PINN_LAMBDA_PHYS: float = 0.05
PINN_LAMBDA_DATA: float = 1.0
GRAD_CLIP: float = 1.0
LR: float = 0.0003
PATIENCE: float = 0.003
N_EPOCH: int = 25
PINN_MAX_EPOCHS: int = 0
```

PINN использует те же `PATIENCE` / `N_EPOCH` по суммарному loss окна (отдельного val-split на окно нет).

Fourier features включаются в `src/models/pinn_fwi.py` (`use_fourier=True`, `n_fourier=32`).

### 9.3. Diffusion (локальный baseline)


| Параметр                  | Что меняет                                 | По умолчанию |
| ------------------------- | ------------------------------------------ | ------------ |
| `DIFFUSION_BASE_CHANNELS` | ширина U-Net                               | 32           |
| `DIFFUSION_TIMESTEPS`     | шаги диффузии                              | 100          |
| `DIFFUSION_MAX_EPOCHS`    | потолок эпох (`0` = early stopping по val) | 0            |
| `DIFFUSION_BATCH_SIZE`    | батч                                       | 4            |
| `DIFFUSION_LR`            | learning rate                              | 1e-4         |


Локальный diffusion обучается с val_loader и той же схемой `PATIENCE` / `N_EPOCH`.

### 9.4. Данные и окна


| Параметр                                | Что меняет                                             |
| --------------------------------------- | ------------------------------------------------------ |
| `WIDTH`                                 | ширина окна по X (трассы)                              |
| `HEIGHT`                                | высота окна по T (время)                               |
| `WINDOW_STRIDE` в `run_cnn_baseline.py` | шаг нарезки (`WIDTH//4` default); меньше → больше окон |
| `SPLIT_MODE`                            | `"spatial"` (честно) / `"random"` (оптимистично)       |
| `VP_MIN`, `VP_MAX`                      | границы нормализации Vp                                |




### 9.5. Чеклист fine-tuning

1. Зафиксируйте `USE_ALL_DATASETS` / `ACTIVE_DATASET` и переконвертируйте данные при необходимости.
2. Начните с CNN baseline — подберите `CNN_BASE_CHANNELS`, `LR`, `PATIENCE`, `N_EPOCH`.
3. Смотрите `outputs/logs/cnn_baseline.log` — `val` должен сходиться; если растёт, уменьшите `LR`.
4. Для PINN стабилизируйте `PINN_LAMBDA_PHYS` и `GRAD_CLIP`; длительность задаётся early stopping.
5. Сравнивайте `test_metrics_per_source` и stitched-метрики в `outputs/summary_*.json`.

---



## 10. Устранение неполадок

`Manifest not found` — запустите `python scripts/convert_sgy_to_numpy.py`.

**PINN loss взрывается** — уменьшите `PINN_LAMBDA_PHYS`, увеличьте `GRAD_CLIP=1.0` в config.

**PhysicsNeMo: нет GPU** — diffusion FWI требует CUDA; используйте локальный baseline или облачный GPU.

**Мало окон для обучения** — уменьшите `WINDOW_STRIDE` в `run_cnn_baseline.py` (по умолчанию WIDTH//4).

---



## 11. Ссылки

- Статья PINN: arXiv:2601.16068 (файл `2601.16068v1 (2).pdf` в корне)
- NVIDIA diffusion FWI: [https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/geophysics/diffusion_fwi/README.html](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/geophysics/diffusion_fwi/README.html)
- E-FWI dataset: [https://github.com/YangFangShu/EFWI](https://github.com/YangFangShu/EFWI)

