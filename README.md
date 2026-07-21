# EnergyCraft FWI Research Pipeline

Проект решает **обратную задачу полноволновой инверсии (FWI)**: восстановление модели скорости Vp по сейсмическим данным. Цель — эволюция от supervised baseline к **FF-PINN** и diffusion-подходам.

## Подходы

| Модель | Скрипт | Статус |
|--------|--------|--------|
| CNN encoder-decoder | `run_cnn_baseline.py` | Основной baseline |
| PINN (волновое уравнение) | `run_pinn_fwi.py` | Реализован |
| Diffusion FWI (NVIDIA + local) | `diffusion_fwi/` | Реализован |

## Быстрый старт

```powershell
pip install -r requirements.txt
python scripts/convert_sgy_to_numpy.py   # SEG-Y → NumPy
python run_cnn_baseline.py             # CNN baseline
python run_pinn_fwi.py                   # PINN
python main.py --convert                 # всё сразу
```

**Полная инструкция:** [RUNNING.md](RUNNING.md)  
**GitHub и Colab:** [GITHUB.md](GITHUB.md) | [notebooks/EnergyCraft_Colab.ipynb](notebooks/EnergyCraft_Colab.ipynb)

## Данные

- Исходные SEG-Y: `data/`
- После конвертации: `data/processed/` (NumPy + manifest.json)
- **Все источники (default):** `USE_ALL_DATASETS=True` — 3 пары, ~сотни окон
- **Zoloto:** `--dataset zoloto` — 2 сейсмики + одна модель Vp
- **Domanic:** `--dataset domanic`
- Переключение одного источника: `--single` или `USE_ALL_DATASETS=False`

Конвертация основана на `sgy_to_np.ipynb` (obspy).

## Структура

```
scripts/convert_sgy_to_numpy.py   # конвертация данных
src/data/                         # загрузка NumPy, мозаика, окна
src/models/                       # CNN, PINN, physics
diffusion_fwi/                    # NVIDIA PhysicsNeMo + local diffusion
outputs/                          # модели, графики, метрики
```

## Ссылки

- PINN-статья: `2601.16068v1 (2).pdf` (arXiv:2601.16068)
- NVIDIA diffusion FWI: https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/geophysics/diffusion_fwi/README.html
