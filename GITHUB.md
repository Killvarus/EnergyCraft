# Выгрузка EnergyCraft на GitHub

## Проблема с виртуальным окружением

Раньше venv назывался `EnergyCraft/` — так же, как корневая папка проекта. Из-за этого:

- Git мог путать пути;
- `.gitignore` с `/EnergyCraft/` работал неочевидно;
- в репозиторий случайно попадали тысячи файлов из `site-packages`.

**Решение:** стандартное имя `.venv/` в корне проекта.

### Переименование (один раз)

Если у вас ещё папка `EnergyCraft/` с Python внутри:

```powershell
cd D:\Desktop\EnergyCraft

# остановите активные процессы обучения (Ctrl+C)

Rename-Item -Path "EnergyCraft" -NewName ".venv"

# активируйте новое окружение
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Если `.venv` уже есть — просто используйте его.

---

## Что попадает в Git, а что нет

| В Git | Не в Git |
|-------|----------|
| `src/`, `scripts/`, `run_*.py`, `main.py` | `.venv/`, `EnergyCraft/` (старое имя) |
| `diffusion_fwi/`, `notebooks/` | `data/*.sgy`, `data/processed/` |
| `requirements.txt`, `RUNNING.md`, `README.md` | `outputs/` (модели, логи, tensorboard) |
| `data/README.md` (описание данных) | `*.npy`, `*.npz`, `*.pt` |

Проверьте `.gitignore` перед первым push.

---

## Первый push на GitHub

### 1. Создайте репозиторий на GitHub

На https://github.com/new создайте пустой репозиторий (без README, без .gitignore).

### 2. Проверьте, что не попадёт мусор

```powershell
cd D:\Desktop\EnergyCraft
git status
```

Убедитесь, что **не** видны:

- `.venv/` или `EnergyCraft/`
- `data/processed/`
- `outputs/`
- `*.npy`, `*.sgy`

Если venv уже был добавлен в индекс:

```powershell
git rm -r --cached EnergyCraft 2>$null
git rm -r --cached .venv 2>$null
git rm -r --cached outputs 2>$null
git rm -r --cached data/processed 2>$null
```

### 3. Закоммитьте код

```powershell
git add .
git status   # ещё раз проверьте список
git commit -m "Add FWI pipeline: CNN, PINN, diffusion, Colab notebook"
```

### 4. Привяжите remote и отправьте

```powershell
git remote add origin https://github.com/<USER>/<REPO>.git
git branch -M main
git push -u origin main
```

Если `origin` уже есть:

```powershell
git remote set-url origin https://github.com/<USER>/<REPO>.git
git push -u origin main
```

---

## Обновление репозитория после изменений

```powershell
git add .
git commit -m "Описание изменений"
git push
```

---

## Google Colab

После push откройте `notebooks/EnergyCraft_Colab.ipynb` в Colab и укажите URL вашего репозитория в первой ячейке конфигурации.

Данные загружайте с Google Drive (шаблон в ноутбуке) — они не хранятся в Git.
