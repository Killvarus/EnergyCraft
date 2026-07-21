# Данные EnergyCraft

В репозиторий **не попадают** сами файлы данных — только этот README.

## Локально

Положите исходные SEG-Y в эту папку:

```
data/
  Vp_zoloto_101shot_501rec_1000ms.sgy
  Vp_zoloto_50sm-50sm.sgy
  Vp_zoloto_50sm-50sm_MODEL.sgy
  Domanic_Vp_2-2.sgy
  Domanic_Vp_2-2_MODEL.sgy
```

Затем конвертируйте:

```powershell
python scripts/convert_sgy_to_numpy.py
```

Результат: `data/processed/` (NumPy + `manifest.json`).

## Google Colab

См. `notebooks/EnergyCraft_Colab.ipynb` — шаблон загрузки с Google Drive.
