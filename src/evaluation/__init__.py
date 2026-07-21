"""
Метрики качества моделей.

Использует sklearn/scipy для расчёта:
- R² (коэффициент детерминации)
- MSE (среднеквадратичная ошибка)
- RMSE (корень из MSE)
- MAE (средняя абсолютная ошибка)
- Pearson r (коэффициент корреляции Пирсона)
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class MetricsCalculator:
    """Калькулятор метрик качества модели."""

    @staticmethod
    def calculate_all(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = "",
    ) -> Dict[str, float]:
        """
        Вычисляет полный набор метрик.

        Параметры
        ----------
        y_true : истинные значения
        y_pred : предсказанные значения
        prefix : префикс для ключей словаря (например "mlp_", "pinn_")

        Возвращает
        ----------
        Словарь с ключами: {prefix}r2, {prefix}mse, {prefix}rmse, {prefix}mae, {prefix}pearson
        """
        y_t = y_true.flatten()
        y_p = y_pred.flatten()

        r2 = float(r2_score(y_t, y_p))
        mse = float(mean_squared_error(y_t, y_p))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_t, y_p))
        pearson = float(pearsonr(y_t, y_p)[0]) if len(y_t) > 1 else 0.0

        return {
            f"{prefix}r2": r2,
            f"{prefix}mse": mse,
            f"{prefix}rmse": rmse,
            f"{prefix}mae": mae,
            f"{prefix}pearson": pearson,
        }
