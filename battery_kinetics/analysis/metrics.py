"""analysis.metrics

常用误差度量：RMSE、MAE、R²、残差频谱。
返回标量或 DataFrame 形式，方便快速评估模型拟合优度。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "rmse",
    "mae",
    "r2_score",
    "calc_metrics",
]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.Series:
    """一次性计算全部度量，返回 pandas.Series。"""
    return pd.Series(
        {
            "RMSE": rmse(y_true, y_pred),
            "MAE": mae(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
        }
    ) 