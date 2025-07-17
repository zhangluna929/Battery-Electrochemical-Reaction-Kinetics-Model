"""io.experiment

实验数据读取器，支持常见电池测试设备格式。
返回统一的 pandas.DataFrame，包含列：time, current, voltage, temperature。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

__all__ = ["read_arbin_csv", "read_biologic_mpr"]


def read_arbin_csv(path: str | Path) -> pd.DataFrame:  # noqa: D401
    """读取 Arbin 导出的 CSV 文件（通道测试记录）。"""
    df = pd.read_csv(path, low_memory=False)
    # Arbin 常见列名：'Date_Time', 'Current(A)', 'Voltage(V)', 'Temperature(C)' 等
    col_map = {
        "Current(A)": "current",
        "Voltage(V)": "voltage",
        "Temperature(C)": "temperature",
        "Test_Time(s)": "time",
        "Time(s)": "time",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    # 只保留需要列
    keep = [c for c in ["time", "current", "voltage", "temperature"] if c in df.columns]
    return df[keep].copy()


def read_biologic_mpr(path: str | Path) -> pd.DataFrame:  # noqa: D401
    """读取 Bio-Logic MPR 文件（需要 pympr 库）。"""
    try:
        from pympr import MPRfile  # type: ignore
    except ImportError as e:
        raise ImportError(
            "请先安装 pympr (`pip install pympr`) 才能解析 .mpr 文件"
        ) from e
    mpr = MPRfile(Path(path))
    rec = mpr.to_pandas()
    # Bio-Logic 默认列名: ['time/s', 'Ewe/V', 'I/mA', 'Temp/°C'] 等
    col_map = {
        "time/s": "time",
        "I/mA": "current",  # 注意单位 mA
        "Ewe/V": "voltage",
        "Temp/°C": "temperature",
    }
    rec = rec.rename(columns=col_map)
    if "current" in rec.columns:
        rec["current"] /= 1000  # 转 A
    keep = [c for c in ["time", "current", "voltage", "temperature"] if c in rec.columns]
    return rec[keep].copy() 