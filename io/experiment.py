"""io.experiment

实验数据读取器，支持常见电池测试设备格式。
返回统一的 pandas.DataFrame，包含列：time, current, voltage, temperature。
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_arbin_csv(path: str | Path) -> pd.DataFrame:
    """Reads a CSV file exported from Arbin battery testing equipment."""
    df = pd.read_csv(path, low_memory=False)
    col_map = {
        "Current(A)": "current",
        "Voltage(V)": "voltage",
        "Temperature(C)": "temperature",
        "Test_Time(s)": "time",
        "Time(s)": "time",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    keep = [c for c in ["time", "current", "voltage", "temperature"] if c in df.columns]
    return df[keep].copy()


def read_biologic_mpr(path: str | Path) -> pd.DataFrame:
    """Reads an MPR file from Bio-Logic equipment (requires `pympr`)."""
    try:
        from pympr import MPRfile  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Reading .mpr files requires the 'pympr' library. Please install it with `pip install pympr`."
        ) from e
    mpr = MPRfile(Path(path))
    rec = mpr.to_pandas()
    col_map = {
        "time/s": "time",
        "I/mA": "current",
        "Ewe/V": "voltage",
        "Temp/°C": "temperature",
    }
    rec = rec.rename(columns=col_map)
    if "current" in rec.columns:
        rec["current"] /= 1000  # Convert mA to A
    keep = [c for c in ["time", "current", "voltage", "temperature"] if c in rec.columns]
    return rec[keep].copy() 