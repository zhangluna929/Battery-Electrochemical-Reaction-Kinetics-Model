"""io.loader
统一的 YAML/JSON 参数加载接口。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml

__all__ = ["load_parameters"]


def load_parameters(path: str | Path) -> Dict[str, Any]:
    """加载参数或材料文件。

    用法支持三种形式：

    1. 绝对/相对文件路径（*.yml/*.yaml/*.json）
    2. 仅文件名（会在当前工作目录查找）
    3. 材料库名称（例如 ``LLZO``），会在 ``battery_kinetics/data/materials``
       中搜索 ``<name>.yml``。
    """

    path = Path(path)

    # 情况 3：材料/电极/电解质库快捷名
    if not path.exists() and path.suffix == "":
        base_dir = Path(__file__).resolve().parent.parent / "data"
        for sub in ("materials", "electrodes", "electrolytes"):
            candidate = base_dir / sub / f"{path.name}.yml"
            if candidate.exists():
                path = candidate
                break

    # 再次检查存在性
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix in {".yml", ".yaml"}:
        with path.open("r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
    elif path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            params = json.load(f)
    else:
        raise ValueError(f"Unsupported file suffix: {path.suffix}")

    return params


# -----------------------------------------------------------------------------
# 材料/电极/电解质快捷加载
# -----------------------------------------------------------------------------


def load_material(name: str) -> Dict[str, Any]:
    """根据名称在 data 子目录中搜索 YAML 并返回参数字典。

    搜索顺序：materials → electrodes → electrolytes。
    若有重名会以首次找到为准。
    """
    base_dir = Path(__file__).resolve().parent.parent / "data"
    for sub in ("materials", "electrodes", "electrolytes"):
        fp = base_dir / sub / f"{name}.yml"
        if fp.exists():
            return load_parameters(fp)
    raise FileNotFoundError(f"Material '{name}' not found in database") 