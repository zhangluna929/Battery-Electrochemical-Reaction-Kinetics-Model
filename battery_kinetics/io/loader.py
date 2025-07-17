"""io.loader
统一的 YAML/JSON 参数加载接口。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import yaml

__all__ = ["load_parameters", "load_material"]


def load_parameters(path: str | Path) -> Dict[str, Any]:
    """Loads parameters from a YAML or JSON file."""
    path = Path(path)

    if not path.exists() and path.suffix == "":
        base_dir = Path(__file__).resolve().parent.parent / "data"
        for sub in ("materials", "electrodes", "electrolytes"):
            candidate = base_dir / sub / f"{path.name}.yml"
            if candidate.exists():
                path = candidate
                break

    if not path.exists():
        raise FileNotFoundError(f"Parameter file not found: {path}")

    if path.suffix in {".yml", ".yaml"}:
        with path.open("r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
    elif path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            params = json.load(f)
    else:
        raise ValueError(f"Unsupported file suffix: {path.suffix}")

    return params


def load_material(name: str) -> Dict[str, Any]:
    """Loads a material, electrode, or electrolyte file from the internal database."""
    return load_parameters(name) 